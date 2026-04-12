"""Cross-topology circuit dataset with Eulerian walk sequences.

Loads circuits from multiple pickle files, builds PyG graphs with
either legacy log10 or admittance-polynomial edge features, and
generates augmented Eulerian walk sequences for autoregressive training.

Supports two edge feature modes:
    'log10'      — [log10(R), log10(C), log10(L)], 0 = absent  (v1)
    'polynomial' — [G/G_ref, C/C_ref, L_inv/L_inv_ref]         (v2)
"""

import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from ml.data.bipartite_graph import from_pickle_circuit
from ml.data.traversal import augment_traversals, enumerate_euler_circuits, hierholzer
from ml.models.constants import G_REF, C_REF, L_INV_REF


class CrossTopoSequenceDataset(Dataset):
    """Load circuits from multiple pkl files, filter by type, with sequences."""

    def __init__(self, pkl_paths, allowed_types, vocab, indices=None,
                 augment=True, max_seq_len=32, max_walks=200,
                 edge_feature_mode='log10', n_augment_walks=32):
        """
        edge_feature_mode:
            'log10'      — legacy: [log10(R), log10(C), log10(L)], 0 = absent.
            'polynomial' — [G/G_ref, C/C_ref, L_inv/L_inv_ref] normalised
                            polynomial coefficients of Y(s). 0 = absent.
        n_augment_walks: Hierholzer-based augmentation target when `augment`
            is True. Up to `n_augment_walks` distinct traversals are stored
            per circuit; __getitem__ samples one uniformly.
        """
        assert edge_feature_mode in ('log10', 'polynomial')
        all_circuits = []
        for path in pkl_paths:
            with open(path, 'rb') as f:
                raw = pickle.load(f)
            for circ in raw:
                if circ['filter_type'] in allowed_types:
                    all_circuits.append(circ)

        if indices is not None:
            all_circuits = [all_circuits[i] for i in indices]

        self.circuits = all_circuits
        self.vocab = vocab
        self.augment = augment
        self.max_seq_len = max_seq_len
        self.edge_feature_mode = edge_feature_mode

        self.pyg_graphs = []
        self.all_walks = []
        self.behavior_targets = []
        self.gain_1k_targets = []

        for circ in all_circuits:
            self.pyg_graphs.append(self._build_pyg_graph(circ, edge_feature_mode))

            bg = from_pickle_circuit(circ)
            if augment:
                walks = augment_traversals(bg, n=n_augment_walks, start='VSS')
                if len(walks) < 8:
                    walks = enumerate_euler_circuits(bg, max_circuits=max_walks)
                    if not walks:
                        walks = [hierholzer(bg, start='VSS', rng=None)]
            else:
                walks = [hierholzer(bg, start='VSS', rng=None)]
            self.all_walks.append(walks)

            freq = circ['characteristic_frequency']
            self.behavior_targets.append(
                torch.tensor([np.log10(max(freq, 1.0))], dtype=torch.float32))

            gain_1k = self._compute_gain_at_freq(circ, 1000.0)
            self.gain_1k_targets.append(
                torch.tensor([gain_1k], dtype=torch.float32))

    @staticmethod
    def _compute_gain_at_freq(circ, freq_hz):
        """Compute |H(j*2*pi*freq_hz)| from poles/zeros/gain."""
        s = 1j * 2 * np.pi * freq_hz
        poles = circ['label']['poles']
        zeros = circ['label']['zeros']
        gain = circ['label']['gain']
        H = gain
        for z in zeros:
            H *= (s - z)
        for p in poles:
            H /= (s - p)
        return float(np.abs(H))

    @staticmethod
    def _compute_transfer_fn(circ, freqs_hz):
        """Return complex H(jw) at each frequency in `freqs_hz`."""
        freqs = np.asarray(freqs_hz, dtype=np.float64)
        s = 1j * 2 * np.pi * freqs
        poles = circ['label']['poles']
        zeros = circ['label']['zeros']
        gain = circ['label']['gain']
        H = np.full_like(s, gain, dtype=np.complex128)
        for z in zeros:
            H *= (s - z)
        for p in poles:
            H /= (s - p)
        return H

    @staticmethod
    def _build_pyg_graph(circ, edge_feature_mode='log10'):
        graph_adj = circ['graph_adj']
        x = torch.tensor([n['features'] for n in graph_adj['nodes']], dtype=torch.float32)
        edge_list, edge_attrs = [], []
        for src_id, neighbors in enumerate(graph_adj['adjacency']):
            for nb in neighbors:
                edge_list.append([src_id, nb['id']])
                C, G, L_inv = nb['impedance_den']
                if edge_feature_mode == 'log10':
                    log_R = np.log10(1.0 / G) if G > 1e-12 else 0.0
                    log_C = np.log10(C) if C > 1e-12 else 0.0
                    log_L = np.log10(1.0 / L_inv) if L_inv > 1e-12 else 0.0
                    edge_attrs.append([log_R, log_C, log_L])
                elif edge_feature_mode == 'polynomial':
                    g_n = (G / G_REF) if G > 1e-12 else 0.0
                    c_n = (C / C_REF) if C > 1e-12 else 0.0
                    l_n = (L_inv / L_INV_REF) if L_inv > 1e-12 else 0.0
                    edge_attrs.append([g_n, c_n, l_n])
                else:
                    raise ValueError(
                        f"unknown edge_feature_mode={edge_feature_mode!r}")
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, idx):
        walks = self.all_walks[idx]
        walk = random.choice(walks) if self.augment and len(walks) > 1 else walks[0]
        token_ids = self.vocab.encode(walk + ['EOS'])
        seq_len = len(token_ids)
        if seq_len < self.max_seq_len:
            token_ids += [self.vocab.pad_id] * (self.max_seq_len - seq_len)
        elif seq_len > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            seq_len = self.max_seq_len
        return {
            'graph': self.pyg_graphs[idx],
            'seq': torch.tensor(token_ids, dtype=torch.long),
            'seq_len': seq_len,
            'behavior': self.behavior_targets[idx],
            'gain_1k': self.gain_1k_targets[idx],
            'filter_type': self.circuits[idx]['filter_type'],
        }


def collate_fn(batch):
    """Collate CrossTopoSequenceDataset samples into a batch."""
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    seq = torch.stack([item['seq'] for item in batch])
    seq_len = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    behavior = torch.stack([item['behavior'] for item in batch]).squeeze(-1)
    gain_1k = torch.stack([item['gain_1k'] for item in batch]).squeeze(-1)
    filter_types = [item['filter_type'] for item in batch]
    return {
        'graph': graphs, 'seq': seq, 'seq_len': seq_len,
        'behavior': behavior, 'gain_1k': gain_1k, 'filter_types': filter_types,
    }
