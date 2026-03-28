"""
Sequence dataset for training the GPT-style circuit decoder.

Loads circuits from the existing pickle dataset, converts each to a
bipartite graph, pre-computes exhaustive Eulerian circuit walks for
augmentation, and tokenizes them.
"""

from __future__ import annotations

import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.bipartite_graph import from_pickle_circuit
from ml.data.traversal import enumerate_euler_circuits, hierholzer
from ml.models.vocabulary import CircuitVocabulary


class SequenceDataset(Dataset):
    """
    Dataset of (circuit_graph, walk_token_ids) pairs.

    Each item returns:
        - graph: PyG Data object (for the encoder, same format as CircuitDataset)
        - seq: LongTensor of token IDs [L] (the Eulerian circuit walk)
        - seq_len: int (un-padded length)

    When augment=True, pre-computes all distinct Euler circuits per circuit
    (exhaustive enumeration) and samples uniformly from them at each
    __getitem__ call.

    Args:
        dataset_path: Path to filter_dataset.pkl.
        circuit_indices: List of indices into the pickle list.
        vocab: CircuitVocabulary instance.
        augment: If True, pre-compute exhaustive walks and sample from them.
        max_seq_len: Maximum sequence length (for padding). Default 64.
        max_walks_per_circuit: Cap on exhaustive enumeration per circuit.
    """

    def __init__(
        self,
        dataset_path: str,
        circuit_indices: List[int],
        vocab: CircuitVocabulary,
        augment: bool = True,
        max_seq_len: int = 64,
        max_walks_per_circuit: int = 200,
    ):
        with open(dataset_path, 'rb') as f:
            all_circuits = pickle.load(f)

        self.circuits = all_circuits
        self.circuit_indices = circuit_indices
        self.vocab = vocab
        self.augment = augment
        self.max_seq_len = max_seq_len

        # Pre-build bipartite graphs, PyG graphs, and exhaustive walks
        self.bipartite_graphs = []
        self.pyg_graphs = []
        self.all_walks: List[List[List[str]]] = []  # per-circuit list of walks

        total_walks = 0
        for idx in circuit_indices:
            circuit = all_circuits[idx]
            bg = from_pickle_circuit(circuit)
            self.bipartite_graphs.append(bg)
            self.pyg_graphs.append(self._build_pyg_graph(circuit))

            if augment:
                walks = enumerate_euler_circuits(bg, max_circuits=max_walks_per_circuit)
                if not walks:
                    walks = [hierholzer(bg, start='VSS', rng=None)]
            else:
                walks = [hierholzer(bg, start='VSS', rng=None)]

            self.all_walks.append(walks)
            total_walks += len(walks)

        if augment:
            print(f"  Exhaustive augmentation: {total_walks} total walks "
                  f"for {len(circuit_indices)} circuits "
                  f"(avg {total_walks / len(circuit_indices):.1f}/circuit)")

    @staticmethod
    def _build_pyg_graph(circuit: dict):
        """Build PyG Data object from circuit dict (same as CircuitDataset)."""
        from torch_geometric.data import Data

        graph_adj = circuit['graph_adj']
        nodes = graph_adj['nodes']
        adjacency = graph_adj['adjacency']

        x = torch.tensor(
            [n['features'] for n in nodes], dtype=torch.float32
        )

        edge_list = []
        edge_attrs = []
        for src_id, neighbors in enumerate(adjacency):
            for nb in neighbors:
                tgt_id = nb['id']
                edge_list.append([src_id, tgt_id])
                C, G, L_inv = nb['impedance_den']
                log_R = np.log10(1.0 / G) if G > 1e-12 else 0.0
                log_C = np.log10(C) if C > 1e-12 else 0.0
                log_L = np.log10(1.0 / L_inv) if L_inv > 1e-12 else 0.0
                edge_attrs.append([log_R, log_C, log_L])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self) -> int:
        return len(self.circuit_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pyg_graph = self.pyg_graphs[idx]
        walks = self.all_walks[idx]

        # Pick a walk: random from pre-computed set if augmenting
        if self.augment and len(walks) > 1:
            walk = random.choice(walks)
        else:
            walk = walks[0]

        # Tokenize: walk + EOS
        token_ids = self.vocab.encode(walk + ['EOS'])
        seq_len = len(token_ids)

        # Pad to max_seq_len
        if seq_len < self.max_seq_len:
            token_ids = token_ids + [self.vocab.pad_id] * (self.max_seq_len - seq_len)
        elif seq_len > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            seq_len = self.max_seq_len

        seq = torch.tensor(token_ids, dtype=torch.long)

        return {
            'graph': pyg_graph,
            'seq': seq,           # [max_seq_len]
            'seq_len': seq_len,   # int (un-padded length)
        }


def collate_sequence_batch(
    batch: List[Dict],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for SequenceDataset.

    Returns:
        graph: Batched PyG graph (for encoder).
        seq: [B, max_seq_len] padded token IDs.
        seq_len: [B] un-padded lengths.
    """
    from torch_geometric.data import Batch

    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    seq = torch.stack([item['seq'] for item in batch])
    seq_len = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)

    return {
        'graph': batched_graph,
        'seq': seq,
        'seq_len': seq_len,
    }
