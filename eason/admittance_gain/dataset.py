"""
Admittance dataset for gain prediction experiment.

Each circuit has 100 frequency points. This dataset expands to
(circuit, freq) pairs so each sample is a graph with complex admittance
[Re(Y), Im(Y)] edge features at a specific frequency, targeting the
gain |H(jw)| at that freq.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Dict


class AdmittanceDataset(Dataset):
    """
    Dataset of (circuit, frequency) pairs for gain prediction.

    Each item returns:
        - PyG Data with node features [N, 4] and edge_attr [E, 2] = [Re(Y), Im(Y)]
        - target: log10(|H(jw)|) clipped to [-10, 5]
        - filter_type_idx: integer filter type for analysis
    """

    FILTER_TYPES = [
        'low_pass', 'high_pass', 'band_pass', 'band_stop',
        'rlc_series', 'rlc_parallel', 'lc_lowpass', 'cl_highpass'
    ]

    def __init__(self, dataset_path: str, circuit_indices: List[int]):
        with open(dataset_path, 'rb') as f:
            all_circuits = pickle.load(f)

        self.circuit_indices = circuit_indices
        self.circuits = all_circuits
        self.num_freqs = len(all_circuits[0]['frequency_response']['freqs'])

    def __len__(self):
        return len(self.circuit_indices) * self.num_freqs

    def __getitem__(self, idx):
        circuit_idx = self.circuit_indices[idx // self.num_freqs]
        freq_idx = idx % self.num_freqs

        circuit = self.circuits[circuit_idx]
        fr = circuit['frequency_response']
        freq_hz = fr['freqs'][freq_idx]
        omega = 2.0 * np.pi * freq_hz

        graph_adj = circuit['graph_adj']
        nodes = graph_adj['nodes']
        adjacency = graph_adj['adjacency']

        # Node features: 4D one-hot [is_GND, is_VIN, is_VOUT, is_INTERNAL]
        x = torch.tensor([n['features'] for n in nodes], dtype=torch.float32)

        # Build edges with admittance magnitudes
        edge_list = []
        edge_attrs = []

        for source_id, neighbors in enumerate(adjacency):
            for neighbor in neighbors:
                target_id = neighbor['id']
                edge_list.append([source_id, target_id])

                # impedance_den = [C, G, L_inv]
                C, G, L_inv = neighbor['impedance_den']

                # Y(jw) = G + j(wC - L_inv/w)
                reactive = omega * C - L_inv / omega if omega > 0 else 0.0

                edge_attrs.append([G, reactive])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        # Target: log10(|H(jw)|) clipped
        H_mag = fr['H_magnitude'][freq_idx]
        target = np.clip(np.log10(H_mag + 1e-10), -10.0, 5.0)
        target = torch.tensor(target, dtype=torch.float32)

        # Filter type for analysis
        filter_type_idx = self.FILTER_TYPES.index(circuit['filter_type'])

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return graph, target, filter_type_idx


def collate_admittance(batch):
    """Collate (graph, target, filter_type_idx) tuples."""
    graphs, targets, ftypes = zip(*batch)
    batched = Batch.from_data_list(graphs)
    targets = torch.stack(targets)
    ftypes = torch.tensor(ftypes, dtype=torch.long)
    return batched, targets, ftypes
