"""
Yubo dataset for auxiliary pole prediction and filter classification.

Wraps CircuitDataset to return per-circuit:
    - graph:              PyG Data (node features [N,4], edge_attr [E,3])
    - pole_real:          signed-log dominant pole real part (scalar)
    - pole_imag:          signed-log dominant pole imag part (scalar)
    - filter_type_label:  integer class index 0-7
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import List

from ml.data.dataset import CircuitDataset


class YuboDataset(Dataset):
    """
    Dataset of circuit graphs with pole and filter-type targets.

    Wraps CircuitDataset and exposes only the fields needed for the
    auxiliary-head experiment: graph, pole_real, pole_imag, filter_type_label.

    Args:
        dataset_path:     Path to filter_dataset.pkl
        circuit_indices:  List of circuit indices to include (train or val split)
    """

    FILTER_TYPES = CircuitDataset.FILTER_TYPES  # 8 types

    def __init__(self, dataset_path: str, circuit_indices: List[int]):
        # normalize_features=False: we only use pz_target (sign-log scaled),
        # which is independent of the magnitude normalisation flag.
        self._base = CircuitDataset(dataset_path, normalize_features=False)
        self.circuit_indices = circuit_indices

    def __len__(self) -> int:
        return len(self.circuit_indices)

    def __getitem__(self, idx: int):
        item = self._base[self.circuit_indices[idx]]

        graph = item['graph']

        # pz_target = [sigma_p, omega_p, sigma_z, omega_z] in signed-log scale
        # We only supervise on the dominant pole: pz_target[0:2]
        pz = item['pz_target']
        pole_real = pz[0]
        pole_imag = pz[1]

        # filter_type is one-hot [8]; convert to class index
        filter_type_label = item['filter_type'].argmax().long()

        return graph, pole_real, pole_imag, filter_type_label

    def get_train_val_split(self, train_ratio: float = 0.8, seed: int = 42):
        """
        Stratified split of circuit_indices into train / val subsets.

        Returns (train_indices, val_indices) — both are lists of raw
        dataset indices suitable for constructing new YuboDataset objects.
        """
        import numpy as np
        np.random.seed(seed)

        train_out, val_out = [], []
        circuits = self._base.circuits

        for ftype in self.FILTER_TYPES:
            type_indices = [
                i for i in self.circuit_indices
                if circuits[i]['filter_type'] == ftype
            ]
            np.random.shuffle(type_indices)
            n_train = int(len(type_indices) * train_ratio)
            train_out.extend(type_indices[:n_train])
            val_out.extend(type_indices[n_train:])

        return train_out, val_out


def collate_yubo(batch):
    """Collate (graph, pole_real, pole_imag, filter_type_label) tuples."""
    graphs, pole_reals, pole_imags, labels = zip(*batch)
    batched = Batch.from_data_list(graphs)
    pole_real = torch.stack(pole_reals)
    pole_imag = torch.stack(pole_imags)
    labels = torch.stack(labels)
    return batched, pole_real, pole_imag, labels
