"""
PyTorch Dataset for circuit graphs with multi-modal features.

This dataset loads circuits from the pickle file and converts them to
PyTorch Geometric Data objects with additional features for training
the GraphVAE.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add tools to path for imports
sys.path.insert(0, 'tools')


class CircuitDataset(Dataset):
    """
    PyTorch Dataset for circuit graphs with multi-modal features.

    Returns per circuit:
        - graph: PyG Data object with node/edge features
        - poles: Variable-length tensor of complex poles
        - zeros: Variable-length tensor of complex zeros
        - gain: Scalar gain value
        - freq_response: [701, 2] tensor (magnitude, phase)
        - filter_type: One-hot [6] tensor
        - circuit_id: String ID
        - ged_neighbors: (Optional) k-nearest neighbor indices
    """

    # Filter type to index mapping
    FILTER_TYPES = [
        'low_pass',
        'high_pass',
        'band_pass',
        'band_stop',
        'rlc_series',
        'rlc_parallel'
    ]

    def __init__(
        self,
        dataset_path: str = 'rlc_dataset/filter_dataset.pkl',
        ged_matrix_path: Optional[str] = None,
        k_neighbors: int = 5,
        normalize_features: bool = True,
        log_scale_impedance: bool = True
    ):
        """
        Initialize the circuit dataset.

        Args:
            dataset_path: Path to the pickle file with circuits
            ged_matrix_path: Optional path to precomputed GED matrix
            k_neighbors: Number of nearest neighbors to include
            normalize_features: Whether to normalize node/edge features
            log_scale_impedance: Whether to log-scale impedance features
        """
        self.dataset_path = dataset_path
        self.k_neighbors = k_neighbors
        self.normalize_features = normalize_features
        self.log_scale_impedance = log_scale_impedance

        # Load circuit data
        with open(dataset_path, 'rb') as f:
            self.circuits = pickle.load(f)

        print(f"Loaded {len(self.circuits)} circuits from {dataset_path}")

        # Load GED matrix if provided
        self.ged_matrix = None
        if ged_matrix_path and Path(ged_matrix_path).exists():
            self.ged_matrix = np.load(ged_matrix_path)
            print(f"Loaded GED matrix from {ged_matrix_path}")

        # Compute normalization statistics if needed
        if normalize_features:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean/std for feature normalization."""
        # Collect all impedance features
        all_impedances = []

        for circuit in self.circuits:
            for neighbors in circuit['graph_adj']['adjacency']:
                for edge in neighbors:
                    imp_den = edge['impedance_den']
                    all_impedances.append(imp_den)

        all_impedances = np.array(all_impedances)  # Shape: [N, 3]

        if self.log_scale_impedance:
            # Add small epsilon to avoid log(0)
            all_impedances = np.log(all_impedances + 1e-15)

        self.impedance_mean = torch.tensor(all_impedances.mean(axis=0), dtype=torch.float32)
        self.impedance_std = torch.tensor(all_impedances.std(axis=0) + 1e-8, dtype=torch.float32)

        print(f"Impedance normalization:")
        print(f"  Mean: {self.impedance_mean.numpy()}")
        print(f"  Std:  {self.impedance_std.numpy()}")

        # Collect pole/zero magnitudes for normalization
        all_pole_mags = []
        all_zero_mags = []

        for circuit in self.circuits:
            if 'label' in circuit and circuit['label'] is not None:
                poles = circuit['label'].get('poles', [])
                zeros = circuit['label'].get('zeros', [])

                for p in poles:
                    all_pole_mags.append(abs(p))
                for z in zeros:
                    all_zero_mags.append(abs(z))

        # Compute log-scale statistics for poles/zeros
        if all_pole_mags:
            log_pole_mags = np.log(np.array(all_pole_mags) + 1e-8)
            self.log_pole_mean = float(log_pole_mags.mean())
            self.log_pole_std = float(log_pole_mags.std() + 1e-8)
        else:
            self.log_pole_mean = 0.0
            self.log_pole_std = 1.0

        if all_zero_mags:
            log_zero_mags = np.log(np.array(all_zero_mags) + 1e-8)
            self.log_zero_mean = float(log_zero_mags.mean())
            self.log_zero_std = float(log_zero_mags.std() + 1e-8)
        else:
            self.log_zero_mean = 0.0
            self.log_zero_std = 1.0

        print(f"Pole/Zero normalization (log-scale magnitudes):")
        print(f"  Pole: mean={self.log_pole_mean:.2f}, std={self.log_pole_std:.2f}")
        print(f"  Zero: mean={self.log_zero_mean:.2f}, std={self.log_zero_std:.2f}")

    def get_normalization_stats(self) -> Dict:
        """Get normalization statistics for denormalization."""
        return {
            'log_pole_mean': self.log_pole_mean if hasattr(self, 'log_pole_mean') else 0.0,
            'log_pole_std': self.log_pole_std if hasattr(self, 'log_pole_std') else 1.0,
            'log_zero_mean': self.log_zero_mean if hasattr(self, 'log_zero_mean') else 0.0,
            'log_zero_std': self.log_zero_std if hasattr(self, 'log_zero_std') else 1.0
        }

    @staticmethod
    def denormalize_poles_zeros(
        poles_zeros: np.ndarray,
        log_mean: float,
        log_std: float
    ) -> np.ndarray:
        """
        Denormalize poles or zeros back to original scale.

        Args:
            poles_zeros: Normalized [real, imag] pairs [N, 2]
            log_mean: Mean of log-magnitudes
            log_std: Std of log-magnitudes

        Returns:
            Denormalized [real, imag] pairs [N, 2]
        """
        if len(poles_zeros) == 0:
            return poles_zeros

        # Get magnitude and phase
        mags = np.sqrt((poles_zeros**2).sum(axis=-1))
        phases = np.arctan2(poles_zeros[:, 1], poles_zeros[:, 0])

        # Denormalize magnitude
        # normalized_log_mag = (log_mag - mean) / std
        # log_mag = normalized_log_mag * std + mean
        # mag = exp(log_mag)

        # Get log of normalized magnitude
        log_norm_mags = np.log(mags + 1e-8)

        # Denormalize in log-space
        log_mags = log_norm_mags * log_std + log_mean

        # Convert back to linear scale
        denorm_mags = np.exp(log_mags)

        # Reconstruct complex numbers
        denormalized = np.stack([
            denorm_mags * np.cos(phases),
            denorm_mags * np.sin(phases)
        ], axis=-1)

        return denormalized

    def __len__(self) -> int:
        """Return number of circuits in dataset."""
        return len(self.circuits)

    def _convert_graph_to_pyg(self, graph_adj: Dict) -> Data:
        """
        Convert graph adjacency dict to PyTorch Geometric Data object.

        Args:
            graph_adj: Dictionary with 'nodes' and 'adjacency' keys

        Returns:
            PyG Data object with x (node features) and edge_index/edge_attr
        """
        nodes = graph_adj['nodes']
        adjacency = graph_adj['adjacency']

        # Node features: 4D one-hot [is_GND, is_VIN, is_VOUT, is_INTERNAL]
        node_features = []
        for node in nodes:
            node_features.append(node['features'])

        x = torch.tensor(node_features, dtype=torch.float32)

        # Edge index and edge features
        edge_index = []
        edge_attr = []

        for source_id, neighbors in enumerate(adjacency):
            for neighbor in neighbors:
                target_id = neighbor['id']
                edge_index.append([source_id, target_id])

                # Edge features: impedance_den = [C, G, L_inv]
                imp_den = neighbor['impedance_den']

                if self.log_scale_impedance:
                    # Log scale with small epsilon
                    imp_den = np.log(np.array(imp_den) + 1e-15)

                imp_den = torch.tensor(imp_den, dtype=torch.float32)

                # Normalize if requested
                if self.normalize_features:
                    imp_den = (imp_den - self.impedance_mean) / self.impedance_std

                edge_attr.append(imp_den)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _encode_filter_type(self, filter_type: str) -> torch.Tensor:
        """Convert filter type string to one-hot encoding."""
        one_hot = torch.zeros(len(self.FILTER_TYPES), dtype=torch.float32)

        if filter_type in self.FILTER_TYPES:
            idx = self.FILTER_TYPES.index(filter_type)
            one_hot[idx] = 1.0
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return one_hot

    def _get_ged_neighbors(self, idx: int) -> Optional[torch.Tensor]:
        """Get k-nearest neighbor indices based on GED."""
        if self.ged_matrix is None:
            return None

        # Get distances for this circuit
        distances = self.ged_matrix[idx]

        # Get k+1 nearest (including self), then exclude self
        neighbor_indices = np.argsort(distances)[:self.k_neighbors + 1]
        neighbor_indices = neighbor_indices[neighbor_indices != idx][:self.k_neighbors]

        return torch.tensor(neighbor_indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single circuit from the dataset.

        Args:
            idx: Index of circuit to retrieve

        Returns:
            Dictionary with:
                - graph: PyG Data object
                - poles: Complex tensor of poles
                - zeros: Complex tensor of zeros
                - gain: Scalar gain
                - freq_response: [701, 2] magnitude and phase
                - filter_type: One-hot [6]
                - circuit_id: String ID
                - ged_neighbors: (Optional) k-nearest indices
        """
        circuit = self.circuits[idx]

        # Convert graph to PyG format
        graph = self._convert_graph_to_pyg(circuit['graph_adj'])

        # Extract poles and zeros (complex numbers)
        poles = circuit['label']['poles']
        zeros = circuit['label']['zeros']
        gain = circuit['label']['gain']

        # Convert to PyTorch tensors with normalization
        # Normalize poles/zeros by log-scaling their magnitudes
        if len(poles) > 0:
            poles_list = []
            for p in poles:
                mag = abs(p)
                phase = np.angle(p)
                # Normalize magnitude in log-space
                if self.normalize_features and mag > 1e-10:
                    log_mag = np.log(mag + 1e-8)
                    norm_log_mag = (log_mag - self.log_pole_mean) / self.log_pole_std
                    norm_mag = np.exp(norm_log_mag)
                else:
                    norm_mag = mag
                # Reconstruct as [real, imag]
                poles_list.append([norm_mag * np.cos(phase), norm_mag * np.sin(phase)])
            poles_tensor = torch.tensor(poles_list, dtype=torch.float32)
        else:
            poles_tensor = torch.zeros(0, 2, dtype=torch.float32)

        if len(zeros) > 0:
            zeros_list = []
            for z in zeros:
                mag = abs(z)
                phase = np.angle(z)
                # Normalize magnitude in log-space
                if self.normalize_features and mag > 1e-10:
                    log_mag = np.log(mag + 1e-8)
                    norm_log_mag = (log_mag - self.log_zero_mean) / self.log_zero_std
                    norm_mag = np.exp(norm_log_mag)
                else:
                    norm_mag = mag
                # Reconstruct as [real, imag]
                zeros_list.append([norm_mag * np.cos(phase), norm_mag * np.sin(phase)])
            zeros_tensor = torch.tensor(zeros_list, dtype=torch.float32)
        else:
            zeros_tensor = torch.zeros(0, 2, dtype=torch.float32)

        gain_tensor = torch.tensor([gain], dtype=torch.float32)

        # Frequency response: combine magnitude and phase
        freq_resp = circuit['frequency_response']
        freq_response = torch.tensor(
            np.stack([freq_resp['H_magnitude'], freq_resp['H_phase']], axis=1),
            dtype=torch.float32
        )  # Shape: [701, 2]

        # Filter type one-hot
        filter_type = self._encode_filter_type(circuit['filter_type'])

        # Get GED neighbors if available
        ged_neighbors = self._get_ged_neighbors(idx)

        return {
            'graph': graph,
            'poles': poles_tensor,
            'zeros': zeros_tensor,
            'gain': gain_tensor,
            'freq_response': freq_response,
            'filter_type': filter_type,
            'circuit_id': circuit['id'],
            'ged_neighbors': ged_neighbors,
            'idx': idx
        }

    def get_filter_type_indices(self, filter_type: str) -> List[int]:
        """
        Get all indices of circuits with a specific filter type.

        Args:
            filter_type: One of FILTER_TYPES

        Returns:
            List of indices
        """
        indices = []
        for idx, circuit in enumerate(self.circuits):
            if circuit['filter_type'] == filter_type:
                indices.append(idx)
        return indices

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Get stratified train/val/test split indices.

        Ensures each filter type is represented in all splits.

        Args:
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            seed: Random seed for reproducibility

        Returns:
            (train_indices, val_indices, test_indices)
        """
        np.random.seed(seed)

        train_indices = []
        val_indices = []
        test_indices = []

        # Split each filter type separately (stratified)
        for filter_type in self.FILTER_TYPES:
            indices = self.get_filter_type_indices(filter_type)
            np.random.shuffle(indices)

            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        return train_indices, val_indices, test_indices

    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*70)
        print("CIRCUIT DATASET STATISTICS")
        print("="*70)

        print(f"\nTotal circuits: {len(self.circuits)}")

        # Filter type distribution
        print(f"\nFilter type distribution:")
        for ftype in self.FILTER_TYPES:
            count = len(self.get_filter_type_indices(ftype))
            print(f"  {ftype:<15}: {count:>3} circuits")

        # Graph statistics
        num_nodes = [len(c['graph_adj']['nodes']) for c in self.circuits]
        num_edges = [sum(len(neighbors) for neighbors in c['graph_adj']['adjacency'])
                     for c in self.circuits]

        print(f"\nGraph structure:")
        print(f"  Nodes:  min={min(num_nodes)}, max={max(num_nodes)}, "
              f"mean={np.mean(num_nodes):.1f}")
        print(f"  Edges:  min={min(num_edges)}, max={max(num_edges)}, "
              f"mean={np.mean(num_edges):.1f}")

        # Pole/zero statistics
        num_poles = [len(c['label']['poles']) for c in self.circuits]
        num_zeros = [len(c['label']['zeros']) for c in self.circuits]

        print(f"\nTransfer function:")
        print(f"  Poles:  min={min(num_poles)}, max={max(num_poles)}, "
              f"mean={np.mean(num_poles):.1f}")
        print(f"  Zeros:  min={min(num_zeros)}, max={max(num_zeros)}, "
              f"mean={np.mean(num_zeros):.1f}")

        print()


def collate_circuit_batch(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for DataLoader.

    Handles variable-length poles/zeros by keeping them as lists.
    Batches graphs using PyG batching.

    Args:
        batch: List of dataset items

    Returns:
        Batched dictionary
    """
    from torch_geometric.data import Batch

    # Batch graphs using PyG
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    # Keep variable-length data as lists
    poles = [item['poles'] for item in batch]
    zeros = [item['zeros'] for item in batch]

    # Stack fixed-size tensors
    gains = torch.stack([item['gain'] for item in batch])
    freq_responses = torch.stack([item['freq_response'] for item in batch])
    filter_types = torch.stack([item['filter_type'] for item in batch])

    # Keep circuit IDs as list
    circuit_ids = [item['circuit_id'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch], dtype=torch.long)

    # GED neighbors (if available)
    ged_neighbors = [item['ged_neighbors'] for item in batch]
    if ged_neighbors[0] is not None:
        ged_neighbors = torch.stack(ged_neighbors)
    else:
        ged_neighbors = None

    return {
        'graph': batched_graph,
        'poles': poles,
        'zeros': zeros,
        'gain': gains,
        'freq_response': freq_responses,
        'filter_type': filter_types,
        'circuit_id': circuit_ids,
        'ged_neighbors': ged_neighbors,
        'idx': indices
    }
