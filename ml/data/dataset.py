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
        """
        Compute mean/std for feature normalization with practical range clipping.

        This implements Option 3 fix for component value normalization:
        1. Clip to practical ranges BEFORE logging
        2. Use proper z-score normalization (mean=0, std=1)
        """
        # Collect all impedance features
        all_C = []
        all_G = []
        all_L_inv = []

        for circuit in self.circuits:
            for neighbors in circuit['graph_adj']['adjacency']:
                for edge in neighbors:
                    imp_den = edge['impedance_den']  # [C, G, L_inv]
                    C, G, L_inv = imp_den
                    all_C.append(C)
                    all_G.append(G)
                    all_L_inv.append(L_inv)

        all_C = np.array(all_C)
        all_G = np.array(all_G)
        all_L_inv = np.array(all_L_inv)

        if self.log_scale_impedance:
            # Option 3: Clip to practical ranges BEFORE logging
            # This prevents extreme values from polluting the normalization

            # Convert G and L_inv back to R and L for clipping
            # (avoiding division by zero with epsilon)
            all_R = 1.0 / (all_G + 1e-15)
            all_L = 1.0 / (all_L_inv + 1e-15)

            # Clip to practical component ranges
            # These ranges are based on common off-the-shelf components
            R_practical = np.clip(all_R, 10, 100e3)        # 10Ω to 100kΩ
            L_practical = np.clip(all_L, 1e-9, 10e-3)      # 1nH to 10mH
            C_practical = np.clip(all_C, 1e-12, 1e-6)      # 1pF to 1μF

            # Convert back to G and L_inv
            G_practical = 1.0 / R_practical
            L_inv_practical = 1.0 / L_practical

            # Log transform
            log_C = np.log(C_practical + 1e-15)
            log_G = np.log(G_practical + 1e-15)
            log_L_inv = np.log(L_inv_practical + 1e-15)

            # Z-score normalization (centered at 0, std=1)
            # This is the key fix - previous normalization didn't center at 0
            C_mean = log_C.mean()
            C_std = log_C.std() + 1e-8
            G_mean = log_G.mean()
            G_std = log_G.std() + 1e-8
            L_inv_mean = log_L_inv.mean()
            L_inv_std = log_L_inv.std() + 1e-8

            # Store as tensors
            self.impedance_mean = torch.tensor([C_mean, G_mean, L_inv_mean], dtype=torch.float32)
            self.impedance_std = torch.tensor([C_std, G_std, L_inv_std], dtype=torch.float32)

            print(f"Impedance normalization (with practical range clipping):")
            print(f"  C:     mean={C_mean:.3f}, std={C_std:.3f}")
            print(f"  G:     mean={G_mean:.3f}, std={G_std:.3f}")
            print(f"  L_inv: mean={L_inv_mean:.3f}, std={L_inv_std:.3f}")

            # Print practical ranges for reference
            print(f"\nPractical component ranges:")
            print(f"  R: 10Ω to 100kΩ")
            print(f"  L: 1nH to 10mH")
            print(f"  C: 1pF to 1μF")
        else:
            # If not log-scaling, just use simple mean/std
            all_impedances = np.stack([all_C, all_G, all_L_inv], axis=1)
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

                # Create binary masks indicating which components are present
                # Original values before log-scaling for accurate thresholding
                C, G, L_inv = imp_den
                has_C = 1.0 if C > 1e-12 else 0.0
                has_R = 1.0 if G > 1e-12 else 0.0  # G = 1/R
                has_L = 1.0 if L_inv > 1e-12 else 0.0  # L_inv = 1/L
                is_parallel = 1.0  # All components between two nodes are parallel

                if self.log_scale_impedance:
                    # Apply same practical range clipping as in normalization (Option 3 fix)
                    # Convert G and L_inv to R and L
                    R = 1.0 / (G + 1e-15)
                    L = 1.0 / (L_inv + 1e-15)

                    # Clip to practical ranges
                    R_practical = np.clip(R, 10, 100e3)        # 10Ω to 100kΩ
                    L_practical = np.clip(L, 1e-9, 10e-3)      # 1nH to 10mH
                    C_practical = np.clip(C, 1e-12, 1e-6)      # 1pF to 1μF

                    # Convert back to G and L_inv
                    G_practical = 1.0 / R_practical
                    L_inv_practical = 1.0 / L_practical

                    # Log transform
                    imp_den = np.log(np.array([C_practical, G_practical, L_inv_practical]) + 1e-15)

                imp_den = torch.tensor(imp_den, dtype=torch.float32)

                # Normalize if requested (z-score normalization)
                if self.normalize_features:
                    imp_den = (imp_den - self.impedance_mean) / self.impedance_std

                # Concatenate continuous features + binary masks
                # Result: [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
                binary_masks = torch.tensor([has_C, has_R, has_L, is_parallel], dtype=torch.float32)
                edge_features = torch.cat([imp_den, binary_masks], dim=0)

                edge_attr.append(edge_features)

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
                - poles: Complex tensor of poles [num_poles, 2]
                - zeros: Complex tensor of zeros [num_zeros, 2]
                - num_poles: Number of poles (scalar)
                - num_zeros: Number of zeros (scalar)
                - gain: Scalar gain
                - freq_response: [701, 2] magnitude and phase
                - filter_type: One-hot [6]
                - circuit_id: String ID
                - ged_neighbors: (Optional) k-nearest indices
                - idx: Dataset index
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

        # Add pole/zero counts for variable-length decoder
        num_poles = len(poles)
        num_zeros = len(zeros)

        return {
            'graph': graph,
            'poles': poles_tensor,
            'zeros': zeros_tensor,
            'num_poles': torch.tensor(num_poles, dtype=torch.long),
            'num_zeros': torch.tensor(num_zeros, dtype=torch.long),
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
    num_poles = torch.stack([item['num_poles'] for item in batch])
    num_zeros = torch.stack([item['num_zeros'] for item in batch])

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
        'num_poles': num_poles,
        'num_zeros': num_zeros,
        'gain': gains,
        'freq_response': freq_responses,
        'filter_type': filter_types,
        'circuit_id': circuit_ids,
        'ged_neighbors': ged_neighbors,
        'idx': indices
    }


def collate_graphgpt_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for GraphGPT training.

    Converts PyG Data objects into the format expected by GraphGPT decoder:
    - Node types as integers [batch, max_nodes]
    - Edge existence as adjacency matrix [batch, max_nodes, max_nodes]
    - Edge values as matrix [batch, max_nodes, max_nodes, 7]
    - Pole/zero counts and values as tensors

    Args:
        batch: List of dataset items

    Returns:
        Dictionary with GraphGPT-compatible format
    """
    from torch_geometric.data import Batch

    batch_size = len(batch)

    # Always use fixed max_nodes (for GraphGPT decoder)
    # The decoder expects exactly 5 nodes
    max_nodes = 5

    # Initialize tensors
    node_features_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_idx_list = []

    node_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    edge_existence = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
    edge_values = torch.zeros(batch_size, max_nodes, max_nodes, 7, dtype=torch.float32)

    # Extract specifications (cutoff frequency, Q-factor) from filter_type
    # For now, use dummy values - should be extracted from actual circuit specs
    specifications = torch.zeros(batch_size, 2, dtype=torch.float32)

    for b, item in enumerate(batch):
        graph = item['graph']
        num_nodes = graph.x.shape[0]

        # Convert node features (one-hot) to node types
        # graph.x is [num_nodes, 4] one-hot: [is_GND, is_VIN, is_VOUT, is_INTERNAL]
        node_type_indices = torch.argmax(graph.x, dim=1)  # [num_nodes]
        node_types[b, :num_nodes] = node_type_indices
        # Pad with MASK token (index 4)
        node_types[b, num_nodes:] = 4

        # Store for encoder (PyG format)
        node_features_list.append(graph.x)
        edge_index_list.append(graph.edge_index)
        edge_attr_list.append(graph.edge_attr)
        batch_idx_list.extend([b] * num_nodes)

        # Convert edge_index to adjacency matrix
        # graph.edge_index is [2, num_edges]
        # graph.edge_attr is [num_edges, 7]: [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
        for edge_idx in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, edge_idx].item()
            dst = graph.edge_index[1, edge_idx].item()

            edge_existence[b, src, dst] = 1.0
            edge_values[b, src, dst] = graph.edge_attr[edge_idx]

    # Batch graphs for encoder (PyG format)
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    batch_idx = torch.tensor(batch_idx_list, dtype=torch.long)

    # Keep poles and zeros as lists for encoder
    poles_list = [item['poles'] for item in batch]
    zeros_list = [item['zeros'] for item in batch]

    # Also create padded versions for decoder
    # Use fixed max values from decoder config (max_poles=4, max_zeros=4)
    max_poles = 4
    max_zeros = 4

    pole_count = torch.stack([item['num_poles'] for item in batch])
    zero_count = torch.stack([item['num_zeros'] for item in batch])

    pole_values = torch.zeros(batch_size, max_poles, 2, dtype=torch.float32)
    zero_values = torch.zeros(batch_size, max_zeros, 2, dtype=torch.float32)

    for b, item in enumerate(batch):
        n_poles = item['num_poles'].item()
        n_zeros = item['num_zeros'].item()

        if n_poles > 0:
            pole_values[b, :n_poles] = item['poles']
        if n_zeros > 0:
            zero_values[b, :n_zeros] = item['zeros']

    # Extract specifications from filter_type
    # Use dummy values for now: cutoff=1000Hz, Q=0.707
    # These should ideally come from the circuit's actual specifications
    for b, item in enumerate(batch):
        # Normalize: log(cutoff)/4.0, log(Q)/2.0
        specifications[b, 0] = np.log10(1000.0) / 4.0  # Normalized cutoff
        specifications[b, 1] = np.log10(0.707) / 2.0   # Normalized Q

    return {
        # For encoder (PyG format)
        'node_features': batched_graph.x,
        'edge_index': batched_graph.edge_index,
        'edge_attr': batched_graph.edge_attr,
        'batch_idx': batched_graph.batch,
        'poles_list': poles_list,
        'zeros_list': zeros_list,

        # For decoder (GraphGPT format)
        'node_types': node_types,
        'edge_existence': edge_existence,
        'edge_values': edge_values,
        'specifications': specifications,

        # For loss computation
        'pole_count': pole_count,
        'zero_count': zero_count,
        'pole_values': pole_values,
        'zero_values': zero_values,
    }
