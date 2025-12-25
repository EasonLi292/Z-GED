"""
Variable-length decoder for GraphVAE.

Extends HybridDecoder to support variable-length pole/zero predictions.
Key improvement: Predicts COUNT of poles/zeros, then decodes that many.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from torch_geometric.data import Data

from .decoder import FILTER_TYPES, CIRCUIT_TEMPLATES


class VariableLengthDecoder(nn.Module):
    """
    Hybrid decoder with variable-length pole/zero prediction.

    Improvements over HybridDecoder:
    1. Predicts number of poles/zeros (0-4)
    2. Predicts up to max_poles/max_zeros values
    3. Uses predicted count to mask invalid predictions
    4. Enables accurate transfer function reconstruction

    Args:
        latent_dim: Total latent dimension
        edge_feature_dim: Edge feature dimension
        hidden_dim: Hidden dimension for MLPs
        max_poles: Maximum number of poles to predict (default: 4)
        max_zeros: Maximum number of zeros to predict (default: 4)
        topo_latent_dim: Topology branch dimension
        values_latent_dim: Component values branch dimension
        pz_latent_dim: Poles/zeros branch dimension
    """

    def __init__(
        self,
        latent_dim: int = 8,
        edge_feature_dim: int = 7,
        hidden_dim: int = 128,
        max_nodes: int = 5,
        max_edges: int = 10,
        max_poles: int = 4,
        max_zeros: int = 4,
        dropout: float = 0.1,
        topo_latent_dim: Optional[int] = None,
        values_latent_dim: Optional[int] = None,
        pz_latent_dim: Optional[int] = None
    ):
        super().__init__()

        # Set branch dimensions
        if topo_latent_dim is None or values_latent_dim is None or pz_latent_dim is None:
            assert latent_dim % 3 == 0
            self.topo_latent_dim = latent_dim // 3
            self.values_latent_dim = latent_dim // 3
            self.pz_latent_dim = latent_dim // 3
        else:
            self.topo_latent_dim = topo_latent_dim
            self.values_latent_dim = values_latent_dim
            self.pz_latent_dim = pz_latent_dim
            assert topo_latent_dim + values_latent_dim + pz_latent_dim == latent_dim

        self.latent_dim = latent_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_poles = max_poles
        self.max_zeros = max_zeros
        self.dropout = dropout

        # Stage 1: Topology prediction (same as HybridDecoder)
        self.topo_head = nn.Sequential(
            nn.Linear(self.topo_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(FILTER_TYPES))
        )

        # Stage 2: Component value prediction (same as HybridDecoder)
        self.value_decoder = nn.Sequential(
            nn.Linear(self.values_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_edges * edge_feature_dim)
        )

        # Stage 3: Variable-length poles/zeros (NEW)
        # 3a: Count prediction heads
        self.pole_count_head = nn.Sequential(
            nn.Linear(self.pz_latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poles + 1)  # 0, 1, 2, 3, 4
        )

        self.zero_count_head = nn.Sequential(
            nn.Linear(self.pz_latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_zeros + 1)  # 0, 1, 2, 3, 4
        )

        # 3b: Value prediction heads (predict up to max_poles/zeros)
        self.pole_decoder = nn.Sequential(
            nn.Linear(self.pz_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_poles * 2)  # max_poles × [real, imag]
        )

        self.zero_decoder = nn.Sequential(
            nn.Linear(self.pz_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_zeros * 2)  # max_zeros × [real, imag]
        )

    def forward(
        self,
        z: torch.Tensor,
        hard: bool = False,
        gt_filter_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent vector to circuit representation.

        Args:
            z: Latent vector [B, latent_dim]
            hard: Use hard assignments (argmax) vs. soft (Gumbel-softmax)
            gt_filter_type: Ground truth filter type for teacher forcing [B, 6]

        Returns:
            Dictionary containing:
                - topo_logits: Topology logits [B, 6]
                - topo_probs: Topology probabilities [B, 6]
                - edge_features: Predicted edge features [B, max_edges, edge_feature_dim]
                - poles_all: All pole predictions [B, max_poles, 2]
                - zeros_all: All zero predictions [B, max_zeros, 2]
                - pole_count_logits: Pole count logits [B, max_poles+1]
                - zero_count_logits: Zero count logits [B, max_zeros+1]
                - num_poles: Predicted pole counts [B]
                - num_zeros: Predicted zero counts [B]
                - graphs: List of PyG Data objects (if hard=True)
        """
        batch_size = z.size(0)

        # Split latent vector into branches
        z_topo = z[:, :self.topo_latent_dim]
        z_values = z[:, self.topo_latent_dim:self.topo_latent_dim+self.values_latent_dim]
        z_pz = z[:, -self.pz_latent_dim:]

        # Stage 1: Topology prediction
        topo_logits = self.topo_head(z_topo)

        if gt_filter_type is not None:
            # Teacher forcing: use ground truth topology
            topo_probs = gt_filter_type
        else:
            if hard:
                # Hard assignment during generation
                topo_idx = topo_logits.argmax(dim=-1)
                topo_probs = F.one_hot(topo_idx, num_classes=len(FILTER_TYPES)).float()
            else:
                # Soft assignment during training (Gumbel-softmax)
                topo_probs = F.gumbel_softmax(topo_logits, tau=1.0, hard=False)

        # Stage 2: Component value prediction
        edge_features_flat = self.value_decoder(z_values)
        edge_features = edge_features_flat.view(batch_size, self.max_edges, self.edge_feature_dim)

        # Stage 3: Variable-length poles/zeros
        # 3a: Predict counts
        pole_count_logits = self.pole_count_head(z_pz)  # [B, max_poles+1]
        zero_count_logits = self.zero_count_head(z_pz)  # [B, max_zeros+1]

        if hard:
            num_poles = pole_count_logits.argmax(dim=-1)  # [B]
            num_zeros = zero_count_logits.argmax(dim=-1)  # [B]
        else:
            # During training: use soft counts for gradients
            pole_count_probs = F.softmax(pole_count_logits, dim=-1)
            zero_count_probs = F.softmax(zero_count_logits, dim=-1)

            # Expected count: E[count] = Σ i × P(count=i)
            counts_range = torch.arange(self.max_poles + 1, device=z.device, dtype=torch.float32)
            num_poles = (pole_count_probs * counts_range).sum(dim=-1)
            num_zeros = (zero_count_probs * counts_range).sum(dim=-1)

        # 3b: Predict values (up to max)
        poles_flat = self.pole_decoder(z_pz)
        zeros_flat = self.zero_decoder(z_pz)

        poles_all = poles_flat.view(batch_size, self.max_poles, 2)  # [B, max_poles, 2]
        zeros_all = zeros_flat.view(batch_size, self.max_zeros, 2)  # [B, max_zeros, 2]

        output = {
            'topo_logits': topo_logits,
            'topo_probs': topo_probs,
            'edge_features': edge_features,
            'poles_all': poles_all,
            'zeros_all': zeros_all,
            'pole_count_logits': pole_count_logits,
            'zero_count_logits': zero_count_logits,
            'num_poles': num_poles,
            'num_zeros': num_zeros,
        }

        # If hard sampling, construct actual graphs
        if hard or not self.training:
            # For hard mode, num_poles/num_zeros are integers
            # Extract valid poles/zeros for each sample
            poles_list = []
            zeros_list = []

            for i in range(batch_size):
                if isinstance(num_poles, torch.Tensor) and num_poles.dtype in [torch.long, torch.int]:
                    n_p = num_poles[i].item()
                    n_z = num_zeros[i].item()
                else:
                    # Soft counts during training
                    n_p = int(num_poles[i].round().item())
                    n_z = int(num_zeros[i].round().item())

                # Clamp to valid range
                n_p = max(0, min(n_p, self.max_poles))
                n_z = max(0, min(n_z, self.max_zeros))

                # Extract valid predictions
                poles_i = poles_all[i, :n_p] if n_p > 0 else torch.zeros(0, 2, device=z.device)
                zeros_i = zeros_all[i, :n_z] if n_z > 0 else torch.zeros(0, 2, device=z.device)

                poles_list.append(poles_i)
                zeros_list.append(zeros_i)

            output['poles'] = poles_list  # List of [n_poles_i, 2] tensors
            output['zeros'] = zeros_list  # List of [n_zeros_i, 2] tensors

            # Construct graphs
            graphs = self._construct_graphs(topo_probs, edge_features)
            output['graphs'] = graphs

        return output

    def _construct_graphs(
        self,
        topo_probs: torch.Tensor,
        edge_features: torch.Tensor
    ) -> List[Data]:
        """Construct PyG Data objects (same as HybridDecoder)."""
        batch_size = topo_probs.size(0)
        graphs = []

        for i in range(batch_size):
            # Select filter type (argmax)
            filter_idx = topo_probs[i].argmax().item()
            filter_type = FILTER_TYPES[filter_idx]
            template = CIRCUIT_TEMPLATES[filter_type]

            # Get template structure
            num_nodes = template['num_nodes']
            edges = template['edges']
            node_types = template['node_types']

            # Create node features (one-hot encoding)
            x = torch.zeros(num_nodes, 4, device=edge_features.device)
            for node_idx, node_type in enumerate(node_types):
                x[node_idx, node_type] = 1.0

            # Create edge index
            edge_index = torch.tensor(edges, dtype=torch.long, device=edge_features.device).t()

            # Get edge features
            num_edges = len(edges)
            edge_attr = edge_features[i, :num_edges, :]

            # Create PyG Data object
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                filter_type=filter_type
            )

            graphs.append(graph)

        return graphs
