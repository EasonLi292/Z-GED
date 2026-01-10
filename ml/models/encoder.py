"""
Hierarchical Encoder for GraphVAE.

Encodes circuit graphs into a hierarchical latent space.

Production configuration (8D):
    z = [z_topo (2D) | z_values (2D) | z_pz (4D)]

where:
    - z_topo: Encodes graph topology and filter type
    - z_values: Encodes component values and their distributions
    - z_pz: Encodes poles/zeros (transfer function behavior)

The latent dimensions are configurable via constructor parameters.
Default total latent_dim=8 splits as 2D + 2D + 4D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .gnn_layers import ImpedanceGNN, GlobalPooling, DeepSets


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for circuit graphs.

    Architecture:
        Stage 1: Impedance-aware GNN (3 layers)
            Input: Node features [4D] + Edge features [7D]
            Output: Node embeddings [64D]

        Stage 2: Hierarchical latent encoding
            Branch 1 (Topology): Global pooling → MLP → μ_topo, log_σ_topo → z_topo [2D]
            Branch 2 (Values): Edge aggregation → MLP → μ_values, log_σ_values → z_values [2D]
            Branch 3 (Poles/Zeros): DeepSets → MLP → μ_pz, log_σ_pz → z_pz [4D]

    Args:
        node_feature_dim: Input node feature dimension (default: 4)
        edge_feature_dim: Input edge feature dimension (default: 7)
        gnn_hidden_dim: Hidden dimension for GNN (default: 64)
        gnn_num_layers: Number of GNN layers (default: 3)
        latent_dim: Total latent dimension (default: 8)
        topo_latent_dim: Topology latent dimension (default: 2)
        values_latent_dim: Values latent dimension (default: 2)
        pz_latent_dim: Poles/zeros latent dimension (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 8,
        dropout: float = 0.1,
        # Variable branch dimensions (defaults: 2D + 2D + 4D for 8D total)
        topo_latent_dim: Optional[int] = None,
        values_latent_dim: Optional[int] = None,
        pz_latent_dim: Optional[int] = None
    ):
        super().__init__()

        # Set branch dimensions
        if topo_latent_dim is None or values_latent_dim is None or pz_latent_dim is None:
            # Production defaults: 2D topology + 2D values + 4D TF = 8D total
            if latent_dim == 8:
                self.topo_latent_dim = 2
                self.values_latent_dim = 2
                self.pz_latent_dim = 4
            else:
                # Fall back to equal split for other latent dimensions
                assert latent_dim % 3 == 0, "Latent dim must be divisible by 3 for equal split"
                self.topo_latent_dim = latent_dim // 3
                self.values_latent_dim = latent_dim // 3
                self.pz_latent_dim = latent_dim // 3
        else:
            self.topo_latent_dim = topo_latent_dim
            self.values_latent_dim = values_latent_dim
            self.pz_latent_dim = pz_latent_dim
            assert topo_latent_dim + values_latent_dim + pz_latent_dim == latent_dim, \
                f"Branch dims {topo_latent_dim}+{values_latent_dim}+{pz_latent_dim} != {latent_dim}"

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.latent_dim = latent_dim

        # Stage 1: Impedance-Aware GNN
        self.gnn = ImpedanceGNN(
            in_channels=node_feature_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            edge_dim=edge_feature_dim,
            dropout=dropout
        )

        # Stage 2: Hierarchical encoding branches

        # Branch 1: Topology encoding (from node embeddings)
        self.pooling = GlobalPooling(pooling_types=['mean', 'max'])
        topo_input_dim = gnn_hidden_dim * 2  # mean + max pooling

        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_input_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
        )

        self.topo_mu = nn.Linear(gnn_hidden_dim // 2, self.topo_latent_dim)
        self.topo_logvar = nn.Linear(gnn_hidden_dim // 2, self.topo_latent_dim)

        # Branch 2: Component values encoding (from edge features)
        # Aggregate edge features across the graph
        self.values_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
        )

        self.values_mu = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)
        self.values_logvar = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)

        # Branch 3: Poles/Zeros encoding (DeepSets for variable-length)
        self.pz_encoder_poles = DeepSets(
            input_dim=2,  # [real, imag]
            hidden_dim=32,
            output_dim=16
        )

        self.pz_encoder_zeros = DeepSets(
            input_dim=2,
            hidden_dim=32,
            output_dim=16
        )

        # Combine poles and zeros encoding
        pz_combined_dim = 16 + 16  # poles + zeros
        self.pz_combine = nn.Sequential(
            nn.Linear(pz_combined_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pz_mu = nn.Linear(gnn_hidden_dim // 2, self.pz_latent_dim)
        self.pz_logvar = nn.Linear(gnn_hidden_dim // 2, self.pz_latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        poles_list: List[torch.Tensor],
        zeros_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode circuit graphs to hierarchical latent space.

        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_feature_dim]
            batch: Batch assignment for nodes [N]
            poles_list: List of pole tensors [num_poles, 2] for each graph
            zeros_list: List of zero tensors [num_zeros, 2] for each graph

        Returns:
            z: Sampled latent vector [B, latent_dim]
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log-variance of latent distribution [B, latent_dim]
        """
        batch_size = batch.max().item() + 1

        # Stage 1: GNN encoding
        h_nodes = self.gnn(x, edge_index, edge_attr, batch)  # [N, gnn_hidden_dim]

        # Branch 1: Topology encoding (from node embeddings)
        h_topo = self.pooling(h_nodes, batch)  # [B, gnn_hidden_dim * 2]
        h_topo = self.topo_encoder(h_topo)     # [B, gnn_hidden_dim // 2]

        mu_topo = self.topo_mu(h_topo)         # [B, 8]
        logvar_topo = self.topo_logvar(h_topo) # [B, 8]

        # Branch 2: Component values encoding (from edge features)
        # Aggregate edge features per graph
        h_values_list = []
        edge_batch = batch[edge_index[0]]  # Compute once for all graphs
        for i in range(batch_size):
            # Find edges where source node belongs to this graph
            edge_mask = edge_batch == i
            edges_in_graph = edge_attr[edge_mask]  # [E_i, edge_feature_dim]

            # Aggregate (mean pooling over edges)
            if edges_in_graph.size(0) > 0:
                h_values_graph = edges_in_graph.mean(dim=0)  # [edge_feature_dim]
            else:
                h_values_graph = torch.zeros(self.edge_feature_dim, device=edge_attr.device)

            h_values_list.append(h_values_graph)

        h_values = torch.stack(h_values_list)   # [B, edge_feature_dim]
        h_values = self.values_encoder(h_values) # [B, gnn_hidden_dim // 2]

        mu_values = self.values_mu(h_values)         # [B, 8]
        logvar_values = self.values_logvar(h_values) # [B, 8]

        # Branch 3: Poles/Zeros encoding (DeepSets)
        h_poles_list = []
        h_zeros_list = []

        for poles, zeros in zip(poles_list, zeros_list):
            # Encode poles
            if poles.size(0) > 0:
                h_poles = self.pz_encoder_poles(poles).squeeze(0)  # [16]
            else:
                h_poles = torch.zeros(16, device=x.device)
            h_poles_list.append(h_poles)

            # Encode zeros
            if zeros.size(0) > 0:
                h_zeros = self.pz_encoder_zeros(zeros).squeeze(0)  # [16]
            else:
                h_zeros = torch.zeros(16, device=x.device)
            h_zeros_list.append(h_zeros)

        h_poles = torch.stack(h_poles_list)  # [B, 16]
        h_zeros = torch.stack(h_zeros_list)  # [B, 16]

        h_pz = torch.cat([h_poles, h_zeros], dim=-1)  # [B, 32]
        h_pz = self.pz_combine(h_pz)                   # [B, gnn_hidden_dim // 2]

        mu_pz = self.pz_mu(h_pz)                       # [B, 8]
        logvar_pz = self.pz_logvar(h_pz)               # [B, 8]

        # Combine all branches
        mu = torch.cat([mu_topo, mu_values, mu_pz], dim=-1)        # [B, 24]
        logvar = torch.cat([logvar_topo, logvar_values, logvar_pz], dim=-1)  # [B, 24]

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)

        Args:
            mu: Mean [B, latent_dim]
            logvar: Log-variance [B, latent_dim]

        Returns:
            z: Sampled latent vector [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use the mean
            return mu

    def encode_deterministic(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        poles_list: List[torch.Tensor],
        zeros_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode to latent space without sampling (deterministic).

        Returns the mean μ instead of sampling z ~ N(μ, σ²).

        Args:
            (same as forward)

        Returns:
            mu: Mean latent vector [B, latent_dim]
        """
        _, mu, _ = self.forward(x, edge_index, edge_attr, batch, poles_list, zeros_list)
        return mu

    def get_latent_split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split latent vector into three components.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            z_topo: Topology latent [B, topo_latent_dim]
            z_values: Values latent [B, values_latent_dim]
            z_pz: Poles/zeros latent [B, pz_latent_dim]
        """
        topo_end = self.topo_latent_dim
        values_end = topo_end + self.values_latent_dim
        z_topo = z[:, :topo_end]
        z_values = z[:, topo_end:values_end]
        z_pz = z[:, values_end:]
        return z_topo, z_values, z_pz
