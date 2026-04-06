"""
Hierarchical Encoder for GraphVAE.

Encodes circuit graphs into a hierarchical latent space.

Production configuration (8D):
    z = [z_topo (2D) | z_structure (2D) | z_pz (4D)]

where:
    - z_topo: Encodes graph topology and filter type
    - z_structure: Encodes terminal structure (GND/VIN/VOUT node embeddings)
    - z_pz: Encodes poles/zeros (transfer function behavior)

The latent dimensions are configurable via constructor parameters.
Default total latent_dim=8 splits as 2D + 2D + 4D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .gnn_layers import ImpedanceGNN, GlobalPooling


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for circuit graphs.

    Architecture:
        Stage 1: Impedance-aware GNN (3 layers)
            Input: Node features [4D] + Edge features [3D: log10(R), log10(C), log10(L)]
            Output: Node embeddings [64D]

        Stage 2: Hierarchical latent encoding
            Branch 1 (Topology): Global pooling → MLP → μ_topo, log_σ_topo → z_topo [2D]
            Branch 2 (Structure): GND/VIN/VOUT node embeddings → MLP → μ_structure, log_σ_structure → z_structure [2D]
            Branch 3 (Poles/Zeros): Vin'-MLP attention pooling + terminals → MLP → μ_pz, log_σ_pz → z_pz [4D]
                                    Built-in pole_head: μ_pz → (σ_p, ω_p) for auxiliary supervision

    Args:
        node_feature_dim: Input node feature dimension (default: 4)
        edge_feature_dim: Input edge feature dimension (default: 3)
        gnn_hidden_dim: Hidden dimension for GNN (default: 64)
        gnn_num_layers: Number of GNN layers (default: 3)
        latent_dim: Total latent dimension (default: 8)
        topo_latent_dim: Topology latent dimension (default: 2)
        structure_latent_dim: Structure latent dimension (default: 2)
        values_latent_dim: Alias for structure_latent_dim (default: None)
        pz_latent_dim: Poles/zeros latent dimension (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 3,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 8,
        dropout: float = 0.1,
        # Variable branch dimensions (defaults: 2D + 2D + 4D for 8D total)
        topo_latent_dim: Optional[int] = None,
        structure_latent_dim: Optional[int] = None,
        values_latent_dim: Optional[int] = None,
        pz_latent_dim: Optional[int] = None
    ):
        super().__init__()

        # Set branch dimensions
        if values_latent_dim is not None:
            if structure_latent_dim is not None and structure_latent_dim != values_latent_dim:
                raise ValueError("values_latent_dim and structure_latent_dim must match when both are set")
            structure_latent_dim = values_latent_dim

        if topo_latent_dim is None or structure_latent_dim is None or pz_latent_dim is None:
            # Production defaults: 2D topology + 2D structure + 4D TF = 8D total
            if latent_dim == 8:
                self.topo_latent_dim = 2
                self.structure_latent_dim = 2
                self.pz_latent_dim = 4
            else:
                # Fall back to equal split for other latent dimensions
                assert latent_dim % 3 == 0, "Latent dim must be divisible by 3 for equal split"
                self.topo_latent_dim = latent_dim // 3
                self.structure_latent_dim = latent_dim // 3
                self.pz_latent_dim = latent_dim // 3
        else:
            self.topo_latent_dim = topo_latent_dim
            self.structure_latent_dim = structure_latent_dim
            self.pz_latent_dim = pz_latent_dim
            assert topo_latent_dim + structure_latent_dim + pz_latent_dim == latent_dim, \
                f"Branch dims {topo_latent_dim}+{structure_latent_dim}+{pz_latent_dim} != {latent_dim}"

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.latent_dim = latent_dim
        # Backward compatibility for older configs/tests
        self.values_latent_dim = self.structure_latent_dim

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

        # Branch 2: Component values encoding (from GND, VIN, VOUT node embeddings)
        # The GNN already propagates edge/component information into node embeddings
        # via message passing, so GND/VIN/VOUT embeddings capture position-specific
        # component info (e.g., what's connected to ground vs input vs output).
        # Node types: GND=0, VIN=1, VOUT=2

        # Concatenate h_GND, h_VIN, h_VOUT node embeddings
        values_combined_dim = 3 * gnn_hidden_dim
        self.values_combine = nn.Sequential(
            nn.Linear(values_combined_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.values_mu = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)
        self.values_logvar = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)

        # Branch 3: Poles/Zeros encoding — Vin'-MLP attention pooling
        # Computes VIN-conditioned attention weights over all nodes to pool
        # a signal-path-aware representation for the pz branch.
        self.vin_pool_attn = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.Tanh(),
            nn.Linear(gnn_hidden_dim, 1)
        )
        # Input: h_vin_prime [D] + GND/VIN/VOUT embeddings [3D] = 4D
        pz_input_dim = gnn_hidden_dim * 4
        pz_hidden = gnn_hidden_dim * 2   # 128
        pz_drop = 0.2
        self.pz_encoder = nn.Sequential(
            nn.Linear(pz_input_dim, pz_hidden),
            nn.ReLU(),
            nn.Dropout(pz_drop),
            nn.Linear(pz_hidden, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(pz_drop)
        )

        self.pz_mu = nn.Linear(gnn_hidden_dim, self.pz_latent_dim)
        self.pz_logvar = nn.Linear(gnn_hidden_dim, self.pz_latent_dim)

        # Built-in pole prediction head: mu_pz -> (sigma_p, omega_p)
        self.pole_head = nn.Sequential(
            nn.Linear(self.pz_latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode circuit graphs to hierarchical latent space.

        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_feature_dim]
            batch: Batch assignment for nodes [N]

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

        mu_topo = self.topo_mu(h_topo)         # [B, topo_latent_dim]
        logvar_topo = self.topo_logvar(h_topo) # [B, topo_latent_dim]

        # Branch 2 & 3: Extract GND/VIN/VOUT terminal embeddings (shared)
        # Node types: GND=0, VIN=1, VOUT=2
        h_terminal_list = []

        for i in range(batch_size):
            node_mask = batch == i
            graph_x = x[node_mask]
            graph_h = h_nodes[node_mask]  # [num_nodes, gnn_hidden_dim]
            node_types = graph_x.argmax(dim=-1)  # [num_nodes]

            # Extract embeddings for GND, VIN, VOUT nodes
            h_gnd = torch.zeros(self.gnn_hidden_dim, device=x.device)
            h_vin = torch.zeros(self.gnn_hidden_dim, device=x.device)
            h_vout = torch.zeros(self.gnn_hidden_dim, device=x.device)

            for n_idx in range(node_types.size(0)):
                ntype = node_types[n_idx].item()
                if ntype == 0:    # GND
                    h_gnd = graph_h[n_idx]
                elif ntype == 1:  # VIN
                    h_vin = graph_h[n_idx]
                elif ntype == 2:  # VOUT
                    h_vout = graph_h[n_idx]

            h_terminal_list.append((h_gnd, h_vin, h_vout))

        # Branch 2: Component values encoding (from GND, VIN, VOUT node embeddings)
        h_values_list = [torch.cat([h_gnd, h_vin, h_vout], dim=-1)
                         for h_gnd, h_vin, h_vout in h_terminal_list]
        h_values = torch.stack(h_values_list)
        h_values = self.values_combine(h_values)

        mu_values = self.values_mu(h_values)         # [B, structure_latent_dim]
        logvar_values = self.values_logvar(h_values)  # [B, structure_latent_dim]

        # Branch 3: Poles/Zeros encoding — Vin'-MLP attention pooling
        # Pool all nodes using VIN-conditioned attention to capture signal path
        h_vin_prime_list = []
        for i in range(batch_size):
            node_mask = batch == i
            graph_h = h_nodes[node_mask]                                          # [n, D]
            h_vin_i = h_terminal_list[i][1]                                       # [D]
            h_vin_exp = h_vin_i.unsqueeze(0).expand(graph_h.size(0), -1)         # [n, D]
            attn_in = torch.cat([graph_h, h_vin_exp], dim=-1)                    # [n, 2D]
            attn_w = torch.softmax(
                self.vin_pool_attn(attn_in).squeeze(-1), dim=0
            )                                                                      # [n]
            h_vin_prime_list.append((attn_w.unsqueeze(-1) * graph_h).sum(0))     # [D]
        h_vin_prime = torch.stack(h_vin_prime_list)  # [B, D]

        h_terminals = torch.stack([torch.cat([h_gnd, h_vin, h_vout], dim=-1)
                                   for h_gnd, h_vin, h_vout in h_terminal_list])  # [B, 3*D]
        h_pz_input = torch.cat([h_vin_prime, h_terminals], dim=-1)               # [B, 4*D]
        h_pz = self.pz_encoder(h_pz_input)                                        # [B, D]

        mu_pz = self.pz_mu(h_pz)             # [B, pz_latent_dim]
        logvar_pz = self.pz_logvar(h_pz)     # [B, pz_latent_dim]

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
    ) -> torch.Tensor:
        """
        Encode to latent space without sampling (deterministic).

        Returns the mean μ instead of sampling z ~ N(μ, σ²).

        Args:
            (same as forward)

        Returns:
            mu: Mean latent vector [B, latent_dim]
        """
        _, mu, _ = self.forward(x, edge_index, edge_attr, batch)
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

    def predict_poles(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Predict dominant pole (sigma_p, omega_p) from latent mean.

        Args:
            mu: Latent mean [B, latent_dim]

        Returns:
            pole_pred: Predicted (signed-log sigma_p, signed-log omega_p) [B, 2]
        """
        mu_pz = mu[:, self.topo_latent_dim + self.structure_latent_dim:]
        return self.pole_head(mu_pz)
