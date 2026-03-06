"""
GainEncoder: Hierarchical VAE encoder with admittance-based GNN + gain head.

Mirrors HierarchicalEncoder structure but uses AdmittanceGNN instead of
ImpedanceGNN, and adds a gain prediction head on the 8D latent.
"""

import torch
import torch.nn as nn
from typing import Tuple

from eason.admittance_gain.admittance_conv import AdmittanceGNN


class GlobalPooling(nn.Module):
    """Mean + max pooling for graph-level representations."""

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        pooled = []

        # Mean
        mean_feat = torch.zeros(batch_size, x.size(1), dtype=x.dtype, device=x.device)
        mean_feat.index_add_(0, batch, x)
        counts = torch.zeros(batch_size, dtype=torch.float, device=x.device)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        mean_feat = mean_feat / counts.unsqueeze(1).clamp(min=1)
        pooled.append(mean_feat)

        # Max
        max_feat = torch.full((batch_size, x.size(1)), float('-inf'), dtype=x.dtype, device=x.device)
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                max_feat[i] = x[mask].max(dim=0)[0]
        pooled.append(max_feat)

        return torch.cat(pooled, dim=-1)


class GainEncoder(nn.Module):
    """
    Hierarchical VAE encoder with gain prediction.

    Architecture:
        AdmittanceGNN (3 layers) -> 3 VAE branches (2D + 2D + 4D = 8D)
        -> gain_head MLP predicts |H(jw)| from z

    Returns (z, mu, logvar, gain_pred).
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gnn_hidden_dim = gnn_hidden_dim
        self.latent_dim = latent_dim
        self.topo_dim = 2
        self.struct_dim = 2
        self.pz_dim = 4

        # Stage 1: Admittance GNN
        self.gnn = AdmittanceGNN(
            in_channels=node_feature_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout
        )

        self.pooling = GlobalPooling()
        pool_dim = gnn_hidden_dim * 2  # mean + max

        # Branch 1: Topology (from global pool)
        self.topo_encoder = nn.Sequential(
            nn.Linear(pool_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
        )
        self.topo_mu = nn.Linear(gnn_hidden_dim // 2, self.topo_dim)
        self.topo_logvar = nn.Linear(gnn_hidden_dim // 2, self.topo_dim)

        # Branch 2: Structure (from GND/VIN/VOUT embeddings)
        terminal_dim = 3 * gnn_hidden_dim
        self.struct_encoder = nn.Sequential(
            nn.Linear(terminal_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.struct_mu = nn.Linear(gnn_hidden_dim // 2, self.struct_dim)
        self.struct_logvar = nn.Linear(gnn_hidden_dim // 2, self.struct_dim)

        # Branch 3: PZ (global pool + terminals)
        pz_input_dim = pool_dim + terminal_dim
        self.pz_encoder = nn.Sequential(
            nn.Linear(pz_input_dim, gnn_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gnn_hidden_dim * 4, gnn_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pz_mu = nn.Linear(gnn_hidden_dim, self.pz_dim)
        self.pz_logvar = nn.Linear(gnn_hidden_dim, self.pz_dim)

        # Gain prediction head: z (8D) -> scalar gain
        self.gain_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch.max().item() + 1

        # GNN encoding
        h_nodes = self.gnn(x, edge_index, edge_attr, batch)

        # Branch 1: Topology
        h_pool = self.pooling(h_nodes, batch)
        h_topo = self.topo_encoder(h_pool)
        mu_topo = self.topo_mu(h_topo)
        logvar_topo = self.topo_logvar(h_topo)

        # Extract terminal embeddings (GND=0, VIN=1, VOUT=2)
        h_terminals_list = []
        for i in range(batch_size):
            node_mask = batch == i
            graph_x = x[node_mask]
            graph_h = h_nodes[node_mask]
            node_types = graph_x.argmax(dim=-1)

            h_gnd = torch.zeros(self.gnn_hidden_dim, device=x.device)
            h_vin = torch.zeros(self.gnn_hidden_dim, device=x.device)
            h_vout = torch.zeros(self.gnn_hidden_dim, device=x.device)

            for n_idx in range(node_types.size(0)):
                ntype = node_types[n_idx].item()
                if ntype == 0:
                    h_gnd = graph_h[n_idx]
                elif ntype == 1:
                    h_vin = graph_h[n_idx]
                elif ntype == 2:
                    h_vout = graph_h[n_idx]

            h_terminals_list.append(torch.cat([h_gnd, h_vin, h_vout]))

        h_terminals = torch.stack(h_terminals_list)

        # Branch 2: Structure
        h_struct = self.struct_encoder(h_terminals)
        mu_struct = self.struct_mu(h_struct)
        logvar_struct = self.struct_logvar(h_struct)

        # Branch 3: PZ
        h_pz_input = torch.cat([h_pool, h_terminals], dim=-1)
        h_pz = self.pz_encoder(h_pz_input)
        mu_pz = self.pz_mu(h_pz)
        logvar_pz = self.pz_logvar(h_pz)

        # Combine
        mu = torch.cat([mu_topo, mu_struct, mu_pz], dim=-1)
        logvar = torch.cat([logvar_topo, logvar_struct, logvar_pz], dim=-1)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # Gain prediction
        gain_pred = self.gain_head(z).squeeze(-1)

        return z, mu, logvar, gain_pred
