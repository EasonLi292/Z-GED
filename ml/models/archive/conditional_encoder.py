"""
Conditional Hierarchical Encoder for GraphVAE.

Extends HierarchicalEncoder to accept conditions (specifications) as input.
Conditions are embedded and concatenated to intermediate representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .encoder import HierarchicalEncoder


class ConditionalHierarchicalEncoder(HierarchicalEncoder):
    """
    Conditional encoder that accepts circuit specifications as additional input.

    Architecture:
        1. Embed conditions: [cutoff_freq, q_factor] → condition_embedding
        2. Process graph with GNN (same as base encoder)
        3. Concatenate condition embedding to each branch before μ/σ computation
        4. Encode to latent space conditioned on specifications

    Args:
        conditions_dim: Dimension of condition vector (default: 2 for [cutoff, q])
        condition_embed_dim: Dimension of condition embedding (default: 64)
        All other args same as HierarchicalEncoder
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 3,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 24,
        dropout: float = 0.1,
        topo_latent_dim: Optional[int] = None,
        values_latent_dim: Optional[int] = None,
        pz_latent_dim: Optional[int] = None,
        # Conditional parameters
        conditions_dim: int = 2,
        condition_embed_dim: int = 64
    ):
        # Initialize base encoder
        super().__init__(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            topo_latent_dim=topo_latent_dim,
            values_latent_dim=values_latent_dim,
            pz_latent_dim=pz_latent_dim
        )

        self.conditions_dim = conditions_dim
        self.condition_embed_dim = condition_embed_dim

        # Condition embedding network
        self.condition_embed = nn.Sequential(
            nn.Linear(conditions_dim, condition_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_embed_dim // 2, condition_embed_dim),
            nn.ReLU()
        )

        # Update branch encoders to accept concatenated condition embedding
        # Need to adjust input dimensions for μ/σ MLPs

        # Topology branch: originally [128] → now [128 + 64]
        topo_input_dim = gnn_hidden_dim * 2  # mean + max pooling
        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_input_dim + condition_embed_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
        )

        # Redefine μ/σ for topology
        self.topo_mu = nn.Linear(gnn_hidden_dim // 2, self.topo_latent_dim)
        self.topo_logvar = nn.Linear(gnn_hidden_dim // 2, self.topo_latent_dim)

        # Values branch: originally [edge features] → now [edge features + 64]
        # Keep original edge aggregation, add condition before μ/σ
        values_input_dim = edge_feature_dim  # From edge aggregation

        self.values_encoder = nn.Sequential(
            nn.Linear(values_input_dim + condition_embed_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
        )

        self.values_mu = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)
        self.values_logvar = nn.Linear(gnn_hidden_dim // 2, self.values_latent_dim)

        # P&Z branch: originally [DeepSets output] → now [DeepSets output + 64]
        # DeepSets already has correct dimensions, add condition before μ/σ
        pz_input_dim = 32  # From pz_joint MLP

        self.pz_joint = nn.Sequential(
            nn.Linear(pz_input_dim + condition_embed_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU()
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
        zeros_list: List[torch.Tensor],
        conditions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional conditions.

        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_feature_dim]
            batch: Batch assignment [N]
            poles_list: List of pole tensors [num_poles, 2] for each circuit
            zeros_list: List of zero tensors [num_zeros, 2] for each circuit
            conditions: Optional condition tensor [B, conditions_dim]
                       If None, uses zero padding (backward compatible)

        Returns:
            z: Latent code [B, latent_dim] (sampled from posterior)
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        """
        batch_size = batch.max().item() + 1

        # Embed conditions (or use zeros if not provided)
        if conditions is not None:
            cond_embed = self.condition_embed(conditions)  # [B, condition_embed_dim]
        else:
            # Backward compatibility: if no conditions, use zeros
            cond_embed = torch.zeros(batch_size, self.condition_embed_dim,
                                     device=x.device, dtype=x.dtype)

        # Stage 1: GNN processing (same as base encoder)
        node_embeddings = self.gnn(x, edge_index, edge_attr)  # [N, gnn_hidden_dim]

        # Branch 1: Topology encoding (with conditions)
        pooled = self.pooling(node_embeddings, batch)  # [B, 128]
        topo_concat = torch.cat([pooled, cond_embed], dim=-1)  # [B, 128+64]
        topo_encoding = self.topo_encoder(topo_concat)  # [B, 32]

        topo_mu = self.topo_mu(topo_encoding)  # [B, topo_latent_dim]
        topo_logvar = self.topo_logvar(topo_encoding)

        # Branch 2: Component values encoding (with conditions)
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
                h_values_graph = torch.zeros(edge_attr.size(1), device=edge_attr.device)

            h_values_list.append(h_values_graph)

        h_values = torch.stack(h_values_list)   # [B, edge_feature_dim]
        values_concat = torch.cat([h_values, cond_embed], dim=-1)  # [B, edge_feature_dim+64]
        values_encoding = self.values_encoder(values_concat)  # [B, 32]

        values_mu = self.values_mu(values_encoding)  # [B, values_latent_dim]
        values_logvar = self.values_logvar(values_encoding)

        # Branch 3: Poles/Zeros encoding (with conditions)
        h_poles_list = []
        h_zeros_list = []

        for poles, zeros in zip(poles_list, zeros_list):
            # Encode poles
            if poles.size(0) > 0:
                h_poles_single = self.pz_encoder_poles(poles).squeeze(0)  # [16]
            else:
                h_poles_single = torch.zeros(16, device=x.device)
            h_poles_list.append(h_poles_single)

            # Encode zeros
            if zeros.size(0) > 0:
                h_zeros_single = self.pz_encoder_zeros(zeros).squeeze(0)  # [16]
            else:
                h_zeros_single = torch.zeros(16, device=x.device)
            h_zeros_list.append(h_zeros_single)

        h_poles = torch.stack(h_poles_list)  # [B, 16]
        h_zeros = torch.stack(h_zeros_list)  # [B, 16]

        # Combine poles and zeros
        h_pz = torch.cat([h_poles, h_zeros], dim=-1)  # [B, 32]
        pz_concat = torch.cat([h_pz, cond_embed], dim=-1)  # [B, 32+64]
        pz_encoding = self.pz_joint(pz_concat)  # [B, 32]

        pz_mu = self.pz_mu(pz_encoding)  # [B, pz_latent_dim]
        pz_logvar = self.pz_logvar(pz_encoding)

        # Concatenate all branches
        mu = torch.cat([topo_mu, values_mu, pz_mu], dim=-1)  # [B, latent_dim]
        logvar = torch.cat([topo_logvar, values_logvar, pz_logvar], dim=-1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        return z, mu, logvar
