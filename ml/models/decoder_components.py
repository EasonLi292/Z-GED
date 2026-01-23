"""
Simplified Edge Decoder for Topology-Only Generation.

Predicts only edge existence and component type (topology).
Component values are NOT predicted - only topology matters.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LatentGuidedEdgeDecoder(nn.Module):
    """
    Minimal edge decoder for topology prediction.

    Predicts:
    - edge_component_logits: 8-way classification (0=no edge, 1-7=component type)

    Derived (not predicted):
    - is_parallel: True if component type >= 4
    - masks: deterministic from component type
    - values: NOT needed for topology-only generation
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Edge encoder from node pair
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Context projection
        self.context_proj = nn.Linear(latent_dim + conditions_dim, hidden_dim)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Single output head: 8-way edge-component classification
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8)
        )

    def forward(
        self,
        node_i: torch.Tensor,
        node_j: torch.Tensor,
        latent: torch.Tensor,
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict edge topology.

        Args:
            node_i: [batch, hidden_dim]
            node_j: [batch, hidden_dim]
            latent: [batch, latent_dim]
            conditions: [batch, conditions_dim]

        Returns:
            edge_component_logits: [batch, 8] (0=no edge, 1-7=component type)
        """
        # Encode node pair
        edge = self.edge_encoder(torch.cat([node_i, node_j], dim=-1))

        # Project context
        context = self.context_proj(torch.cat([latent, conditions], dim=-1)).unsqueeze(1)

        # Cross-attention
        attended, _ = self.cross_attention(
            query=edge.unsqueeze(1),
            key=context,
            value=context
        )
        attended = self.norm(attended.squeeze(1))

        # Fuse and predict
        fused = self.fusion(torch.cat([edge, attended], dim=-1))
        return self.output_head(fused)


if __name__ == '__main__':
    """Test edge decoder."""
    print("Testing Edge Decoder...")

    batch_size = 2
    hidden_dim = 256

    edge_decoder = LatentGuidedEdgeDecoder(
        hidden_dim=hidden_dim,
        latent_dim=8,
        conditions_dim=2
    )

    node_i = torch.randn(batch_size, hidden_dim)
    node_j = torch.randn(batch_size, hidden_dim)
    latent = torch.randn(batch_size, 8)
    conditions = torch.randn(batch_size, 2)

    logits = edge_decoder(node_i, node_j, latent, conditions)

    assert logits.shape == (batch_size, 8)
    print(f"  Output shape: {logits.shape}")

    num_params = sum(p.numel() for p in edge_decoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("\n✅ Test passed!")
