"""
Simplified Edge Decoder for Topology-Only Generation.

Predicts only edge existence and component type (topology).
Component values are NOT predicted - only topology matters.

Uses autoregressive (GRU-based) edge generation so that each edge
decision is conditioned on all previous edge decisions in the sequence.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LatentGuidedEdgeDecoder(nn.Module):
    """
    Autoregressive edge decoder for topology prediction.

    Uses a GRU cell to maintain sequential state across edge decisions,
    following the GraphRNN "Edge-level RNN" concept. Each edge decision
    is conditioned on all previous edge decisions via the GRU hidden state.

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
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        num_edge_classes: int = 8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_classes = num_edge_classes

        # Edge encoder from node pair
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Context projection (latent only, no conditions)
        self.context_proj = nn.Linear(latent_dim, hidden_dim)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Autoregressive components
        self.edge_token_embedding = nn.Embedding(num_edge_classes, hidden_dim)
        self.edge_gru = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)

        # Fusion: [edge, attended_context, gru_hidden_state]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Single output head: 8-way edge-component classification
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_edge_classes)
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state for a new edge sequence."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        node_i: torch.Tensor,
        node_j: torch.Tensor,
        latent: torch.Tensor,
        edge_hidden_state: torch.Tensor,
        previous_edge_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edge topology for a single node pair, conditioned on
        previous edge decisions via the GRU hidden state.

        Args:
            node_i: [batch, hidden_dim]
            node_j: [batch, hidden_dim]
            latent: [batch, latent_dim]
            edge_hidden_state: [batch, hidden_dim] - GRU hidden state
            previous_edge_token: [batch] - previous edge class index (0-7)

        Returns:
            edge_component_logits: [batch, 8] (0=no edge, 1-7=component type)
            new_hidden_state: [batch, hidden_dim] - updated GRU hidden state
        """
        # Encode node pair
        edge = self.edge_encoder(torch.cat([node_i, node_j], dim=-1))

        # Project context (latent only)
        context = self.context_proj(latent).unsqueeze(1)

        # Cross-attention
        attended, _ = self.cross_attention(
            query=edge.unsqueeze(1),
            key=context,
            value=context
        )
        attended = self.norm(attended.squeeze(1))

        # Embed previous edge decision
        prev_embed = self.edge_token_embedding(previous_edge_token)

        # GRU step: update hidden state with current context + previous decision
        gru_input = torch.cat([attended, prev_embed], dim=-1)
        new_hidden_state = self.edge_gru(gru_input, edge_hidden_state)

        # Fuse: [node pair repr, latent-attended context, sequential state]
        fused = self.fusion(torch.cat([edge, attended, new_hidden_state], dim=-1))

        return self.output_head(fused), new_hidden_state


if __name__ == '__main__':
    """Test edge decoder."""
    print("Testing Edge Decoder...")

    batch_size = 2
    hidden_dim = 256

    edge_decoder = LatentGuidedEdgeDecoder(
        hidden_dim=hidden_dim,
        latent_dim=8
    )

    node_i = torch.randn(batch_size, hidden_dim)
    node_j = torch.randn(batch_size, hidden_dim)
    latent = torch.randn(batch_size, 8)

    # Initialize autoregressive state
    hidden = edge_decoder.init_hidden(batch_size, node_i.device)
    prev_token = torch.zeros(batch_size, dtype=torch.long)

    logits, new_hidden = edge_decoder(node_i, node_j, latent, hidden, prev_token)

    assert logits.shape == (batch_size, 8)
    assert new_hidden.shape == (batch_size, hidden_dim)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden shape: {new_hidden.shape}")

    # Test sequential: feed output back in
    logits2, new_hidden2 = edge_decoder(
        node_i, node_j, latent, new_hidden, torch.argmax(logits, dim=-1)
    )
    assert logits2.shape == (batch_size, 8)
    print(f"  Sequential step 2 logits: {logits2.shape}")

    num_params = sum(p.numel() for p in edge_decoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("\nTest passed!")
