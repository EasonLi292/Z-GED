"""
Simplified Edge Decoder for Topology-Only Generation.

Predicts only edge existence and component type (topology).
Component values are NOT predicted - only topology matters.

Uses autoregressive (Transformer-based) edge generation so that each edge
decision is conditioned on all previous edge decisions in the sequence.
"""

import torch
import torch.nn as nn


class LatentGuidedEdgeDecoder(nn.Module):
    """
    Autoregressive edge decoder for topology prediction.

    Uses causal self-attention over the edge sequence (Transformer encoder).
    Each edge decision is conditioned on all previous edge decisions via
    the attention mask.

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
        num_layers: int = 4,
        dropout: float = 0.1,
        num_edge_classes: int = 8,
        max_edges: int = 128
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_classes = num_edge_classes
        self.max_edges = max_edges

        # Edge encoder from node pair
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Context projection (latent only, no conditions)
        self.context_proj = nn.Linear(latent_dim, hidden_dim)

        # Autoregressive components
        self.edge_token_embedding = nn.Embedding(num_edge_classes, hidden_dim)
        self.position_embedding = nn.Embedding(max_edges, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Single output head: 8-way edge-component classification
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_edge_classes)
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask for self-attention (True blocks attending to future)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        node_i: torch.Tensor,
        node_j: torch.Tensor,
        latent: torch.Tensor,
        previous_edge_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict edge topology for a sequence of node pairs, conditioned on
        previous edge decisions via causal self-attention.

        Args:
            node_i: [batch, seq_len, hidden_dim]
            node_j: [batch, seq_len, hidden_dim]
            latent: [batch, latent_dim]
            previous_edge_tokens: [batch, seq_len] - previous edge class index (0-7),
                shifted right so each position sees prior decisions

        Returns:
            edge_component_logits: [batch, seq_len, 8] (0=no edge, 1-7=component type)
        """
        batch_size, seq_len, _ = node_i.shape
        if seq_len > self.max_edges:
            raise ValueError(f"seq_len {seq_len} exceeds max_edges {self.max_edges}")

        # Encode node pairs
        edge = self.edge_encoder(torch.cat([node_i, node_j], dim=-1))

        # Project context (latent only)
        context = self.context_proj(latent).unsqueeze(1)

        # Position + previous edge decision embeddings
        pos_ids = torch.arange(seq_len, device=latent.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(pos_ids)
        prev_embed = self.edge_token_embedding(previous_edge_tokens)

        x = edge + context + pos_embed + prev_embed
        x = self.input_norm(x)

        mask = self._causal_mask(seq_len, latent.device)
        x = self.transformer(x, mask=mask)

        return self.output_head(x)


if __name__ == '__main__':
    """Test edge decoder."""
    print("Testing Edge Decoder...")

    batch_size = 2
    hidden_dim = 256

    edge_decoder = LatentGuidedEdgeDecoder(
        hidden_dim=hidden_dim,
        latent_dim=8,
        num_layers=2,
        max_edges=16
    )

    node_i = torch.randn(batch_size, 1, hidden_dim)
    node_j = torch.randn(batch_size, 1, hidden_dim)
    latent = torch.randn(batch_size, 8)

    # Initialize autoregressive tokens
    prev_tokens = torch.zeros(batch_size, 1, dtype=torch.long)

    logits = edge_decoder(node_i, node_j, latent, prev_tokens)

    assert logits.shape == (batch_size, 1, 8)
    print(f"  Logits shape: {logits.shape}")

    # Test sequential: feed output back in (length 2)
    node_i2 = torch.cat([node_i, node_i], dim=1)
    node_j2 = torch.cat([node_j, node_j], dim=1)
    prev_tokens2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prev_tokens2[:, 1] = torch.argmax(logits[:, 0], dim=-1)
    logits2 = edge_decoder(node_i2, node_j2, latent, prev_tokens2)
    assert logits2.shape == (batch_size, 2, 8)
    print(f"  Sequential step 2 logits: {logits2.shape}")

    num_params = sum(p.numel() for p in edge_decoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("\nTest passed!")
