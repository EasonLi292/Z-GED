"""
Graph Transformer layers for diffusion denoising network.

Implements permutation-equivariant transformer layers for processing
circuit graphs during diffusion denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head self-attention for graphs.

    Computes attention between all pairs of nodes in a graph, providing
    permutation-equivariant feature aggregation.

    Args:
        hidden_dim: Dimension of node features
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_edge_features: Whether to incorporate edge features in attention

    Example:
        >>> attn = MultiHeadGraphAttention(hidden_dim=256, num_heads=8)
        >>> node_features = torch.randn(4, 5, 256)  # [batch, num_nodes, hidden_dim]
        >>> out = attn(node_features)  # Shape: [4, 5, 256]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_feature_dim: int = 0
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_features = use_edge_features

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature projection (if used)
        if use_edge_features:
            self.edge_proj = nn.Linear(edge_feature_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head graph attention.

        Args:
            node_features: Node features [batch_size, num_nodes, hidden_dim]
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            attention_mask: Optional mask [batch_size, num_nodes, num_nodes]
                           (1 = attend, 0 = mask out)

        Returns:
            out: Updated node features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = node_features.shape

        # Project to Q, K, V
        Q = self.q_proj(node_features)  # [batch, num_nodes, hidden_dim]
        K = self.k_proj(node_features)
        V = self.v_proj(node_features)

        # Reshape for multi-head attention
        # [batch, num_nodes, num_heads, head_dim] -> [batch, num_heads, num_nodes, head_dim]
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, num_heads, num_nodes, num_nodes]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add edge features to attention scores (if provided)
        if self.use_edge_features and edge_features is not None:
            # Project edge features to attention bias
            # [batch, num_nodes, num_nodes, edge_dim] -> [batch, num_nodes, num_nodes, num_heads]
            edge_bias = self.edge_proj(edge_features)
            # Transpose to [batch, num_heads, num_nodes, num_nodes]
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            attn_scores = attn_scores + edge_bias

        # Apply attention mask (if provided)
        if attention_mask is not None:
            # Expand mask to [batch, num_heads, num_nodes, num_nodes]
            mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, num_nodes, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        # [batch, num_nodes, num_heads, head_dim] -> [batch, num_nodes, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.hidden_dim
        )

        # Final projection
        out = self.out_proj(attn_output)

        return out


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with self-attention and feedforward network.

    Implements a single transformer layer for graph processing with
    residual connections and layer normalization.

    Args:
        hidden_dim: Dimension of node features
        num_heads: Number of attention heads
        ff_dim: Feedforward network hidden dimension
        dropout: Dropout probability
        use_edge_features: Whether to use edge features
        edge_feature_dim: Dimension of edge features

    Example:
        >>> layer = GraphTransformerLayer(hidden_dim=256, num_heads=8, ff_dim=512)
        >>> x = torch.randn(4, 5, 256)  # [batch, nodes, features]
        >>> out = layer(x)  # Shape: [4, 5, 256]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_feature_dim: int = 0
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * hidden_dim

        self.hidden_dim = hidden_dim

        # Self-attention
        self.self_attn = MultiHeadGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_edge_features=use_edge_features,
            edge_feature_dim=edge_feature_dim
        )

        # Feedforward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through graph transformer layer.

        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            attention_mask: Optional attention mask [batch_size, num_nodes, num_nodes]

        Returns:
            out: Updated node features [batch_size, num_nodes, hidden_dim]
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(x, edge_features, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual connection
        ff_out = self.ff_network(x)
        x = self.norm2(x + ff_out)

        return x


class GraphTransformerStack(nn.Module):
    """
    Stack of Graph Transformer layers.

    Args:
        num_layers: Number of transformer layers
        hidden_dim: Dimension of node features
        num_heads: Number of attention heads
        ff_dim: Feedforward network hidden dimension
        dropout: Dropout probability
        use_edge_features: Whether to use edge features
        edge_feature_dim: Dimension of edge features

    Example:
        >>> model = GraphTransformerStack(num_layers=6, hidden_dim=256, num_heads=8)
        >>> x = torch.randn(4, 5, 256)
        >>> out = model(x)  # Shape: [4, 5, 256]
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_feature_dim: int = 0
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Create stack of transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_edge_features=use_edge_features,
                edge_feature_dim=edge_feature_dim
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            attention_mask: Optional attention mask [batch_size, num_nodes, num_nodes]

        Returns:
            out: Final node features [batch_size, num_nodes, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x, edge_features, attention_mask)

        return x


class GraphPooling(nn.Module):
    """
    Graph-level pooling for aggregating node features.

    Supports multiple pooling strategies: mean, max, sum, attention.

    Args:
        pooling_type: Type of pooling ('mean', 'max', 'sum', 'attention')
        hidden_dim: Dimension of node features (required for attention pooling)

    Example:
        >>> pooling = GraphPooling(pooling_type='attention', hidden_dim=256)
        >>> node_features = torch.randn(4, 5, 256)  # [batch, nodes, features]
        >>> graph_features = pooling(node_features)  # Shape: [4, 256]
    """

    def __init__(self, pooling_type: str = 'mean', hidden_dim: int = None):
        super().__init__()

        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            if hidden_dim is None:
                raise ValueError("hidden_dim required for attention pooling")

            # Attention-based pooling
            self.attn_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, node_features: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            node_features: Node features [batch_size, num_nodes, hidden_dim]
            node_mask: Optional mask for valid nodes [batch_size, num_nodes]
                      (1 = valid, 0 = padding)

        Returns:
            graph_features: Graph-level features [batch_size, hidden_dim]
        """
        if self.pooling_type == 'mean':
            if node_mask is not None:
                # Masked mean pooling
                mask = node_mask.unsqueeze(-1)  # [batch, num_nodes, 1]
                sum_features = (node_features * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)
                return sum_features / count
            else:
                return node_features.mean(dim=1)

        elif self.pooling_type == 'max':
            if node_mask is not None:
                # Masked max pooling
                mask = node_mask.unsqueeze(-1)  # [batch, num_nodes, 1]
                masked_features = node_features.clone()
                masked_features[mask.expand_as(node_features) == 0] = float('-inf')
                return masked_features.max(dim=1)[0]
            else:
                return node_features.max(dim=1)[0]

        elif self.pooling_type == 'sum':
            if node_mask is not None:
                mask = node_mask.unsqueeze(-1)
                return (node_features * mask).sum(dim=1)
            else:
                return node_features.sum(dim=1)

        elif self.pooling_type == 'attention':
            # Compute attention weights
            attn_scores = self.attn_weights(node_features)  # [batch, num_nodes, 1]

            if node_mask is not None:
                # Mask out invalid nodes
                mask = node_mask.unsqueeze(-1)  # [batch, num_nodes, 1]
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, num_nodes, 1]

            # Weighted sum
            graph_features = (node_features * attn_weights).sum(dim=1)  # [batch, hidden_dim]

            return graph_features

        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
