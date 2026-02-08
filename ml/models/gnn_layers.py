"""
Custom GNN layers for circuit graph processing.

Implements impedance-aware message passing that respects the
physical properties of electrical components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class ImpedanceConv(MessagePassing):
    """
    Component-aware graph convolution layer.

    Uses separate message transformations for each component type (R, C, L)
    to ensure the network distinguishes between different components.
    Component values directly condition the message via concatenation.

    Edge attr format: [log10(R), log10(C), log10(L)]
    where 0 = component absent (log10 values never naturally equal 0).
    Presence masks are derived internally from nonzero values.

    Args:
        in_channels: Input node feature dimension
        out_channels: Output node feature dimension
        edge_dim: Edge feature dimension (default: 3)
        aggr: Aggregation method (default: 'add')
        bias: Whether to add bias (default: True)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 3,
        aggr: str = 'add',
        bias: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout = dropout

        # Node feature transformation
        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)

        # Component-type-specific message transformations
        # Each takes [x_j, component_value] as input (out_channels + 1)
        # The value directly conditions the message
        self.lin_R = nn.Linear(out_channels + 1, out_channels)
        self.lin_C = nn.Linear(out_channels + 1, out_channels)
        self.lin_L = nn.Linear(out_channels + 1, out_channels)

        # Attention for edge importance (based on node features only)
        self.att = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_node.reset_parameters()
        self.lin_R.reset_parameters()
        self.lin_C.reset_parameters()
        self.lin_L.reset_parameters()

        # Initialize attention MLP
        for layer in self.att:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
            return_attention_weights: Whether to return attention weights

        Returns:
            out: Updated node features [N, out_channels]
            attention_weights: (Optional) Edge attention weights
        """
        # Transform node features
        x = self.lin_node(x)

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=None
        )

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Apply dropout
        out = F.dropout(out, p=self.dropout, training=self.training)

        if return_attention_weights:
            # Compute attention weights for visualization
            row, col = edge_index
            att_weights = self._compute_attention(x[row], x[col])
            return out, (edge_index, att_weights)

        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct messages from neighbors using value-conditioned component transforms.

        Args:
            x_i: Target node features [E, out_channels]
            x_j: Source node features [E, out_channels]
            edge_attr: Edge features [E, 3]
                Format: [log10(R), log10(C), log10(L)] where 0 = absent

        Returns:
            messages: Weighted messages [E, out_channels]
        """
        # Extract log10 component values
        val_R = edge_attr[:, 0:1]  # [E, 1]
        val_C = edge_attr[:, 1:2]  # [E, 1]
        val_L = edge_attr[:, 2:3]  # [E, 1]

        # Derive presence masks (0 = absent, nonzero = present)
        is_R = (val_R.abs() > 0.01).float()  # [E, 1]
        is_C = (val_C.abs() > 0.01).float()  # [E, 1]
        is_L = (val_L.abs() > 0.01).float()  # [E, 1]

        # Value-conditioned component transforms: cat([x_j, value])
        msg_R = self.lin_R(torch.cat([x_j, val_R], dim=-1))  # [E, out_channels]
        msg_C = self.lin_C(torch.cat([x_j, val_C], dim=-1))  # [E, out_channels]
        msg_L = self.lin_L(torch.cat([x_j, val_L], dim=-1))  # [E, out_channels]

        # Route by presence mask (compound components activate multiple transforms)
        message = is_R * msg_R + is_C * msg_C + is_L * msg_L

        # Compute attention weights (based on node features, not edge type)
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_weight = self.att(att_input)

        # Apply attention weighting
        message = att_weight * message

        return message

    def _compute_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights for visualization."""
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_weight = self.att(att_input)
        return att_weight.squeeze(-1)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class ImpedanceGNN(nn.Module):
    """
    Multi-layer impedance-aware GNN.

    Stacks multiple ImpedanceConv layers with residual connections
    and layer normalization.

    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of ImpedanceConv layers
        edge_dim: Edge feature dimension
        dropout: Dropout probability
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        edge_dim: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual

        # Build layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            ImpedanceConv(in_channels, hidden_channels, edge_dim, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                ImpedanceConv(hidden_channels, hidden_channels, edge_dim, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output layer
        self.convs.append(
            ImpedanceConv(hidden_channels, out_channels, edge_dim, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(out_channels))

        # Projection for residual connections (if dimensions don't match)
        if use_residual:
            self.residual_proj = nn.ModuleList()
            dims = [in_channels] + [hidden_channels] * (num_layers - 1)
            for i, (in_dim, out_dim) in enumerate(zip(dims, [hidden_channels] * (num_layers - 1) + [out_channels])):
                if in_dim != out_dim:
                    self.residual_proj.append(nn.Linear(in_dim, out_dim))
                else:
                    self.residual_proj.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch assignment for nodes [N]

        Returns:
            out: Output node features [N, out_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_input = x

            # Convolution
            x = conv(x, edge_index, edge_attr)

            # Layer normalization
            x = norm(x)

            # Activation (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)

            # Residual connection
            if self.use_residual:
                x = x + self.residual_proj[i](x_input)

        return x


class GlobalPooling(nn.Module):
    """
    Global pooling layer for graph-level representations.

    Combines mean, max, and sum pooling for a richer representation.

    Args:
        pooling_types: List of pooling types to use (default: ['mean', 'max', 'sum'])
    """

    def __init__(self, pooling_types: list = ['mean', 'max']):
        super().__init__()
        self.pooling_types = pooling_types

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            x: Node features [N, D]
            batch: Batch assignment for nodes [N]

        Returns:
            out: Graph-level features [B, D * num_pooling_types]
        """
        pooled = []

        # Get unique batch indices
        batch_size = batch.max().item() + 1

        for pool_type in self.pooling_types:
            if pool_type == 'mean':
                # Mean pooling
                pooled_feat = torch.zeros(
                    batch_size, x.size(1),
                    dtype=x.dtype, device=x.device
                )
                pooled_feat.index_add_(0, batch, x)
                counts = torch.zeros(batch_size, dtype=torch.float, device=x.device)
                counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
                pooled_feat = pooled_feat / counts.unsqueeze(1).clamp(min=1)
                pooled.append(pooled_feat)

            elif pool_type == 'max':
                # Max pooling
                pooled_feat = torch.full(
                    (batch_size, x.size(1)),
                    float('-inf'),
                    dtype=x.dtype, device=x.device
                )
                for i in range(batch_size):
                    mask = batch == i
                    if mask.any():
                        pooled_feat[i] = x[mask].max(dim=0)[0]
                pooled.append(pooled_feat)

            elif pool_type == 'sum':
                # Sum pooling
                pooled_feat = torch.zeros(
                    batch_size, x.size(1),
                    dtype=x.dtype, device=x.device
                )
                pooled_feat.index_add_(0, batch, x)
                pooled.append(pooled_feat)

        # Concatenate different pooling types
        out = torch.cat(pooled, dim=-1)
        return out


class DeepSets(nn.Module):
    """
    DeepSets architecture for permutation-invariant encoding.

    Used for encoding variable-length sets of poles/zeros.
    Reference: Zaheer et al., "Deep Sets" (2017)

    Args:
        input_dim: Dimension of each element (2 for complex numbers as [real, imag])
        hidden_dim: Hidden dimension for element encoding
        output_dim: Output dimension for aggregated representation
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 8
    ):
        super().__init__()

        # Element-wise encoder (φ function)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Aggregated encoder (ρ function)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a variable-length set.

        Args:
            x: Elements [N, input_dim] or [B, max_len, input_dim]
            mask: Optional mask for padded elements [N] or [B, max_len]

        Returns:
            out: Aggregated representation [output_dim] or [B, output_dim]
        """
        # Encode each element
        h = self.phi(x)  # [N, hidden_dim] or [B, max_len, hidden_dim]

        # Apply mask if provided
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        # Aggregate (sum pooling for permutation invariance)
        if h.dim() == 3:
            # Batched input: [B, max_len, hidden_dim]
            h_agg = h.sum(dim=1)  # [B, hidden_dim]
            if mask is not None:
                # Normalize by count
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                h_agg = h_agg / counts
        else:
            # Single set: [N, hidden_dim]
            h_agg = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Aggregate encoding
        out = self.rho(h_agg)

        return out
