"""
Admittance-based GNN layers for gain prediction experiment.

Instead of component-specific MLPs (lin_R, lin_C, lin_L), messages use
a single unified edge MLP on [x_j, Re(Y), Im(Y)] — testing whether the
GNN can aggregate complex admittance into a global frequency response.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional


class AdmittanceConv(MessagePassing):
    """
    Admittance-weighted graph convolution.

    Message: att_weight * lin_edge([x_j, Re(Y), Im(Y)])
    where edge_attr = [Re(Y), Im(Y)] (complex admittance components).

    Single unified edge MLP — no component-specific paths.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)

        # Edge MLP: [x_j, Re(Y), Im(Y)] -> out_channels
        self.lin_edge = nn.Linear(out_channels + 2, out_channels, bias=False)

        # Attention on [x_i, x_j]
        self.att = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.bias is not None:
            out = out + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=-1)  # [E, out_channels + 2]
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_weight = self.att(att_input)
        return att_weight * self.lin_edge(msg_input)


class AdmittanceGNN(nn.Module):
    """
    Multi-layer admittance-weighted GNN with residual + LayerNorm.

    Mirrors ImpedanceGNN structure exactly, substituting AdmittanceConv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(AdmittanceConv(in_channels, hidden_channels, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(AdmittanceConv(hidden_channels, hidden_channels, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output layer
        self.convs.append(AdmittanceConv(hidden_channels, out_channels, dropout=dropout))
        self.norms.append(nn.LayerNorm(out_channels))

        # Residual projections
        if use_residual:
            self.residual_proj = nn.ModuleList()
            dims = [in_channels] + [hidden_channels] * (num_layers - 1)
            out_dims = [hidden_channels] * (num_layers - 1) + [out_channels]
            for in_dim, o_dim in zip(dims, out_dims):
                if in_dim != o_dim:
                    self.residual_proj.append(nn.Linear(in_dim, o_dim))
                else:
                    self.residual_proj.append(nn.Identity())

    def forward(self, x, edge_index, edge_attr, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_input = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
            if self.use_residual:
                x = x + self.residual_proj[i](x_input)
        return x
