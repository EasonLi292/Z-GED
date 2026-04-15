"""
Physics-informed admittance encoder (v2).

Each edge is characterised by its admittance-polynomial coefficients
    [G, C, L_inv]  =  [1/R, C_farads, 1/L]
so that Y(s) = G + sC + L_inv/s. These coefficients are ADDITIVE under
parallel combination — sum-aggregation in a GNN at a shared node
computes the exact parallel admittance of incident branches.

Coefficient scaling uses a fixed Box-Cox transform with gamma=0.5:
    f(x) = ((1+x)^0.5 - 1) / 0.5 = 2*(sqrt(1+x) - 1)
This provides mild compression of outlier admittance values while
preserving magnitude differences. Gamma=0.5 was selected by grid search
over [-1.0, 1.0] (see ARCHITECTURE.md).

Key properties:
  - f(0)=0: absent components contribute nothing.
  - f'(0)=1: small admittances are treated linearly.
  - f(1)=0.83, f(10)=4.63, f(100)=18.1: moderate compression.
  - Not learnable — avoids init-bias trap where learned scaling
    parameters stay near their initialization regardless of optimum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# Node-feature one-hot index → semantic label
# Matches `ml/data/dataset.py:194` — [is_GND, is_VIN, is_VOUT, is_INTERNAL]
NODE_GND = 0
NODE_VIN = 1
NODE_VOUT = 2


def _make_phi(dim: int) -> nn.Sequential:
    """2-layer MLP for per-coefficient neighbour transform.

    Inner layer has bias + ReLU for expressiveness; outer layer is
    bias=False so that coeff * φ(x_j) produces zero when coeff is zero.
    """
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim, bias=False),
    )


BOXCOX_GAMMA = 0.5


def _boxcox(x: torch.Tensor) -> torch.Tensor:
    """Fixed Box-Cox transform: ((1+x)^gamma - 1) / gamma with gamma=0.5.

    Equivalent to 2*(sqrt(1+x) - 1). f(0)=0, f'(0)=1.
    """
    return (torch.sqrt(1 + x) - 1) * 2  # closed-form for gamma=0.5


class AdmittanceConv(MessagePassing):
    """Message passing with message proportional to each edge coefficient.

        msg = f(g) · φ_G(x_j) + f(c) · φ_C(x_j) + f(l) · φ_L(x_j)

    where f is a fixed Box-Cox transform (gamma=0.5) and
        φ_k = 2-layer MLP (nonlinear in x_j, bias=False on outer layer)

    Aggregation is SUM. The scalar product f(coeff)·φ(x_j) preserves the
    parallel-admittance prior at small values (f'(0)=1, so near-zero
    coefficients add linearly). For larger values, f compresses outliers
    to help the GNN generalise across the admittance range.

    Post-aggregation (at the target node): bias, LayerNorm, ReLU, residual.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: int = 3, dropout: float = 0.1):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)

        # Per-coefficient neighbour transform (2-layer MLP).
        self.phi_G = _make_phi(out_channels)
        self.phi_C = _make_phi(out_channels)
        self.phi_L = _make_phi(out_channels)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def message(self, x_j, edge_attr):
        g = _boxcox(edge_attr[:, 0:1])
        c = _boxcox(edge_attr[:, 1:2])
        l = _boxcox(edge_attr[:, 2:3])

        return g * self.phi_G(x_j) + c * self.phi_C(x_j) + l * self.phi_L(x_j)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'({self.in_channels}, {self.out_channels}, '
                f'edge_dim={self.edge_dim})')


class AdmittanceEncoder(nn.Module):
    """Physics-informed GNN encoder with optional VAE bottleneck.

    Inputs:
        x          : [N, 4] one-hot node types [GND, VIN, VOUT, INTERNAL]
        edge_index : [2, E]
        edge_attr  : [E, 3] normalised [G/G_ref, C/C_ref, L_inv/L_inv_ref]
        batch      : [N] graph ids

    Output (vae=True):
        z      : [B, latent_dim]  sampled latent (= mu at eval)
        mu     : [B, latent_dim]  posterior mean
        logvar : [B, latent_dim]  posterior log-variance

    Output (vae=False):
        h : [B, latent_dim]  deterministic signature (v1 compat)
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 5,
        num_layers: int = 3,
        dropout: float = 0.1,
        vae: bool = True,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.vae = vae

        # GNN stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [node_feature_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(
                AdmittanceConv(dims[i], dims[i + 1], dropout=dropout))
            self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                self.residual_projs.append(nn.Identity())

        # Terminal readout
        if vae:
            # Structured 5D latent: 2D topo + 1D VIN + 1D VOUT + 1D GND
            # Topology: from global terminal concatenation
            self.mu_topo = nn.Linear(3 * hidden_dim, 2)
            self.logvar_topo = nn.Linear(3 * hidden_dim, 2)
            # Terminals: from individual GNN embeddings
            self.mu_vin = nn.Linear(hidden_dim, 1)
            self.logvar_vin = nn.Linear(hidden_dim, 1)
            self.mu_vout = nn.Linear(hidden_dim, 1)
            self.logvar_vout = nn.Linear(hidden_dim, 1)
            self.mu_gnd = nn.Linear(hidden_dim, 1)
            self.logvar_gnd = nn.Linear(hidden_dim, 1)
            # Init logvar bias to -2 → initial std ≈ 0.37
            for m in [self.logvar_topo, self.logvar_vin,
                       self.logvar_vout, self.logvar_gnd]:
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, -2.0)
        else:
            self.readout = nn.Linear(3 * hidden_dim, latent_dim)

    @property
    def h_dim(self):
        """Backward-compatible alias."""
        return self.latent_dim

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def _extract_terminals(self, x, h_nodes, batch):
        """Pick VIN, VOUT, GND embeddings per graph in the batch."""
        node_types = x.argmax(dim=-1)
        batch_size = int(batch.max().item()) + 1
        device = h_nodes.device

        vin_feat = torch.zeros(batch_size, self.hidden_dim, device=device)
        vout_feat = torch.zeros(batch_size, self.hidden_dim, device=device)
        gnd_feat = torch.zeros(batch_size, self.hidden_dim, device=device)

        vin_feat[batch[node_types == NODE_VIN]] = h_nodes[node_types == NODE_VIN]
        vout_feat[batch[node_types == NODE_VOUT]] = h_nodes[node_types == NODE_VOUT]
        gnd_feat[batch[node_types == NODE_GND]] = h_nodes[node_types == NODE_GND]

        return vin_feat, vout_feat, gnd_feat

    def forward(self, x, edge_index, edge_attr, batch):
        # --- GNN stack ---
        h_nodes = x
        for conv, norm, proj in zip(
            self.convs, self.norms, self.residual_projs
        ):
            h_in = h_nodes
            h_nodes = conv(h_nodes, edge_index, edge_attr)
            h_nodes = norm(h_nodes)
            h_nodes = F.relu(h_nodes) + proj(h_in)

        # --- Terminal readout ---
        vin_feat, vout_feat, gnd_feat = self._extract_terminals(
            x, h_nodes, batch)

        if not self.vae:
            h = self.readout(
                torch.cat([vin_feat, vout_feat, gnd_feat], dim=-1))
            return h

        # --- Structured VAE readout ---
        terminal_cat = torch.cat([vin_feat, vout_feat, gnd_feat], dim=-1)

        mu_topo = self.mu_topo(terminal_cat)          # [B, 2]
        lv_topo = self.logvar_topo(terminal_cat)       # [B, 2]

        mu_vin = self.mu_vin(vin_feat)                 # [B, 1]
        lv_vin = self.logvar_vin(vin_feat)             # [B, 1]
        mu_vout = self.mu_vout(vout_feat)              # [B, 1]
        lv_vout = self.logvar_vout(vout_feat)          # [B, 1]
        mu_gnd = self.mu_gnd(gnd_feat)                 # [B, 1]
        lv_gnd = self.logvar_gnd(gnd_feat)             # [B, 1]

        mu = torch.cat([mu_topo, mu_vin, mu_vout, mu_gnd], dim=-1)      # [B, 5]
        logvar = torch.cat([lv_topo, lv_vin, lv_vout, lv_gnd], dim=-1)  # [B, 5]
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar
