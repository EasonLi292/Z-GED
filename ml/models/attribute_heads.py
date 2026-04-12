"""Attribute prediction heads for the v2 admittance encoder.

These lightweight MLPs predict circuit characteristics from the 5D
latent mean (mu), enabling spec-driven generation via gradient descent.

Heads are trained jointly with the encoder/decoder and operate on mu
(not z) so that predictions are deterministic at inference time.
"""

import torch
import torch.nn as nn


class FreqHead(nn.Module):
    """Predict log10(characteristic_frequency) from mu."""

    def __init__(self, latent_dim=5, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, mu):
        return self.net(mu).squeeze(-1)


class GainHead(nn.Module):
    """Predict |H(1kHz)| from mu."""

    def __init__(self, latent_dim=5, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, mu):
        return self.net(mu).squeeze(-1)


class TypeHead(nn.Module):
    """Predict filter_type (10-way) from mu."""

    def __init__(self, latent_dim=5, n_types=10):
        super().__init__()
        self.linear = nn.Linear(latent_dim, n_types)

    def forward(self, mu):
        return self.linear(mu)


def kl_divergence(mu, logvar):
    """Standard VAE KL: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar), mean over batch."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=-1).mean()
