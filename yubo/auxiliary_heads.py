"""
Auxiliary prediction heads attached to the HierarchicalEncoder latent z.

RegressionMLP:    z [B, 8] -> predicted [pole_real, pole_imag]
ClassificationMLP: z [B, 8] -> predicted filter type logits [B, 8]
"""

import torch.nn as nn


class RegressionMLP(nn.Module):
    """
    Predicts dominant pole location [pole_real, pole_imag] from latent z.

    Input:  z [B, 8]
    Output: [B, 2]  — (signed-log pole_real, signed-log pole_imag)
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        return self.net(z)


class ClassificationMLP(nn.Module):
    """
    Predicts filter type (8 classes) from latent z.

    Input:  z [B, 8]
    Output: logits [B, 8]
    """

    def __init__(self, latent_dim: int = 8, num_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, z):
        return self.net(z)
