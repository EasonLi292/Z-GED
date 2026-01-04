"""Loss functions for circuit generation training.

Current production loss:
- GumbelSoftmaxCircuitLoss: Main training loss for joint edge-component prediction
- ConnectivityLoss: Ensures VIN/VOUT connectivity (used internally by GumbelSoftmax)
"""

from .gumbel_softmax_loss import GumbelSoftmaxCircuitLoss
from .connectivity_loss import ConnectivityLoss, SpectralConnectivityLoss

__all__ = [
    'GumbelSoftmaxCircuitLoss',
    'ConnectivityLoss',
    'SpectralConnectivityLoss',
]
