"""Loss functions for circuit generation training."""

from .gumbel_softmax_loss import GumbelSoftmaxCircuitLoss
from .connectivity_loss import ConnectivityLoss

__all__ = [
    'GumbelSoftmaxCircuitLoss',
    'ConnectivityLoss',
]
