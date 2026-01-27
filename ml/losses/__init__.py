"""Loss functions for circuit generation training."""

from .circuit_loss import CircuitLoss
from .connectivity_loss import ConnectivityLoss

__all__ = [
    'CircuitLoss',
    'ConnectivityLoss',
]
