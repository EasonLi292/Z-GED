"""Loss functions for GraphVAE training."""

from .reconstruction import (
    ReconstructionLoss,
    TemplateAwareReconstructionLoss,
    EdgeFeatureReconstructionLoss
)
from .transfer_function import (
    TransferFunctionLoss,
    SimplifiedTransferFunctionLoss,
    chamfer_distance
)
from .ged_metric import (
    GEDMetricLoss,
    SoftGEDMetricLoss,
    RegularizedGEDMetricLoss
)
from .composite import (
    CompositeLoss,
    SimplifiedCompositeLoss,
    WeightScheduler
)

__all__ = [
    'ReconstructionLoss',
    'TemplateAwareReconstructionLoss',
    'EdgeFeatureReconstructionLoss',
    'TransferFunctionLoss',
    'SimplifiedTransferFunctionLoss',
    'chamfer_distance',
    'GEDMetricLoss',
    'SoftGEDMetricLoss',
    'RegularizedGEDMetricLoss',
    'CompositeLoss',
    'SimplifiedCompositeLoss',
    'WeightScheduler'
]
