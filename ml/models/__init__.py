"""Neural network architectures for circuit generation."""

# Core layers
from .gnn_layers import ImpedanceConv, ImpedanceGNN, GlobalPooling, DeepSets
from .constants import FILTER_TYPES, CIRCUIT_TEMPLATES

# Encoder
from .encoder import HierarchicalEncoder

# Decoder (production models)
from .decoder import LatentGuidedGraphGPTDecoder
from .decoder_components import LatentDecomposer, LatentGuidedEdgeDecoder

__all__ = [
    # Layers
    'ImpedanceConv',
    'ImpedanceGNN',
    'GlobalPooling',
    'DeepSets',
    # Encoder
    'HierarchicalEncoder',
    # Decoder
    'LatentGuidedGraphGPTDecoder',
    'LatentDecomposer',
    'LatentGuidedEdgeDecoder',
    # Constants
    'FILTER_TYPES',
    'CIRCUIT_TEMPLATES',
]
