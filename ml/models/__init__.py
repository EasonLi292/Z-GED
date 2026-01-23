"""Neural network architectures for circuit generation."""

# Core layers
from .gnn_layers import ImpedanceConv, ImpedanceGNN, GlobalPooling, DeepSets
from .constants import FILTER_TYPES, CIRCUIT_TEMPLATES

# Encoder
from .encoder import HierarchicalEncoder

# Decoder (simplified architecture)
from .decoder import SimplifiedCircuitDecoder, LatentGuidedGraphGPTDecoder
from .decoder_components import LatentGuidedEdgeDecoder

__all__ = [
    # Layers
    'ImpedanceConv',
    'ImpedanceGNN',
    'GlobalPooling',
    'DeepSets',
    # Encoder
    'HierarchicalEncoder',
    # Decoder
    'SimplifiedCircuitDecoder',
    'LatentGuidedGraphGPTDecoder',  # Alias for backward compatibility
    'LatentGuidedEdgeDecoder',
    # Constants
    'FILTER_TYPES',
    'CIRCUIT_TEMPLATES',
]
