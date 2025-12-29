"""Neural network architectures for GraphVAE."""

from .gnn_layers import ImpedanceConv, ImpedanceGNN, GlobalPooling, DeepSets
from .encoder import HierarchicalEncoder
from .conditional_encoder import ConditionalHierarchicalEncoder
from .constants import FILTER_TYPES, CIRCUIT_TEMPLATES

# New latent-guided models (current approach)
from .graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from .latent_guided_decoder import LatentDecomposer, LatentGuidedEdgeDecoder

# Note: ConditionalVariableLengthDecoder removed (old approach, depends on deleted variable_decoder)

__all__ = [
    'ImpedanceConv',
    'ImpedanceGNN',
    'GlobalPooling',
    'DeepSets',
    'HierarchicalEncoder',
    'ConditionalHierarchicalEncoder',
    'FILTER_TYPES',
    'CIRCUIT_TEMPLATES',
    'LatentGuidedGraphGPTDecoder',
    'LatentDecomposer',
    'LatentGuidedEdgeDecoder'
]
