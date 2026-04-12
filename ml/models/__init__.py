"""Neural network architectures for circuit generation."""

# Core layers
from .gnn_layers import ImpedanceConv, ImpedanceGNN, GlobalPooling
from .constants import FILTER_TYPES, CIRCUIT_TEMPLATES

# Encoder (v1 — 8D latent, impedance-aware GNN)
from .encoder import HierarchicalEncoder

# Encoder (v2 — 5D structured VAE, admittance-polynomial features)
from .admittance_encoder import AdmittanceConv, AdmittanceEncoder

# Attribute heads (v2 — predict circuit specs from latent mu)
from .attribute_heads import FreqHead, GainHead, TypeHead, kl_divergence

# Decoder (sequence-based, shared by v1 and v2)
from .decoder import SequenceDecoder
from .vocabulary import CircuitVocabulary

# v2 constants
from .constants import (
    G_REF, C_REF, L_INV_REF,
    FILTER_TYPES_V2, TYPE_TO_IDX,
)

__all__ = [
    # Layers
    'ImpedanceConv',
    'ImpedanceGNN',
    'GlobalPooling',
    # Encoder (v1)
    'HierarchicalEncoder',
    # Encoder (v2)
    'AdmittanceConv',
    'AdmittanceEncoder',
    # Attribute heads (v2)
    'FreqHead',
    'GainHead',
    'TypeHead',
    'kl_divergence',
    # Decoder
    'SequenceDecoder',
    'CircuitVocabulary',
    # Constants
    'FILTER_TYPES',
    'CIRCUIT_TEMPLATES',
    'G_REF',
    'C_REF',
    'L_INV_REF',
    'FILTER_TYPES_V2',
    'TYPE_TO_IDX',
]
