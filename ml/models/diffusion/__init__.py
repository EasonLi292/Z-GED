"""
Diffusion models for circuit graph generation.

This package implements autoregressive diffusion models for generating
arbitrary circuit topologies with variable-length poles and zeros.
"""

from .noise_schedules import (
    get_cosine_schedule,
    get_discrete_transition_matrix,
    get_cumulative_transition_matrix,
    extract_index,
    add_noise_continuous,
    add_noise_discrete,
    compute_posterior_mean_variance
)
from .time_embedding import (
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    AdaptiveGroupNorm
)
from .graph_transformer import (
    MultiHeadGraphAttention,
    GraphTransformerLayer,
    GraphTransformerStack,
    GraphPooling
)
from .denoising_network import (
    DiffusionGraphTransformer,
    ConditionalDiffusionDecoder
)

__all__ = [
    # Noise schedules
    'get_cosine_schedule',
    'get_discrete_transition_matrix',
    'get_cumulative_transition_matrix',
    'extract_index',
    'add_noise_continuous',
    'add_noise_discrete',
    'compute_posterior_mean_variance',

    # Time embeddings
    'SinusoidalTimeEmbedding',
    'TimeEmbeddingMLP',
    'AdaptiveGroupNorm',

    # Graph transformer
    'MultiHeadGraphAttention',
    'GraphTransformerLayer',
    'GraphTransformerStack',
    'GraphPooling',

    # Denoising network
    'DiffusionGraphTransformer',
    'ConditionalDiffusionDecoder',
]
