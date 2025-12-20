"""
Analysis tools for GraphVAE latent space.

Includes:
- Diffusion map analysis
- Latent space visualization
- Dimensionality reduction
"""

from .diffusion_map import (
    DiffusionMap,
    compute_diffusion_map,
    analyze_eigenspectrum,
    estimate_intrinsic_dimension,
    visualize_diffusion_coordinates
)

__all__ = [
    'DiffusionMap',
    'compute_diffusion_map',
    'analyze_eigenspectrum',
    'estimate_intrinsic_dimension',
    'visualize_diffusion_coordinates'
]
