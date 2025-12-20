"""
Diffusion Map Analysis for Latent Space Dimensionality.

Diffusion maps use spectral decomposition of a diffusion operator to reveal
the intrinsic geometry of high-dimensional data. By examining the eigenvalue
spectrum, we can estimate the true dimensionality of the latent space.

References:
    Coifman & Lafon (2006). "Diffusion maps." Applied and computational harmonic analysis.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from pathlib import Path


class DiffusionMap:
    """
    Diffusion map for analyzing latent space geometry.

    The diffusion map constructs a Markov chain on the data manifold and uses
    the eigenvalues/eigenvectors of the transition matrix to find low-dimensional
    embeddings that preserve the diffusion distance.

    Args:
        epsilon: Kernel bandwidth (default: auto-select using median distance)
        n_components: Number of diffusion components to compute (default: 10)
        alpha: Normalization parameter (0=no normalization, 1=Laplace-Beltrami)

    Attributes:
        eigenvalues_: Eigenvalues of diffusion operator (sorted descending)
        eigenvectors_: Eigenvectors (diffusion coordinates)
        epsilon_: Selected kernel bandwidth
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        n_components: int = 10,
        alpha: float = 0.5
    ):
        self.epsilon = epsilon
        self.n_components = n_components
        self.alpha = alpha

        # Fitted attributes
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.epsilon_ = None
        self.distances_ = None

    def fit(self, X: np.ndarray) -> 'DiffusionMap':
        """
        Fit diffusion map to latent vectors.

        Args:
            X: Latent vectors [N, D]

        Returns:
            self
        """
        N = X.shape[0]

        # Step 1: Compute pairwise distances
        distances = squareform(pdist(X, metric='euclidean'))
        self.distances_ = distances

        # Step 2: Select epsilon (kernel bandwidth) if not provided
        if self.epsilon is None:
            # Use median heuristic: epsilon = median(distances)^2
            self.epsilon_ = np.median(distances[distances > 0]) ** 2
        else:
            self.epsilon_ = self.epsilon

        # Step 3: Compute affinity matrix (Gaussian kernel)
        K = np.exp(-distances ** 2 / self.epsilon_)

        # Step 4: Normalize (anisotropic diffusion)
        # D^(-alpha) K D^(-alpha) where D is degree matrix
        D = np.sum(K, axis=1)
        D_alpha = np.diag(D ** (-self.alpha))
        K_norm = D_alpha @ K @ D_alpha

        # Step 5: Compute row-normalized transition matrix
        D_norm = np.sum(K_norm, axis=1)
        P = K_norm / D_norm[:, np.newaxis]

        # Step 6: Eigenvalue decomposition
        # Use symmetric matrix for numerical stability
        D_sqrt = np.diag(np.sqrt(D_norm))
        D_sqrt_inv = np.diag(1.0 / np.sqrt(D_norm))
        M = D_sqrt @ P @ D_sqrt_inv

        # Compute eigenvectors/eigenvalues
        eigenvalues, eigenvectors = eigh(M)

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Transform back to eigenvectors of P
        eigenvectors = D_sqrt_inv @ eigenvectors

        # Keep top n_components
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.eigenvectors_ = eigenvectors[:, :self.n_components]

        return self

    def transform(self, t: int = 1) -> np.ndarray:
        """
        Get diffusion coordinates at time t.

        Args:
            t: Diffusion time (default: 1)

        Returns:
            Diffusion coordinates [N, n_components]
        """
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() before transform()")

        # Diffusion coordinates: Ψ_i(t) = λ_i^t * φ_i
        diffusion_coords = self.eigenvectors_ * (self.eigenvalues_ ** t)[np.newaxis, :]

        return diffusion_coords

    def fit_transform(self, X: np.ndarray, t: int = 1) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(t)


def compute_diffusion_map(
    latent_vectors: np.ndarray,
    epsilon: Optional[float] = None,
    n_components: int = 10,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute diffusion map decomposition.

    Args:
        latent_vectors: Latent space embeddings [N, D]
        epsilon: Kernel bandwidth (auto if None)
        n_components: Number of components to compute
        alpha: Normalization parameter

    Returns:
        eigenvalues: Top eigenvalues [n_components]
        eigenvectors: Top eigenvectors [N, n_components]
        epsilon: Selected kernel bandwidth
    """
    dmap = DiffusionMap(epsilon=epsilon, n_components=n_components, alpha=alpha)
    dmap.fit(latent_vectors)

    return dmap.eigenvalues_, dmap.eigenvectors_, dmap.epsilon_


def analyze_eigenspectrum(
    eigenvalues: np.ndarray,
    threshold: float = 0.05,
    plot: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Analyze eigenvalue spectrum to determine intrinsic dimensionality.

    Three methods:
        1. Eigenvalue decay (spectral gap detection)
        2. Cumulative variance explained
        3. Elbow detection (maximum curvature)

    Args:
        eigenvalues: Sorted eigenvalues (descending)
        threshold: Threshold for eigenvalue significance (default: 0.05)
        plot: Whether to plot spectrum (default: True)
        save_path: Path to save plot (optional)

    Returns:
        Dictionary with:
            - intrinsic_dim_gap: Dimension from spectral gap
            - intrinsic_dim_variance: Dimension for 95% variance
            - intrinsic_dim_elbow: Dimension from elbow method
            - eigenvalue_ratios: Consecutive eigenvalue ratios
            - cumulative_variance: Cumulative variance explained
    """
    n = len(eigenvalues)

    # Method 1: Spectral gap (largest drop in eigenvalues)
    eigenvalue_ratios = eigenvalues[1:] / eigenvalues[:-1]
    spectral_gaps = 1 - eigenvalue_ratios
    intrinsic_dim_gap = np.argmax(spectral_gaps) + 1

    # Method 2: Cumulative variance explained (95% threshold)
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    intrinsic_dim_variance = np.argmax(cumulative_variance >= 0.95) + 1

    # Method 3: Elbow method (maximum curvature)
    # Normalize eigenvalues to [0, 1] range
    eigenvalues_norm = (eigenvalues - eigenvalues.min()) / (eigenvalues.max() - eigenvalues.min())

    # Compute curvature using finite differences
    indices = np.arange(n)
    # Fit line from first to last point
    line = np.linspace(eigenvalues_norm[0], eigenvalues_norm[-1], n)
    distances = np.abs(eigenvalues_norm - line)
    intrinsic_dim_elbow = np.argmax(distances) + 1

    # Plot eigenvalue spectrum
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Eigenvalue decay
        ax = axes[0, 0]
        ax.plot(range(1, n+1), eigenvalues, 'bo-', linewidth=2, markersize=8)
        ax.axvline(intrinsic_dim_gap, color='r', linestyle='--',
                   label=f'Spectral gap: {intrinsic_dim_gap}')
        ax.axvline(intrinsic_dim_elbow, color='g', linestyle='--',
                   label=f'Elbow: {intrinsic_dim_elbow}')
        ax.set_xlabel('Component Index', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title('Eigenvalue Spectrum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Log-scale eigenvalues
        ax = axes[0, 1]
        ax.semilogy(range(1, n+1), eigenvalues, 'ro-', linewidth=2, markersize=8)
        ax.axvline(intrinsic_dim_gap, color='r', linestyle='--', alpha=0.5)
        ax.axvline(intrinsic_dim_elbow, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('Component Index', fontsize=12)
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
        ax.set_title('Eigenvalue Spectrum (Log Scale)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 3: Cumulative variance explained
        ax = axes[1, 0]
        ax.plot(range(1, n+1), cumulative_variance * 100, 'go-',
                linewidth=2, markersize=8)
        ax.axhline(95, color='r', linestyle='--', label='95% threshold')
        ax.axvline(intrinsic_dim_variance, color='r', linestyle='--',
                   label=f'95% at d={intrinsic_dim_variance}')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 4: Eigenvalue ratios (spectral gaps)
        ax = axes[1, 1]
        ax.plot(range(2, n+1), spectral_gaps, 'mo-', linewidth=2, markersize=8)
        ax.axvline(intrinsic_dim_gap + 1, color='r', linestyle='--',
                   label=f'Max gap after d={intrinsic_dim_gap}')
        ax.set_xlabel('Component Index', fontsize=12)
        ax.set_ylabel('Spectral Gap (1 - λ_i+1/λ_i)', fontsize=12)
        ax.set_title('Spectral Gaps', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eigenspectrum plot saved to: {save_path}")
        else:
            plt.savefig('/tmp/eigenspectrum.png', dpi=300, bbox_inches='tight')
            print(f"Eigenspectrum plot saved to: /tmp/eigenspectrum.png")

        plt.close()

    return {
        'intrinsic_dim_gap': int(intrinsic_dim_gap),
        'intrinsic_dim_variance': int(intrinsic_dim_variance),
        'intrinsic_dim_elbow': int(intrinsic_dim_elbow),
        'eigenvalue_ratios': eigenvalue_ratios,
        'cumulative_variance': cumulative_variance,
        'spectral_gaps': spectral_gaps
    }


def estimate_intrinsic_dimension(
    latent_vectors: np.ndarray,
    n_components: int = 24,
    epsilon: Optional[float] = None,
    plot: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Estimate intrinsic dimensionality of latent space using diffusion maps.

    This is the main function to use for analyzing latent space dimensionality.

    Args:
        latent_vectors: Latent space embeddings [N, D]
        n_components: Number of components to analyze (default: 24)
        epsilon: Kernel bandwidth (auto if None)
        plot: Whether to generate plots (default: True)
        save_path: Path to save plots (optional)

    Returns:
        Dictionary with:
            - eigenvalues: Eigenvalues
            - eigenvectors: Eigenvectors
            - epsilon: Kernel bandwidth used
            - intrinsic_dim_gap: Estimated dimension (spectral gap method)
            - intrinsic_dim_variance: Estimated dimension (variance method)
            - intrinsic_dim_elbow: Estimated dimension (elbow method)
            - recommended_dim: Consensus estimate
            - analysis: Full eigenspectrum analysis
    """
    print("\n" + "="*70)
    print("DIFFUSION MAP ANALYSIS - INTRINSIC DIMENSIONALITY ESTIMATION")
    print("="*70)
    print(f"Latent vectors: {latent_vectors.shape}")
    print(f"Analyzing top {n_components} components...")

    # Compute diffusion map
    eigenvalues, eigenvectors, epsilon = compute_diffusion_map(
        latent_vectors,
        epsilon=epsilon,
        n_components=n_components,
        alpha=0.5
    )

    print(f"\nKernel bandwidth (epsilon): {epsilon:.4f}")
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")

    # Analyze eigenspectrum
    analysis = analyze_eigenspectrum(
        eigenvalues,
        plot=plot,
        save_path=save_path
    )

    # Consensus estimate (use spectral gap as primary)
    recommended_dim = analysis['intrinsic_dim_gap']

    print("\n" + "-"*70)
    print("INTRINSIC DIMENSION ESTIMATES:")
    print("-"*70)
    print(f"  Spectral gap method:     {analysis['intrinsic_dim_gap']} dimensions")
    print(f"  Variance method (95%):   {analysis['intrinsic_dim_variance']} dimensions")
    print(f"  Elbow method:            {analysis['intrinsic_dim_elbow']} dimensions")
    print(f"\n  Recommended:             {recommended_dim} dimensions")
    print("-"*70)

    # Interpret results
    current_latent_dim = latent_vectors.shape[1]
    if recommended_dim < current_latent_dim:
        reduction = (1 - recommended_dim / current_latent_dim) * 100
        print(f"\nInterpretation:")
        print(f"  Current latent space: {current_latent_dim}D")
        print(f"  Intrinsic dimension:  ~{recommended_dim}D")
        print(f"  Potential reduction:  {reduction:.1f}%")
        print(f"\n  The latent space has ~{current_latent_dim - recommended_dim} redundant dimensions.")
        print(f"  Consider reducing to {recommended_dim}D for more efficient representation.")
    else:
        print(f"\nInterpretation:")
        print(f"  Current latent space ({current_latent_dim}D) is well-matched to intrinsic complexity.")
        print(f"  No dimensionality reduction recommended.")

    print("="*70 + "\n")

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'epsilon': epsilon,
        'intrinsic_dim_gap': analysis['intrinsic_dim_gap'],
        'intrinsic_dim_variance': analysis['intrinsic_dim_variance'],
        'intrinsic_dim_elbow': analysis['intrinsic_dim_elbow'],
        'recommended_dim': recommended_dim,
        'analysis': analysis
    }


def visualize_diffusion_coordinates(
    latent_vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    n_components: int = 3,
    save_path: Optional[str] = None
):
    """
    Visualize data in diffusion coordinate space.

    Args:
        latent_vectors: Latent space embeddings [N, D]
        labels: Class labels for coloring (optional) [N]
        label_names: Names for each class (optional)
        n_components: Number of diffusion components to plot (2 or 3)
        save_path: Path to save plot (optional)
    """
    # Compute diffusion map
    dmap = DiffusionMap(n_components=max(3, n_components))
    coords = dmap.fit_transform(latent_vectors, t=1)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                               cmap='tab10', s=100, alpha=0.6, edgecolors='k')
            if label_names is not None:
                handles, _ = scatter.legend_elements()
                ax.legend(handles, label_names, title="Filter Type",
                         loc='best', framealpha=0.9)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6, edgecolors='k')

        ax.set_xlabel('Diffusion Coordinate 1', fontsize=12)
        ax.set_ylabel('Diffusion Coordinate 2', fontsize=12)
        ax.set_title('Diffusion Map Embedding', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    else:  # 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if labels is not None:
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=labels, cmap='tab10', s=100, alpha=0.6,
                               edgecolors='k', linewidths=0.5)
            if label_names is not None:
                handles, _ = scatter.legend_elements()
                ax.legend(handles, label_names, title="Filter Type",
                         loc='best', framealpha=0.9)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      s=100, alpha=0.6, edgecolors='k', linewidths=0.5)

        ax.set_xlabel('Diffusion Coord 1', fontsize=12)
        ax.set_ylabel('Diffusion Coord 2', fontsize=12)
        ax.set_zlabel('Diffusion Coord 3', fontsize=12)
        ax.set_title('Diffusion Map Embedding (3D)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diffusion coordinates plot saved to: {save_path}")
    else:
        plt.savefig('/tmp/diffusion_coords.png', dpi=300, bbox_inches='tight')
        print(f"Diffusion coordinates plot saved to: /tmp/diffusion_coords.png")

    plt.close()
