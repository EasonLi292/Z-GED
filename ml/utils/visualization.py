"""
Visualization utilities for GraphVAE analysis.

Implements visualizations for:
    1. Latent space exploration (t-SNE, PCA)
    2. Reconstruction quality
    3. Training dynamics
    4. Latent dimension interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class LatentSpaceVisualizer:
    """
    Visualize latent space using dimensionality reduction.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        self.figsize = figsize
        self.filter_type_colors = {
            0: '#1f77b4',  # low_pass - blue
            1: '#ff7f0e',  # high_pass - orange
            2: '#2ca02c',  # band_pass - green
            3: '#d62728',  # band_stop - red
            4: '#9467bd',  # rlc_series - purple
            5: '#8c564b'   # rlc_parallel - brown
        }
        self.filter_type_names = [
            'Low-pass',
            'High-pass',
            'Band-pass',
            'Band-stop',
            'RLC Series',
            'RLC Parallel'
        ]

    def plot_tsne_pca(
        self,
        latent_vectors: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        perplexity: int = 30
    ):
        """
        Plot t-SNE and PCA side by side.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            labels: Filter type labels [N]
            save_path: Optional path to save figure
            perplexity: t-SNE perplexity parameter
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)

        # Compute PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Plot t-SNE
        for label_id in np.unique(labels):
            mask = labels == label_id
            axes[0].scatter(
                latent_tsne[mask, 0],
                latent_tsne[mask, 1],
                c=self.filter_type_colors[label_id],
                label=self.filter_type_names[label_id],
                alpha=0.7,
                s=50
            )
        axes[0].set_title('t-SNE Projection')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Plot PCA
        for label_id in np.unique(labels):
            mask = labels == label_id
            axes[1].scatter(
                latent_pca[mask, 0],
                latent_pca[mask, 1],
                c=self.filter_type_colors[label_id],
                label=self.filter_type_names[label_id],
                alpha=0.7,
                s=50
            )
        axes[1].set_title(f'PCA Projection (Var: {pca.explained_variance_ratio_.sum():.1%})')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved latent space visualization to {save_path}")

        plt.show()

    def plot_latent_dimensions(
        self,
        latent_vectors: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        max_dims: int = 8
    ):
        """
        Plot distributions of individual latent dimensions by filter type.

        Args:
            latent_vectors: Latent representations [N, latent_dim]
            labels: Filter type labels [N]
            save_path: Optional path to save figure
            max_dims: Maximum number of dimensions to plot
        """
        latent_dim = min(latent_vectors.shape[1], max_dims)

        n_cols = 4
        n_rows = (latent_dim + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for dim in range(latent_dim):
            ax = axes[dim]

            for label_id in np.unique(labels):
                mask = labels == label_id
                values = latent_vectors[mask, dim]

                ax.hist(
                    values,
                    bins=20,
                    alpha=0.5,
                    color=self.filter_type_colors[label_id],
                    label=self.filter_type_names[label_id],
                    density=True
                )

            ax.set_title(f'Latent Dimension {dim}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            if dim == 0:
                ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(latent_dim, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved latent dimension analysis to {save_path}")

        plt.show()

    def plot_hierarchical_structure(
        self,
        latent_vectors: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize hierarchical latent space structure (z_topo, z_values, z_pz).

        Assumes latent_dim = 24 with 8D per branch.

        Args:
            latent_vectors: Latent representations [N, 24]
            labels: Filter type labels [N]
            save_path: Optional path to save figure
        """
        if latent_vectors.shape[1] != 24:
            print(f"Warning: Expected 24D latent space, got {latent_vectors.shape[1]}D")
            return

        from sklearn.decomposition import PCA

        # Split into three branches
        z_topo = latent_vectors[:, :8]
        z_values = latent_vectors[:, 8:16]
        z_pz = latent_vectors[:, 16:24]

        # PCA for each branch
        pca_topo = PCA(n_components=2).fit_transform(z_topo)
        pca_values = PCA(n_components=2).fit_transform(z_values)
        pca_pz = PCA(n_components=2).fit_transform(z_pz)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        branches = [
            (pca_topo, 'Topology Branch (z_topo)'),
            (pca_values, 'Component Values Branch (z_values)'),
            (pca_pz, 'Poles/Zeros Branch (z_pz)')
        ]

        for ax, (pca_data, title) in zip(axes, branches):
            for label_id in np.unique(labels):
                mask = labels == label_id
                ax.scatter(
                    pca_data[mask, 0],
                    pca_data[mask, 1],
                    c=self.filter_type_colors[label_id],
                    label=self.filter_type_names[label_id],
                    alpha=0.7,
                    s=50
                )

            ax.set_title(title)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved hierarchical structure visualization to {save_path}")

        plt.show()


class TrainingVisualizer:
    """
    Visualize training dynamics and loss curves.
    """

    @staticmethod
    def plot_training_history(
        history_path: str,
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation metrics over time.

        Args:
            history_path: Path to training_history.json
            save_path: Optional path to save figure
        """
        with open(history_path, 'r') as f:
            history = json.load(f)

        train_history = history['train']
        val_history = history['val']

        epochs = np.arange(1, len(train_history) + 1)
        val_epochs = np.arange(1, len(val_history) + 1)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        axes[0, 0].plot(epochs, [h['total_loss'] for h in train_history],
                       label='Train', marker='o')
        axes[0, 0].plot(val_epochs, [h['total_loss'] for h in val_history],
                       label='Val', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Reconstruction loss
        axes[0, 1].plot(epochs, [h['recon_total'] for h in train_history],
                       label='Train', marker='o')
        axes[0, 1].plot(val_epochs, [h['recon_total'] for h in val_history],
                       label='Val', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Transfer function loss (log scale)
        axes[1, 0].semilogy(epochs, [h['tf_total'] for h in train_history],
                           label='Train', marker='o')
        axes[1, 0].semilogy(val_epochs, [h['tf_total'] for h in val_history],
                           label='Val', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Transfer Function Loss')
        axes[1, 0].set_title('Transfer Function Loss (log scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # KL divergence
        axes[1, 1].plot(epochs, [h['kl_loss'] for h in train_history],
                       label='Train', marker='o')
        axes[1, 1].plot(val_epochs, [h['kl_loss'] for h in val_history],
                       label='Val', marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('KL Divergence')
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history visualization to {save_path}")

        plt.show()

    @staticmethod
    def plot_loss_components(
        history_path: str,
        save_path: Optional[str] = None
    ):
        """
        Plot individual loss components.

        Args:
            history_path: Path to training_history.json
            save_path: Optional path to save figure
        """
        with open(history_path, 'r') as f:
            history = json.load(f)

        train_history = history['train']
        epochs = np.arange(1, len(train_history) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Topology vs Edge reconstruction
        axes[0, 0].plot(epochs, [h['recon_topo'] for h in train_history],
                       label='Topology', marker='o')
        axes[0, 0].plot(epochs, [h['recon_edge'] for h in train_history],
                       label='Edge Features', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Reconstruction Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Topology accuracy
        if 'topo_accuracy' in train_history[0]:
            axes[0, 1].plot(epochs, [h['topo_accuracy'] * 100 for h in train_history],
                           marker='o', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Topology Classification Accuracy')
            axes[0, 1].grid(True, alpha=0.3)

        # All losses on one plot (normalized)
        total = np.array([h['total_loss'] for h in train_history])
        recon = np.array([h['recon_total'] for h in train_history])
        tf = np.array([h['tf_total'] for h in train_history])
        kl = np.array([h['kl_loss'] for h in train_history])

        axes[1, 0].plot(epochs, total / total[0], label='Total', marker='o')
        axes[1, 0].plot(epochs, recon / recon[0], label='Recon', marker='s')
        axes[1, 0].plot(epochs, tf / tf[0], label='TF', marker='^')
        axes[1, 0].plot(epochs, kl / kl[0], label='KL', marker='d')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Normalized Loss')
        axes[1, 0].set_title('All Loss Components (Normalized)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Loss ratios
        axes[1, 1].plot(epochs, recon / total * 100, label='Recon %', marker='o')
        axes[1, 1].plot(epochs, tf / total * 100, label='TF %', marker='s')
        axes[1, 1].plot(epochs, kl / total * 100, label='KL %', marker='^')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Percentage of Total Loss')
        axes[1, 1].set_title('Loss Component Proportions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved loss components visualization to {save_path}")

        plt.show()


class ReconstructionVisualizer:
    """
    Visualize reconstruction quality.
    """

    @staticmethod
    def plot_pole_zero_comparison(
        pred_poles: List[np.ndarray],
        pred_zeros: List[np.ndarray],
        target_poles: List[np.ndarray],
        target_zeros: List[np.ndarray],
        indices: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot pole-zero diagrams comparing predictions to targets.

        Args:
            pred_poles: Predicted poles [num_poles, 2] per circuit
            pred_zeros: Predicted zeros
            target_poles: Target poles
            target_zeros: Target zeros
            indices: Which circuits to plot (default: first 4)
            save_path: Optional path to save figure
        """
        if indices is None:
            indices = list(range(min(4, len(pred_poles))))

        n_circuits = len(indices)
        n_cols = 2
        n_rows = (n_circuits + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_circuits == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for plot_idx, circuit_idx in enumerate(indices):
            ax = axes[plot_idx]

            # Plot target poles and zeros
            t_poles = target_poles[circuit_idx]
            t_zeros = target_zeros[circuit_idx]

            if len(t_poles) > 0:
                ax.scatter(t_poles[:, 0], t_poles[:, 1],
                          marker='x', s=100, c='red',
                          label='Target Poles', linewidths=2)

            if len(t_zeros) > 0:
                ax.scatter(t_zeros[:, 0], t_zeros[:, 1],
                          marker='o', s=100, c='blue',
                          facecolors='none', linewidths=2,
                          label='Target Zeros')

            # Plot predicted poles and zeros
            p_poles = pred_poles[circuit_idx]
            p_zeros = pred_zeros[circuit_idx]

            if len(p_poles) > 0:
                ax.scatter(p_poles[:, 0], p_poles[:, 1],
                          marker='+', s=100, c='red',
                          label='Pred Poles', linewidths=2, alpha=0.6)

            if len(p_zeros) > 0:
                ax.scatter(p_zeros[:, 0], p_zeros[:, 1],
                          marker='s', s=50, c='blue',
                          facecolors='none', linewidths=2,
                          label='Pred Zeros', alpha=0.6)

            # Draw axes
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.set_title(f'Circuit {circuit_idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_circuits, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved pole-zero comparison to {save_path}")

        plt.show()
