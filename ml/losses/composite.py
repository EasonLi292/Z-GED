"""
Composite loss for GraphVAE training.

Combines multiple loss components with adaptive weighting:
    1. Reconstruction loss (topology + edges)
    2. Transfer function loss (poles/zeros)
    3. KL divergence (VAE regularization)
    4. GED metric learning (optional)

Implements weight scheduling to balance different objectives during training.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from .reconstruction import TemplateAwareReconstructionLoss
from .transfer_function import SimplifiedTransferFunctionLoss
from .ged_metric import GEDMetricLoss


class CompositeLoss(nn.Module):
    """
    Multi-objective composite loss for GraphVAE.

    L_total = λ_recon × L_recon
            + λ_tf × L_transfer_function
            + λ_kl × L_kl
            + λ_ged × L_ged_metric

    Args:
        recon_weight: Weight for reconstruction loss
        tf_weight: Weight for transfer function loss
        kl_weight: Weight for KL divergence
        ged_weight: Weight for GED metric learning (default: 0, disabled)
        use_ged_loss: Whether to use GED metric learning
        weight_schedule: Weight scheduling strategy ('fixed', 'linear', 'cosine')
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        tf_weight: float = 0.1,
        kl_weight: float = 0.01,
        ged_weight: float = 0.0,
        use_ged_loss: bool = False,
        weight_schedule: str = 'linear'
    ):
        super().__init__()

        self.initial_weights = {
            'recon': recon_weight,
            'tf': tf_weight,
            'kl': kl_weight,
            'ged': ged_weight
        }

        self.use_ged_loss = use_ged_loss
        self.weight_schedule = weight_schedule

        # Loss components
        self.recon_loss = TemplateAwareReconstructionLoss(
            topo_weight=1.0,
            edge_weight=1.0
        )

        self.tf_loss = SimplifiedTransferFunctionLoss(
            pole_weight=1.0,
            zero_weight=0.5,
            use_complex_distance=True
        )

        if use_ged_loss:
            self.ged_loss = GEDMetricLoss(
                mode='mse',
                alpha=1.0,
                use_learnable_alpha=True
            )
        else:
            self.ged_loss = None

        # Track current epoch for scheduling
        self.current_epoch = 0
        self.total_epochs = 200  # Will be updated during training

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update current epoch for weight scheduling."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def get_scheduled_weights(self) -> Dict[str, float]:
        """
        Get weights according to schedule.

        Schedule strategies:
            - 'fixed': Constant weights
            - 'linear': Linear interpolation from initial to final
            - 'cosine': Cosine annealing

        Final weights (at end of training):
            recon: 1.0 (constant)
            tf: 2.0 (increase from 0.1)
            kl: 1.0 (increase from 0.01)
            ged: 0.5 (decrease from initial if enabled)
        """
        if self.weight_schedule == 'fixed':
            return self.initial_weights.copy()

        # Compute progress [0, 1]
        progress = self.current_epoch / max(self.total_epochs, 1)

        final_weights = {
            'recon': 1.0,
            'tf': 2.0,
            'kl': 1.0,
            'ged': 0.5 if self.use_ged_loss else 0.0
        }

        if self.weight_schedule == 'linear':
            # Linear interpolation
            weights = {}
            for key in self.initial_weights:
                w_init = self.initial_weights[key]
                w_final = final_weights[key]
                weights[key] = w_init + progress * (w_final - w_init)
            return weights

        elif self.weight_schedule == 'cosine':
            # Cosine annealing
            weights = {}
            for key in self.initial_weights:
                w_init = self.initial_weights[key]
                w_final = final_weights[key]
                # Cosine decay from init to final
                weights[key] = w_final + (w_init - w_final) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            return weights

        else:
            raise ValueError(f"Unknown schedule: {self.weight_schedule}")

    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(q(z|x) || p(z)) where p(z) = N(0, I).

        KL = -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log-variance of latent distribution [B, latent_dim]

        Returns:
            kl_loss: KL divergence (scalar)
        """
        # KL divergence for each latent dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Sum over latent dimensions, mean over batch
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        return kl_loss

    def forward(
        self,
        encoder_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        decoder_output: Dict[str, torch.Tensor],
        target_filter_type: torch.Tensor,
        target_edge_attr: torch.Tensor,
        edge_batch: torch.Tensor,
        target_poles_list: List[torch.Tensor],
        target_zeros_list: List[torch.Tensor],
        ged_matrix: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total composite loss.

        Args:
            encoder_output: (z, mu, logvar) from encoder
            decoder_output: Output from decoder
            target_filter_type: Ground truth filter types [B, 6]
            target_edge_attr: Ground truth edge features [E, 3]
            edge_batch: Batch assignment for edges [E]
            target_poles_list: List of target poles
            target_zeros_list: List of target zeros
            ged_matrix: (Optional) Precomputed GED matrix
            indices: (Optional) Batch indices in dataset

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of all loss components and metrics
        """
        z, mu, logvar = encoder_output

        # Get scheduled weights
        weights = self.get_scheduled_weights()

        # 1. Reconstruction loss
        loss_recon, metrics_recon = self.recon_loss(
            decoder_output,
            target_filter_type,
            target_edge_attr,
            edge_batch
        )

        # 2. Transfer function loss
        loss_tf, metrics_tf = self.tf_loss(
            decoder_output['poles'],
            decoder_output['zeros'],
            target_poles_list,
            target_zeros_list
        )

        # 3. KL divergence
        loss_kl = self.compute_kl_divergence(mu, logvar)

        # 4. GED metric learning (optional)
        if self.use_ged_loss and ged_matrix is not None and indices is not None:
            loss_ged, metrics_ged = self.ged_loss(z, ged_matrix, indices)
        else:
            loss_ged = torch.tensor(0.0, device=z.device)
            metrics_ged = {'ged_loss': 0.0, 'ged_correlation': 0.0}

        # Total loss
        total_loss = (
            weights['recon'] * loss_recon +
            weights['tf'] * loss_tf +
            weights['kl'] * loss_kl +
            weights['ged'] * loss_ged
        )

        # Combine all metrics
        metrics = {
            'total_loss': total_loss.item(),
            'kl_loss': loss_kl.item(),
            **metrics_recon,
            **metrics_tf,
            **metrics_ged,
            'weight_recon': weights['recon'],
            'weight_tf': weights['tf'],
            'weight_kl': weights['kl'],
            'weight_ged': weights['ged']
        }

        return total_loss, metrics


class SimplifiedCompositeLoss(nn.Module):
    """
    Simplified composite loss without GED metric learning.

    Useful for initial training or when GED matrix is not available.

    L_total = λ_recon × L_recon + λ_tf × L_tf + λ_kl × L_kl
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        tf_weight: float = 0.5,
        kl_weight: float = 0.05
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.tf_weight = tf_weight
        self.kl_weight = kl_weight

        self.recon_loss = TemplateAwareReconstructionLoss()
        self.tf_loss = SimplifiedTransferFunctionLoss()

    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence."""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl_per_dim.sum(dim=-1).mean()

    def forward(
        self,
        encoder_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        decoder_output: Dict[str, torch.Tensor],
        target_filter_type: torch.Tensor,
        target_edge_attr: torch.Tensor,
        edge_batch: torch.Tensor,
        target_poles_list: List[torch.Tensor],
        target_zeros_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute simplified composite loss.

        Args:
            (same as CompositeLoss but without GED components)

        Returns:
            total_loss: Combined loss
            metrics: Loss components
        """
        z, mu, logvar = encoder_output

        # Reconstruction loss
        loss_recon, metrics_recon = self.recon_loss(
            decoder_output,
            target_filter_type,
            target_edge_attr,
            edge_batch
        )

        # Transfer function loss
        loss_tf, metrics_tf = self.tf_loss(
            decoder_output['poles'],
            decoder_output['zeros'],
            target_poles_list,
            target_zeros_list
        )

        # KL divergence
        loss_kl = self.compute_kl_divergence(mu, logvar)

        # Total loss
        total_loss = (
            self.recon_weight * loss_recon +
            self.tf_weight * loss_tf +
            self.kl_weight * loss_kl
        )

        metrics = {
            'total_loss': total_loss.item(),
            'kl_loss': loss_kl.item(),
            **metrics_recon,
            **metrics_tf
        }

        return total_loss, metrics


class WeightScheduler:
    """
    Helper class for managing loss weight schedules.

    Implements different scheduling strategies for adapting loss weights
    during training.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        total_epochs: int,
        schedule_type: str = 'linear'
    ):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get weights for current epoch."""
        progress = epoch / max(self.total_epochs, 1)

        if self.schedule_type == 'linear':
            return self._linear_schedule(progress)
        elif self.schedule_type == 'cosine':
            return self._cosine_schedule(progress)
        elif self.schedule_type == 'warmup':
            return self._warmup_schedule(progress)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def _linear_schedule(self, progress: float) -> Dict[str, float]:
        """Linear interpolation from initial to final."""
        weights = {}
        for key in self.initial_weights:
            w_init = self.initial_weights[key]
            w_final = self.final_weights[key]
            weights[key] = w_init + progress * (w_final - w_init)
        return weights

    def _cosine_schedule(self, progress: float) -> Dict[str, float]:
        """Cosine annealing from initial to final."""
        import math
        weights = {}
        for key in self.initial_weights:
            w_init = self.initial_weights[key]
            w_final = self.final_weights[key]
            weights[key] = w_final + (w_init - w_final) * 0.5 * (1 + math.cos(progress * math.pi))
        return weights

    def _warmup_schedule(self, progress: float, warmup_fraction: float = 0.1) -> Dict[str, float]:
        """Warmup then constant."""
        if progress < warmup_fraction:
            # Linear warmup
            warmup_progress = progress / warmup_fraction
            weights = {}
            for key in self.initial_weights:
                w_init = self.initial_weights[key]
                w_final = self.final_weights[key]
                weights[key] = w_init + warmup_progress * (w_final - w_init)
            return weights
        else:
            # Constant at final weights
            return self.final_weights.copy()
