"""
GED metric learning loss.

Encourages latent space distances to correlate with Graph Edit Distance:
    ||z_i - z_j||_2 ∝ GED(G_i, G_j)

This helps the latent space preserve circuit similarity structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class GEDMetricLoss(nn.Module):
    """
    Metric learning loss using precomputed GED matrix.

    Objective: Make latent distance correlate with GED distance.

    Two variants:
        1. Contrastive: Pairs with small GED should have small latent distance
        2. Triplet: For anchor, positive (similar), negative (dissimilar)
        3. MSE: Direct regression ||z_i - z_j||_2 ≈ α × GED(i, j)

    Args:
        mode: Loss mode ('mse', 'contrastive', 'triplet')
        alpha: Learnable scaling factor (default: 1.0)
        margin: Margin for contrastive/triplet loss (default: 1.0)
        use_learnable_alpha: Whether alpha is learnable (default: True)
    """

    def __init__(
        self,
        mode: str = 'mse',
        alpha: float = 1.0,
        margin: float = 1.0,
        use_learnable_alpha: bool = True
    ):
        super().__init__()

        assert mode in ['mse', 'contrastive', 'triplet']
        self.mode = mode
        self.margin = margin

        if use_learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

    def forward(
        self,
        z: torch.Tensor,
        ged_matrix: torch.Tensor,
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GED metric learning loss.

        Args:
            z: Latent vectors [B, latent_dim]
            ged_matrix: Precomputed GED matrix [N, N] for full dataset
            indices: Indices of samples in this batch [B]

        Returns:
            loss: Metric learning loss
            metrics: Dictionary with loss and correlation
        """
        batch_size = z.size(0)
        device = z.device

        # Get GED submatrix for this batch
        ged_batch = ged_matrix[indices][:, indices]  # [B, B]

        # Compute pairwise latent distances
        # [B, B] where entry (i, j) is ||z_i - z_j||_2
        latent_dist = torch.cdist(z, z, p=2)  # [B, B]

        if self.mode == 'mse':
            # MSE between scaled latent distance and GED
            # L = mean((||z_i - z_j||_2 - α × GED(i, j))^2)

            target_dist = self.alpha * ged_batch
            loss = F.mse_loss(latent_dist, target_dist)

        elif self.mode == 'contrastive':
            # Contrastive loss: pull similar pairs close, push dissimilar apart
            # For pairs with small GED: minimize latent distance
            # For pairs with large GED: maximize latent distance (up to margin)

            # Define threshold for "similar" (e.g., GED < median)
            ged_threshold = ged_batch.median()

            similar_mask = (ged_batch < ged_threshold).float()
            dissimilar_mask = 1.0 - similar_mask

            # Exclude diagonal (self-pairs)
            eye_mask = torch.eye(batch_size, device=device)
            similar_mask = similar_mask * (1 - eye_mask)
            dissimilar_mask = dissimilar_mask * (1 - eye_mask)

            # Loss for similar pairs: minimize distance
            loss_similar = (similar_mask * latent_dist.pow(2)).sum()
            num_similar = similar_mask.sum().clamp(min=1)
            loss_similar = loss_similar / num_similar

            # Loss for dissimilar pairs: maximize distance (with margin)
            loss_dissimilar = (dissimilar_mask * F.relu(self.margin - latent_dist).pow(2)).sum()
            num_dissimilar = dissimilar_mask.sum().clamp(min=1)
            loss_dissimilar = loss_dissimilar / num_dissimilar

            loss = loss_similar + loss_dissimilar

        elif self.mode == 'triplet':
            # Triplet loss: For each anchor, find positive and negative
            # Positive: circuit with small GED
            # Negative: circuit with large GED

            losses = []
            for i in range(batch_size):
                # Get GED distances from anchor i to all others
                ged_from_i = ged_batch[i]  # [B]
                ged_from_i[i] = float('inf')  # Exclude self

                # Find positive (smallest GED, excluding self)
                pos_idx = ged_from_i.argmin()

                # Find negative (largest GED)
                ged_from_i[i] = float('-inf')  # Reset for argmax
                neg_idx = ged_from_i.argmax()

                # Latent distances
                d_pos = latent_dist[i, pos_idx]
                d_neg = latent_dist[i, neg_idx]

                # Triplet loss: d_pos + margin < d_neg
                loss_i = F.relu(d_pos - d_neg + self.margin)
                losses.append(loss_i)

            loss = torch.stack(losses).mean()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Compute correlation between latent distance and GED (for monitoring)
        # Flatten and exclude diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        latent_dist_flat = latent_dist[mask]
        ged_flat = ged_batch[mask]

        # Pearson correlation
        if latent_dist_flat.numel() > 1:
            correlation = torch.corrcoef(
                torch.stack([latent_dist_flat, ged_flat])
            )[0, 1]
        else:
            correlation = torch.tensor(0.0, device=device)

        metrics = {
            'ged_loss': loss.item(),
            'ged_correlation': correlation.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, nn.Parameter) else self.alpha.item()
        }

        return loss, metrics


class SoftGEDMetricLoss(nn.Module):
    """
    Soft GED metric loss that works without precomputed GED matrix.

    Instead of using exact GED, uses soft similarities based on:
        - Filter type matching
        - Component value similarity
        - Pole/zero similarity

    This is useful when GED matrix is not available.

    Args:
        topo_weight: Weight for topology similarity
        value_weight: Weight for component value similarity
        pz_weight: Weight for pole/zero similarity
    """

    def __init__(
        self,
        topo_weight: float = 1.0,
        value_weight: float = 0.5,
        pz_weight: float = 0.5
    ):
        super().__init__()

        self.topo_weight = topo_weight
        self.value_weight = value_weight
        self.pz_weight = pz_weight

    def forward(
        self,
        z: torch.Tensor,
        filter_types: torch.Tensor,
        edge_features: torch.Tensor,
        poles_list: list,
        zeros_list: list
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute soft metric loss.

        Args:
            z: Latent vectors [B, latent_dim]
            filter_types: One-hot filter types [B, 6]
            edge_features: Edge features [E, 3]
            poles_list: List of pole tensors
            zeros_list: List of zero tensors

        Returns:
            loss: Metric loss
            metrics: Loss components
        """
        batch_size = z.size(0)
        device = z.device

        # Compute latent distances
        latent_dist = torch.cdist(z, z, p=2)  # [B, B]

        # Compute soft similarities

        # 1. Topology similarity (1 if same type, 0 otherwise)
        filter_type_idx = filter_types.argmax(dim=-1)  # [B]
        topo_similarity = (filter_type_idx.unsqueeze(0) == filter_type_idx.unsqueeze(1)).float()

        # 2. Component value similarity (inverse of distance)
        # This is complex due to variable edge counts - skip for now
        value_similarity = torch.ones(batch_size, batch_size, device=device)

        # 3. Pole/zero similarity (Chamfer distance)
        # Also complex - skip for now
        pz_similarity = torch.ones(batch_size, batch_size, device=device)

        # Combined similarity (weighted)
        total_similarity = (
            self.topo_weight * topo_similarity +
            self.value_weight * value_similarity +
            self.pz_weight * pz_similarity
        )

        # Normalize to [0, 1]
        total_similarity = total_similarity / (self.topo_weight + self.value_weight + self.pz_weight)

        # Loss: Make latent distance inversely proportional to similarity
        # High similarity → low latent distance
        # Low similarity → high latent distance (up to margin)

        target_dist = (1.0 - total_similarity) * 10.0  # Scale factor

        # Exclude diagonal
        eye_mask = torch.eye(batch_size, device=device)
        mask = (1 - eye_mask).bool()

        loss = F.mse_loss(latent_dist[mask], target_dist[mask])

        metrics = {
            'soft_ged_loss': loss.item()
        }

        return loss, metrics


class RegularizedGEDMetricLoss(nn.Module):
    """
    GED metric loss with regularization to prevent collapse.

    Adds a term to encourage spread in latent space.

    Args:
        ged_loss: Underlying GED metric loss
        spread_weight: Weight for spread regularization (default: 0.1)
    """

    def __init__(
        self,
        ged_loss: GEDMetricLoss,
        spread_weight: float = 0.1
    ):
        super().__init__()

        self.ged_loss = ged_loss
        self.spread_weight = spread_weight

    def forward(
        self,
        z: torch.Tensor,
        ged_matrix: torch.Tensor,
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute regularized GED loss.

        Args:
            (same as GEDMetricLoss)

        Returns:
            loss: Total loss
            metrics: Loss components
        """
        # Base GED loss
        loss_ged, metrics = self.ged_loss(z, ged_matrix, indices)

        # Spread regularization: encourage variance in latent space
        # This prevents all points from collapsing to a single point
        z_mean = z.mean(dim=0, keepdim=True)  # [1, latent_dim]
        z_centered = z - z_mean  # [B, latent_dim]
        variance = z_centered.pow(2).mean()

        # Encourage variance (penalize low variance)
        loss_spread = -torch.log(variance + 1e-8)

        total_loss = loss_ged + self.spread_weight * loss_spread

        metrics['spread_loss'] = loss_spread.item()
        metrics['latent_variance'] = variance.item()
        metrics['total_ged_loss'] = total_loss.item()

        return total_loss, metrics
