"""
Transfer function loss for circuit graphs.

Measures how well the predicted circuit matches the target transfer function:
    - Poles: Chamfer distance for variable-length pole matching
    - Zeros: Chamfer distance for variable-length zero matching
    - Gain: MSE on log-scale gain
    - Frequency response: MSE on magnitude and phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


def chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor
) -> torch.Tensor:
    """
    Compute Chamfer distance between two point sets.

    Chamfer distance: For each point in pred, find nearest in target, and vice versa.
    CD(P, Q) = mean(min_q ||p - q||^2) + mean(min_p ||p - q||^2)

    Args:
        pred_points: Predicted points [N1, D]
        target_points: Target points [N2, D]

    Returns:
        distance: Scalar Chamfer distance
    """
    if pred_points.size(0) == 0 and target_points.size(0) == 0:
        return torch.tensor(0.0, device=pred_points.device)

    if pred_points.size(0) == 0:
        # No predicted points, penalize all target points
        return target_points.pow(2).sum()

    if target_points.size(0) == 0:
        # No target points, penalize all predicted points
        return pred_points.pow(2).sum()

    # Compute pairwise distances: [N1, N2]
    dist_matrix = torch.cdist(pred_points, target_points, p=2).pow(2)

    # For each predicted point, find nearest target
    min_dist_pred_to_target = dist_matrix.min(dim=1)[0]  # [N1]

    # For each target point, find nearest prediction
    min_dist_target_to_pred = dist_matrix.min(dim=0)[0]  # [N2]

    # Chamfer distance is the sum of both directions
    chamfer_dist = min_dist_pred_to_target.mean() + min_dist_target_to_pred.mean()

    return chamfer_dist


class TransferFunctionLoss(nn.Module):
    """
    Transfer function matching loss using poles, zeros, and frequency response.

    Components:
        1. Pole matching: Chamfer distance between predicted and target poles
        2. Zero matching: Chamfer distance between predicted and target zeros
        3. Gain matching: MSE on log-scale gain
        4. Frequency response: MSE on magnitude and phase

    Args:
        pole_weight: Weight for pole matching (default: 1.0)
        zero_weight: Weight for zero matching (default: 0.5)
        gain_weight: Weight for gain matching (default: 0.1)
        freq_response_weight: Weight for frequency response (default: 0.5)
    """

    def __init__(
        self,
        pole_weight: float = 1.0,
        zero_weight: float = 0.5,
        gain_weight: float = 0.1,
        freq_response_weight: float = 0.5
    ):
        super().__init__()

        self.pole_weight = pole_weight
        self.zero_weight = zero_weight
        self.gain_weight = gain_weight
        self.freq_response_weight = freq_response_weight

    def forward(
        self,
        pred_poles: torch.Tensor,
        pred_zeros: torch.Tensor,
        target_poles_list: List[torch.Tensor],
        target_zeros_list: List[torch.Tensor],
        pred_gain: torch.Tensor = None,
        target_gain: torch.Tensor = None,
        pred_freq_response: torch.Tensor = None,
        target_freq_response: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute transfer function loss.

        Args:
            pred_poles: Predicted poles [B, max_poles, 2]
            pred_zeros: Predicted zeros [B, max_zeros, 2]
            target_poles_list: List of target pole tensors [num_poles_i, 2]
            target_zeros_list: List of target zero tensors [num_zeros_i, 2]
            pred_gain: (Optional) Predicted gain [B, 1]
            target_gain: (Optional) Target gain [B, 1]
            pred_freq_response: (Optional) Predicted frequency response [B, N, 2]
            target_freq_response: (Optional) Target frequency response [B, N, 2]

        Returns:
            loss: Total transfer function loss
            metrics: Dictionary of loss components
        """
        batch_size = pred_poles.size(0)
        device = pred_poles.device

        # 1. Pole matching (Chamfer distance)
        pole_losses = []
        for i in range(batch_size):
            # Get predicted poles (use all, they might be padded with zeros)
            # Better: use actual non-zero poles
            pred_poles_i = pred_poles[i]  # [max_poles, 2]

            # Filter out zero poles (padding)
            # Check if magnitude is significant
            pole_magnitudes = pred_poles_i.pow(2).sum(dim=-1).sqrt()
            valid_mask = pole_magnitudes > 1e-6
            pred_poles_i = pred_poles_i[valid_mask]  # [num_valid, 2]

            # Get target poles
            target_poles_i = target_poles_list[i]  # [num_poles_i, 2]

            # Chamfer distance
            if pred_poles_i.size(0) > 0 or target_poles_i.size(0) > 0:
                cd = chamfer_distance(pred_poles_i, target_poles_i)
                pole_losses.append(cd)

        loss_poles = torch.stack(pole_losses).mean() if pole_losses else torch.tensor(0.0, device=device)

        # 2. Zero matching (Chamfer distance)
        zero_losses = []
        for i in range(batch_size):
            pred_zeros_i = pred_zeros[i]  # [max_zeros, 2]

            # Filter out padding
            zero_magnitudes = pred_zeros_i.pow(2).sum(dim=-1).sqrt()
            valid_mask = zero_magnitudes > 1e-6
            pred_zeros_i = pred_zeros_i[valid_mask]

            target_zeros_i = target_zeros_list[i]

            if pred_zeros_i.size(0) > 0 or target_zeros_i.size(0) > 0:
                cd = chamfer_distance(pred_zeros_i, target_zeros_i)
                zero_losses.append(cd)

        loss_zeros = torch.stack(zero_losses).mean() if zero_losses else torch.tensor(0.0, device=device)

        # 3. Gain matching (log-scale MSE)
        if pred_gain is not None and target_gain is not None:
            # Use log-scale for gain (spans many orders of magnitude)
            log_pred_gain = torch.log(pred_gain.abs() + 1e-8)
            log_target_gain = torch.log(target_gain.abs() + 1e-8)
            loss_gain = F.mse_loss(log_pred_gain, log_target_gain)
        else:
            loss_gain = torch.tensor(0.0, device=device)

        # 4. Frequency response matching
        if pred_freq_response is not None and target_freq_response is not None:
            # Frequency response is [B, N, 2] where [:, :, 0] is magnitude, [:, :, 1] is phase

            # MSE on magnitude (possibly log-scale)
            pred_mag = pred_freq_response[:, :, 0]
            target_mag = target_freq_response[:, :, 0]

            # Use log-scale for magnitude
            log_pred_mag = torch.log(pred_mag + 1e-8)
            log_target_mag = torch.log(target_mag + 1e-8)
            loss_mag = F.mse_loss(log_pred_mag, log_target_mag)

            # MSE on phase (wrapped to [-π, π])
            pred_phase = pred_freq_response[:, :, 1]
            target_phase = target_freq_response[:, :, 1]

            # Compute phase difference and wrap
            phase_diff = pred_phase - target_phase
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            loss_phase = phase_diff.pow(2).mean()

            loss_freq = loss_mag + loss_phase
        else:
            loss_freq = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = (
            self.pole_weight * loss_poles +
            self.zero_weight * loss_zeros +
            self.gain_weight * loss_gain +
            self.freq_response_weight * loss_freq
        )

        metrics = {
            'tf_total': total_loss.item(),
            'tf_poles': loss_poles.item(),
            'tf_zeros': loss_zeros.item(),
            'tf_gain': loss_gain.item(),
            'tf_freq': loss_freq.item()
        }

        return total_loss, metrics


class SimplifiedTransferFunctionLoss(nn.Module):
    """
    Simplified transfer function loss using only poles and zeros.

    This is more appropriate when we don't explicitly predict gain or
    frequency response from the decoder.

    Args:
        pole_weight: Weight for pole matching
        zero_weight: Weight for zero matching
        use_complex_distance: Use distance in complex plane vs. [real, imag]
    """

    def __init__(
        self,
        pole_weight: float = 1.0,
        zero_weight: float = 0.5,
        use_complex_distance: bool = True,
        use_log_scale: bool = True
    ):
        super().__init__()

        self.pole_weight = pole_weight
        self.zero_weight = zero_weight
        self.use_complex_distance = use_complex_distance
        self.use_log_scale = use_log_scale  # Compute distance in log-magnitude space

    def set_normalization_stats(self, stats: dict):
        """Set normalization statistics for denormalization."""
        self.log_pole_mean = stats.get('log_pole_mean', 0.0)
        self.log_pole_std = stats.get('log_pole_std', 1.0)
        self.log_zero_mean = stats.get('log_zero_mean', 0.0)
        self.log_zero_std = stats.get('log_zero_std', 1.0)

    def _denormalize_poles_zeros(
        self,
        poles_zeros: torch.Tensor,
        log_mean: float,
        log_std: float
    ) -> torch.Tensor:
        """
        Denormalize poles or zeros back to original scale.

        Args:
            poles_zeros: Normalized [real, imag] pairs [N, 2]
            log_mean: Mean of log-magnitudes
            log_std: Std of log-magnitudes

        Returns:
            Denormalized [real, imag] pairs [N, 2]
        """
        if poles_zeros.size(0) == 0:
            return poles_zeros

        # Get magnitude and phase
        mags = torch.sqrt((poles_zeros**2).sum(dim=-1))
        phases = torch.atan2(poles_zeros[:, 1], poles_zeros[:, 0])

        # Denormalize magnitude
        log_norm_mags = torch.log(mags + 1e-8)
        log_mags = log_norm_mags * log_std + log_mean
        denorm_mags = torch.exp(log_mags)

        # Reconstruct complex numbers
        denormalized = torch.stack([
            denorm_mags * torch.cos(phases),
            denorm_mags * torch.sin(phases)
        ], dim=-1)

        return denormalized

    def _complex_chamfer_distance(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Chamfer distance treating [real, imag] as complex numbers.

        For poles/zeros, the distance in the complex plane is more meaningful.
        """
        if pred_points.size(0) == 0 and target_points.size(0) == 0:
            return torch.tensor(0.0, device=pred_points.device)

        if pred_points.size(0) == 0:
            # Penalize missing poles/zeros
            return target_points.pow(2).sum(dim=-1).sum()

        if target_points.size(0) == 0:
            return pred_points.pow(2).sum(dim=-1).sum()

        # Convert to complex numbers
        pred_complex = torch.complex(pred_points[:, 0], pred_points[:, 1])
        target_complex = torch.complex(target_points[:, 0], target_points[:, 1])

        # Compute pairwise distances in complex plane
        # |z1 - z2|^2 = (r1 - r2)^2 + (i1 - i2)^2
        pred_expanded = pred_complex.unsqueeze(1)  # [N1, 1]
        target_expanded = target_complex.unsqueeze(0)  # [1, N2]

        dist_matrix = (pred_expanded - target_expanded).abs().pow(2)  # [N1, N2]

        # Chamfer distance
        min_pred_to_target = dist_matrix.min(dim=1)[0].mean()
        min_target_to_pred = dist_matrix.min(dim=0)[0].mean()

        return min_pred_to_target + min_target_to_pred

    def forward(
        self,
        pred_poles: torch.Tensor,
        pred_zeros: torch.Tensor,
        target_poles_list: List[torch.Tensor],
        target_zeros_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute simplified transfer function loss.

        Args:
            pred_poles: [B, max_poles, 2]
            pred_zeros: [B, max_zeros, 2]
            target_poles_list: List of [num_poles_i, 2]
            target_zeros_list: List of [num_zeros_i, 2]

        Returns:
            loss: Total loss
            metrics: Loss components
        """
        batch_size = pred_poles.size(0)
        device = pred_poles.device

        pole_losses = []
        zero_losses = []

        for i in range(batch_size):
            # Poles
            pred_poles_i = pred_poles[i]
            target_poles_i = target_poles_list[i]

            # Filter padding (magnitude threshold)
            pred_mag = pred_poles_i.pow(2).sum(dim=-1).sqrt()
            pred_poles_i = pred_poles_i[pred_mag > 1e-6]

            if self.use_complex_distance:
                cd_poles = self._complex_chamfer_distance(pred_poles_i, target_poles_i)
            else:
                cd_poles = chamfer_distance(pred_poles_i, target_poles_i)

            pole_losses.append(cd_poles)

            # Zeros
            pred_zeros_i = pred_zeros[i]
            target_zeros_i = target_zeros_list[i]

            pred_mag = pred_zeros_i.pow(2).sum(dim=-1).sqrt()
            pred_zeros_i = pred_zeros_i[pred_mag > 1e-6]

            if self.use_complex_distance:
                cd_zeros = self._complex_chamfer_distance(pred_zeros_i, target_zeros_i)
            else:
                cd_zeros = chamfer_distance(pred_zeros_i, target_zeros_i)

            zero_losses.append(cd_zeros)

        loss_poles = torch.stack(pole_losses).mean()
        loss_zeros = torch.stack(zero_losses).mean()

        total_loss = (
            self.pole_weight * loss_poles +
            self.zero_weight * loss_zeros
        )

        metrics = {
            'tf_total': total_loss.item(),
            'tf_poles': loss_poles.item(),
            'tf_zeros': loss_zeros.item()
        }

        return total_loss, metrics
