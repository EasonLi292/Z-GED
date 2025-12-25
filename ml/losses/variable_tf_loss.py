"""
Variable-length transfer function loss.

Handles variable number of poles and zeros by:
1. Predicting count (cross-entropy loss)
2. Predicting values (Chamfer distance on valid predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class VariableLengthTransferFunctionLoss(nn.Module):
    """
    Transfer function loss for variable-length pole/zero predictions.

    Components:
    1. Count prediction: Cross-entropy for number of poles/zeros
    2. Value prediction: Chamfer distance for pole/zero locations

    Args:
        pole_count_weight: Weight for pole count prediction
        zero_count_weight: Weight for zero count prediction
        pole_value_weight: Weight for pole value prediction
        zero_value_weight: Weight for zero value prediction
        use_complex_distance: Use distance in complex plane
    """

    def __init__(
        self,
        pole_count_weight: float = 5.0,  # Higher initially - structure matters
        zero_count_weight: float = 5.0,
        pole_value_weight: float = 1.0,
        zero_value_weight: float = 1.0,
        use_complex_distance: bool = True
    ):
        super().__init__()

        self.pole_count_weight = pole_count_weight
        self.zero_count_weight = zero_count_weight
        self.pole_value_weight = pole_value_weight
        self.zero_value_weight = zero_value_weight
        self.use_complex_distance = use_complex_distance

    def _complex_chamfer_distance(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Chamfer distance treating [real, imag] as complex numbers.

        Args:
            pred_points: Predicted points [N1, 2]
            target_points: Target points [N2, 2]

        Returns:
            Chamfer distance (scalar)
        """
        if pred_points.size(0) == 0 and target_points.size(0) == 0:
            return torch.tensor(0.0, device=pred_points.device)

        if pred_points.size(0) == 0:
            # Penalty for missing poles/zeros
            return target_points.pow(2).sum()

        if target_points.size(0) == 0:
            # Penalty for extra poles/zeros
            return pred_points.pow(2).sum()

        # Convert to complex numbers
        pred_complex = torch.complex(pred_points[:, 0], pred_points[:, 1])
        target_complex = torch.complex(target_points[:, 0], target_points[:, 1])

        # Pairwise distances
        pred_expanded = pred_complex.unsqueeze(1)  # [N1, 1]
        target_expanded = target_complex.unsqueeze(0)  # [1, N2]
        dist_matrix = (pred_expanded - target_expanded).abs().pow(2)  # [N1, N2]

        # Chamfer distance: bidirectional nearest neighbor
        min_pred_to_target = dist_matrix.min(dim=1)[0].mean()
        min_target_to_pred = dist_matrix.min(dim=0)[0].mean()

        return min_pred_to_target + min_target_to_pred

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_poles_list: List[torch.Tensor],
        target_zeros_list: List[torch.Tensor],
        target_num_poles: torch.Tensor,
        target_num_zeros: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute variable-length transfer function loss.

        Args:
            outputs: Decoder outputs containing:
                - pole_count_logits: [B, max_poles+1]
                - zero_count_logits: [B, max_zeros+1]
                - poles_all: [B, max_poles, 2]
                - zeros_all: [B, max_zeros, 2]
            target_poles_list: List of target poles [num_poles_i, 2]
            target_zeros_list: List of target zeros [num_zeros_i, 2]
            target_num_poles: Ground truth pole counts [B]
            target_num_zeros: Ground truth zero counts [B]

        Returns:
            loss: Total loss
            metrics: Loss components
        """
        batch_size = outputs['pole_count_logits'].size(0)
        device = outputs['pole_count_logits'].device

        # 1. Count prediction loss (cross-entropy)
        loss_pole_count = F.cross_entropy(
            outputs['pole_count_logits'],
            target_num_poles.long()
        )

        loss_zero_count = F.cross_entropy(
            outputs['zero_count_logits'],
            target_num_zeros.long()
        )

        # Count accuracy for metrics
        pred_pole_counts = outputs['pole_count_logits'].argmax(dim=-1)
        pred_zero_counts = outputs['zero_count_logits'].argmax(dim=-1)

        pole_count_acc = (pred_pole_counts == target_num_poles).float().mean()
        zero_count_acc = (pred_zero_counts == target_num_zeros).float().mean()

        # 2. Value prediction loss (Chamfer distance on valid poles/zeros)
        pole_value_losses = []
        zero_value_losses = []

        for i in range(batch_size):
            # Get predicted count (use argmax for hard assignment)
            n_poles_pred = pred_pole_counts[i].item()
            n_zeros_pred = pred_zero_counts[i].item()

            # Extract valid predictions (first n_pred)
            pred_poles = outputs['poles_all'][i, :n_poles_pred] if n_poles_pred > 0 else torch.zeros(0, 2, device=device)
            pred_zeros = outputs['zeros_all'][i, :n_zeros_pred] if n_zeros_pred > 0 else torch.zeros(0, 2, device=device)

            # Get ground truth
            target_poles = target_poles_list[i]
            target_zeros = target_zeros_list[i]

            # Compute Chamfer distance
            pole_loss = self._complex_chamfer_distance(pred_poles, target_poles)
            zero_loss = self._complex_chamfer_distance(pred_zeros, target_zeros)

            pole_value_losses.append(pole_loss)
            zero_value_losses.append(zero_loss)

        # Average over batch
        loss_pole_value = torch.stack(pole_value_losses).mean() if pole_value_losses else torch.tensor(0.0, device=device)
        loss_zero_value = torch.stack(zero_value_losses).mean() if zero_value_losses else torch.tensor(0.0, device=device)

        # 3. Total loss (weighted combination)
        total_loss = (
            self.pole_count_weight * loss_pole_count +
            self.zero_count_weight * loss_zero_count +
            self.pole_value_weight * loss_pole_value +
            self.zero_value_weight * loss_zero_value
        )

        # Metrics
        metrics = {
            'tf_loss': total_loss.item(),
            'pole_count_loss': loss_pole_count.item(),
            'zero_count_loss': loss_zero_count.item(),
            'pole_value_loss': loss_pole_value.item(),
            'zero_value_loss': loss_zero_value.item(),
            'pole_count_acc': pole_count_acc.item(),
            'zero_count_acc': zero_count_acc.item(),
        }

        return total_loss, metrics
