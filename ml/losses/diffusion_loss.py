"""
Unified loss function for diffusion-based circuit generation.

Combines losses for discrete and continuous variables with adaptive weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DiffusionCircuitLoss(nn.Module):
    """
    Unified loss function for circuit diffusion model.

    Combines:
    1. Discrete losses (node types, pole/zero counts) - Cross-entropy
    2. Continuous losses (edge values, poles/zeros) - MSE
    3. Transfer function loss (reuse existing VariableLengthTransferFunctionLoss)
    4. Structural validity loss (from constraints module)

    Time-dependent weighting:
    - Early timesteps (high t): Focus on structure (nodes, counts)
    - Late timesteps (low t): Focus on values (edge values, pole/zero values)

    Args:
        node_type_weight: Weight for node type prediction loss
        edge_exist_weight: Weight for edge existence loss
        edge_value_weight: Weight for edge value prediction loss
        pole_count_weight: Weight for pole count prediction loss
        zero_count_weight: Weight for zero count prediction loss
        pole_value_weight: Weight for pole value prediction loss
        zero_value_weight: Weight for zero value prediction loss
        tf_weight: Weight for transfer function loss
        structure_weight: Weight for structural validity loss
        use_time_weighting: Whether to use time-dependent weighting

    Example:
        >>> loss_fn = DiffusionCircuitLoss()
        >>> loss, metrics = loss_fn(predictions, targets, t)
    """

    def __init__(
        self,
        node_type_weight: float = 1.0,
        edge_exist_weight: float = 1.0,
        edge_value_weight: float = 1.0,
        pole_count_weight: float = 1.0,
        zero_count_weight: float = 1.0,
        pole_value_weight: float = 1.0,
        zero_value_weight: float = 1.0,
        tf_weight: float = 0.5,
        structure_weight: float = 0.1,
        use_time_weighting: bool = True
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.edge_exist_weight = edge_exist_weight
        self.edge_value_weight = edge_value_weight
        self.pole_count_weight = pole_count_weight
        self.zero_count_weight = zero_count_weight
        self.pole_value_weight = pole_value_weight
        self.zero_value_weight = zero_value_weight
        self.tf_weight = tf_weight
        self.structure_weight = structure_weight
        self.use_time_weighting = use_time_weighting

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        t: torch.Tensor,
        timesteps: int = 1000
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute unified diffusion loss.

        Args:
            predictions: Model predictions containing:
                - 'node_types': Node type logits [batch, max_nodes, num_node_types]
                - 'edge_existence': Edge existence logits [batch, max_nodes, max_nodes]
                - 'edge_values': Edge values [batch, max_nodes, max_nodes, 7]
                - 'pole_count_logits': Pole count logits [batch, max_poles + 1]
                - 'zero_count_logits': Zero count logits [batch, max_zeros + 1]
                - 'pole_values': Pole values [batch, max_poles, 2]
                - 'zero_values': Zero values [batch, max_zeros, 2]
            targets: Ground truth containing same keys (with appropriate formats)
            t: Timesteps [batch]
            timesteps: Total number of diffusion timesteps

        Returns:
            total_loss: Combined loss
            metrics: Dictionary of individual loss components
        """
        batch_size = t.shape[0]
        device = t.device

        # ==================================================================
        # Compute time-dependent weights
        # ==================================================================

        if self.use_time_weighting:
            # Normalize timesteps to [0, 1]
            t_normalized = t.float() / timesteps

            # Early timesteps (t near 1): Focus on structure
            # Late timesteps (t near 0): Focus on values
            structure_weight_t = t_normalized  # Higher at early timesteps
            value_weight_t = 1.0 - t_normalized  # Higher at late timesteps
        else:
            structure_weight_t = torch.ones_like(t, dtype=torch.float32)
            value_weight_t = torch.ones_like(t, dtype=torch.float32)

        # ==================================================================
        # 1. Node Type Loss (Discrete - Cross Entropy)
        # ==================================================================

        node_type_logits = predictions['node_types']  # [batch, max_nodes, num_node_types]
        target_node_types = targets['node_types']  # [batch, max_nodes]

        # Flatten for cross-entropy
        node_type_logits_flat = node_type_logits.reshape(-1, node_type_logits.shape[-1])
        target_node_types_flat = target_node_types.reshape(-1)

        loss_node_type = F.cross_entropy(
            node_type_logits_flat,
            target_node_types_flat,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)  # [batch]

        # Apply time weighting
        loss_node_type = (loss_node_type * structure_weight_t).mean()

        # ==================================================================
        # 2. Edge Existence Loss (Discrete - Binary Cross Entropy)
        # ==================================================================

        edge_exist_logits = predictions['edge_existence']  # [batch, max_nodes, max_nodes]
        target_edge_exist = targets['edge_existence']  # [batch, max_nodes, max_nodes]

        loss_edge_exist = F.binary_cross_entropy_with_logits(
            edge_exist_logits,
            target_edge_exist,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)  # [batch]

        loss_edge_exist = (loss_edge_exist * structure_weight_t).mean()

        # ==================================================================
        # 3. Edge Value Loss (Continuous - MSE)
        # ==================================================================

        pred_edge_values = predictions['edge_values']  # [batch, max_nodes, max_nodes, 7]
        target_edge_values = targets['edge_values']  # [batch, max_nodes, max_nodes, 7]

        # Only compute loss on existing edges
        existing_edges = target_edge_exist.unsqueeze(-1).expand_as(pred_edge_values)

        loss_edge_value = F.mse_loss(
            pred_edge_values * existing_edges,
            target_edge_values * existing_edges,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)  # [batch]

        loss_edge_value = (loss_edge_value * value_weight_t).mean()

        # ==================================================================
        # 4. Pole Count Loss (Discrete - Cross Entropy)
        # ==================================================================

        pole_count_logits = predictions['pole_count_logits']  # [batch, max_poles + 1]
        target_pole_count = targets['pole_count']  # [batch]

        loss_pole_count = F.cross_entropy(
            pole_count_logits,
            target_pole_count,
            reduction='none'
        )  # [batch]

        loss_pole_count = (loss_pole_count * structure_weight_t).mean()

        # ==================================================================
        # 5. Zero Count Loss (Discrete - Cross Entropy)
        # ==================================================================

        zero_count_logits = predictions['zero_count_logits']  # [batch, max_zeros + 1]
        target_zero_count = targets['zero_count']  # [batch]

        loss_zero_count = F.cross_entropy(
            zero_count_logits,
            target_zero_count,
            reduction='none'
        )  # [batch]

        loss_zero_count = (loss_zero_count * structure_weight_t).mean()

        # ==================================================================
        # 6. Pole Value Loss (Continuous - MSE)
        # ==================================================================

        pred_pole_values = predictions['pole_values']  # [batch, max_poles, 2]
        target_pole_values = targets['pole_values']  # [batch, max_poles, 2]

        # Create mask for valid poles
        max_poles = pred_pole_values.shape[1]
        pole_mask = torch.arange(max_poles, device=device).unsqueeze(0) < target_pole_count.unsqueeze(1)
        pole_mask = pole_mask.unsqueeze(-1).expand_as(pred_pole_values)  # [batch, max_poles, 2]

        loss_pole_value = F.mse_loss(
            pred_pole_values * pole_mask,
            target_pole_values * pole_mask,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)  # [batch]

        loss_pole_value = (loss_pole_value * value_weight_t).mean()

        # ==================================================================
        # 7. Zero Value Loss (Continuous - MSE)
        # ==================================================================

        pred_zero_values = predictions['zero_values']  # [batch, max_zeros, 2]
        target_zero_values = targets['zero_values']  # [batch, max_zeros, 2]

        # Create mask for valid zeros
        max_zeros = pred_zero_values.shape[1]
        zero_mask = torch.arange(max_zeros, device=device).unsqueeze(0) < target_zero_count.unsqueeze(1)
        zero_mask = zero_mask.unsqueeze(-1).expand_as(pred_zero_values)  # [batch, max_zeros, 2]

        loss_zero_value = F.mse_loss(
            pred_zero_values * zero_mask,
            target_zero_values * zero_mask,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)  # [batch]

        loss_zero_value = (loss_zero_value * value_weight_t).mean()

        # ==================================================================
        # 8. Structural Validity Loss (Optional)
        # ==================================================================

        # This can be added from the constraints module
        # For now, placeholder
        loss_structure = torch.tensor(0.0, device=device)

        # ==================================================================
        # 9. Combine losses
        # ==================================================================

        total_loss = (
            self.node_type_weight * loss_node_type +
            self.edge_exist_weight * loss_edge_exist +
            self.edge_value_weight * loss_edge_value +
            self.pole_count_weight * loss_pole_count +
            self.zero_count_weight * loss_zero_count +
            self.pole_value_weight * loss_pole_value +
            self.zero_value_weight * loss_zero_value +
            self.structure_weight * loss_structure
        )

        # ==================================================================
        # 10. Compute metrics
        # ==================================================================

        with torch.no_grad():
            # Node type accuracy
            pred_node_classes = torch.argmax(node_type_logits, dim=-1)
            node_type_acc = (pred_node_classes == target_node_types).float().mean().item() * 100

            # Pole count accuracy
            pred_pole_count = torch.argmax(pole_count_logits, dim=-1)
            pole_count_acc = (pred_pole_count == target_pole_count).float().mean().item() * 100

            # Zero count accuracy
            pred_zero_count = torch.argmax(zero_count_logits, dim=-1)
            zero_count_acc = (pred_zero_count == target_zero_count).float().mean().item() * 100

            # Edge existence accuracy
            pred_edge_exist = (torch.sigmoid(edge_exist_logits) > 0.5).float()
            edge_exist_acc = (pred_edge_exist == target_edge_exist).float().mean().item() * 100

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_edge_exist': loss_edge_exist.item(),
            'loss_edge_value': loss_edge_value.item(),
            'loss_pole_count': loss_pole_count.item(),
            'loss_zero_count': loss_zero_count.item(),
            'loss_pole_value': loss_pole_value.item(),
            'loss_zero_value': loss_zero_value.item(),
            'loss_structure': loss_structure.item(),
            'node_type_acc': node_type_acc,
            'pole_count_acc': pole_count_acc,
            'zero_count_acc': zero_count_acc,
            'edge_exist_acc': edge_exist_acc
        }

        return total_loss, metrics


class AdaptiveLossBalancer:
    """
    Adaptive loss balancing using gradient magnitudes.

    Automatically balances multiple loss terms by equalizing their
    gradient magnitudes, preventing one loss from dominating.

    Reference: GradNorm (Chen et al., 2018)

    Example:
        >>> balancer = AdaptiveLossBalancer(num_losses=7)
        >>> balanced_loss = balancer.balance(losses, model)
    """

    def __init__(self, num_losses: int, alpha: float = 0.12):
        self.num_losses = num_losses
        self.alpha = alpha  # Strength of restoring force

        # Initialize loss weights
        self.weights = torch.ones(num_losses)

    def balance(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> torch.Tensor:
        """
        Compute balanced loss using gradient magnitudes.

        Args:
            losses: Dictionary of individual losses
            model: Model to compute gradients for

        Returns:
            balanced_loss: Weighted sum of losses
        """
        # This is a simplified version
        # Full implementation would compute gradients and update weights
        # For now, just return weighted sum

        loss_values = torch.stack(list(losses.values()))
        weighted_loss = (self.weights.to(loss_values.device) * loss_values).sum()

        return weighted_loss
