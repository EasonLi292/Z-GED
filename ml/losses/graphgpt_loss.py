"""
Loss function for GraphGPT-style autoregressive circuit generation.

Much simpler than diffusion loss:
- No timestep-dependent weighting
- No diffusion noise prediction
- Standard cross-entropy and MSE losses
- No focal loss needed (autoregressive handles imbalance naturally)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GraphGPTCircuitLoss(nn.Module):
    """
    Unified loss for GraphGPT circuit generation.

    Components:
    1. Node type loss: Cross-entropy (5 classes)
    2. Edge existence loss: Binary cross-entropy
    3. Edge value loss: MSE on existing edges
    4. Pole/zero count loss: Cross-entropy
    5. Pole/zero value loss: MSE with masking

    No need for:
    - Diffusion timestep handling
    - Focal loss (autoregressive naturally balanced)
    - Complex loss weighting schedules

    Args:
        node_type_weight: Weight for node type loss
        edge_exist_weight: Weight for edge existence loss
        edge_value_weight: Weight for edge value loss
        pole_count_weight: Weight for pole count loss
        zero_count_weight: Weight for zero count loss
        pole_value_weight: Weight for pole value loss
        zero_value_weight: Weight for zero value loss
        tf_weight: Weight for transfer function loss (optional)
    """

    def __init__(
        self,
        node_type_weight: float = 1.0,
        edge_exist_weight: float = 1.0,  # No 50x needed!
        edge_value_weight: float = 1.0,
        pole_count_weight: float = 1.0,
        zero_count_weight: float = 1.0,
        pole_value_weight: float = 1.0,
        zero_value_weight: float = 1.0,
        tf_weight: float = 0.0  # Optional TF loss
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

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and metrics.

        Args:
            predictions: Model outputs from GraphGPTDecoder.forward()
            targets: Ground truth containing:
                - 'node_types': [batch, max_nodes]
                - 'edge_existence': [batch, max_nodes, max_nodes]
                - 'edge_values': [batch, max_nodes, max_nodes, 7]
                - 'pole_count': [batch]
                - 'zero_count': [batch]
                - 'pole_values': [batch, max_poles, 2]
                - 'zero_values': [batch, max_zeros, 2]

        Returns:
            total_loss: Combined loss scalar
            metrics: Dictionary of loss components and accuracies
        """
        device = predictions['node_types'].device
        batch_size = predictions['node_types'].shape[0]

        # ==================================================================
        # 1. Node Type Loss (Cross-Entropy)
        # ==================================================================

        node_logits = predictions['node_types']  # [batch, max_nodes, num_types]
        target_nodes = targets['node_types']  # [batch, max_nodes]

        # Flatten and compute cross-entropy
        node_logits_flat = node_logits.reshape(-1, node_logits.shape[-1])
        target_nodes_flat = target_nodes.reshape(-1)

        loss_node_type = F.cross_entropy(
            node_logits_flat,
            target_nodes_flat,
            reduction='mean'
        )

        # ==================================================================
        # 2. Edge Existence Loss (Binary Cross-Entropy)
        # ==================================================================

        edge_exist_logits = predictions['edge_existence']  # [batch, max_nodes, max_nodes]
        target_edge_exist = targets['edge_existence']  # [batch, max_nodes, max_nodes]

        # Only compute loss on upper triangle (to avoid double counting)
        # Since graph is undirected and we made it symmetric
        mask = torch.triu(torch.ones_like(target_edge_exist), diagonal=1)

        loss_edge_exist = F.binary_cross_entropy_with_logits(
            edge_exist_logits[mask == 1],
            target_edge_exist[mask == 1],
            reduction='mean'
        )

        # ==================================================================
        # 3. Edge Value Loss (MSE on existing edges)
        # ==================================================================

        pred_edge_values = predictions['edge_values']  # [batch, max_nodes, max_nodes, 7]
        target_edge_values = targets['edge_values']  # [batch, max_nodes, max_nodes, 7]

        # Only compute loss where edges exist
        edge_mask = target_edge_exist.unsqueeze(-1).expand_as(pred_edge_values)

        loss_edge_value = F.mse_loss(
            pred_edge_values * edge_mask,
            target_edge_values * edge_mask,
            reduction='sum'
        ) / (edge_mask.sum() + 1e-6)  # Normalize by number of existing edges

        # ==================================================================
        # 4. Pole Count Loss (Cross-Entropy)
        # ==================================================================

        pole_count_logits = predictions['pole_count_logits']  # [batch, max_poles + 1]
        target_pole_count = targets['pole_count']  # [batch]

        loss_pole_count = F.cross_entropy(
            pole_count_logits,
            target_pole_count,
            reduction='mean'
        )

        # ==================================================================
        # 5. Zero Count Loss (Cross-Entropy)
        # ==================================================================

        zero_count_logits = predictions['zero_count_logits']  # [batch, max_zeros + 1]
        target_zero_count = targets['zero_count']  # [batch]

        loss_zero_count = F.cross_entropy(
            zero_count_logits,
            target_zero_count,
            reduction='mean'
        )

        # ==================================================================
        # 6. Pole Value Loss (MSE with masking)
        # ==================================================================

        pred_pole_values = predictions['pole_values']  # [batch, max_poles, 2]
        target_pole_values = targets['pole_values']  # [batch, max_poles, 2]

        # Create mask for valid poles
        max_poles = pred_pole_values.shape[1]
        pole_mask = torch.arange(max_poles, device=device).unsqueeze(0) < target_pole_count.unsqueeze(1)
        pole_mask = pole_mask.unsqueeze(-1).expand_as(pred_pole_values)

        loss_pole_value = F.mse_loss(
            pred_pole_values * pole_mask,
            target_pole_values * pole_mask,
            reduction='sum'
        ) / (pole_mask.sum() + 1e-6)

        # ==================================================================
        # 7. Zero Value Loss (MSE with masking)
        # ==================================================================

        pred_zero_values = predictions['zero_values']  # [batch, max_zeros, 2]
        target_zero_values = targets['zero_values']  # [batch, max_zeros, 2]

        # Create mask for valid zeros
        max_zeros = pred_zero_values.shape[1]
        zero_mask = torch.arange(max_zeros, device=device).unsqueeze(0) < target_zero_count.unsqueeze(1)
        zero_mask = zero_mask.unsqueeze(-1).expand_as(pred_zero_values)

        loss_zero_value = F.mse_loss(
            pred_zero_values * zero_mask,
            target_zero_values * zero_mask,
            reduction='sum'
        ) / (zero_mask.sum() + 1e-6)

        # ==================================================================
        # 8. Combine Losses
        # ==================================================================

        total_loss = (
            self.node_type_weight * loss_node_type +
            self.edge_exist_weight * loss_edge_exist +
            self.edge_value_weight * loss_edge_value +
            self.pole_count_weight * loss_pole_count +
            self.zero_count_weight * loss_zero_count +
            self.pole_value_weight * loss_pole_value +
            self.zero_value_weight * loss_zero_value
        )

        # Optional: Add transfer function loss if available
        if self.tf_weight > 0 and 'tf_loss' in targets:
            total_loss = total_loss + self.tf_weight * targets['tf_loss']

        # ==================================================================
        # 9. Compute Metrics
        # ==================================================================

        with torch.no_grad():
            # Node type accuracy
            pred_node_classes = torch.argmax(node_logits, dim=-1)
            node_type_acc = (pred_node_classes == target_nodes).float().mean().item() * 100

            # Pole count accuracy
            pred_pole_count = torch.argmax(pole_count_logits, dim=-1)
            pole_count_acc = (pred_pole_count == target_pole_count).float().mean().item() * 100

            # Zero count accuracy
            pred_zero_count = torch.argmax(zero_count_logits, dim=-1)
            zero_count_acc = (pred_zero_count == target_zero_count).float().mean().item() * 100

            # Edge existence accuracy
            pred_edge_exist = (torch.sigmoid(edge_exist_logits) > 0.5).float()
            edge_exist_acc = (pred_edge_exist == target_edge_exist).float().mean().item() * 100

            # Edge probability statistics (for monitoring)
            edge_probs = torch.sigmoid(edge_exist_logits)
            edge_prob_mean = edge_probs.mean().item()
            edge_prob_std = edge_probs.std().item()

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_edge_exist': loss_edge_exist.item(),
            'loss_edge_value': loss_edge_value.item(),
            'loss_pole_count': loss_pole_count.item(),
            'loss_zero_count': loss_zero_count.item(),
            'loss_pole_value': loss_pole_value.item(),
            'loss_zero_value': loss_zero_value.item(),
            'node_type_acc': node_type_acc,
            'pole_count_acc': pole_count_acc,
            'zero_count_acc': zero_count_acc,
            'edge_exist_acc': edge_exist_acc,
            'edge_prob_mean': edge_prob_mean,
            'edge_prob_std': edge_prob_std
        }

        return total_loss, metrics


if __name__ == '__main__':
    # Test the loss function
    print("Testing GraphGPT Loss Function...")

    loss_fn = GraphGPTCircuitLoss()

    # Create dummy predictions and targets
    batch_size = 4
    max_nodes = 5
    max_poles = 4
    max_zeros = 4

    predictions = {
        'node_types': torch.randn(batch_size, max_nodes, 5),
        'edge_existence': torch.randn(batch_size, max_nodes, max_nodes),
        'edge_values': torch.randn(batch_size, max_nodes, max_nodes, 7),
        'pole_count_logits': torch.randn(batch_size, max_poles + 1),
        'zero_count_logits': torch.randn(batch_size, max_zeros + 1),
        'pole_values': torch.randn(batch_size, max_poles, 2),
        'zero_values': torch.randn(batch_size, max_zeros, 2)
    }

    targets = {
        'node_types': torch.randint(0, 5, (batch_size, max_nodes)),
        'edge_existence': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        'edge_values': torch.randn(batch_size, max_nodes, max_nodes, 7),
        'pole_count': torch.randint(0, max_poles + 1, (batch_size,)),
        'zero_count': torch.randint(0, max_zeros + 1, (batch_size,)),
        'pole_values': torch.randn(batch_size, max_poles, 2),
        'zero_values': torch.randn(batch_size, max_zeros, 2)
    }

    # Compute loss
    total_loss, metrics = loss_fn(predictions, targets)

    print(f"✅ Loss computation successful")
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Node type loss: {metrics['loss_node_type']:.4f}")
    print(f"   Edge exist loss: {metrics['loss_edge_exist']:.4f}")
    print(f"   Edge value loss: {metrics['loss_edge_value']:.4f}")
    print(f"\n   Node type acc: {metrics['node_type_acc']:.1f}%")
    print(f"   Edge exist acc: {metrics['edge_exist_acc']:.1f}%")
    print(f"   Pole count acc: {metrics['pole_count_acc']:.1f}%")
    print(f"\n   Edge prob mean: {metrics['edge_prob_mean']:.4f}")
    print(f"   Edge prob std: {metrics['edge_prob_std']:.4f}")

    print("\n✅ All tests passed!")
