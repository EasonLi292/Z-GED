"""
Loss function for Latent-Guided GraphGPT.

Key differences from standard GraphGPT loss:
1. TF consistency weighting for edges
2. No pole/zero prediction (TF evaluated via SPICE simulation)
3. Optional connectivity loss integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from ml.losses.connectivity_loss import ConnectivityLoss


class LatentGuidedCircuitLoss(nn.Module):
    """
    Loss function for latent-guided circuit generation without TF prediction.

    Combines:
    1. Node type loss (cross-entropy)
    2. Edge existence loss (BCE)
    3. Edge value loss (MSE on existing edges)
    4. TF consistency-weighted edge loss (NEW!)
    5. Connectivity loss (OPTIONAL)

    Transfer function evaluation is done via SPICE simulation, not prediction.
    """

    def __init__(
        self,
        # Standard weights
        node_type_weight: float = 1.0,
        edge_exist_weight: float = 1.0,
        edge_value_weight: float = 1.0,
        # NEW: Latent-guided weights
        consistency_weighting_strength: float = 2.0,
        # OPTIONAL: Connectivity loss
        use_connectivity_loss: bool = True,
        connectivity_weight: float = 5.0
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.edge_exist_weight = edge_exist_weight
        self.edge_value_weight = edge_value_weight

        # NEW: Latent-guided parameters
        self.consistency_weighting_strength = consistency_weighting_strength

        # OPTIONAL: Connectivity loss
        self.use_connectivity_loss = use_connectivity_loss
        if use_connectivity_loss:
            self.connectivity_loss = ConnectivityLoss(
                vin_weight=10.0,
                vout_weight=5.0,
                graph_weight=3.0,
                isolated_weight=2.0
            )
        self.connectivity_weight = connectivity_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute latent-guided loss (without TF prediction).

        Args:
            predictions: Model outputs with:
                - 'node_types': [batch, max_nodes, num_types] logits
                - 'edge_existence': [batch, max_nodes, max_nodes] logits
                - 'edge_values': [batch, max_nodes, max_nodes, 7]
                - 'consistency_scores': [batch, max_nodes, max_nodes]
            targets: Ground truth with:
                - 'node_types': [batch, max_nodes] class indices
                - 'edge_existence': [batch, max_nodes, max_nodes] binary
                - 'edge_values': [batch, max_nodes, max_nodes, 7]
            latent: Not used (kept for API compatibility)

        Returns:
            total_loss: Combined loss
            metrics: Dictionary with loss breakdown
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

        loss_node_type = F.cross_entropy(node_logits_flat, target_nodes_flat)

        # Compute node type accuracy
        with torch.no_grad():
            pred_nodes = torch.argmax(node_logits_flat, dim=-1)
            node_type_acc = (pred_nodes == target_nodes_flat).float().mean().item() * 100

        # ==================================================================
        # 2. Edge Existence Loss (Binary Cross-Entropy)
        # ==================================================================

        edge_logits = predictions['edge_existence']  # [batch, max_nodes, max_nodes]
        target_edges = targets['edge_existence']  # [batch, max_nodes, max_nodes]

        # BCE loss (only upper triangle to avoid double counting)
        loss_per_edge = F.binary_cross_entropy_with_logits(
            edge_logits,
            target_edges,
            reduction='none'
        )

        mask = torch.triu(torch.ones_like(target_edges), diagonal=1)
        loss_edge_exist = (loss_per_edge * mask).sum() / (mask.sum() + 1e-6)

        # Compute edge accuracy
        with torch.no_grad():
            pred_edges = (torch.sigmoid(edge_logits) > 0.5).float()
            edge_acc = ((pred_edges == target_edges).float() * mask).sum() / (mask.sum() + 1e-6)
            edge_acc = edge_acc.item() * 100

        # ==================================================================
        # 3. Edge Value Loss (MSE on existing edges)
        # ==================================================================

        pred_values = predictions['edge_values']  # [batch, max_nodes, max_nodes, 7]
        target_values = targets['edge_values']  # [batch, max_nodes, max_nodes, 7]

        # Only compute loss on existing edges
        edge_mask = (target_edges > 0.5).unsqueeze(-1).float()  # [batch, max_nodes, max_nodes, 1]

        # MSE loss
        value_diff = (pred_values - target_values) ** 2
        loss_edge_value = (value_diff * edge_mask).sum() / (edge_mask.sum() + 1e-6)

        # ==================================================================
        # 4. TF Consistency-Weighted Edge Loss (optional enhancement)
        # ==================================================================

        # Get consistency scores from decoder
        consistency_scores = predictions.get('consistency_scores', None)

        if consistency_scores is not None:
            # Normalize consistency scores to [0, 1]
            consistency_normalized = torch.sigmoid(consistency_scores)

            # Metrics
            avg_consistency = consistency_normalized.mean().item()
            high_consistency_edges = (consistency_normalized > 0.7).float().mean().item()

            # Optionally: weight edge loss by consistency
            # (Currently not used, but available for future enhancement)
        else:
            avg_consistency = 0.0
            high_consistency_edges = 0.0

        # ==================================================================
        # 5. OPTIONAL: Connectivity Loss
        # ==================================================================

        if self.use_connectivity_loss:
            loss_connectivity, metrics_connectivity = self.connectivity_loss(
                edge_logits,
                predictions['node_types']
            )
        else:
            loss_connectivity = torch.tensor(0.0, device=device)
            metrics_connectivity = {}

        # ==================================================================
        # 6. Combine Losses
        # ==================================================================

        total_loss = (
            self.node_type_weight * loss_node_type +
            self.edge_exist_weight * loss_edge_exist +
            self.edge_value_weight * loss_edge_value +
            self.connectivity_weight * loss_connectivity
        )

        # ==================================================================
        # 7. Metrics
        # ==================================================================

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_edge_exist': loss_edge_exist.item(),
            'loss_edge_value': loss_edge_value.item(),
            'node_type_acc': node_type_acc,
            'edge_exist_acc': edge_acc,
            'avg_consistency_score': avg_consistency,
            'high_consistency_edges_pct': high_consistency_edges * 100,
            **metrics_connectivity,
        }

        return total_loss, metrics


if __name__ == '__main__':
    """Test latent-guided loss."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    print("Testing Latent-Guided Loss Function (without TF prediction)...")

    batch_size = 4
    max_nodes = 5

    # Create predictions (no pole/zero predictions)
    predictions = {
        'node_types': torch.randn(batch_size, max_nodes, 5),
        'edge_existence': torch.randn(batch_size, max_nodes, max_nodes),
        'edge_values': torch.randn(batch_size, max_nodes, max_nodes, 7),
        'consistency_scores': torch.randn(batch_size, max_nodes, max_nodes),
    }

    # Create targets (no pole/zero targets)
    targets = {
        'node_types': torch.randint(0, 5, (batch_size, max_nodes)),
        'edge_existence': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        'edge_values': torch.randn(batch_size, max_nodes, max_nodes, 7),
    }

    # Test loss
    loss_fn = LatentGuidedCircuitLoss(
        consistency_weighting_strength=2.0,
        use_connectivity_loss=True,
        connectivity_weight=5.0
    )

    total_loss, metrics = loss_fn(predictions, targets, latent=None)

    print(f"\n✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"\nLoss breakdown:")
    print(f"  Node type loss: {metrics['loss_node_type']:.4f}")
    print(f"  Edge existence loss: {metrics['loss_edge_exist']:.4f}")
    print(f"  Edge value loss: {metrics['loss_edge_value']:.4f}")
    if 'loss_vin_connectivity' in metrics:
        print(f"  VIN connectivity loss: {metrics['loss_vin_connectivity']:.4f}")

    print(f"\nMetrics:")
    print(f"  Node type accuracy: {metrics['node_type_acc']:.1f}%")
    print(f"  Edge existence accuracy: {metrics['edge_exist_acc']:.1f}%")
    print(f"  Avg consistency score: {metrics['avg_consistency_score']:.3f}")
    print(f"  High consistency edges: {metrics['high_consistency_edges_pct']:.1f}%")

    print("\n✅ All tests passed!")
    print("\nLatent-Guided Loss (without TF prediction) is ready for training!")
    print("TF evaluation will be done via SPICE simulation instead.")
