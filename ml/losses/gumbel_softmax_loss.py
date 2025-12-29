"""
Loss function for Gumbel-Softmax Component Selection.

Key differences from standard loss:
1. Component type loss (cross-entropy for discrete selection)
2. Component value loss (MSE only where component exists)
3. Is-parallel loss (BCE for binary)
4. No mask loss (masks are derived from component type)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    from ml.losses.connectivity_loss import ConnectivityLoss
    from ml.models.gumbel_softmax_utils import masks_to_component_type
except ImportError:
    # For testing standalone
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ml.losses.connectivity_loss import ConnectivityLoss
    from ml.models.gumbel_softmax_utils import masks_to_component_type


class GumbelSoftmaxCircuitLoss(nn.Module):
    """
    Loss function for circuit generation with Gumbel-Softmax component selection.

    Combines:
    1. Node type loss (cross-entropy)
    2. Edge existence loss (BCE)
    3. Component type loss (cross-entropy) - NEW!
    4. Component value loss (MSE, masked by component existence)
    5. Is-parallel loss (BCE)
    6. Connectivity loss (optional)
    """

    def __init__(
        self,
        # Standard weights
        node_type_weight: float = 1.0,
        edge_exist_weight: float = 1.0,
        # NEW: Gumbel-Softmax weights
        component_type_weight: float = 2.0,  # Important for discrete selection
        component_value_weight: float = 1.0,
        is_parallel_weight: float = 0.5,
        # Connectivity loss
        use_connectivity_loss: bool = True,
        connectivity_weight: float = 5.0,
        # Gumbel-Softmax parameters
        gumbel_temperature: float = 0.5
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.edge_exist_weight = edge_exist_weight
        self.component_type_weight = component_type_weight
        self.component_value_weight = component_value_weight
        self.is_parallel_weight = is_parallel_weight
        self.gumbel_temperature = gumbel_temperature

        # Connectivity loss
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
        Compute Gumbel-Softmax loss.

        Args:
            predictions: Model outputs with:
                - 'node_types': [batch, max_nodes, num_types] logits
                - EITHER (Phase 3 - joint prediction):
                    - 'edge_component_logits': [batch, max_nodes, max_nodes, 8]
                      (class 0=no edge, 1-7=edge with component type)
                - OR (legacy - separate heads):
                    - 'edge_existence': [batch, max_nodes, max_nodes] logits
                    - 'component_type_logits': [batch, max_nodes, max_nodes, 8]
                - 'component_values': [batch, max_nodes, max_nodes, 3]
                - 'is_parallel_logits': [batch, max_nodes, max_nodes]
            targets: Ground truth with:
                - 'node_types': [batch, max_nodes] class indices
                - 'edge_existence': [batch, max_nodes, max_nodes] binary
                - 'component_types': [batch, max_nodes, max_nodes] component type indices (0-7)
                - 'component_values': [batch, max_nodes, max_nodes, 3] continuous values
                - 'is_parallel': [batch, max_nodes, max_nodes] binary

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
        # 2-3. Edge-Component Loss (UNIFIED - Phase 3)
        # ==================================================================

        # Check if using Phase 3 (joint prediction) or legacy (separate heads)
        use_joint_prediction = 'edge_component_logits' in predictions

        target_edges = targets['edge_existence']  # [batch, max_nodes, max_nodes]
        target_component_types = targets['component_types']  # [batch, max_nodes, max_nodes]

        if use_joint_prediction:
            # PHASE 3: Joint edge-component prediction
            # ==========================================

            edge_component_logits = predictions['edge_component_logits']  # [batch, N, N, 8]

            # Create unified target:
            # - Class 0: No edge
            # - Classes 1-7: Edge exists with component type
            target_edge_component = torch.where(
                target_edges.unsqueeze(-1) > 0.5,
                target_component_types.unsqueeze(-1),  # 1-7 for existing edges
                torch.zeros_like(target_component_types.unsqueeze(-1))  # 0 for non-existing edges
            ).squeeze(-1).long()  # [batch, N, N]

            # Upper triangle mask (avoid double counting)
            triu_mask = torch.triu(torch.ones_like(target_edges), diagonal=1)  # [batch, N, N]

            # Flatten for cross-entropy
            edge_comp_logits_flat = edge_component_logits.reshape(-1, 8)  # [batch*N*N, 8]
            target_edge_comp_flat = target_edge_component.reshape(-1)  # [batch*N*N]
            triu_mask_flat = triu_mask.reshape(-1)  # [batch*N*N]

            # Compute cross-entropy on all potential edges (upper triangle)
            if triu_mask_flat.sum() > 0:
                loss_per_edge_comp = F.cross_entropy(
                    edge_comp_logits_flat,
                    target_edge_comp_flat,
                    reduction='none'
                )
                loss_edge_component = (loss_per_edge_comp * triu_mask_flat).sum() / (triu_mask_flat.sum() + 1e-6)
            else:
                loss_edge_component = torch.tensor(0.0, device=device)

            # For compatibility, split into edge existence and component type losses
            # (used in total loss computation)
            loss_edge_exist = loss_edge_component * 0.5  # Half weight to edge existence
            loss_component_type = loss_edge_component * 0.5  # Half weight to component type

            # Compute accuracies
            with torch.no_grad():
                pred_edge_comp = torch.argmax(edge_comp_logits_flat, dim=-1)  # [batch*N*N]

                # Edge existence accuracy (class 0 vs 1-7)
                pred_has_edge = (pred_edge_comp > 0).float()
                target_has_edge = (target_edge_comp_flat > 0).float()
                edge_acc = ((pred_has_edge == target_has_edge).float() * triu_mask_flat).sum()
                edge_acc = (edge_acc / (triu_mask_flat.sum() + 1e-6)).item() * 100

                # Component type accuracy (only on existing edges)
                edge_mask_flat = (target_edge_comp_flat > 0).float()
                if edge_mask_flat.sum() > 0:
                    comp_type_acc = ((pred_edge_comp == target_edge_comp_flat).float() * edge_mask_flat).sum()
                    comp_type_acc = (comp_type_acc / (edge_mask_flat.sum() + 1e-6)).item() * 100
                else:
                    comp_type_acc = 0.0

            # Create edge_mask for later use (component values, is_parallel)
            edge_mask = (target_edges > 0.5).float()  # [batch, N, N]

        else:
            # LEGACY: Separate edge existence and component type heads
            # =========================================================

            edge_logits = predictions['edge_existence']  # [batch, max_nodes, max_nodes]
            component_type_logits = predictions['component_type_logits']  # [batch, N, N, 8]

            # Edge Existence Loss (Binary Cross-Entropy)
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

            # Component Type Loss (Cross-Entropy on existing edges only)
            edge_mask = (target_edges > 0.5).float()  # [batch, max_nodes, max_nodes]

            # Flatten for cross-entropy
            comp_logits_flat = component_type_logits.reshape(-1, 8)  # [batch*N*N, 8]
            comp_targets_flat = target_component_types.reshape(-1)  # [batch*N*N]
            edge_mask_flat = edge_mask.reshape(-1)  # [batch*N*N]

            # Compute loss only on existing edges
            if edge_mask_flat.sum() > 0:
                loss_per_comp = F.cross_entropy(
                    comp_logits_flat,
                    comp_targets_flat,
                    reduction='none'
                )
                loss_component_type = (loss_per_comp * edge_mask_flat).sum() / (edge_mask_flat.sum() + 1e-6)
            else:
                loss_component_type = torch.tensor(0.0, device=device)

            # Compute component type accuracy
            with torch.no_grad():
                if edge_mask_flat.sum() > 0:
                    pred_comp_types = torch.argmax(comp_logits_flat, dim=-1)
                    comp_type_acc = ((pred_comp_types == comp_targets_flat).float() * edge_mask_flat).sum()
                    comp_type_acc = (comp_type_acc / (edge_mask_flat.sum() + 1e-6)).item() * 100
                else:
                    comp_type_acc = 0.0

        # ==================================================================
        # 4. Component Values Loss (MSE, masked by component existence)
        # ==================================================================

        pred_values = predictions['component_values']  # [batch, max_nodes, max_nodes, 3]
        target_values = targets['component_values']  # [batch, max_nodes, max_nodes, 3]

        # Get component types to create masks for which values to use
        # target_component_types: 0=None, 1=R, 2=C, 3=L, 4=RC, 5=RL, 6=CL, 7=RCL
        # Component value indices: [0]=log(C), [1]=G, [2]=log(L_inv)

        # Create masks for each component value
        # mask_C: component types {2, 4, 6, 7} (C, RC, CL, RCL)
        mask_C = ((target_component_types == 2) | (target_component_types == 4) |
                  (target_component_types == 6) | (target_component_types == 7)).float()

        # mask_G: component types {1, 4, 5, 7} (R, RC, RL, RCL)
        mask_G = ((target_component_types == 1) | (target_component_types == 4) |
                  (target_component_types == 5) | (target_component_types == 7)).float()

        # mask_L: component types {3, 5, 6, 7} (L, RL, CL, RCL)
        mask_L = ((target_component_types == 3) | (target_component_types == 5) |
                  (target_component_types == 6) | (target_component_types == 7)).float()

        # Combine masks [batch, max_nodes, max_nodes, 3]
        value_masks = torch.stack([mask_C, mask_G, mask_L], dim=-1)  # [batch, N, N, 3]

        # MSE loss only on existing components
        value_diff = (pred_values - target_values) ** 2  # [batch, N, N, 3]
        loss_component_value = (value_diff * value_masks).sum() / (value_masks.sum() + 1e-6)

        # ==================================================================
        # 5. Is-Parallel Loss (Binary Cross-Entropy)
        # ==================================================================

        is_parallel_logits = predictions['is_parallel_logits']  # [batch, max_nodes, max_nodes]
        target_is_parallel = targets['is_parallel']  # [batch, max_nodes, max_nodes]

        # BCE only on existing edges
        loss_per_parallel = F.binary_cross_entropy_with_logits(
            is_parallel_logits,
            target_is_parallel,
            reduction='none'
        )
        loss_is_parallel = (loss_per_parallel * edge_mask).sum() / (edge_mask.sum() + 1e-6)

        # ==================================================================
        # 6. Connectivity Loss (Optional)
        # ==================================================================

        if self.use_connectivity_loss:
            # For Phase 3, convert joint prediction to edge existence logits
            if use_joint_prediction:
                # Derive edge existence probability: P(edge exists) = P(class > 0)
                # Convert to logits: log(p / (1-p))
                edge_probs = F.softmax(edge_component_logits, dim=-1)  # [batch, N, N, 8]
                edge_exist_probs = 1.0 - edge_probs[..., 0]  # [batch, N, N]
                edge_exist_probs = torch.clamp(edge_exist_probs, 1e-6, 1.0 - 1e-6)
                edge_logits = torch.log(edge_exist_probs / (1.0 - edge_exist_probs))
            # else: edge_logits already defined in legacy branch

            loss_connectivity, metrics_connectivity = self.connectivity_loss(
                edge_logits,
                predictions['node_types']
            )
        else:
            loss_connectivity = torch.tensor(0.0, device=device)
            metrics_connectivity = {}

        # ==================================================================
        # 7. Combine Losses
        # ==================================================================

        total_loss = (
            self.node_type_weight * loss_node_type +
            self.edge_exist_weight * loss_edge_exist +
            self.component_type_weight * loss_component_type +  # NEW!
            self.component_value_weight * loss_component_value +
            self.is_parallel_weight * loss_is_parallel +
            self.connectivity_weight * loss_connectivity
        )

        # ==================================================================
        # 8. Metrics
        # ==================================================================

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_edge_exist': loss_edge_exist.item(),
            'loss_component_type': loss_component_type.item(),  # NEW!
            'loss_component_value': loss_component_value.item(),
            'loss_is_parallel': loss_is_parallel.item(),
            'loss_connectivity': loss_connectivity.item() if self.use_connectivity_loss else 0.0,
            'node_type_acc': node_type_acc,
            'edge_exist_acc': edge_acc,
            'component_type_acc': comp_type_acc,  # NEW!
        }

        # Add connectivity metrics if available
        if self.use_connectivity_loss:
            metrics.update(metrics_connectivity)

        return total_loss, metrics


if __name__ == '__main__':
    """Test the Gumbel-Softmax loss function."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    print("Testing Gumbel-Softmax Circuit Loss\n")

    batch_size = 2
    max_nodes = 5

    # Create dummy predictions
    predictions = {
        'node_types': torch.randn(batch_size, max_nodes, 5),  # 5 node types
        'edge_existence': torch.randn(batch_size, max_nodes, max_nodes),
        'component_type_logits': torch.randn(batch_size, max_nodes, max_nodes, 8),  # 8 component types
        'component_values': torch.randn(batch_size, max_nodes, max_nodes, 3),
        'is_parallel_logits': torch.randn(batch_size, max_nodes, max_nodes),
    }

    # Create dummy targets
    targets = {
        'node_types': torch.randint(0, 5, (batch_size, max_nodes)),
        'edge_existence': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        'component_types': torch.randint(0, 8, (batch_size, max_nodes, max_nodes)),
        'component_values': torch.randn(batch_size, max_nodes, max_nodes, 3),
        'is_parallel': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
    }

    # Create loss function
    loss_fn = GumbelSoftmaxCircuitLoss(
        component_type_weight=2.0,
        use_connectivity_loss=False  # Disable for testing
    )

    # Compute loss
    total_loss, metrics = loss_fn(predictions, targets)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"\nLoss breakdown:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ Loss function test passed!")
