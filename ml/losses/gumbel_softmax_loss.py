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
        # Learned stopping criterion
        stop_weight: float = 1.0,  # Weight for stopping criterion loss
        # NEW: Stop-Node correlation loss (Fix 2 from Generation Plan)
        # Forces consistency between stop prediction and node type prediction
        stop_node_correlation_weight: float = 2.0,
        # NEW: Direct node count prediction loss
        node_count_weight: float = 5.0,  # High weight - this is CRITICAL for correct stopping
        # Connectivity loss
        use_connectivity_loss: bool = True,
        connectivity_weight: float = 5.0,
        # Gumbel-Softmax parameters
        gumbel_temperature: float = 0.5,
        # VAE regularization
        kl_weight: float = 0.005
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.edge_exist_weight = edge_exist_weight
        self.component_type_weight = component_type_weight
        self.component_value_weight = component_value_weight
        self.is_parallel_weight = is_parallel_weight
        self.stop_weight = stop_weight
        self.stop_node_correlation_weight = stop_node_correlation_weight
        self.node_count_weight = node_count_weight  # NEW
        self.gumbel_temperature = gumbel_temperature
        self.kl_weight = kl_weight

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
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
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
        # 1.5 Stopping Criterion Loss (NEW - Method 1)
        # ==================================================================

        if 'stop_logits' in predictions and predictions['stop_logits'] is not None:
            stop_logits = predictions['stop_logits']  # [batch, max_nodes]

            # Create stop targets: 1 if position should be MASK, 0 if real node
            # Node type 4 = MASK
            target_stop = (target_nodes == 4).float()  # [batch, max_nodes]

            # BCE loss for stopping criterion
            loss_stop = F.binary_cross_entropy_with_logits(
                stop_logits,
                target_stop,
                reduction='mean'
            )

            # Compute stop accuracy
            with torch.no_grad():
                pred_stop = (torch.sigmoid(stop_logits) > 0.5).float()
                stop_acc = (pred_stop == target_stop).float().mean().item() * 100

            # ==================================================================
            # 1.6 Stop-Node Correlation Loss (Fix 2 from Generation Plan)
            # ==================================================================
            # This loss forces consistency between stop prediction and node type:
            # - When should_stop=1 (MASK position): penalize high INTERNAL prob
            # - When should_stop=0 (real node): penalize high MASK prob
            #
            # Without this, the model can have:
            #   stop_prob=0.55 (predicts stop) but node_logits favor INTERNAL
            # Leading to: "100% stop accuracy" but wrong node count during generation

            # Only apply to positions >= 3 (where stopping is possible)
            # Positions 0,1,2 are always GND, VIN, VOUT
            max_nodes = node_logits.shape[1]

            if max_nodes > 3:
                # Get node probabilities for positions 3+
                node_probs = F.softmax(node_logits[:, 3:, :], dim=-1)  # [batch, max_nodes-3, 5]
                internal_prob = node_probs[:, :, 3]  # INTERNAL class = 3
                mask_prob = node_probs[:, :, 4]      # MASK class = 4

                # Get stop targets for positions 3+
                stop_targets_3plus = target_stop[:, 3:]  # [batch, max_nodes-3]

                # Correlation loss:
                # 1. When should_stop=1: penalize high INTERNAL probability
                #    Loss = stop_target * internal_prob
                # 2. When should_stop=0: penalize high MASK probability
                #    Loss = (1 - stop_target) * mask_prob

                loss_stop_internal = (stop_targets_3plus * internal_prob).mean()
                loss_stop_mask = ((1 - stop_targets_3plus) * mask_prob).mean()

                loss_stop_node_correlation = loss_stop_internal + loss_stop_mask

                # Compute correlation accuracy: do stop and node type agree?
                with torch.no_grad():
                    # Check if predictions are consistent
                    pred_node_types_3plus = torch.argmax(node_logits[:, 3:, :], dim=-1)  # [batch, max_nodes-3]
                    pred_is_mask = (pred_node_types_3plus == 4).float()
                    pred_stop_3plus = (torch.sigmoid(stop_logits[:, 3:]) > 0.5).float()

                    # Correlation = both predict stop (stop=1 AND node=MASK)
                    #            OR both predict continue (stop=0 AND node!=MASK)
                    consistent = ((pred_stop_3plus == 1) & (pred_is_mask == 1)) | \
                                 ((pred_stop_3plus == 0) & (pred_is_mask == 0))
                    stop_node_correlation_acc = consistent.float().mean().item() * 100
            else:
                loss_stop_node_correlation = torch.tensor(0.0, device=device)
                stop_node_correlation_acc = 100.0  # No positions to check

        else:
            loss_stop = torch.tensor(0.0, device=device)
            loss_stop_node_correlation = torch.tensor(0.0, device=device)
            stop_acc = 0.0
            stop_node_correlation_acc = 0.0

        # ==================================================================
        # 1.7 Direct Node Count Loss (NEW - avoids train/test mismatch!)
        # ==================================================================
        # This loss trains a head to directly predict the number of nodes
        # Target: count non-MASK nodes in target_nodes → 3, 4, or 5
        # Output: 3-class classification (0=3 nodes, 1=4 nodes, 2=5 nodes)

        if 'node_count_logits' in predictions and predictions['node_count_logits'] is not None:
            node_count_logits = predictions['node_count_logits']  # [batch, 3]

            # Compute target node count for each sample in batch
            # Count nodes that are NOT MASK (type 4)
            with torch.no_grad():
                target_node_counts = (target_nodes != 4).sum(dim=1)  # [batch]
                # Convert to class index: 3 nodes → 0, 4 nodes → 1, 5 nodes → 2
                target_count_class = (target_node_counts - 3).clamp(0, 2).long()

            # Cross-entropy loss for node count prediction
            loss_node_count = F.cross_entropy(node_count_logits, target_count_class)

            # Compute node count accuracy
            with torch.no_grad():
                pred_count_class = torch.argmax(node_count_logits, dim=-1)
                node_count_acc = (pred_count_class == target_count_class).float().mean().item() * 100
        else:
            loss_node_count = torch.tensor(0.0, device=device)
            node_count_acc = 0.0

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
        # 7. KL Divergence (VAE Regularization)
        # ==================================================================

        if mu is not None and logvar is not None:
            # Standard analytical KL divergence for VAE
            # KL(q(z|x) || p(z)) where p(z) = N(0,1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.shape[0]  # Average over batch
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # 8. Combine Losses
        # ==================================================================

        total_loss = (
            self.node_type_weight * loss_node_type +
            self.stop_weight * loss_stop +  # Stopping criterion
            self.stop_node_correlation_weight * loss_stop_node_correlation +  # Fix 2 correlation
            self.node_count_weight * loss_node_count +  # NEW: Direct node count prediction
            self.edge_exist_weight * loss_edge_exist +
            self.component_type_weight * loss_component_type +
            self.component_value_weight * loss_component_value +
            self.is_parallel_weight * loss_is_parallel +
            self.connectivity_weight * loss_connectivity +
            self.kl_weight * kl_loss
        )

        # ==================================================================
        # 8. Metrics
        # ==================================================================

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_stop': loss_stop.item(),
            'loss_stop_correlation': loss_stop_node_correlation.item(),
            'loss_node_count': loss_node_count.item(),  # NEW: Direct count
            'loss_edge_exist': loss_edge_exist.item(),
            'loss_component_type': loss_component_type.item(),
            'loss_component_value': loss_component_value.item(),
            'loss_is_parallel': loss_is_parallel.item(),
            'loss_connectivity': loss_connectivity.item() if self.use_connectivity_loss else 0.0,
            'loss_kl': kl_loss.item(),
            'node_type_acc': node_type_acc,
            'stop_acc': stop_acc,
            'stop_node_corr_acc': stop_node_correlation_acc,
            'node_count_acc': node_count_acc,  # NEW: This is the KEY metric for generation!
            'edge_exist_acc': edge_acc,
            'component_type_acc': comp_type_acc,
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
