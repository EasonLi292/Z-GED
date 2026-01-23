"""
Simplified Loss Function for Topology-Only Circuit Generation.

Only topology losses - no component value prediction.

Losses:
1. Node type loss (cross-entropy)
2. Node count loss (cross-entropy for 3/4/5 nodes)
3. Edge-component loss (joint 8-way classification)
4. Connectivity loss (optional)
5. KL loss (VAE regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    from ml.losses.connectivity_loss import ConnectivityLoss
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ml.losses.connectivity_loss import ConnectivityLoss


class GumbelSoftmaxCircuitLoss(nn.Module):
    """
    Topology-only loss function.

    Losses:
    1. Node type (cross-entropy)
    2. Node count (cross-entropy)
    3. Edge-component (8-way cross-entropy)
    4. Connectivity (optional)
    5. KL (VAE regularization)
    """

    def __init__(
        self,
        node_type_weight: float = 1.0,
        node_count_weight: float = 5.0,
        edge_component_weight: float = 2.0,
        connectivity_weight: float = 5.0,
        kl_weight: float = 0.01,  # Low KL for topology separation in latent space
        use_connectivity_loss: bool = True
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.node_count_weight = node_count_weight
        self.edge_component_weight = edge_component_weight
        self.connectivity_weight = connectivity_weight
        self.kl_weight = kl_weight

        self.use_connectivity_loss = use_connectivity_loss
        if use_connectivity_loss:
            self.connectivity_loss = ConnectivityLoss(
                vin_weight=10.0,
                vout_weight=5.0,
                graph_weight=3.0,
                isolated_weight=2.0
            )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute topology-only loss.

        Args:
            predictions:
                - 'node_types': [batch, num_nodes, 5] logits
                - 'node_count_logits': [batch, 3]
                - 'edge_component_logits': [batch, num_nodes, num_nodes, 8]
            targets:
                - 'node_types': [batch, num_nodes]
                - 'edge_existence': [batch, num_nodes, num_nodes]
                - 'component_types': [batch, num_nodes, num_nodes]
        """
        device = predictions['node_types'].device

        # 1. Node Type Loss
        node_logits = predictions['node_types']
        target_nodes = targets['node_types']

        node_logits_flat = node_logits.reshape(-1, node_logits.shape[-1])
        target_nodes_flat = target_nodes.reshape(-1)

        loss_node_type = F.cross_entropy(node_logits_flat, target_nodes_flat)

        with torch.no_grad():
            pred_nodes = torch.argmax(node_logits_flat, dim=-1)
            node_type_acc = (pred_nodes == target_nodes_flat).float().mean().item() * 100

        # 2. Node Count Loss
        if 'node_count_logits' in predictions:
            node_count_logits = predictions['node_count_logits']

            with torch.no_grad():
                target_node_counts = (target_nodes != 4).sum(dim=1)
                target_count_class = (target_node_counts - 3).clamp(0, 2).long()

            loss_node_count = F.cross_entropy(node_count_logits, target_count_class)

            with torch.no_grad():
                pred_count = torch.argmax(node_count_logits, dim=-1)
                node_count_acc = (pred_count == target_count_class).float().mean().item() * 100
        else:
            loss_node_count = torch.tensor(0.0, device=device)
            node_count_acc = 0.0

        # 3. Edge-Component Loss (8-way)
        edge_component_logits = predictions['edge_component_logits']
        target_edges = targets['edge_existence']
        target_component_types = targets['component_types']

        # Unified target: 0=no edge, 1-7=edge with component type
        target_edge_component = torch.where(
            target_edges > 0.5,
            target_component_types,
            torch.zeros_like(target_component_types)
        ).long()

        triu_mask = torch.triu(torch.ones_like(target_edges), diagonal=1)

        edge_comp_logits_flat = edge_component_logits.reshape(-1, 8)
        target_edge_comp_flat = target_edge_component.reshape(-1)
        triu_mask_flat = triu_mask.reshape(-1)

        if triu_mask_flat.sum() > 0:
            loss_per_edge = F.cross_entropy(edge_comp_logits_flat, target_edge_comp_flat, reduction='none')
            loss_edge_component = (loss_per_edge * triu_mask_flat).sum() / (triu_mask_flat.sum() + 1e-6)
        else:
            loss_edge_component = torch.tensor(0.0, device=device)

        with torch.no_grad():
            pred_edge_comp = torch.argmax(edge_comp_logits_flat, dim=-1)

            pred_has_edge = (pred_edge_comp > 0).float()
            target_has_edge = (target_edge_comp_flat > 0).float()
            edge_acc = ((pred_has_edge == target_has_edge).float() * triu_mask_flat).sum()
            edge_acc = (edge_acc / (triu_mask_flat.sum() + 1e-6)).item() * 100

            edge_mask_flat = (target_edge_comp_flat > 0).float()
            if edge_mask_flat.sum() > 0:
                comp_type_acc = ((pred_edge_comp == target_edge_comp_flat).float() * edge_mask_flat).sum()
                comp_type_acc = (comp_type_acc / (edge_mask_flat.sum() + 1e-6)).item() * 100
            else:
                comp_type_acc = 0.0

        # 4. Connectivity Loss
        if self.use_connectivity_loss:
            edge_probs = F.softmax(edge_component_logits, dim=-1)
            edge_exist_probs = torch.clamp(1.0 - edge_probs[..., 0], 1e-6, 1.0 - 1e-6)
            edge_logits = torch.log(edge_exist_probs / (1.0 - edge_exist_probs))

            loss_connectivity, metrics_connectivity = self.connectivity_loss(
                edge_logits, predictions['node_types']
            )
        else:
            loss_connectivity = torch.tensor(0.0, device=device)
            metrics_connectivity = {}

        # 5. KL Loss
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # Total
        total_loss = (
            self.node_type_weight * loss_node_type +
            self.node_count_weight * loss_node_count +
            self.edge_component_weight * loss_edge_component +
            self.connectivity_weight * loss_connectivity +
            self.kl_weight * kl_loss
        )

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_node_count': loss_node_count.item(),
            'loss_edge_component': loss_edge_component.item(),
            'loss_connectivity': loss_connectivity.item() if self.use_connectivity_loss else 0.0,
            'loss_kl': kl_loss.item(),
            'node_type_acc': node_type_acc,
            'node_count_acc': node_count_acc,
            'edge_exist_acc': edge_acc,
            'component_type_acc': comp_type_acc,
        }

        if self.use_connectivity_loss:
            metrics.update(metrics_connectivity)

        return total_loss, metrics


if __name__ == '__main__':
    print("Testing Topology Loss...")

    batch_size = 2
    max_nodes = 5

    predictions = {
        'node_types': torch.randn(batch_size, max_nodes, 5),
        'node_count_logits': torch.randn(batch_size, 3),
        'edge_component_logits': torch.randn(batch_size, max_nodes, max_nodes, 8),
    }

    targets = {
        'node_types': torch.randint(0, 5, (batch_size, max_nodes)),
        'edge_existence': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        'component_types': torch.randint(0, 8, (batch_size, max_nodes, max_nodes)),
    }

    loss_fn = GumbelSoftmaxCircuitLoss(use_connectivity_loss=False)
    total_loss, metrics = loss_fn(predictions, targets)

    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ Test passed!")
