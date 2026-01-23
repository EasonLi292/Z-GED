"""
Connectivity Loss for Circuit Generation.

Ensures generated circuits have proper connectivity (VIN/VOUT connected).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConnectivityLoss(nn.Module):
    """
    Loss component that penalizes disconnected circuits.

    Key constraints for RLC circuits:
    1. VIN must have at least 1 edge (CRITICAL)
    2. VOUT must have at least 1 edge (CRITICAL)
    3. Graph should be connected (IMPORTANT)
    4. No isolated nodes (except MASK)

    Args:
        vin_weight: Weight for VIN connectivity constraint
        vout_weight: Weight for VOUT connectivity constraint
        graph_weight: Weight for overall graph connectivity
        isolated_weight: Weight for isolated node penalty
    """

    def __init__(
        self,
        vin_weight: float = 10.0,
        vout_weight: float = 5.0,
        graph_weight: float = 3.0,
        isolated_weight: float = 2.0
    ):
        super().__init__()

        self.vin_weight = vin_weight
        self.vout_weight = vout_weight
        self.graph_weight = graph_weight
        self.isolated_weight = isolated_weight

    def forward(
        self,
        edge_logits: torch.Tensor,
        node_types: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute connectivity loss.

        Args:
            edge_logits: Predicted edge existence logits [batch, max_nodes, max_nodes]
            node_types: Node type predictions [batch, max_nodes]
                        (can be logits or hard labels)

        Returns:
            loss: Total connectivity loss scalar
            metrics: Dictionary with loss breakdown
        """
        batch_size = edge_logits.shape[0]
        max_nodes = edge_logits.shape[1]
        device = edge_logits.device

        # Convert edge logits to probabilities
        edge_probs = torch.sigmoid(edge_logits)

        # If node_types are logits, convert to hard labels
        if node_types.dim() == 3:  # [batch, max_nodes, num_types]
            node_labels = torch.argmax(node_types, dim=-1)
        else:
            node_labels = node_types

        # 1. VIN Connectivity Loss (MOST CRITICAL)
        loss_vin = torch.tensor(0.0, device=device)
        vin_violations = 0

        for batch_idx in range(batch_size):
            vin_mask = (node_labels[batch_idx] == 1)

            if vin_mask.sum() > 0:
                vin_id = vin_mask.nonzero()[0].item()
                vin_degree = edge_probs[batch_idx, vin_id, :].sum() + \
                            edge_probs[batch_idx, :, vin_id].sum()
                penalty = F.relu(1.0 - vin_degree)
                loss_vin += penalty

                if penalty > 0.5:
                    vin_violations += 1

        loss_vin = loss_vin / batch_size

        # 2. VOUT Connectivity Loss
        loss_vout = torch.tensor(0.0, device=device)
        vout_violations = 0

        for batch_idx in range(batch_size):
            vout_mask = (node_labels[batch_idx] == 2)

            if vout_mask.sum() > 0:
                vout_id = vout_mask.nonzero()[0].item()
                vout_degree = edge_probs[batch_idx, vout_id, :].sum() + \
                             edge_probs[batch_idx, :, vout_id].sum()
                penalty = F.relu(1.0 - vout_degree)
                loss_vout += penalty

                if penalty > 0.5:
                    vout_violations += 1

        loss_vout = loss_vout / batch_size

        # 3. Graph Connectivity Loss (Differentiable)
        loss_graph = torch.tensor(0.0, device=device)

        for batch_idx in range(batch_size):
            A = edge_probs[batch_idx]
            A_with_loops = A + torch.eye(max_nodes, device=device)

            # Compute k-hop connectivity matrix
            A_pow = A_with_loops
            for _ in range(max_nodes - 2):
                A_pow = A_pow @ A_with_loops

            valid_mask = (node_labels[batch_idx] < 4).float()
            valid_pairs = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(0)
            connectivity_strength = A_pow * valid_pairs
            penalty = F.relu(0.1 - connectivity_strength).sum()
            loss_graph += penalty

        loss_graph = loss_graph / (batch_size * max_nodes * max_nodes)

        # 4. Isolated Node Loss
        loss_isolated = torch.tensor(0.0, device=device)
        isolated_violations = 0

        for batch_idx in range(batch_size):
            for node_id in range(max_nodes):
                if node_labels[batch_idx, node_id] < 4:
                    degree = edge_probs[batch_idx, node_id, :].sum() + \
                            edge_probs[batch_idx, :, node_id].sum()
                    penalty = F.relu(0.5 - degree)
                    loss_isolated += penalty

                    if penalty > 0.25:
                        isolated_violations += 1

        loss_isolated = loss_isolated / (batch_size * max_nodes)

        # 5. Combine Losses
        total_loss = (
            self.vin_weight * loss_vin +
            self.vout_weight * loss_vout +
            self.graph_weight * loss_graph +
            self.isolated_weight * loss_isolated
        )

        metrics = {
            'loss_vin_connectivity': loss_vin.item(),
            'loss_vout_connectivity': loss_vout.item(),
            'loss_graph_connectivity': loss_graph.item(),
            'loss_isolated_nodes': loss_isolated.item(),
            'vin_violations': vin_violations,
            'vout_violations': vout_violations,
            'isolated_violations': isolated_violations
        }

        return total_loss, metrics
