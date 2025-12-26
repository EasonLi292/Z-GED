"""
Structural validity loss for circuit generation.

Implements differentiable losses that encourage structurally valid circuits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class StructuralValidityLoss(nn.Module):
    """
    Differentiable loss for structural circuit validity.

    Encourages:
    1. Presence of required nodes (GND, VIN, VOUT)
    2. Graph connectivity
    3. Node diversity (minimum 3 unique nodes)
    4. No self-loops
    5. Balanced edge distribution

    Args:
        required_nodes_weight: Weight for required nodes loss
        connectivity_weight: Weight for connectivity loss
        diversity_weight: Weight for node diversity loss
        self_loop_weight: Weight for self-loop penalty
        balance_weight: Weight for edge balance loss

    Example:
        >>> loss_fn = StructuralValidityLoss()
        >>> loss = loss_fn(predictions)
    """

    def __init__(
        self,
        required_nodes_weight: float = 1.0,
        connectivity_weight: float = 1.0,
        diversity_weight: float = 0.5,
        self_loop_weight: float = 1.0,
        balance_weight: float = 0.1
    ):
        super().__init__()

        self.required_nodes_weight = required_nodes_weight
        self.connectivity_weight = connectivity_weight
        self.diversity_weight = diversity_weight
        self.self_loop_weight = self_loop_weight
        self.balance_weight = balance_weight

        # Node type indices
        self.NODE_GND = 0
        self.NODE_VIN = 1
        self.NODE_VOUT = 2
        self.NODE_INTERNAL = 3
        self.NODE_MASK = 4

    def forward(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute structural validity loss.

        Args:
            predictions: Model predictions containing:
                - 'node_types': Node type logits [batch, max_nodes, num_node_types]
                - 'edge_existence': Edge existence logits [batch, max_nodes, max_nodes]

        Returns:
            loss: Total structural validity loss
        """
        batch_size = predictions['node_types'].shape[0]
        device = predictions['node_types'].device

        total_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # 1. Required Nodes Loss
        # ==================================================================

        # Get node type probabilities
        node_type_probs = F.softmax(predictions['node_types'], dim=-1)  # [batch, max_nodes, num_node_types]

        # Maximum probability for each node type across all nodes
        max_type_probs = node_type_probs.max(dim=1)[0]  # [batch, num_node_types]

        # Encourage high probability for required nodes
        required_node_types = [self.NODE_GND, self.NODE_VIN, self.NODE_VOUT]

        for node_type in required_node_types:
            # Loss = -log(max_prob) encourages at least one node to have this type
            prob = max_type_probs[:, node_type]
            loss = -torch.log(prob + 1e-8).mean()
            total_loss = total_loss + self.required_nodes_weight * loss

        # ==================================================================
        # 2. Connectivity Loss (Approximate)
        # ==================================================================

        # Encourage sufficient edges for connectivity
        edge_probs = torch.sigmoid(predictions['edge_existence'])  # [batch, max_nodes, max_nodes]

        # Make symmetric (undirected graph)
        edge_probs_sym = (edge_probs + edge_probs.transpose(1, 2)) / 2

        # Number of edges per graph
        num_edges = edge_probs_sym.sum(dim=[1, 2])  # [batch]

        # For n nodes, need at least n-1 edges for connectivity
        # Approximate n as number of non-MASK nodes
        non_mask_probs = 1.0 - node_type_probs[..., self.NODE_MASK]  # [batch, max_nodes]
        approx_num_nodes = non_mask_probs.sum(dim=1)  # [batch]

        min_edges = torch.clamp(approx_num_nodes - 1, min=2)  # At least 2 edges

        # Penalize if not enough edges
        connectivity_loss = F.relu(min_edges - num_edges).mean()
        total_loss = total_loss + self.connectivity_weight * connectivity_loss

        # ==================================================================
        # 3. Node Diversity Loss
        # ==================================================================

        # Encourage diversity in node types (entropy)
        # Average type distribution across nodes
        avg_type_dist = node_type_probs.mean(dim=1)  # [batch, num_node_types]

        # Entropy: H = -sum(p * log(p))
        entropy = -(avg_type_dist * torch.log(avg_type_dist + 1e-8)).sum(dim=1)  # [batch]

        # Maximize entropy (encourage diversity) -> minimize negative entropy
        diversity_loss = -entropy.mean()
        total_loss = total_loss + self.diversity_weight * diversity_loss

        # ==================================================================
        # 4. Self-Loop Penalty
        # ==================================================================

        # Penalize self-loops (diagonal edges)
        edge_logits = predictions['edge_existence']
        diagonal_logits = edge_logits.diagonal(dim1=1, dim2=2)  # [batch, max_nodes]

        # Encourage diagonal to be 0 (no self-loops)
        self_loop_loss = F.binary_cross_entropy_with_logits(
            diagonal_logits,
            torch.zeros_like(diagonal_logits),
            reduction='mean'
        )
        total_loss = total_loss + self.self_loop_weight * self_loop_loss

        # ==================================================================
        # 5. Edge Balance Loss
        # ==================================================================

        # Encourage balanced edge distribution (not all edges or no edges)
        # Penalize if edge probability mean is too close to 0 or 1
        mean_edge_prob = edge_probs.mean(dim=[1, 2])  # [batch]

        # Optimal is around 0.3-0.5 (sparse but connected graphs)
        target_density = 0.4
        balance_loss = (mean_edge_prob - target_density).pow(2).mean()
        total_loss = total_loss + self.balance_weight * balance_loss

        return total_loss


class PhysicalConstraintLoss(nn.Module):
    """
    Physical constraints for circuit validity.

    Enforces:
    1. Positive component values (R, L, C > 0)
    2. Stable poles (Re(pole) < 0)
    3. Reasonable pole/zero magnitudes

    Args:
        positive_value_weight: Weight for positive component value loss
        pole_stability_weight: Weight for pole stability loss
        magnitude_weight: Weight for reasonable magnitude loss

    Example:
        >>> loss_fn = PhysicalConstraintLoss()
        >>> loss = loss_fn(predictions)
    """

    def __init__(
        self,
        positive_value_weight: float = 1.0,
        pole_stability_weight: float = 1.0,
        magnitude_weight: float = 0.1
    ):
        super().__init__()

        self.positive_value_weight = positive_value_weight
        self.pole_stability_weight = pole_stability_weight
        self.magnitude_weight = magnitude_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physical constraint loss.

        Args:
            predictions: Model predictions

        Returns:
            loss: Total physical constraint loss
        """
        device = predictions['edge_values'].device
        total_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # 1. Positive Component Values
        # ==================================================================

        if 'edge_values' in predictions:
            # Edge values: C, G, L_inv (first 3 channels)
            component_values = predictions['edge_values'][..., :3]  # [batch, max_nodes, max_nodes, 3]

            # Penalize negative values using ReLU
            negative_loss = F.relu(-component_values).mean()
            total_loss = total_loss + self.positive_value_weight * negative_loss

        # ==================================================================
        # 2. Pole Stability (Negative Real Parts)
        # ==================================================================

        if 'pole_values' in predictions:
            pole_reals = predictions['pole_values'][..., 0]  # [batch, max_poles]

            # Penalize positive real parts (unstable poles)
            unstable_loss = F.relu(pole_reals + 0.1).mean()  # +0.1 to encourage more negative
            total_loss = total_loss + self.pole_stability_weight * unstable_loss

        # ==================================================================
        # 3. Reasonable Magnitudes
        # ==================================================================

        if 'pole_values' in predictions:
            # Pole magnitudes should be reasonable (not too large)
            pole_magnitudes = torch.sqrt(
                predictions['pole_values'][..., 0] ** 2 +
                predictions['pole_values'][..., 1] ** 2
            )

            # Penalize very large magnitudes (log scale)
            log_mag = torch.log(pole_magnitudes.abs() + 1e-8)
            large_mag_loss = F.relu(log_mag - 10).mean()  # Penalize |pole| > e^10
            total_loss = total_loss + self.magnitude_weight * large_mag_loss

        if 'zero_values' in predictions:
            # Same for zeros
            zero_magnitudes = torch.sqrt(
                predictions['zero_values'][..., 0] ** 2 +
                predictions['zero_values'][..., 1] ** 2
            )

            log_mag = torch.log(zero_magnitudes.abs() + 1e-8)
            large_mag_loss = F.relu(log_mag - 10).mean()
            total_loss = total_loss + self.magnitude_weight * large_mag_loss

        return total_loss


class CombinedStructuralLoss(nn.Module):
    """
    Combined structural and physical constraint loss.

    Combines StructuralValidityLoss and PhysicalConstraintLoss.

    Args:
        structural_weight: Weight for structural validity loss
        physical_weight: Weight for physical constraint loss

    Example:
        >>> loss_fn = CombinedStructuralLoss()
        >>> loss = loss_fn(predictions)
    """

    def __init__(
        self,
        structural_weight: float = 1.0,
        physical_weight: float = 1.0
    ):
        super().__init__()

        self.structural_weight = structural_weight
        self.physical_weight = physical_weight

        self.structural_loss = StructuralValidityLoss()
        self.physical_loss = PhysicalConstraintLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions

        Returns:
            loss: Combined structural and physical loss
        """
        struct_loss = self.structural_loss(predictions)
        phys_loss = self.physical_loss(predictions)

        total_loss = (
            self.structural_weight * struct_loss +
            self.physical_weight * phys_loss
        )

        return total_loss
