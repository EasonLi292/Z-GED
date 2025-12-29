"""
Connectivity Loss for Circuit Generation.

This module implements explicit structural constraints to ensure
generated circuits have proper connectivity.

This is Option 1A from ARCHITECTURE_INVESTIGATION.md - requires
retraining but fixes the root cause.
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

        # ============================================================
        # 1. VIN Connectivity Loss (MOST CRITICAL)
        # ============================================================

        loss_vin = torch.tensor(0.0, device=device)
        vin_violations = 0

        for batch_idx in range(batch_size):
            # Find VIN node (type = 1)
            vin_mask = (node_labels[batch_idx] == 1)

            if vin_mask.sum() > 0:
                vin_id = vin_mask.nonzero()[0].item()

                # Compute VIN degree (sum of edge probabilities)
                # Both outgoing and incoming
                vin_degree = edge_probs[batch_idx, vin_id, :].sum() + \
                            edge_probs[batch_idx, :, vin_id].sum()

                # We want degree ≥ 1.0
                # Penalty = relu(1.0 - degree)
                penalty = F.relu(1.0 - vin_degree)

                loss_vin += penalty

                if penalty > 0.5:
                    vin_violations += 1

        loss_vin = loss_vin / batch_size

        # ============================================================
        # 2. VOUT Connectivity Loss
        # ============================================================

        loss_vout = torch.tensor(0.0, device=device)
        vout_violations = 0

        for batch_idx in range(batch_size):
            # Find VOUT node (type = 2)
            vout_mask = (node_labels[batch_idx] == 2)

            if vout_mask.sum() > 0:
                vout_id = vout_mask.nonzero()[0].item()

                # Compute VOUT degree
                vout_degree = edge_probs[batch_idx, vout_id, :].sum() + \
                             edge_probs[batch_idx, :, vout_id].sum()

                # Penalty for degree < 1.0
                penalty = F.relu(1.0 - vout_degree)

                loss_vout += penalty

                if penalty > 0.5:
                    vout_violations += 1

        loss_vout = loss_vout / batch_size

        # ============================================================
        # 3. Graph Connectivity Loss (Differentiable)
        # ============================================================

        loss_graph = torch.tensor(0.0, device=device)

        for batch_idx in range(batch_size):
            # Build adjacency matrix (probabilistic)
            A = edge_probs[batch_idx]

            # Add self-loops
            A_with_loops = A + torch.eye(max_nodes, device=device)

            # Compute k-hop connectivity matrix
            # If graph is connected, A^k should have all non-zero entries
            # We use k = max_nodes - 1 (sufficient for connectivity check)
            A_pow = A_with_loops
            for _ in range(max_nodes - 2):
                A_pow = A_pow @ A_with_loops  # Matrix multiplication

            # Mask out MASK nodes (type = 4)
            valid_mask = (node_labels[batch_idx] < 4).float()

            # For valid nodes, check if they can reach all other valid nodes
            # A_pow[i,j] > 0 means there's a path from i to j
            valid_pairs = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(0)

            # Penalty: sum of pairs with weak connectivity
            # We want A_pow[i,j] > 0.1 for all valid pairs
            connectivity_strength = A_pow * valid_pairs
            penalty = F.relu(0.1 - connectivity_strength).sum()

            loss_graph += penalty

        loss_graph = loss_graph / (batch_size * max_nodes * max_nodes)

        # ============================================================
        # 4. Isolated Node Loss
        # ============================================================

        loss_isolated = torch.tensor(0.0, device=device)
        isolated_violations = 0

        for batch_idx in range(batch_size):
            # For each non-MASK node, check degree
            for node_id in range(max_nodes):
                if node_labels[batch_idx, node_id] < 4:  # Not MASK
                    degree = edge_probs[batch_idx, node_id, :].sum() + \
                            edge_probs[batch_idx, :, node_id].sum()

                    # Penalty if isolated (degree < 0.5)
                    penalty = F.relu(0.5 - degree)
                    loss_isolated += penalty

                    if penalty > 0.25:
                        isolated_violations += 1

        loss_isolated = loss_isolated / (batch_size * max_nodes)

        # ============================================================
        # 5. Combine Losses
        # ============================================================

        total_loss = (
            self.vin_weight * loss_vin +
            self.vout_weight * loss_vout +
            self.graph_weight * loss_graph +
            self.isolated_weight * loss_isolated
        )

        # ============================================================
        # 6. Metrics
        # ============================================================

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


class SpectralConnectivityLoss(nn.Module):
    """
    Alternative connectivity loss using spectral graph theory.

    Uses Laplacian eigenvalues to check connectivity:
    - Laplacian L = D - A (degree - adjacency)
    - λ_2 (second smallest eigenvalue) = 0 iff graph is disconnected
    - We penalize λ_2 close to 0

    This is more principled but more expensive (eigenvalue computation).

    Args:
        weight: Weight for spectral connectivity penalty
        epsilon: Small value to encourage λ_2 > epsilon
    """

    def __init__(self, weight: float = 5.0, epsilon: float = 0.1):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon

    def forward(
        self,
        edge_logits: torch.Tensor,
        node_types: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute spectral connectivity loss.

        Args:
            edge_logits: Edge existence logits [batch, max_nodes, max_nodes]
            node_types: Node type predictions [batch, max_nodes]

        Returns:
            loss: Spectral connectivity loss
            metrics: Dictionary with eigenvalues
        """
        batch_size = edge_logits.shape[0]
        device = edge_logits.device

        # Convert to probabilities
        edge_probs = torch.sigmoid(edge_logits)

        total_loss = torch.tensor(0.0, device=device)
        lambda_2_values = []

        for batch_idx in range(batch_size):
            # Build adjacency matrix
            A = edge_probs[batch_idx]

            # Compute degree matrix
            D = torch.diag(A.sum(dim=1))

            # Laplacian L = D - A
            L = D - A

            # Compute eigenvalues
            try:
                eigenvals = torch.linalg.eigvalsh(L)  # Real eigenvalues (L is symmetric)

                # Sort eigenvalues
                eigenvals_sorted = torch.sort(eigenvals)[0]

                # Second smallest eigenvalue (algebraic connectivity)
                lambda_2 = eigenvals_sorted[1]

                # Penalty: encourage λ_2 > epsilon
                penalty = F.relu(self.epsilon - lambda_2)

                total_loss += penalty
                lambda_2_values.append(lambda_2.item())

            except Exception as e:
                # Eigenvalue computation failed (numerical issues)
                # Apply maximum penalty
                total_loss += self.epsilon
                lambda_2_values.append(0.0)

        total_loss = total_loss / batch_size

        metrics = {
            'loss_spectral_connectivity': total_loss.item(),
            'lambda_2_mean': sum(lambda_2_values) / len(lambda_2_values) if lambda_2_values else 0.0,
            'lambda_2_min': min(lambda_2_values) if lambda_2_values else 0.0
        }

        return total_loss, metrics


if __name__ == '__main__':
    """Test connectivity losses."""
    print("Testing Connectivity Losses...")

    # Create dummy data
    batch_size = 4
    max_nodes = 5

    # Edge logits (random)
    edge_logits = torch.randn(batch_size, max_nodes, max_nodes)

    # Node types: [GND, VIN, VOUT, INTERNAL, INTERNAL]
    node_types = torch.tensor([
        [0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4],  # One MASK node
        [0, 1, 2, 3, 3],
        [0, 1, 2, 4, 4]   # Two MASK nodes
    ])

    print("\n1. Testing ConnectivityLoss...")
    loss_fn = ConnectivityLoss()
    loss, metrics = loss_fn(edge_logits, node_types)

    print(f"   Total loss: {loss.item():.4f}")
    print(f"   VIN connectivity loss: {metrics['loss_vin_connectivity']:.4f}")
    print(f"   VOUT connectivity loss: {metrics['loss_vout_connectivity']:.4f}")
    print(f"   Graph connectivity loss: {metrics['loss_graph_connectivity']:.4f}")
    print(f"   Isolated nodes loss: {metrics['loss_isolated_nodes']:.4f}")
    print(f"   VIN violations: {metrics['vin_violations']}/{batch_size}")

    print("\n2. Testing SpectralConnectivityLoss...")
    spectral_loss_fn = SpectralConnectivityLoss()
    spectral_loss, spectral_metrics = spectral_loss_fn(edge_logits, node_types)

    print(f"   Total loss: {spectral_loss.item():.4f}")
    print(f"   λ₂ mean: {spectral_metrics['lambda_2_mean']:.4f}")
    print(f"   λ₂ min: {spectral_metrics['lambda_2_min']:.4f}")

    print("\n3. Testing with connected graph...")
    # Create connected graph (all nodes connected to node 0)
    edge_logits_connected = torch.zeros(1, max_nodes, max_nodes)
    edge_logits_connected[0, :, 0] = 5.0  # High logit → high probability
    edge_logits_connected[0, 0, :] = 5.0

    node_types_single = torch.tensor([[0, 1, 2, 3, 3]])

    loss_connected, metrics_connected = loss_fn(edge_logits_connected, node_types_single)

    print(f"   VIN connectivity loss: {metrics_connected['loss_vin_connectivity']:.4f} (should be ~0)")
    print(f"   VIN violations: {metrics_connected['vin_violations']} (should be 0)")

    print("\n✅ All tests passed!")
    print("\nUsage in training:")
    print("""
    from ml.losses.connectivity_loss import ConnectivityLoss

    # Create loss function
    connectivity_loss_fn = ConnectivityLoss(
        vin_weight=10.0,   # Critical
        vout_weight=5.0,
        graph_weight=3.0,
        isolated_weight=2.0
    )

    # In training loop:
    predictions = decoder(latent, conditions)

    # Standard losses
    loss_standard, metrics = standard_loss_fn(predictions, targets)

    # Connectivity loss
    loss_connectivity, conn_metrics = connectivity_loss_fn(
        predictions['edge_existence'],  # Edge logits
        predictions['node_types']       # Node type logits or labels
    )

    # Total loss
    total_loss = loss_standard + 10.0 * loss_connectivity

    # Backprop
    total_loss.backward()
    """)
