"""
Circuit validity constraints for diffusion model.

Implements constraint checking and enforcement for generated circuits.
Ensures circuits are electrically valid and structurally sound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import networkx as nx
import numpy as np


class CircuitConstraints:
    """
    Circuit validity constraints and enforcement.

    Checks and enforces:
    1. Node requirements: Must have GND, VIN, VOUT
    2. Connectivity: Circuit must be a connected graph
    3. Node diversity: At least 3 nodes (GND, VIN, VOUT minimum)
    4. Edge validity: Edges should connect different nodes
    5. Component values: R, L, C must be positive
    6. Pole/zero stability: Poles should have negative real parts (for stability)

    Example:
        >>> constraints = CircuitConstraints()
        >>> is_valid, violations = constraints.check_validity(circuit)
        >>> if not is_valid:
        ...     circuit = constraints.enforce_constraints(circuit)
    """

    def __init__(self):
        # Node type indices
        self.NODE_GND = 0
        self.NODE_VIN = 1
        self.NODE_VOUT = 2
        self.NODE_INTERNAL = 3
        self.NODE_MASK = 4

    def check_validity(
        self,
        circuit: Dict[str, torch.Tensor],
        verbose: bool = False
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if circuit satisfies all validity constraints.

        Args:
            circuit: Circuit dictionary containing:
                - 'node_types': Node type indices [batch, max_nodes]
                - 'edge_existence': Binary edge matrix [batch, max_nodes, max_nodes]
                - 'edge_values': Edge values [batch, max_nodes, max_nodes, 7]
                - 'pole_count': Number of poles [batch]
                - 'zero_count': Number of zeros [batch]
                - 'pole_values': Pole values [batch, max_poles, 2]
                - 'zero_values': Zero values [batch, max_zeros, 2]
            verbose: Print violation details

        Returns:
            is_valid: Boolean indicating overall validity
            violations: Dictionary of constraint violations per check
        """
        batch_size = circuit['node_types'].shape[0]

        violations = {
            'has_gnd': torch.zeros(batch_size, dtype=torch.bool),
            'has_vin': torch.zeros(batch_size, dtype=torch.bool),
            'has_vout': torch.zeros(batch_size, dtype=torch.bool),
            'is_connected': torch.zeros(batch_size, dtype=torch.bool),
            'has_diversity': torch.zeros(batch_size, dtype=torch.bool),
            'valid_edges': torch.zeros(batch_size, dtype=torch.bool),
            'positive_values': torch.zeros(batch_size, dtype=torch.bool),
            'stable_poles': torch.zeros(batch_size, dtype=torch.bool)
        }

        for b in range(batch_size):
            # Extract circuit for this batch
            node_types_b = circuit['node_types'][b]  # [max_nodes]
            edge_exist_b = circuit['edge_existence'][b]  # [max_nodes, max_nodes]
            edge_values_b = circuit['edge_values'][b]  # [max_nodes, max_nodes, 7]
            pole_values_b = circuit['pole_values'][b]  # [max_poles, 2]
            pole_count_b = circuit['pole_count'][b].item()

            # ==================================================================
            # 1. Check required nodes (GND, VIN, VOUT)
            # ==================================================================

            has_gnd = (node_types_b == self.NODE_GND).any()
            has_vin = (node_types_b == self.NODE_VIN).any()
            has_vout = (node_types_b == self.NODE_VOUT).any()

            violations['has_gnd'][b] = has_gnd
            violations['has_vin'][b] = has_vin
            violations['has_vout'][b] = has_vout

            # ==================================================================
            # 2. Check connectivity
            # ==================================================================

            # Only consider non-MASK nodes
            valid_nodes = (node_types_b != self.NODE_MASK).nonzero(as_tuple=True)[0]
            num_valid_nodes = len(valid_nodes)

            if num_valid_nodes >= 2:
                # Build adjacency matrix
                adj_matrix = edge_exist_b[valid_nodes][:, valid_nodes]

                # Check if graph is connected using DFS
                is_connected = self._is_connected(adj_matrix.cpu().numpy())
                violations['is_connected'][b] = is_connected
            else:
                violations['is_connected'][b] = False

            # ==================================================================
            # 3. Check node diversity (at least 3 nodes)
            # ==================================================================

            violations['has_diversity'][b] = (num_valid_nodes >= 3)

            # ==================================================================
            # 4. Check edge validity (no self-loops unless needed)
            # ==================================================================

            # Edges should generally not be self-loops
            self_loops = edge_exist_b.diagonal().sum()
            violations['valid_edges'][b] = (self_loops == 0)

            # ==================================================================
            # 5. Check component values are positive
            # ==================================================================

            # Edge values: C, G, L_inv (first 3 channels)
            # They should be positive (or zero if masked)
            component_values = edge_values_b[..., :3]  # [max_nodes, max_nodes, 3]
            existing_edges = edge_exist_b.unsqueeze(-1).expand_as(component_values)

            # Check positive values where edges exist
            positive_where_exists = (component_values[existing_edges > 0] > 0).all()
            violations['positive_values'][b] = positive_where_exists

            # ==================================================================
            # 6. Check pole stability (negative real parts)
            # ==================================================================

            if pole_count_b > 0:
                pole_reals = pole_values_b[:pole_count_b, 0]  # Real parts
                stable = (pole_reals < 0).all()  # Should be negative for stability
                violations['stable_poles'][b] = stable
            else:
                violations['stable_poles'][b] = True  # No poles = trivially stable

        # Overall validity: all constraints satisfied
        all_valid = torch.stack([v.all() for v in violations.values()]).all()

        if verbose:
            print("\n" + "=" * 70)
            print("Circuit Validity Check")
            print("=" * 70)
            for constraint, valid in violations.items():
                pass_rate = valid.float().mean().item() * 100
                print(f"{constraint:20s}: {pass_rate:.1f}% pass")
            print(f"\nOverall validity: {all_valid.item()}")

        return all_valid.item(), violations

    def _is_connected(self, adj_matrix: np.ndarray) -> bool:
        """
        Check if graph is connected using NetworkX.

        Args:
            adj_matrix: Adjacency matrix [n, n]

        Returns:
            is_connected: Boolean indicating connectivity
        """
        n = adj_matrix.shape[0]

        if n == 0:
            return False

        # Create undirected graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Add edges
        edges = np.argwhere(adj_matrix > 0)
        for i, j in edges:
            if i != j:  # Ignore self-loops
                G.add_edge(i, j)

        # Check connectivity
        return nx.is_connected(G)

    def enforce_constraints(
        self,
        circuit: Dict[str, torch.Tensor],
        mode: str = 'soft'
    ) -> Dict[str, torch.Tensor]:
        """
        Enforce circuit validity constraints.

        Args:
            circuit: Circuit dictionary
            mode: 'soft' (probabilistic fixes) or 'hard' (deterministic fixes)

        Returns:
            constrained_circuit: Circuit with constraints enforced
        """
        constrained = circuit.copy()

        batch_size = circuit['node_types'].shape[0]
        max_nodes = circuit['node_types'].shape[1]

        for b in range(batch_size):
            node_types_b = constrained['node_types'][b].clone()  # [max_nodes]

            # ==================================================================
            # 1. Ensure required nodes exist
            # ==================================================================

            # Check if GND exists
            if not (node_types_b == self.NODE_GND).any():
                # Set first node to GND
                node_types_b[0] = self.NODE_GND

            # Check if VIN exists
            if not (node_types_b == self.NODE_VIN).any():
                # Set second node to VIN
                node_types_b[1] = self.NODE_VIN

            # Check if VOUT exists
            if not (node_types_b == self.NODE_VOUT).any():
                # Set third node to VOUT
                node_types_b[2] = self.NODE_VOUT

            constrained['node_types'][b] = node_types_b

            # ==================================================================
            # 2. Remove self-loops
            # ==================================================================

            edge_exist_b = constrained['edge_existence'][b].clone()
            edge_exist_b.fill_diagonal_(0)  # Remove self-loops
            constrained['edge_existence'][b] = edge_exist_b

            # ==================================================================
            # 3. Ensure positive component values
            # ==================================================================

            edge_values_b = constrained['edge_values'][b].clone()
            # Clamp component values to be positive
            edge_values_b[..., :3] = torch.clamp(edge_values_b[..., :3], min=1e-6)
            constrained['edge_values'][b] = edge_values_b

            # ==================================================================
            # 4. Ensure stable poles (negative real parts)
            # ==================================================================

            pole_values_b = constrained['pole_values'][b].clone()
            pole_count_b = circuit['pole_count'][b].item()

            if pole_count_b > 0:
                # Clamp real parts to be negative
                pole_values_b[:pole_count_b, 0] = torch.clamp(
                    pole_values_b[:pole_count_b, 0],
                    max=-1e-6  # Slightly negative
                )

            constrained['pole_values'][b] = pole_values_b

        return constrained

    def compute_constraint_loss(
        self,
        circuit: Dict[str, torch.Tensor],
        weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute differentiable loss for constraint violations.

        This can be used during training to encourage valid circuits.

        Args:
            circuit: Circuit dictionary
            weight: Weight for constraint loss

        Returns:
            loss: Constraint violation loss
        """
        batch_size = circuit['node_types'].shape[0]
        device = circuit['node_types'].device

        total_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # 1. Loss for missing required nodes (via entropy on node types)
        # ==================================================================

        # We want high probability for GND, VIN, VOUT somewhere in the graph
        # This is approximated by encouraging diversity in node types

        node_type_probs = F.softmax(circuit.get('node_type_logits', circuit['node_types']), dim=-1)
        # Average probability of each node type across the graph
        avg_type_probs = node_type_probs.mean(dim=1)  # [batch, num_node_types]

        # We want non-zero probability for GND, VIN, VOUT
        required_types = [self.NODE_GND, self.NODE_VIN, self.NODE_VOUT]
        for node_type in required_types:
            # Penalize if average probability is too low
            prob = avg_type_probs[:, node_type]
            loss = -torch.log(prob + 1e-6).mean()
            total_loss = total_loss + loss

        # ==================================================================
        # 2. Loss for self-loops
        # ==================================================================

        if 'edge_existence' in circuit:
            edge_logits = circuit.get('edge_existence_logits', circuit['edge_existence'])
            # Penalize diagonal edges
            diagonal_logits = edge_logits.diagonal(dim1=1, dim2=2)  # [batch, max_nodes]
            self_loop_loss = F.binary_cross_entropy_with_logits(
                diagonal_logits,
                torch.zeros_like(diagonal_logits),
                reduction='mean'
            )
            total_loss = total_loss + self_loop_loss

        # ==================================================================
        # 3. Loss for negative component values
        # ==================================================================

        if 'edge_values' in circuit:
            component_values = circuit['edge_values'][..., :3]  # C, G, L_inv
            # ReLU loss: penalize negative values
            negative_loss = F.relu(-component_values).mean()
            total_loss = total_loss + negative_loss

        # ==================================================================
        # 4. Loss for unstable poles
        # ==================================================================

        if 'pole_values' in circuit:
            pole_reals = circuit['pole_values'][..., 0]  # Real parts
            # Penalize positive real parts
            unstable_loss = F.relu(pole_reals).mean()
            total_loss = total_loss + unstable_loss

        return weight * total_loss


def post_process_circuit(
    circuit: Dict[str, torch.Tensor],
    enforce_constraints: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Post-process generated circuit for validity.

    Args:
        circuit: Generated circuit
        enforce_constraints: Whether to enforce hard constraints

    Returns:
        processed_circuit: Post-processed circuit
    """
    if enforce_constraints:
        constraints = CircuitConstraints()
        circuit = constraints.enforce_constraints(circuit, mode='hard')

    return circuit
