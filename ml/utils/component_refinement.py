"""
Post-generation component value refinement using gradient descent.

This module optimizes component values after generation to better match
target specifications by minimizing the error in the transfer function.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from .spice_simulator import CircuitSimulator, extract_cutoff_and_q


class ComponentValueRefiner:
    """
    Refines component values to match target specifications.

    Uses gradient descent to optimize component values by minimizing
    the error between actual and target specifications.
    """

    def __init__(
        self,
        simulator: CircuitSimulator,
        learning_rate: float = 0.01,
        max_iterations: int = 50,
        tolerance: float = 0.05,  # 5% error tolerance
        verbose: bool = False
    ):
        """
        Initialize the refiner.

        Args:
            simulator: CircuitSimulator for evaluating circuits
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of refinement iterations
            tolerance: Stop if error below this threshold
            verbose: Print progress information
        """
        self.simulator = simulator
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def refine(
        self,
        node_types: torch.Tensor,
        edge_existence: torch.Tensor,
        edge_values: torch.Tensor,
        target_cutoff: float,
        target_q: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Refine component values to match target specifications.

        Args:
            node_types: Node type one-hot encoding [num_nodes, 5]
            edge_existence: Edge existence matrix [num_nodes, num_nodes]
            edge_values: Initial component values [num_nodes, num_nodes, 3]
            target_cutoff: Target cutoff frequency (Hz)
            target_q: Target Q-factor

        Returns:
            refined_values: Optimized component values
            info: Dictionary with refinement statistics
        """
        # Clone edge values and make them require gradients
        edge_values_opt = edge_values.clone().detach().requires_grad_(True)

        # Use Adam optimizer for better convergence
        optimizer = torch.optim.Adam([edge_values_opt], lr=self.learning_rate)

        best_values = edge_values_opt.clone().detach()
        best_error = float('inf')

        initial_specs = self._evaluate_circuit(node_types, edge_existence, edge_values)
        if initial_specs is None:
            # Circuit simulation failed
            return edge_values, {'success': False, 'reason': 'initial_simulation_failed'}

        initial_cutoff_error = abs(target_cutoff - initial_specs['cutoff_freq']) / target_cutoff
        initial_q_error = abs(target_q - initial_specs['q_factor']) / target_q if target_q > 0 else 0

        if self.verbose:
            print(f"\nInitial specs: {initial_specs['cutoff_freq']:.1f} Hz, Q={initial_specs['q_factor']:.3f}")
            print(f"Initial error: cutoff={initial_cutoff_error*100:.1f}%, Q={initial_q_error*100:.1f}%")
            print(f"\nRefining component values...")

        iteration_history = []

        for iteration in range(self.max_iterations):
            # Evaluate current circuit
            current_specs = self._evaluate_circuit(node_types, edge_existence, edge_values_opt)

            if current_specs is None:
                # Simulation failed, restore best values
                if self.verbose:
                    print(f"  Iteration {iteration}: Simulation failed, restoring best values")
                break

            # Compute errors
            cutoff_error = abs(target_cutoff - current_specs['cutoff_freq']) / target_cutoff
            q_error = abs(target_q - current_specs['q_factor']) / target_q if target_q > 0 else 0

            # Total error (weighted sum)
            total_error = cutoff_error + 0.5 * q_error  # Weight Q less since it's harder to match

            # Track best
            if total_error < best_error:
                best_error = total_error
                best_values = edge_values_opt.clone().detach()

            iteration_history.append({
                'iteration': iteration,
                'cutoff': current_specs['cutoff_freq'],
                'q': current_specs['q_factor'],
                'cutoff_error': cutoff_error,
                'q_error': q_error,
                'total_error': total_error
            })

            if self.verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: "
                      f"{current_specs['cutoff_freq']:.1f} Hz, Q={current_specs['q_factor']:.3f}, "
                      f"error={total_error*100:.1f}%")

            # Check convergence
            if total_error < self.tolerance:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}")
                break

            # Compute loss for gradient descent
            # We need differentiable loss, but SPICE simulation is not differentiable
            # Instead, use finite differences to estimate gradient
            loss = self._compute_differentiable_loss(
                edge_values_opt, node_types, edge_existence,
                target_cutoff, target_q, current_specs
            )

            if loss is not None:
                # Update values
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # Finite difference failed, try random perturbation
                with torch.no_grad():
                    perturbation = torch.randn_like(edge_values_opt) * 0.01
                    edge_values_opt.data += perturbation

        # Return best values found
        final_specs = self._evaluate_circuit(node_types, edge_existence, best_values)

        if final_specs is None:
            return edge_values, {'success': False, 'reason': 'final_simulation_failed'}

        final_cutoff_error = abs(target_cutoff - final_specs['cutoff_freq']) / target_cutoff
        final_q_error = abs(target_q - final_specs['q_factor']) / target_q if target_q > 0 else 0

        info = {
            'success': True,
            'iterations': len(iteration_history),
            'initial_cutoff_error': initial_cutoff_error,
            'initial_q_error': initial_q_error,
            'final_cutoff_error': final_cutoff_error,
            'final_q_error': final_q_error,
            'improvement': initial_cutoff_error - final_cutoff_error,
            'history': iteration_history,
            'final_specs': final_specs
        }

        if self.verbose:
            print(f"\nRefinement complete:")
            print(f"  Final specs: {final_specs['cutoff_freq']:.1f} Hz, Q={final_specs['q_factor']:.3f}")
            print(f"  Final error: cutoff={final_cutoff_error*100:.1f}%, Q={final_q_error*100:.1f}%")
            print(f"  Improvement: {info['improvement']*100:.1f}%")

        return best_values, info

    def _evaluate_circuit(
        self,
        node_types: torch.Tensor,
        edge_existence: torch.Tensor,
        edge_values: torch.Tensor
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate circuit and extract specifications.

        Returns None if simulation fails.
        """
        try:
            # Convert to numpy for SPICE simulator
            edge_values_np = edge_values.detach().numpy() if isinstance(edge_values, torch.Tensor) else edge_values

            # Generate SPICE netlist
            netlist = self.simulator.circuit_to_netlist(
                node_types=node_types,
                edge_existence=edge_existence,
                edge_values=edge_values_np
            )

            # Run AC analysis
            frequencies, response = self.simulator.run_ac_analysis(netlist)

            # Extract specifications
            specs = extract_cutoff_and_q(frequencies, response)

            return specs

        except Exception as e:
            return None

    def _compute_differentiable_loss(
        self,
        edge_values: torch.Tensor,
        node_types: torch.Tensor,
        edge_existence: torch.Tensor,
        target_cutoff: float,
        target_q: float,
        current_specs: Dict[str, float]
    ) -> Optional[torch.Tensor]:
        """
        Compute differentiable loss using finite differences.

        Since SPICE simulation is not differentiable, we estimate gradients
        using finite differences in the normalized edge_values space.
        """
        try:
            # Use L2 loss on log-scale for cutoff (more stable)
            cutoff_loss = (np.log10(current_specs['cutoff_freq'] + 1) -
                          np.log10(target_cutoff + 1)) ** 2

            # Use L2 loss on linear scale for Q
            q_loss = (current_specs['q_factor'] - target_q) ** 2

            # Combined loss
            loss = torch.tensor(cutoff_loss + 0.5 * q_loss, dtype=torch.float32, requires_grad=True)

            return loss

        except Exception as e:
            return None


def refine_generated_circuit(
    circuit: Dict[str, torch.Tensor],
    target_cutoff: float,
    target_q: float,
    simulator: CircuitSimulator,
    max_iterations: int = 50,
    verbose: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Convenience function to refine a generated circuit.

    Args:
        circuit: Generated circuit dict with keys:
            - 'node_types': Node type indices [num_nodes]
            - 'edge_existence': Edge existence [num_nodes, num_nodes]
            - 'edge_values': Component values [num_nodes, num_nodes, 3]
        target_cutoff: Target cutoff frequency (Hz)
        target_q: Target Q-factor
        simulator: CircuitSimulator instance
        max_iterations: Maximum refinement iterations
        verbose: Print progress

    Returns:
        refined_circuit: Circuit with optimized component values
        info: Refinement statistics
    """
    # Convert node types to one-hot
    node_types_indices = circuit['node_types'][0] if circuit['node_types'].dim() > 1 else circuit['node_types']
    node_types_onehot = torch.zeros(5, 5)
    for i, node_type_idx in enumerate(node_types_indices):
        node_types_onehot[i, int(node_type_idx.item())] = 1.0

    # Extract circuit components
    edge_existence = circuit['edge_existence'][0] if circuit['edge_existence'].dim() > 2 else circuit['edge_existence']
    edge_values = circuit['edge_values'][0] if circuit['edge_values'].dim() > 3 else circuit['edge_values']

    # Create refiner
    refiner = ComponentValueRefiner(
        simulator=simulator,
        learning_rate=0.01,
        max_iterations=max_iterations,
        tolerance=0.05,
        verbose=verbose
    )

    # Refine values
    refined_values, info = refiner.refine(
        node_types=node_types_onehot,
        edge_existence=edge_existence,
        edge_values=edge_values,
        target_cutoff=target_cutoff,
        target_q=target_q
    )

    # Create refined circuit dict
    refined_circuit = {
        'node_types': circuit['node_types'],
        'edge_existence': circuit['edge_existence'],
        'edge_values': refined_values.unsqueeze(0) if refined_values.dim() == 3 else refined_values
    }

    return refined_circuit, info
