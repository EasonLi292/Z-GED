"""
Forward diffusion process for circuit graph generation.

Implements noise injection for training the diffusion model.
Handles both discrete (node types, counts) and continuous (values) variables.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .noise_schedules import (
    get_cosine_schedule,
    get_discrete_transition_matrix,
    get_cumulative_transition_matrix,
    add_noise_continuous,
    add_noise_discrete
)


class CircuitForwardDiffusion:
    """
    Forward diffusion process for circuit graphs.

    Handles noise injection for all circuit components:
    - Node types (discrete categorical)
    - Edge existence (discrete binary)
    - Edge values (continuous: C, G, L_inv)
    - Pole/zero counts (discrete categorical)
    - Pole/zero values (continuous: real, imag)

    Args:
        timesteps: Number of diffusion timesteps (default: 1000)
        max_nodes: Maximum number of nodes (default: 5)
        max_poles: Maximum number of poles (default: 4)
        max_zeros: Maximum number of zeros (default: 4)
        device: Device to use (cpu/cuda/mps)

    Example:
        >>> forward_diff = CircuitForwardDiffusion(timesteps=1000)
        >>> noisy_circuit, noise_dict = forward_diff.add_noise(clean_circuit, t)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        max_nodes: int = 5,
        max_poles: int = 4,
        max_zeros: int = 4,
        device: str = 'cpu'
    ):
        self.timesteps = timesteps
        self.max_nodes = max_nodes
        self.max_poles = max_poles
        self.max_zeros = max_zeros
        self.device = device

        # Node type constants
        self.num_node_types = 5  # GND, VIN, VOUT, INTERNAL, MASK

        # Edge feature dimension
        self.edge_feature_dim = 7  # C, G, L_inv + 4 masks

        # ==================================================================
        # Continuous Diffusion Schedules
        # ==================================================================

        # Get cosine schedule for continuous variables
        betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_cosine_schedule(timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Precompute useful values
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # ==================================================================
        # Discrete Diffusion Schedules
        # ==================================================================

        # Transition matrices for node types (5 classes)
        Q_node_types = get_discrete_transition_matrix(
            self.num_node_types, timesteps, schedule_type='uniform'
        )
        Q_bar_node_types = get_cumulative_transition_matrix(Q_node_types)
        self.register_buffer('Q_bar_node_types', Q_bar_node_types)

        # Transition matrices for counts (max_poles + 1 classes: 0, 1, 2, 3, 4)
        Q_pole_counts = get_discrete_transition_matrix(
            max_poles + 1, timesteps, schedule_type='uniform'
        )
        Q_bar_pole_counts = get_cumulative_transition_matrix(Q_pole_counts)
        self.register_buffer('Q_bar_pole_counts', Q_bar_pole_counts)

        Q_zero_counts = get_discrete_transition_matrix(
            max_zeros + 1, timesteps, schedule_type='uniform'
        )
        Q_bar_zero_counts = get_cumulative_transition_matrix(Q_zero_counts)
        self.register_buffer('Q_bar_zero_counts', Q_bar_zero_counts)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper to register buffers (for non-module class)."""
        setattr(self, name, tensor.to(self.device))

    def add_noise_to_circuit(
        self,
        clean_circuit: Dict[str, torch.Tensor],
        t: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Add noise to clean circuit at timestep t.

        Args:
            clean_circuit: Dictionary containing:
                - 'node_types': One-hot encoded [batch, max_nodes, num_node_types]
                - 'edge_values': Continuous edge features [batch, max_nodes, max_nodes, 7]
                - 'pole_count': Integer counts [batch]
                - 'zero_count': Integer counts [batch]
                - 'pole_values': Pole values [batch, max_poles, 2]
                - 'zero_values': Zero values [batch, max_zeros, 2]
            t: Timesteps [batch]

        Returns:
            noisy_circuit: Dictionary with noisy versions of all components
            noise_dict: Dictionary containing sampled noise for continuous variables
        """
        batch_size = t.shape[0]

        noisy_circuit = {}
        noise_dict = {}

        # ==================================================================
        # 1. Add noise to node types (discrete categorical)
        # ==================================================================

        clean_node_types = clean_circuit['node_types']  # [batch, max_nodes, num_node_types]

        # For each node in each graph
        noisy_node_types_list = []
        for b in range(batch_size):
            noisy_nodes_b = []
            for n in range(self.max_nodes):
                node_one_hot = clean_node_types[b, n].unsqueeze(0)  # [1, num_node_types]
                t_b = t[b].unsqueeze(0)  # [1]

                # Add discrete noise
                noisy_node = add_noise_discrete(node_one_hot, t_b, self.Q_bar_node_types)
                noisy_nodes_b.append(noisy_node.squeeze(0))

            noisy_node_types_list.append(torch.stack(noisy_nodes_b))

        noisy_circuit['node_types'] = torch.stack(noisy_node_types_list)  # [batch, max_nodes, num_node_types]

        # ==================================================================
        # 2. Add noise to edge values (continuous)
        # ==================================================================

        clean_edge_values = clean_circuit['edge_values']  # [batch, max_nodes, max_nodes, 7]

        # Flatten for noise addition
        edge_shape = clean_edge_values.shape
        clean_edge_flat = clean_edge_values.reshape(batch_size, -1)  # [batch, max_nodes * max_nodes * 7]

        # Expand timesteps to match
        t_expanded = t.unsqueeze(1).expand(batch_size, clean_edge_flat.shape[1])
        t_expanded_flat = t_expanded[:, 0]  # [batch] (all same per sample)

        noisy_edge_flat, edge_noise = add_noise_continuous(
            clean_edge_flat, t_expanded_flat, self.alphas_cumprod
        )

        noisy_circuit['edge_values'] = noisy_edge_flat.reshape(edge_shape)
        noise_dict['edge_noise'] = edge_noise.reshape(edge_shape)

        # ==================================================================
        # 3. Add noise to pole count (discrete categorical)
        # ==================================================================

        clean_pole_count = clean_circuit['pole_count']  # [batch]

        # Convert to one-hot
        pole_count_one_hot = F.one_hot(clean_pole_count, num_classes=self.max_poles + 1).float()

        # Add discrete noise
        noisy_pole_count_one_hot = add_noise_discrete(
            pole_count_one_hot, t, self.Q_bar_pole_counts
        )

        noisy_circuit['pole_count_one_hot'] = noisy_pole_count_one_hot  # [batch, max_poles + 1]

        # ==================================================================
        # 4. Add noise to zero count (discrete categorical)
        # ==================================================================

        clean_zero_count = clean_circuit['zero_count']  # [batch]

        # Convert to one-hot
        zero_count_one_hot = F.one_hot(clean_zero_count, num_classes=self.max_zeros + 1).float()

        # Add discrete noise
        noisy_zero_count_one_hot = add_noise_discrete(
            zero_count_one_hot, t, self.Q_bar_zero_counts
        )

        noisy_circuit['zero_count_one_hot'] = noisy_zero_count_one_hot  # [batch, max_zeros + 1]

        # ==================================================================
        # 5. Add noise to pole values (continuous)
        # ==================================================================

        clean_pole_values = clean_circuit['pole_values']  # [batch, max_poles, 2]

        # Flatten
        pole_shape = clean_pole_values.shape
        clean_pole_flat = clean_pole_values.reshape(batch_size, -1)  # [batch, max_poles * 2]

        noisy_pole_flat, pole_noise = add_noise_continuous(
            clean_pole_flat, t, self.alphas_cumprod
        )

        noisy_circuit['pole_values'] = noisy_pole_flat.reshape(pole_shape)
        noise_dict['pole_noise'] = pole_noise.reshape(pole_shape)

        # ==================================================================
        # 6. Add noise to zero values (continuous)
        # ==================================================================

        clean_zero_values = clean_circuit['zero_values']  # [batch, max_zeros, 2]

        # Flatten
        zero_shape = clean_zero_values.shape
        clean_zero_flat = clean_zero_values.reshape(batch_size, -1)  # [batch, max_zeros * 2]

        noisy_zero_flat, zero_noise = add_noise_continuous(
            clean_zero_flat, t, self.alphas_cumprod
        )

        noisy_circuit['zero_values'] = noisy_zero_flat.reshape(zero_shape)
        noise_dict['zero_noise'] = zero_noise.reshape(zero_shape)

        return noisy_circuit, noise_dict

    def prepare_circuit_from_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert batch data to clean circuit format for noise injection.

        Args:
            batch: Batch dictionary from DataLoader containing circuit data

        Returns:
            clean_circuit: Dictionary ready for noise injection
        """
        # This will be implemented based on the actual data format
        # Placeholder for now
        raise NotImplementedError("Will be implemented based on actual data format")

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample

        Returns:
            t: Sampled timesteps [batch_size]
        """
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)


def create_forward_diffusion(
    timesteps: int = 1000,
    max_nodes: int = 5,
    max_poles: int = 4,
    max_zeros: int = 4,
    device: str = 'cpu'
) -> CircuitForwardDiffusion:
    """
    Factory function to create CircuitForwardDiffusion.

    Args:
        timesteps: Number of diffusion timesteps
        max_nodes: Maximum number of nodes
        max_poles: Maximum number of poles
        max_zeros: Maximum number of zeros
        device: Device (cpu/cuda/mps)

    Returns:
        forward_diffusion: CircuitForwardDiffusion instance
    """
    return CircuitForwardDiffusion(
        timesteps=timesteps,
        max_nodes=max_nodes,
        max_poles=max_poles,
        max_zeros=max_zeros,
        device=device
    )
