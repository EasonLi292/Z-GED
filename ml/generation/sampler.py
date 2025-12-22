"""
Circuit Sampler for GraphVAE.

Provides utilities for generating novel circuits through:
1. Prior sampling (unconditional generation)
2. Conditional generation (specify filter type)
3. Latent space interpolation
4. Controlled generation (modify specific branches)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Filter type mapping (must match decoder)
FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']


class CircuitSampler:
    """
    Sampler for generating circuits from trained GraphVAE.

    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        device: Device to run on
        latent_dim: Total latent dimension
        branch_dims: Optional tuple of (topo_dim, values_dim, pz_dim)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: str = 'cpu',
        latent_dim: int = 8,
        branch_dims: Optional[Tuple[int, int, int]] = None
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.latent_dim = latent_dim

        # Set branch dimensions
        if branch_dims is None:
            # Equal split if not specified
            assert latent_dim % 3 == 0, "Latent dim must be divisible by 3"
            branch_dim = latent_dim // 3
            self.branch_dims = (branch_dim, branch_dim, branch_dim)
        else:
            self.branch_dims = branch_dims

        self.topo_dim, self.values_dim, self.pz_dim = self.branch_dims

        # Set models to eval mode
        self.encoder.eval()
        self.decoder.eval()

    def sample_prior(
        self,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sample circuits from prior distribution N(0, I).

        Args:
            num_samples: Number of circuits to generate
            temperature: Sampling temperature (default: 1.0)
                - < 1.0: More conservative (closer to mean)
                - > 1.0: More exploratory (more variance)

        Returns:
            Dictionary with decoder outputs
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=self.device) * temperature

        # Decode to circuits
        with torch.no_grad():
            outputs = self.decoder(z, hard=True)

        return outputs

    def sample_conditional(
        self,
        filter_type: Union[str, int],
        num_samples: int = 1,
        temperature: float = 1.0,
        exemplar_circuits: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuits conditioned on filter type.

        Strategy:
        1. If exemplars provided: Use mean of encoded exemplars as centroid
        2. Otherwise: Sample randomly and force topology

        Args:
            filter_type: Filter type name (str) or index (int)
            num_samples: Number of circuits to generate
            temperature: Sampling temperature
            exemplar_circuits: Optional list of example circuits of this type

        Returns:
            Dictionary with decoder outputs
        """
        # Convert filter type to index
        if isinstance(filter_type, str):
            if filter_type not in FILTER_TYPES:
                raise ValueError(f"Unknown filter type: {filter_type}")
            filter_idx = FILTER_TYPES.index(filter_type)
        else:
            filter_idx = filter_type

        # Create one-hot filter type
        filter_onehot = torch.zeros(num_samples, len(FILTER_TYPES), device=self.device)
        filter_onehot[:, filter_idx] = 1.0

        if exemplar_circuits is not None:
            # Use exemplar-based generation
            z = self._sample_from_exemplars(
                exemplar_circuits,
                num_samples,
                temperature
            )
        else:
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=self.device) * temperature

        # Decode with teacher forcing on topology
        with torch.no_grad():
            outputs = self.decoder(z, hard=True, gt_filter_type=filter_onehot)

        return outputs

    def interpolate(
        self,
        circuit1: Dict,
        circuit2: Dict,
        num_steps: int = 5,
        interpolation_type: str = 'linear'
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Interpolate between two circuits in latent space.

        Args:
            circuit1: First circuit (PyG Data object or batch dict)
            circuit2: Second circuit
            num_steps: Number of interpolation steps (including endpoints)
            interpolation_type: 'linear', 'spherical', or 'branch'

        Returns:
            List of decoder outputs (interpolated circuits)
        """
        # Encode both circuits
        with torch.no_grad():
            z1 = self._encode_circuit(circuit1)
            z2 = self._encode_circuit(circuit2)

        # Generate interpolation
        alphas = torch.linspace(0, 1, num_steps)
        interpolated = []

        for alpha in alphas:
            if interpolation_type == 'linear':
                z_interp = (1 - alpha) * z1 + alpha * z2
            elif interpolation_type == 'spherical':
                # Spherical linear interpolation (slerp)
                z_interp = self._slerp(z1, z2, alpha)
            elif interpolation_type == 'branch':
                # Interpolate each branch independently
                z_interp = self._interpolate_branches(z1, z2, alpha)
            else:
                raise ValueError(f"Unknown interpolation type: {interpolation_type}")

            # Decode
            with torch.no_grad():
                outputs = self.decoder(z_interp, hard=True)
                interpolated.append(outputs)

        return interpolated

    def modify_branch(
        self,
        circuit: Dict,
        branch: str,
        modification: Union[torch.Tensor, float],
        operation: str = 'add'
    ) -> Dict[str, torch.Tensor]:
        """
        Modify specific branch of latent vector.

        Args:
            circuit: Input circuit to modify
            branch: Which branch to modify ('topo', 'values', 'pz')
            modification: Value to add/multiply/replace
            operation: 'add', 'multiply', 'replace'

        Returns:
            Dictionary with decoder outputs
        """
        # Encode circuit
        with torch.no_grad():
            z = self._encode_circuit(circuit)

        # Get branch indices
        if branch == 'topo':
            start, end = 0, self.topo_dim
        elif branch == 'values':
            start, end = self.topo_dim, self.topo_dim + self.values_dim
        elif branch == 'pz':
            start, end = self.topo_dim + self.values_dim, self.latent_dim
        else:
            raise ValueError(f"Unknown branch: {branch}")

        # Apply modification
        z_modified = z.clone()

        if operation == 'add':
            if isinstance(modification, float):
                z_modified[:, start:end] += modification
            else:
                z_modified[:, start:end] += modification
        elif operation == 'multiply':
            if isinstance(modification, float):
                z_modified[:, start:end] *= modification
            else:
                z_modified[:, start:end] *= modification
        elif operation == 'replace':
            z_modified[:, start:end] = modification
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Decode
        with torch.no_grad():
            outputs = self.decoder(z_modified, hard=True)

        return outputs

    def encode(self, circuit: Dict) -> torch.Tensor:
        """Encode circuit to latent space (deterministic)."""
        with torch.no_grad():
            z = self._encode_circuit(circuit)
        return z

    def decode(self, z: torch.Tensor, hard: bool = True) -> Dict[str, torch.Tensor]:
        """Decode latent vector to circuit."""
        with torch.no_grad():
            outputs = self.decoder(z, hard=hard)
        return outputs

    # Helper methods

    def _encode_circuit(self, circuit: Dict) -> torch.Tensor:
        """Encode circuit to latent space (uses mean, not sampled)."""
        # Move circuit to device
        if 'graph' in circuit:
            graph = circuit['graph'].to(self.device)
            # Poles and zeros are tensors, wrap in list for batch size 1
            poles = [circuit['poles'].to(self.device)]
            zeros = [circuit['zeros'].to(self.device)]
        else:
            # Assume it's already a batch dict
            graph = circuit
            # Extract poles/zeros from metadata if available
            poles = [circuit.get('poles', torch.zeros(0, 2, device=self.device))]
            zeros = [circuit.get('zeros', torch.zeros(0, 2, device=self.device))]

        # Create batch tensor if not present (single graph)
        batch = graph.batch if hasattr(graph, 'batch') and graph.batch is not None else torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)

        # Encode (get mean, not sampled z)
        _, mu, _ = self.encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            batch,
            poles,
            zeros
        )

        return mu

    def _sample_from_exemplars(
        self,
        exemplars: List,
        num_samples: int,
        temperature: float
    ) -> torch.Tensor:
        """Sample around mean of exemplar encodings."""
        # Encode all exemplars
        exemplar_zs = []
        for circuit in exemplars:
            z = self._encode_circuit(circuit)
            exemplar_zs.append(z)

        # Compute mean
        z_mean = torch.stack(exemplar_zs).mean(dim=0)

        # Sample around mean
        noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        z = z_mean + temperature * noise

        return z

    def _slerp(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Spherical linear interpolation."""
        # Normalize
        z1_norm = z1 / (z1.norm(dim=-1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute angle
        dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)

        # Slerp formula
        sin_theta = torch.sin(theta)
        s1 = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
        s2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)

        z_interp = s1 * z1 + s2 * z2

        return z_interp

    def _interpolate_branches(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Interpolate each branch independently."""
        z_topo = (1 - alpha) * z1[:, :self.topo_dim] + alpha * z2[:, :self.topo_dim]
        z_values = (1 - alpha) * z1[:, self.topo_dim:self.topo_dim+self.values_dim] + \
                   alpha * z2[:, self.topo_dim:self.topo_dim+self.values_dim]
        z_pz = (1 - alpha) * z1[:, -self.pz_dim:] + alpha * z2[:, -self.pz_dim:]

        z_interp = torch.cat([z_topo, z_values, z_pz], dim=-1)
        return z_interp


# Standalone functions for convenience

def sample_prior(
    decoder: nn.Module,
    num_samples: int,
    latent_dim: int,
    device: str = 'cpu',
    temperature: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Sample from prior N(0, I)."""
    sampler = CircuitSampler(None, decoder, device, latent_dim)
    return sampler.sample_prior(num_samples, temperature)


def sample_conditional(
    decoder: nn.Module,
    filter_type: Union[str, int],
    num_samples: int,
    latent_dim: int,
    device: str = 'cpu',
    temperature: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Sample conditioned on filter type."""
    sampler = CircuitSampler(None, decoder, device, latent_dim)
    return sampler.sample_conditional(filter_type, num_samples, temperature)


def interpolate_circuits(
    encoder: nn.Module,
    decoder: nn.Module,
    circuit1: Dict,
    circuit2: Dict,
    num_steps: int,
    latent_dim: int,
    device: str = 'cpu'
) -> List[Dict[str, torch.Tensor]]:
    """Interpolate between two circuits."""
    sampler = CircuitSampler(encoder, decoder, device, latent_dim)
    return sampler.interpolate(circuit1, circuit2, num_steps)
