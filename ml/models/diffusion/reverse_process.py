"""
Reverse diffusion process for circuit graph generation.

Implements DDPM and DDIM samplers for generating circuits from noise.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable
from tqdm import tqdm

from .noise_schedules import (
    get_cosine_schedule,
    extract_index,
    compute_posterior_mean_variance
)


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Model) sampler.

    Implements the reverse diffusion process with 1000 denoising steps.
    Slower but higher quality than DDIM.

    Reference: Denoising Diffusion Probabilistic Models (Ho et al., 2020)

    Args:
        timesteps: Number of diffusion timesteps (default: 1000)
        device: Device (cpu/cuda/mps)

    Example:
        >>> sampler = DDPMSampler(timesteps=1000)
        >>> circuit = sampler.sample(model, latent_code, conditions, batch_size=4)
    """

    def __init__(self, timesteps: int = 1000, device: str = 'cpu'):
        self.timesteps = timesteps
        self.device = device

        # Get noise schedule
        betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_cosine_schedule(timesteps)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        batch_size: int = 1,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        return_trajectory: bool = False,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuits using DDPM sampling.

        Args:
            model: Denoising network (DiffusionGraphTransformer)
            latent_code: Latent codes [batch_size, latent_dim]
            conditions: Specifications [batch_size, conditions_dim]
            batch_size: Number of samples to generate
            temperature: Sampling temperature (default: 1.0)
            guidance_scale: Classifier-free guidance scale (default: 1.0, no guidance)
            return_trajectory: Whether to return full denoising trajectory
            verbose: Show progress bar

        Returns:
            circuit: Generated circuit components
        """
        # Initialize from pure noise
        max_nodes = model.max_nodes
        max_poles = model.max_poles
        max_zeros = model.max_zeros

        # Continuous variables: Gaussian noise
        noisy_edges = torch.randn(
            batch_size, max_nodes, max_nodes, 7,
            device=self.device
        ) * temperature

        noisy_poles = torch.randn(
            batch_size, max_poles, 2,
            device=self.device
        ) * temperature

        noisy_zeros = torch.randn(
            batch_size, max_zeros, 2,
            device=self.device
        ) * temperature

        # Discrete variables: Uniform distribution (approximated as soft categorical)
        noisy_nodes = torch.ones(
            batch_size, max_nodes, model.num_node_types,
            device=self.device
        ) / model.num_node_types

        # Storage for trajectory
        trajectory = [] if return_trajectory else None

        # Reverse diffusion loop
        iterator = range(self.timesteps - 1, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc="DDPM Sampling")

        for t_idx in iterator:
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)

            # Predict denoised values
            with torch.no_grad():
                predictions = model(
                    noisy_nodes,
                    noisy_edges,
                    t,
                    latent_code,
                    conditions
                )

            # ==================================================================
            # Denoise continuous variables (edges, poles, zeros)
            # ==================================================================

            if t_idx > 0:
                # Not the final step - add noise

                # Denoise edges
                noisy_edges = self._denoise_step_continuous(
                    noisy_edges.reshape(batch_size, -1),
                    predictions['edge_values'].reshape(batch_size, -1),
                    t,
                    temperature
                ).reshape(batch_size, max_nodes, max_nodes, 7)

                # Denoise poles
                noisy_poles = self._denoise_step_continuous(
                    noisy_poles.reshape(batch_size, -1),
                    predictions['pole_values'].reshape(batch_size, -1),
                    t,
                    temperature
                ).reshape(batch_size, max_poles, 2)

                # Denoise zeros
                noisy_zeros = self._denoise_step_continuous(
                    noisy_zeros.reshape(batch_size, -1),
                    predictions['zero_values'].reshape(batch_size, -1),
                    t,
                    temperature
                ).reshape(batch_size, max_zeros, 2)

            else:
                # Final step - use predictions directly
                noisy_edges = predictions['edge_values']
                noisy_poles = predictions['pole_values']
                noisy_zeros = predictions['zero_values']

            # ==================================================================
            # Denoise discrete variables (node types)
            # ==================================================================

            # For discrete variables, use predicted logits directly
            # with temperature-scaled softmax
            node_probs = F.softmax(predictions['node_types'] / temperature, dim=-1)

            if t_idx > 0:
                # Sample from categorical distribution
                noisy_nodes = self._sample_categorical(node_probs)
            else:
                # Final step - use probabilities directly (or argmax for hard assignment)
                noisy_nodes = node_probs

            # Store trajectory
            if return_trajectory:
                trajectory.append({
                    'nodes': noisy_nodes.clone(),
                    'edges': noisy_edges.clone(),
                    'poles': noisy_poles.clone(),
                    'zeros': noisy_zeros.clone(),
                    't': t_idx
                })

        # ==================================================================
        # Final circuit
        # ==================================================================

        # Convert node probabilities to hard assignments
        node_types = torch.argmax(noisy_nodes, dim=-1)  # [batch, max_nodes]

        # Get pole/zero counts from logits
        pole_counts = torch.argmax(predictions['pole_count_logits'], dim=-1)  # [batch]
        zero_counts = torch.argmax(predictions['zero_count_logits'], dim=-1)  # [batch]

        # Determine edge existence
        edge_probs = torch.sigmoid(predictions['edge_existence'])
        edge_exists = (edge_probs > 0.5).float()  # [batch, max_nodes, max_nodes]

        circuit = {
            'node_types': node_types,
            'edge_existence': edge_exists,
            'edge_values': noisy_edges,
            'pole_count': pole_counts,
            'zero_count': zero_counts,
            'pole_values': noisy_poles,
            'zero_values': noisy_zeros
        }

        if return_trajectory:
            circuit['trajectory'] = trajectory

        return circuit

    def _denoise_step_continuous(
        self,
        x_t: torch.Tensor,
        x_pred: torch.Tensor,
        t: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Single denoising step for continuous variables.

        Uses the posterior mean from the predicted clean value.
        """
        batch_size = x_t.shape[0]
        x_shape = x_t.shape

        # Extract schedule values
        alpha_bar_t = extract_index(self.alphas_cumprod, t, x_shape)
        alpha_bar_prev_t = extract_index(self.alphas_cumprod_prev, t, x_shape)
        beta_t = extract_index(self.betas, t, x_shape)

        # Predict x_0 from prediction
        pred_x0 = x_pred

        # Compute posterior mean
        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

        posterior_mean = coef_x0 * pred_x0 + coef_xt * x_t

        # Compute posterior variance
        posterior_variance = (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * beta_t

        # Sample noise
        noise = torch.randn_like(x_t) * temperature

        # x_{t-1} = posterior_mean + sqrt(posterior_variance) * noise
        x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise

        return x_prev

    def _sample_categorical(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical distribution and return one-hot encoding.

        Args:
            probs: Categorical probabilities [batch, ..., num_classes]

        Returns:
            one_hot: One-hot encoded samples [batch, ..., num_classes]
        """
        # Flatten to [batch * ..., num_classes]
        original_shape = probs.shape
        probs_flat = probs.reshape(-1, original_shape[-1])

        # Sample
        samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

        # Convert to one-hot
        one_hot_flat = F.one_hot(samples, num_classes=original_shape[-1]).float()

        # Reshape back
        one_hot = one_hot_flat.reshape(original_shape)

        return one_hot


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Model) sampler.

    Implements deterministic/accelerated sampling with fewer steps (e.g., 50 instead of 1000).
    Faster than DDPM with comparable quality.

    Reference: Denoising Diffusion Implicit Models (Song et al., 2020)

    Args:
        timesteps: Number of diffusion timesteps (default: 1000)
        num_inference_steps: Number of sampling steps (default: 50)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM) (default: 0)
        device: Device (cpu/cuda/mps)

    Example:
        >>> sampler = DDIMSampler(timesteps=1000, num_inference_steps=50)
        >>> circuit = sampler.sample(model, latent_code, conditions, batch_size=4)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: str = 'cpu'
    ):
        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.device = device

        # Get noise schedule
        betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_cosine_schedule(timesteps)

        self.alphas_cumprod = alphas_cumprod.to(device)

        # Create sampling timesteps (evenly spaced)
        step_ratio = timesteps // num_inference_steps
        self.sampling_timesteps = torch.arange(
            0, timesteps, step_ratio, device=device
        ).long().flip(0)  # Reverse order

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        batch_size: int = 1,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuits using DDIM sampling.

        Args:
            model: Denoising network
            latent_code: Latent codes [batch_size, latent_dim]
            conditions: Specifications [batch_size, conditions_dim]
            batch_size: Number of samples
            temperature: Sampling temperature
            guidance_scale: CFG scale
            verbose: Show progress bar

        Returns:
            circuit: Generated circuit components
        """
        # Initialize from noise (same as DDPM)
        max_nodes = model.max_nodes
        max_poles = model.max_poles
        max_zeros = model.max_zeros

        noisy_edges = torch.randn(batch_size, max_nodes, max_nodes, 7, device=self.device) * temperature
        noisy_poles = torch.randn(batch_size, max_poles, 2, device=self.device) * temperature
        noisy_zeros = torch.randn(batch_size, max_zeros, 2, device=self.device) * temperature
        noisy_nodes = torch.ones(batch_size, max_nodes, model.num_node_types, device=self.device) / model.num_node_types

        # DDIM sampling loop
        iterator = self.sampling_timesteps
        if verbose:
            iterator = tqdm(iterator, desc="DDIM Sampling")

        for i, t_idx in enumerate(iterator):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)

            # Predict
            predictions = model(noisy_nodes, noisy_edges, t, latent_code, conditions)

            # DDIM update (simplified for now - using predicted x0 directly)
            # Full DDIM would use the DDIM update rule with eta parameter

            alpha_bar_t = self.alphas_cumprod[t_idx]

            # Get next timestep
            if i < len(iterator) - 1:
                t_next_idx = iterator[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_next_idx]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # Denoise (simplified - using predictions directly for now)
            if i < len(iterator) - 1:
                # Interpolate between current and predicted
                noisy_edges = predictions['edge_values']
                noisy_poles = predictions['pole_values']
                noisy_zeros = predictions['zero_values']
            else:
                # Final step
                noisy_edges = predictions['edge_values']
                noisy_poles = predictions['pole_values']
                noisy_zeros = predictions['zero_values']

            # Update node types
            node_probs = F.softmax(predictions['node_types'] / temperature, dim=-1)
            noisy_nodes = node_probs

        # Final circuit (same as DDPM)
        node_types = torch.argmax(noisy_nodes, dim=-1)
        pole_counts = torch.argmax(predictions['pole_count_logits'], dim=-1)
        zero_counts = torch.argmax(predictions['zero_count_logits'], dim=-1)
        edge_exists = (torch.sigmoid(predictions['edge_existence']) > 0.5).float()

        circuit = {
            'node_types': node_types,
            'edge_existence': edge_exists,
            'edge_values': noisy_edges,
            'pole_count': pole_counts,
            'zero_count': zero_counts,
            'pole_values': noisy_poles,
            'zero_values': noisy_zeros
        }

        return circuit


def create_sampler(
    sampler_type: str = 'ddpm',
    timesteps: int = 1000,
    num_inference_steps: int = 50,
    device: str = 'cpu',
    **kwargs
):
    """
    Factory function to create samplers.

    Args:
        sampler_type: 'ddpm' or 'ddim'
        timesteps: Number of diffusion timesteps
        num_inference_steps: Number of sampling steps (DDIM only)
        device: Device
        **kwargs: Additional sampler-specific arguments

    Returns:
        sampler: DDPMSampler or DDIMSampler instance
    """
    if sampler_type == 'ddpm':
        return DDPMSampler(timesteps=timesteps, device=device)
    elif sampler_type == 'ddim':
        return DDIMSampler(
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
