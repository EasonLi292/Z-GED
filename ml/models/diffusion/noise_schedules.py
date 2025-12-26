"""
Noise schedules for diffusion models.

Implements cosine schedule for continuous diffusion and discrete
transition matrices for categorical diffusion on node types and edges.
"""

import torch
import numpy as np
from typing import Tuple


def get_cosine_schedule(
    timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute cosine noise schedule for continuous diffusion.

    This schedule provides smoother transitions than linear schedules,
    leading to better sample quality.

    Reference: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)

    Args:
        timesteps: Number of diffusion timesteps (e.g., 1000)
        s: Small offset to prevent beta from being too small near t=0
        max_beta: Maximum value for beta to ensure numerical stability

    Returns:
        betas: Noise schedule [timesteps]
        alphas: 1 - betas [timesteps]
        alphas_cumprod: Cumulative product of alphas [timesteps]
        alphas_cumprod_prev: Shifted cumulative product [timesteps]
    """
    # Compute alpha_bar using cosine schedule
    def alpha_bar(t):
        return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

    # Create timestep array
    t = np.arange(0, timesteps + 1, dtype=np.float64)
    t = t / timesteps

    # Compute alpha_bar for each timestep
    alphas_cumprod = alpha_bar(t)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize

    # Compute betas from alpha_bar
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, max_beta)

    # Convert to torch tensors
    betas = torch.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.from_numpy(alphas_cumprod[1:]).float()

    # Create shifted version for DDPM sampling
    alphas_cumprod_prev = torch.cat([
        torch.tensor([1.0]),
        alphas_cumprod[:-1]
    ])

    return betas, alphas, alphas_cumprod, alphas_cumprod_prev


def get_discrete_transition_matrix(
    num_classes: int,
    timesteps: int,
    schedule_type: str = 'uniform'
) -> torch.Tensor:
    """
    Compute transition matrices for discrete diffusion on categorical variables.

    For node types, edges, etc., we use categorical diffusion where the forward
    process gradually transitions to a uniform distribution.

    Args:
        num_classes: Number of categories (e.g., 5 for node types)
        timesteps: Number of diffusion timesteps
        schedule_type: Type of schedule ('uniform' or 'absorbing')
            - uniform: Transitions toward uniform distribution
            - absorbing: Transitions toward a single absorbing state (MASK)

    Returns:
        Q_matrices: Transition matrices [timesteps, num_classes, num_classes]
                   Q[t, i, j] = P(x_t = j | x_{t-1} = i)
    """
    if schedule_type == 'uniform':
        # Uniform transition schedule (DiGress-style)
        # Q_t = (1 - beta_t) * I + beta_t * U
        # where U is uniform distribution

        # Get beta schedule
        betas, _, _, _ = get_cosine_schedule(timesteps)

        Q_matrices = []
        for t in range(timesteps):
            beta_t = betas[t].item()

            # Create transition matrix
            Q_t = torch.zeros(num_classes, num_classes)

            # Diagonal: probability of staying in same state
            Q_t = Q_t + (1 - beta_t) * torch.eye(num_classes)

            # Off-diagonal: probability of transitioning to other states
            Q_t = Q_t + (beta_t / num_classes) * torch.ones(num_classes, num_classes)

            Q_matrices.append(Q_t)

        return torch.stack(Q_matrices)  # [timesteps, num_classes, num_classes]

    elif schedule_type == 'absorbing':
        # Absorbing state schedule (PARD-style)
        # Transitions toward MASK token (assumed to be last class)

        betas, _, _, _ = get_cosine_schedule(timesteps)

        Q_matrices = []
        for t in range(timesteps):
            beta_t = betas[t].item()

            Q_t = torch.zeros(num_classes, num_classes)

            # Non-absorbing states transition to absorbing state
            for i in range(num_classes - 1):
                Q_t[i, i] = 1 - beta_t  # Stay in current state
                Q_t[i, -1] = beta_t      # Transition to MASK

            # Absorbing state stays absorbing
            Q_t[-1, -1] = 1.0

            Q_matrices.append(Q_t)

        return torch.stack(Q_matrices)

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def get_cumulative_transition_matrix(
    Q_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Compute cumulative transition matrices Q_bar[t] = Q[0] @ Q[1] @ ... @ Q[t].

    This allows direct computation of q(x_t | x_0) without iterating through
    all intermediate timesteps.

    Args:
        Q_matrices: Transition matrices [timesteps, num_classes, num_classes]

    Returns:
        Q_bar: Cumulative transition matrices [timesteps, num_classes, num_classes]
    """
    timesteps, num_classes, _ = Q_matrices.shape

    Q_bar = []
    Q_cumulative = torch.eye(num_classes)

    for t in range(timesteps):
        Q_cumulative = Q_cumulative @ Q_matrices[t]
        Q_bar.append(Q_cumulative.clone())

    return torch.stack(Q_bar)


def extract_index(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """
    Extract values from tensor 'a' at indices 't' and reshape to broadcast.

    This is a helper function for indexing noise schedule values at specific
    timesteps during training.

    Args:
        a: Tensor of values (e.g., alphas_cumprod) [timesteps]
        t: Timestep indices [batch_size]
        x_shape: Shape to broadcast to (e.g., [batch_size, latent_dim])

    Returns:
        out: Values at timesteps t, reshaped to broadcast [batch_size, 1, 1, ...]

    Example:
        >>> alphas = torch.tensor([0.9, 0.8, 0.7, 0.6])
        >>> t = torch.tensor([1, 3, 2])  # batch_size = 3
        >>> x_shape = (3, 8)
        >>> extract_index(alphas, t, x_shape)
        tensor([[0.8],
                [0.6],
                [0.7]])  # Shape: [3, 1] for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)

    # Reshape to [batch_size, 1, 1, ...] for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def compute_posterior_mean_variance(
    x_t: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    alphas_cumprod_prev: torch.Tensor,
    betas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute posterior mean and variance for reverse diffusion step.

    Given x_t and predicted noise, compute parameters of q(x_{t-1} | x_t, x_0).

    Args:
        x_t: Noisy sample at timestep t [batch_size, ...]
        t: Timestep [batch_size]
        predicted_noise: Noise predicted by model [batch_size, ...]
        alphas_cumprod: Cumulative product of alphas [timesteps]
        alphas_cumprod_prev: Shifted cumulative product [timesteps]
        betas: Noise schedule [timesteps]

    Returns:
        posterior_mean: Mean of q(x_{t-1} | x_t, x_0) [batch_size, ...]
        posterior_variance: Variance of q(x_{t-1} | x_t, x_0) [batch_size, ...]
        posterior_log_variance_clipped: Log variance (clipped) [batch_size, ...]
    """
    # Extract values at timestep t
    x_shape = x_t.shape
    alpha_bar_t = extract_index(alphas_cumprod, t, x_shape)
    alpha_bar_prev_t = extract_index(alphas_cumprod_prev, t, x_shape)
    beta_t = extract_index(betas, t, x_shape)

    # Predict x_0 from x_t and predicted noise
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # Compute posterior mean
    # μ(x_t, x_0) = (sqrt(α_bar_{t-1}) * β_t / (1 - α_bar_t)) * x_0
    #             + (sqrt(α_t) * (1 - α_bar_{t-1}) / (1 - α_bar_t)) * x_t
    coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

    posterior_mean = coef_x0 * pred_x0 + coef_xt * x_t

    # Compute posterior variance
    # σ²_t = (1 - α_bar_{t-1}) / (1 - α_bar_t) * β_t
    posterior_variance = (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * beta_t

    # Clip log variance for numerical stability
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )

    return posterior_mean, posterior_variance, posterior_log_variance_clipped


def add_noise_continuous(
    x_0: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    noise: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to continuous variables using forward diffusion.

    Implements q(x_t | x_0) = N(sqrt(α_bar_t) * x_0, (1 - α_bar_t) * I)

    Args:
        x_0: Original clean data [batch_size, ...]
        t: Timesteps [batch_size]
        alphas_cumprod: Cumulative product of alphas [timesteps]
        noise: Optional pre-sampled noise (for reproducibility)

    Returns:
        x_t: Noisy data at timestep t [batch_size, ...]
        noise: Sampled noise [batch_size, ...]
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Extract alpha_bar at timestep t
    x_shape = x_0.shape
    sqrt_alpha_bar_t = torch.sqrt(extract_index(alphas_cumprod, t, x_shape))
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - extract_index(alphas_cumprod, t, x_shape))

    # Apply noise: x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    return x_t, noise


def add_noise_discrete(
    x_0: torch.Tensor,
    t: torch.Tensor,
    Q_bar: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to discrete variables using categorical diffusion.

    Implements q(x_t | x_0) = Cat(x_0 @ Q_bar[t])

    Args:
        x_0: Original one-hot encoded data [batch_size, num_classes]
        t: Timesteps [batch_size]
        Q_bar: Cumulative transition matrices [timesteps, num_classes, num_classes]

    Returns:
        x_t: Noisy categorical distribution [batch_size, num_classes]
    """
    batch_size = x_0.shape[0]

    # Get transition matrix for each sample's timestep
    Q_t = Q_bar[t]  # [batch_size, num_classes, num_classes]

    # Compute categorical distribution at timestep t
    # x_t ~ Cat(x_0 @ Q_bar[t])
    # For one-hot x_0, this selects the row corresponding to the original class
    x_t_prob = torch.bmm(x_0.unsqueeze(1), Q_t).squeeze(1)  # [batch_size, num_classes]

    # Sample from categorical distribution
    x_t_sampled = torch.multinomial(x_t_prob, num_samples=1).squeeze(1)  # [batch_size]

    # Convert back to one-hot
    x_t = torch.zeros_like(x_0)
    x_t.scatter_(1, x_t_sampled.unsqueeze(1), 1.0)

    return x_t
