"""
Sinusoidal time embedding for diffusion models.

Encodes discrete timesteps into continuous representations using
sinusoidal functions, similar to positional encodings in transformers.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for timestep conditioning.

    Converts discrete timesteps into continuous embeddings using
    sinusoidal functions at different frequencies.

    Reference: Attention Is All You Need (Vaswani et al., 2017)
               Denoising Diffusion Probabilistic Models (Ho et al., 2020)

    Args:
        embedding_dim: Dimension of time embedding (must be even)
        max_period: Maximum period for sinusoidal functions (default: 10000)

    Example:
        >>> time_embed = SinusoidalTimeEmbedding(embedding_dim=128)
        >>> t = torch.tensor([0, 10, 100, 999])  # Timesteps
        >>> emb = time_embed(t)  # Shape: [4, 128]
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # Pre-compute frequency scaling factors
        half_dim = embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal time embedding.

        Args:
            t: Timesteps [batch_size] or [batch_size, 1]

        Returns:
            emb: Time embeddings [batch_size, embedding_dim]
        """
        # Ensure t is shape [batch_size]
        if t.dim() == 2:
            t = t.squeeze(1)

        # Convert to float
        t = t.float()

        # Compute arguments for sin/cos
        # Shape: [batch_size, half_dim]
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0)

        # Interleave sin and cos
        # [batch_size, half_dim, 2]
        embedding = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)

        # Flatten to [batch_size, embedding_dim]
        embedding = embedding.view(t.shape[0], -1)

        return embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, max_period={self.max_period})"


class TimeEmbeddingMLP(nn.Module):
    """
    MLP for projecting time embeddings to desired dimension.

    Combines sinusoidal time embedding with MLP projection to create
    rich temporal representations for conditioning the denoising network.

    Args:
        embedding_dim: Dimension of sinusoidal embedding
        hidden_dim: Hidden dimension for MLP
        output_dim: Output dimension (default: same as hidden_dim)
        activation: Activation function (default: SiLU/Swish)

    Example:
        >>> time_mlp = TimeEmbeddingMLP(embedding_dim=128, hidden_dim=256)
        >>> t = torch.tensor([0, 10, 100, 999])
        >>> time_features = time_mlp(t)  # Shape: [4, 256]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        activation: nn.Module = None
    ):
        super().__init__()

        if output_dim is None:
            output_dim = hidden_dim

        if activation is None:
            activation = nn.SiLU()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Sinusoidal time embedding
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time features.

        Args:
            t: Timesteps [batch_size]

        Returns:
            time_features: Projected time embeddings [batch_size, output_dim]
        """
        # Get sinusoidal embedding
        emb = self.time_embed(t)  # [batch_size, embedding_dim]

        # Project through MLP
        time_features = self.mlp(emb)  # [batch_size, output_dim]

        return time_features

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"embedding_dim={self.embedding_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim})")


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization conditioned on time embeddings.

    Applies FiLM (Feature-wise Linear Modulation) using time embeddings
    to modulate the normalized features.

    Reference: FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)

    Args:
        num_groups: Number of groups for GroupNorm
        num_channels: Number of channels to normalize
        time_embed_dim: Dimension of time embedding

    Example:
        >>> ada_gn = AdaptiveGroupNorm(num_groups=8, num_channels=256, time_embed_dim=128)
        >>> x = torch.randn(4, 256, 10)  # [batch, channels, nodes]
        >>> time_emb = torch.randn(4, 128)  # [batch, time_embed_dim]
        >>> out = ada_gn(x, time_emb)  # Shape: [4, 256, 10]
    """

    def __init__(self, num_groups: int, num_channels: int, time_embed_dim: int):
        super().__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels

        # Group normalization
        self.norm = nn.GroupNorm(num_groups, num_channels)

        # Time embedding projection for scale and shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * num_channels)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive group normalization.

        Args:
            x: Input features [batch_size, num_channels, ...]
            time_emb: Time embedding [batch_size, time_embed_dim]

        Returns:
            out: Normalized and modulated features [batch_size, num_channels, ...]
        """
        # Normalize
        x_norm = self.norm(x)  # [batch_size, num_channels, ...]

        # Get scale and shift from time embedding
        time_out = self.time_proj(time_emb)  # [batch_size, 2 * num_channels]
        scale, shift = time_out.chunk(2, dim=1)  # Each: [batch_size, num_channels]

        # Reshape for broadcasting
        # [batch_size, num_channels, 1, 1, ...]
        ndim = x.dim()
        scale = scale.view(x.shape[0], x.shape[1], *([1] * (ndim - 2)))
        shift = shift.view(x.shape[0], x.shape[1], *([1] * (ndim - 2)))

        # Apply FiLM: y = scale * x_norm + shift
        out = scale * x_norm + shift

        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"num_groups={self.num_groups}, "
                f"num_channels={self.num_channels})")


def create_timestep_encoding(
    timestep: int,
    embedding_dim: int,
    max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal encoding for a single timestep (utility function).

    Args:
        timestep: Single timestep value
        embedding_dim: Dimension of embedding
        max_period: Maximum period for sinusoidal functions

    Returns:
        encoding: Sinusoidal encoding [embedding_dim]

    Example:
        >>> enc_0 = create_timestep_encoding(0, 128)
        >>> enc_500 = create_timestep_encoding(500, 128)
        >>> enc_999 = create_timestep_encoding(999, 128)
    """
    time_embed = SinusoidalTimeEmbedding(embedding_dim, max_period)
    t = torch.tensor([timestep], dtype=torch.long)
    encoding = time_embed(t).squeeze(0)  # [embedding_dim]

    return encoding
