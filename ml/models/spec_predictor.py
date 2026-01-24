"""
Specification Predictor for Transfer Function Latent.

Predicts cutoff frequency and Q-factor from the transfer function
portion of the latent space (z[4:8]).

This auxiliary task forces the encoder to encode meaningful transfer
function information in z[4:8], preventing posterior collapse.
"""

import torch
import torch.nn as nn


class SpecPredictor(nn.Module):
    """
    Predict circuit specifications from transfer function latent.

    Takes z[4:8] (the poles/zeros latent) and predicts:
    - log10(cutoff_frequency) - log scale for numerical stability
    - Q-factor - linear scale

    Args:
        pz_latent_dim: Dimension of transfer function latent (default: 4)
        hidden_dim: Hidden layer dimension (default: 32)
        num_layers: Number of hidden layers (default: 2)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        pz_latent_dim: int = 4,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.pz_latent_dim = pz_latent_dim

        # Build MLP
        layers = []
        in_dim = pz_latent_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output: [log10(cutoff), Q]
        layers.append(nn.Linear(hidden_dim, 2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z_pz: torch.Tensor) -> torch.Tensor:
        """
        Predict specifications from transfer function latent.

        Args:
            z_pz: Transfer function latent [batch, pz_latent_dim]

        Returns:
            specs: Predicted specifications [batch, 2]
                   specs[:, 0] = log10(cutoff_frequency)
                   specs[:, 1] = Q-factor
        """
        return self.mlp(z_pz)

    def predict_specs(self, z_pz: torch.Tensor) -> dict:
        """
        Predict specifications and return as dictionary.

        Args:
            z_pz: Transfer function latent [batch, pz_latent_dim]

        Returns:
            dict with 'cutoff' (Hz) and 'q_factor'
        """
        with torch.no_grad():
            preds = self.forward(z_pz)
            cutoff = 10 ** preds[:, 0]  # Convert from log scale
            q_factor = preds[:, 1]

        return {
            'cutoff': cutoff,
            'q_factor': q_factor
        }


class FullLatentSpecPredictor(nn.Module):
    """
    Predict specifications from full latent (extracts z[4:8] internally).

    Convenience wrapper that takes full 8D latent and extracts the
    transfer function portion before prediction.

    Args:
        latent_dim: Full latent dimension (default: 8)
        pz_start: Start index of transfer function latent (default: 4)
        hidden_dim: Hidden layer dimension (default: 32)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        pz_start: int = 4,
        hidden_dim: int = 32
    ):
        super().__init__()

        self.pz_start = pz_start
        pz_latent_dim = latent_dim - pz_start

        self.predictor = SpecPredictor(
            pz_latent_dim=pz_latent_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict specifications from full latent.

        Args:
            z: Full latent [batch, latent_dim]

        Returns:
            specs: Predicted specifications [batch, 2]
        """
        z_pz = z[:, self.pz_start:]
        return self.predictor(z_pz)


if __name__ == '__main__':
    print("Testing SpecPredictor...")

    # Test basic predictor
    predictor = SpecPredictor(pz_latent_dim=4, hidden_dim=32)
    z_pz = torch.randn(8, 4)  # batch of 8, 4D latent

    preds = predictor(z_pz)
    print(f"Input shape: {z_pz.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Predicted log10(cutoff): {preds[:, 0].mean():.3f} +/- {preds[:, 0].std():.3f}")
    print(f"Predicted Q: {preds[:, 1].mean():.3f} +/- {preds[:, 1].std():.3f}")

    # Test full latent predictor
    full_predictor = FullLatentSpecPredictor(latent_dim=8, pz_start=4)
    z_full = torch.randn(8, 8)

    preds_full = full_predictor(z_full)
    print(f"\nFull latent input shape: {z_full.shape}")
    print(f"Output shape: {preds_full.shape}")

    # Test predict_specs
    specs = predictor.predict_specs(z_pz)
    print(f"\nPredicted cutoff (Hz): {specs['cutoff'].mean():.1f}")
    print(f"Predicted Q-factor: {specs['q_factor'].mean():.3f}")

    print("\n✓ All tests passed!")
