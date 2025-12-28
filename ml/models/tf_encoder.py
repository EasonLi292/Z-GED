"""
Transfer Function Encoder: Maps poles/zeros to latent space.

This allows targeted circuit generation:
1. You specify exact poles/zeros (transfer function)
2. TF encoder maps them to latent[4:8]
3. Random latent[0:4] provides topology/values variation
4. Decoder generates novel circuits implementing your TF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransferFunctionEncoder(nn.Module):
    """
    Encode transfer function (poles/zeros) to latent space.

    Maps variable-length poles/zeros to fixed 4D latent representation
    (corresponding to latent[4:8] in the full hierarchical latent).

    Architecture:
        DeepSets for permutation-invariant pole/zero encoding
        → MLP → 4D latent (μ and log_σ for VAE sampling)

    Args:
        max_poles: Maximum number of poles (default: 4)
        max_zeros: Maximum number of zeros (default: 4)
        hidden_dim: Hidden dimension (default: 64)
        latent_dim: Output latent dimension (default: 4)
    """

    def __init__(
        self,
        max_poles: int = 4,
        max_zeros: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 4
    ):
        super().__init__()

        self.max_poles = max_poles
        self.max_zeros = max_zeros
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Pole encoder (DeepSets for permutation invariance)
        self.pole_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [real, imag] → hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Zero encoder
        self.zero_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Count embeddings (important for TF order)
        self.pole_count_embedding = nn.Embedding(max_poles + 1, hidden_dim)
        self.zero_count_embedding = nn.Embedding(max_zeros + 1, hidden_dim)

        # Aggregation and encoding
        self.tf_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # poles + zeros + counts
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # VAE outputs
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        pole_values: torch.Tensor,
        pole_count: torch.Tensor,
        zero_values: torch.Tensor,
        zero_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode transfer function to latent space.

        Args:
            pole_values: [batch, max_poles, 2] - pole locations [real, imag]
            pole_count: [batch] - number of poles (0 to max_poles)
            zero_values: [batch, max_zeros, 2] - zero locations [real, imag]
            zero_count: [batch] - number of zeros (0 to max_zeros)

        Returns:
            mu: [batch, latent_dim] - latent mean
            logvar: [batch, latent_dim] - latent log variance
        """
        batch_size = pole_values.shape[0]
        device = pole_values.device

        # Encode poles using DeepSets (sum aggregation)
        pole_embeddings = self.pole_embedding(pole_values)  # [batch, max_poles, hidden]

        # Mask out unused poles
        pole_mask = torch.arange(self.max_poles, device=device).unsqueeze(0).expand(batch_size, -1)
        pole_mask = (pole_mask < pole_count.unsqueeze(1)).float().unsqueeze(-1)
        pole_embeddings = pole_embeddings * pole_mask

        pole_repr = pole_embeddings.sum(dim=1)  # [batch, hidden] - permutation invariant

        # Encode zeros
        zero_embeddings = self.zero_embedding(zero_values)
        zero_mask = torch.arange(self.max_zeros, device=device).unsqueeze(0).expand(batch_size, -1)
        zero_mask = (zero_mask < zero_count.unsqueeze(1)).float().unsqueeze(-1)
        zero_embeddings = zero_embeddings * zero_mask
        zero_repr = zero_embeddings.sum(dim=1)

        # Encode counts
        pole_count_repr = self.pole_count_embedding(pole_count.long())
        zero_count_repr = self.zero_count_embedding(zero_count.long())

        # Combine all TF information
        tf_repr = torch.cat([
            pole_repr,
            zero_repr,
            pole_count_repr,
            zero_count_repr
        ], dim=-1)

        # Encode to latent
        hidden = self.tf_encoder(tf_repr)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        pole_values: torch.Tensor,
        pole_count: torch.Tensor,
        zero_values: torch.Tensor,
        zero_count: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Encode TF to latent (convenience method).

        Args:
            deterministic: If True, return mu. If False, sample from distribution.

        Returns:
            latent: [batch, latent_dim]
        """
        mu, logvar = self.forward(pole_values, pole_count, zero_values, zero_count)

        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, logvar)


def train_tf_encoder_from_dataset(
    dataset_path: str,
    checkpoint_path: str,
    num_epochs: int = 100,
    device: str = 'mps'
):
    """
    Train the TF encoder by learning the pole/zero → latent[4:8] mapping
    from the existing trained encoder.

    This extracts the inverse mapping from the full encoder's pole/zero branch.
    """
    import torch.optim as optim
    from ml.data.dataset import CircuitDataset
    from ml.models.encoder import HierarchicalEncoder

    device = torch.device(device)

    # Load pretrained full encoder (to get ground truth latents)
    print("Loading pretrained encoder...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    full_encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['encoder']['node_feature_dim'],
        edge_feature_dim=config['model']['encoder']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['encoder']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['encoder']['gnn_num_layers'],
        latent_dim=config['model']['encoder']['latent_dim'],
        topo_latent_dim=config['model']['encoder']['topo_latent_dim'],
        values_latent_dim=config['model']['encoder']['values_latent_dim'],
        pz_latent_dim=config['model']['encoder']['pz_latent_dim'],
        dropout=config['model']['encoder']['dropout']
    ).to(device)

    full_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    full_encoder.eval()

    # Create TF encoder to train
    tf_encoder = TransferFunctionEncoder(
        max_poles=4,
        max_zeros=4,
        hidden_dim=64,
        latent_dim=config['model']['encoder']['pz_latent_dim']  # Should be 4
    ).to(device)

    # Load dataset
    print("Loading dataset...")
    dataset = CircuitDataset(dataset_path)

    optimizer = optim.Adam(tf_encoder.parameters(), lr=1e-3)

    print(f"\nTraining TF Encoder for {num_epochs} epochs...")
    print(f"Learning mapping: (poles, zeros) → latent[4:8]")
    print(f"{'='*70}\n")

    tf_encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_kl = 0.0
        total_mse = 0.0

        for i, data in enumerate(dataset):
            data = data.to(device)

            # Get ground truth latent from full encoder (pole/zero part only)
            with torch.no_grad():
                mu_full, logvar_full = full_encoder(data)
                # Extract pole/zero latent (last 4 dimensions)
                pz_latent_gt = mu_full[:, -4:]  # latent[4:8]

            # Get poles/zeros from data
            pole_count = data.pole_count.unsqueeze(0)
            zero_count = data.zero_count.unsqueeze(0)
            pole_values = data.pole_values.unsqueeze(0)
            zero_values = data.zero_values.unsqueeze(0)

            # Encode with TF encoder
            mu_pred, logvar_pred = tf_encoder(
                pole_values, pole_count, zero_values, zero_count
            )

            # Loss: MSE to match ground truth + KL regularization
            mse_loss = F.mse_loss(mu_pred, pz_latent_gt)
            kl_loss = -0.5 * torch.sum(1 + logvar_pred - mu_pred.pow(2) - logvar_pred.exp())
            kl_loss = kl_loss / mu_pred.shape[0]

            loss = mse_loss + 0.01 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tf_encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_mse += mse_loss.item()

        avg_loss = total_loss / len(dataset)
        avg_kl = total_kl / len(dataset)
        avg_mse = total_mse / len(dataset)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KL: {avg_kl:.4f}")

    # Save trained TF encoder
    save_path = 'checkpoints/tf_encoder/best.pt'
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'model_state_dict': tf_encoder.state_dict(),
        'config': {
            'max_poles': 4,
            'max_zeros': 4,
            'hidden_dim': 64,
            'latent_dim': config['model']['encoder']['pz_latent_dim']
        }
    }, save_path)

    print(f"\n✅ TF Encoder trained and saved to: {save_path}")
    return tf_encoder


if __name__ == '__main__':
    # Test TF encoder
    tf_encoder = TransferFunctionEncoder()

    batch_size = 4
    pole_values = torch.randn(batch_size, 4, 2)  # 4 poles with [real, imag]
    pole_count = torch.tensor([2, 3, 1, 2])
    zero_values = torch.randn(batch_size, 4, 2)
    zero_count = torch.tensor([0, 1, 0, 2])

    mu, logvar = tf_encoder(pole_values, pole_count, zero_values, zero_count)

    print(f"✅ TF Encoder test passed")
    print(f"   Input: {pole_count.tolist()} poles, {zero_count.tolist()} zeros")
    print(f"   Output latent mu: {mu.shape}")
    print(f"   Output latent logvar: {logvar.shape}")
