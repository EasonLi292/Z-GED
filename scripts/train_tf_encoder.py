"""
Train Transfer Function Encoder.

This learns the mapping: (poles, zeros) → latent[4:8]
by distilling knowledge from the pretrained full encoder.

Usage:
    python scripts/train_tf_encoder.py \\
        --pretrained-encoder checkpoints/graphgpt_decoder/best.pt \\
        --dataset rlc_dataset/filter_dataset.pkl \\
        --epochs 100 \\
        --device mps
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.tf_encoder import TransferFunctionEncoder
from ml.models.encoder import HierarchicalEncoder
from ml.data.dataset import CircuitDataset


def train_epoch(tf_encoder, full_encoder, dataset_indices, dataset, optimizer, device):
    """Train for one epoch."""
    tf_encoder.train()
    full_encoder.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    num_samples = 0

    for idx in dataset_indices:
        sample = dataset[idx]
        graph = sample['graph'].to(device)

        # Get ground truth latent from full encoder (pole/zero part only)
        with torch.no_grad():
            # Prepare encoder inputs
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)  # Single graph

            poles_tensor = torch.tensor(sample['poles'], dtype=torch.float32, device=device)
            zeros_tensor = torch.tensor(sample['zeros'], dtype=torch.float32, device=device)
            poles_list = [poles_tensor]
            zeros_list = [zeros_tensor]

            _, mu_full, logvar_full = full_encoder(
                x, edge_index, edge_attr, batch, poles_list, zeros_list
            )
            # Extract pole/zero latent (last 4 dimensions)
            pz_latent_gt = mu_full[0, -4:]  # [4] - first sample, last 4 dims

        # Get poles/zeros from sample dict (add batch dimension)
        pole_count = torch.tensor([sample['num_poles']], device=device)  # [1]
        zero_count = torch.tensor([sample['num_zeros']], device=device)  # [1]

        # Pad poles/zeros to max length
        poles = torch.tensor(sample['poles'], dtype=torch.float32, device=device)  # [num_poles, 2]
        zeros = torch.tensor(sample['zeros'], dtype=torch.float32, device=device)  # [num_zeros, 2]

        pole_values = torch.zeros(1, 4, 2, device=device)
        zero_values = torch.zeros(1, 4, 2, device=device)

        if poles.shape[0] > 0:
            pole_values[0, :poles.shape[0], :] = poles[:4]  # Limit to max 4
        if zeros.shape[0] > 0:
            zero_values[0, :zeros.shape[0], :] = zeros[:4]  # Limit to max 4

        # Encode with TF encoder
        mu_pred, logvar_pred = tf_encoder(
            pole_values, pole_count, zero_values, zero_count
        )

        # Remove batch dimension for loss computation
        mu_pred = mu_pred.squeeze(0)  # [4]
        logvar_pred = logvar_pred.squeeze(0)  # [4]

        # Loss: MSE to match ground truth + KL regularization
        mse_loss = F.mse_loss(mu_pred, pz_latent_gt)

        # KL divergence (regularization to keep distribution reasonable)
        kl_loss = -0.5 * torch.sum(1 + logvar_pred - mu_pred.pow(2) - logvar_pred.exp())

        loss = mse_loss + 0.01 * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tf_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_kl += kl_loss.item()
        num_samples += 1

    return {
        'loss': total_loss / num_samples,
        'mse': total_mse / num_samples,
        'kl': total_kl / num_samples
    }


def validate(tf_encoder, full_encoder, dataset_indices, dataset, device):
    """Validate the TF encoder."""
    tf_encoder.eval()
    full_encoder.eval()

    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx in dataset_indices:
            sample = dataset[idx]
            graph = sample['graph'].to(device)

            # Get ground truth
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            poles_tensor = torch.tensor(sample['poles'], dtype=torch.float32, device=device)
            zeros_tensor = torch.tensor(sample['zeros'], dtype=torch.float32, device=device)
            poles_list = [poles_tensor]
            zeros_list = [zeros_tensor]

            _, mu_full, _ = full_encoder(
                x, edge_index, edge_attr, batch, poles_list, zeros_list
            )
            pz_latent_gt = mu_full[0, -4:]

            # Predict
            pole_count = torch.tensor([sample['num_poles']], device=device)
            zero_count = torch.tensor([sample['num_zeros']], device=device)

            poles = torch.tensor(sample['poles'], dtype=torch.float32, device=device)
            zeros = torch.tensor(sample['zeros'], dtype=torch.float32, device=device)

            pole_values = torch.zeros(1, 4, 2, device=device)
            zero_values = torch.zeros(1, 4, 2, device=device)

            if poles.shape[0] > 0:
                pole_values[0, :poles.shape[0], :] = poles[:4]
            if zeros.shape[0] > 0:
                zero_values[0, :zeros.shape[0], :] = zeros[:4]

            mu_pred, _ = tf_encoder(
                pole_values, pole_count, zero_values, zero_count
            )

            mu_pred = mu_pred.squeeze(0)
            mse_loss = F.mse_loss(mu_pred, pz_latent_gt)

            total_mse += mse_loss.item()
            num_samples += 1

    return {
        'mse': total_mse / num_samples
    }


def main(args):
    """Main training function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("Transfer Function Encoder Training")
    print(f"{'='*70}\n")

    # Load pretrained full encoder
    print(f"📂 Loading pretrained encoder from: {args.pretrained_encoder}")
    checkpoint = torch.load(args.pretrained_encoder, map_location=device)
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
    print(f"✅ Encoder loaded (epoch {checkpoint['epoch']})")

    # Create TF encoder
    print(f"\n🔨 Creating TF encoder...")
    tf_encoder = TransferFunctionEncoder(
        max_poles=config['model']['decoder']['max_poles'],
        max_zeros=config['model']['decoder']['max_zeros'],
        hidden_dim=args.hidden_dim,
        latent_dim=config['model']['encoder']['pz_latent_dim']  # Should be 4
    ).to(device)
    print(f"✅ TF encoder created (latent_dim={config['model']['encoder']['pz_latent_dim']})")

    # Load dataset
    print(f"\n📂 Loading dataset from: {args.dataset}")
    dataset = CircuitDataset(args.dataset)

    # Split into train/val indices
    import random

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"✅ Dataset loaded: {train_size} train, {val_size} val")

    # Optimizer
    optimizer = optim.Adam(tf_encoder.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training for {args.epochs} epochs")
    print(f"Learning mapping: (poles, zeros) → latent[4:8]")
    print(f"{'='*70}\n")

    best_val_mse = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Shuffle train indices each epoch
        random.shuffle(train_indices)

        # Train
        train_metrics = train_epoch(
            tf_encoder, full_encoder, train_indices, dataset, optimizer, device
        )

        # Validate
        val_metrics = validate(
            tf_encoder, full_encoder, val_indices, dataset, device
        )

        # Log
        if (epoch + 1) % args.log_frequency == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(MSE: {train_metrics['mse']:.4f}, KL: {train_metrics['kl']:.4f}) | "
                  f"Val MSE: {val_metrics['mse']:.4f}")

        # Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']

            save_path = os.path.join(save_dir, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_mse,
                'config': {
                    'max_poles': config['model']['decoder']['max_poles'],
                    'max_zeros': config['model']['decoder']['max_zeros'],
                    'hidden_dim': args.hidden_dim,
                    'latent_dim': config['model']['encoder']['pz_latent_dim']
                }
            }, save_path)

            if (epoch + 1) % args.log_frequency == 0:
                print(f"  ✅ Best model saved (val_mse: {best_val_mse:.4f})")

    # Final summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation MSE: {best_val_mse:.4f}")
    print(f"Model saved to: {save_dir}/best.pt")
    print(f"\nTo use for targeted generation:")
    print(f"  python scripts/generate_targeted_tf.py \\")
    print(f"    --tf-encoder-checkpoint {save_dir}/best.pt \\")
    print(f"    --poles \"(-1000+2000j)\" \"(-1000-2000j)\" \\")
    print(f"    --zeros \"(-500+0j)\" \\")
    print(f"    --cutoff 1000 --q-factor 0.707 --num-samples 10")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TF Encoder')

    # Data
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl')
    parser.add_argument('--pretrained-encoder', type=str,
                       default='checkpoints/graphgpt_decoder/best.pt')
    parser.add_argument('--train-split', type=float, default=0.8)

    # Model
    parser.add_argument('--hidden-dim', type=int, default=64)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    # Logging
    parser.add_argument('--log-frequency', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='checkpoints/tf_encoder')

    # Device
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()
    main(args)
