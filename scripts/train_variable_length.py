#!/usr/bin/env python3
"""
Train GraphVAE with variable-length pole/zero decoder.

This script trains the improved model that can predict variable numbers of poles/zeros.

Usage:
    python scripts/train_variable_length.py --config configs/8d_variable_length.yaml
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
from datetime import datetime
import json

from ml.models import HierarchicalEncoder, VariableLengthDecoder
from ml.losses.variable_tf_loss import VariableLengthTransferFunctionLoss
from ml.losses.reconstruction import TemplateAwareReconstructionLoss
from ml.data import CircuitDataset, collate_circuit_batch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train/val/test dataloaders."""
    dataset = CircuitDataset(
        dataset_path=config['data']['dataset_path'],
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(config['data']['train_ratio'] * total_size)
    val_size = int(config['data']['val_ratio'] * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(config['data']['split_seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_circuit_batch,
        num_workers=config['hardware'].get('num_workers', 0),
        pin_memory=config['hardware'].get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=config['hardware'].get('num_workers', 0),
        pin_memory=config['hardware'].get('pin_memory', False)
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    return train_loader, val_loader, dataset


def create_models(config: dict, device: str):
    """Create encoder and decoder models."""
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    decoder = VariableLengthDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        max_poles=config['model'].get('max_poles', 4),
        max_zeros=config['model'].get('max_zeros', 4),
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())

    print(f"\nModel architecture:")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Total parameters:   {encoder_params + decoder_params:,}")

    return encoder, decoder


def create_loss_functions(config: dict, device: str):
    """Create loss functions."""
    # Reconstruction loss
    recon_loss = TemplateAwareReconstructionLoss(
        use_curriculum=config['loss'].get('use_topo_curriculum', False),
        curriculum_warmup_epochs=config['loss'].get('topo_curriculum_warmup_epochs', 20),
        curriculum_initial_multiplier=config['loss'].get('topo_curriculum_initial_multiplier', 3.0)
    )

    # Variable-length transfer function loss
    tf_loss = VariableLengthTransferFunctionLoss(
        pole_count_weight=config['loss'].get('pole_count_weight', 5.0),
        zero_count_weight=config['loss'].get('zero_count_weight', 5.0),
        pole_value_weight=config['loss'].get('pole_value_weight', 0.1),
        zero_value_weight=config['loss'].get('zero_value_weight', 0.1)
    )

    # Loss weights
    recon_weight = config['loss'].get('recon_weight', 1.0)
    tf_weight = config['loss'].get('tf_weight', 1.0)
    kl_weight = config['loss'].get('kl_weight', 0.1)

    # Curriculum for pole/zero weights
    use_pz_curriculum = config['loss'].get('use_pz_weight_curriculum', False)
    pz_warmup_epochs = config['loss'].get('pz_weight_warmup_epochs', 50)

    return recon_loss.to(device), tf_loss.to(device), recon_weight, tf_weight, kl_weight, use_pz_curriculum, pz_warmup_epochs


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim.sum(dim=-1).mean()


def get_pz_weights(epoch: int, config: dict, use_curriculum: bool, warmup_epochs: int):
    """Get pole/zero weights with curriculum."""
    if not use_curriculum:
        return (
            config['loss'].get('pole_count_weight', 5.0),
            config['loss'].get('zero_count_weight', 5.0),
            config['loss'].get('pole_value_weight', 0.1),
            config['loss'].get('zero_value_weight', 0.1)
        )

    # Curriculum: Start structure-focused, move to balanced
    progress = min(1.0, epoch / warmup_epochs)

    # Initial: high count weight, low value weight
    initial_count_weight = 5.0
    initial_value_weight = 0.1

    # Final: balanced weights
    final_count_weight = 1.0
    final_value_weight = 1.0

    # Linear interpolation
    pole_count_weight = initial_count_weight + progress * (final_count_weight - initial_count_weight)
    zero_count_weight = initial_count_weight + progress * (final_count_weight - initial_count_weight)
    pole_value_weight = initial_value_weight + progress * (final_value_weight - initial_value_weight)
    zero_value_weight = initial_value_weight + progress * (final_value_weight - initial_value_weight)

    return pole_count_weight, zero_count_weight, pole_value_weight, zero_value_weight


def train_epoch(
    encoder, decoder, train_loader, optimizer, recon_loss, tf_loss,
    recon_weight, tf_weight, kl_weight, epoch, config, device
):
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    total_loss = 0
    metrics_sum = {}

    # Get PZ weights (with curriculum if enabled)
    use_pz_curriculum = config['loss'].get('use_pz_weight_curriculum', False)
    pz_warmup_epochs = config['loss'].get('pz_weight_warmup_epochs', 50)

    pole_count_w, zero_count_w, pole_value_w, zero_value_w = get_pz_weights(
        epoch, config, use_pz_curriculum, pz_warmup_epochs
    )

    # Update TF loss weights
    tf_loss.pole_count_weight = pole_count_w
    tf_loss.zero_count_weight = zero_count_w
    tf_loss.pole_value_weight = pole_value_w
    tf_loss.zero_value_weight = zero_value_w

    # Set epoch for curriculum
    recon_loss.set_epoch(epoch)

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device (graph is already on device via PyG batching)
        graph = batch['graph'].to(device)
        poles_list = [p.to(device) for p in batch['poles']]
        zeros_list = [z.to(device) for z in batch['zeros']]
        filter_type = batch['filter_type'].to(device)
        edge_attr_targets = graph.edge_attr
        edge_batch = graph.batch[graph.edge_index[0]]  # Edge batch assignment
        num_poles = batch['num_poles'].to(device)
        num_zeros = batch['num_zeros'].to(device)

        # Forward pass
        z, mu, logvar = encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            poles_list,
            zeros_list
        )

        # Teacher forcing during training
        use_teacher_forcing = config['training'].get('use_teacher_forcing', True)
        gt_filter_type = filter_type if use_teacher_forcing else None

        outputs = decoder(mu, hard=False, gt_filter_type=gt_filter_type)

        # Compute losses
        loss_recon, metrics_recon = recon_loss(
            outputs,
            filter_type,
            edge_attr_targets,
            edge_batch,
            graph.edge_index
        )

        loss_tf, metrics_tf = tf_loss(
            outputs,
            poles_list,
            zeros_list,
            num_poles,
            num_zeros
        )

        loss_kl = compute_kl_divergence(mu, logvar)

        # Total loss
        loss = recon_weight * loss_recon + tf_weight * loss_tf + kl_weight * loss_kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config['regularization'].get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                config['regularization']['max_grad_norm']
            )

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in {**metrics_recon, **metrics_tf}.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v

        metrics_sum['kl_loss'] = metrics_sum.get('kl_loss', 0) + loss_kl.item()

    # Average metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    avg_metrics['total_loss'] = avg_loss

    return avg_loss, avg_metrics


def validate(
    encoder, decoder, val_loader, recon_loss, tf_loss,
    recon_weight, tf_weight, kl_weight, epoch, config, device
):
    """Validate the model."""
    encoder.eval()
    decoder.eval()

    total_loss = 0
    metrics_sum = {}

    # Get current PZ weights
    use_pz_curriculum = config['loss'].get('use_pz_weight_curriculum', False)
    pz_warmup_epochs = config['loss'].get('pz_weight_warmup_epochs', 50)

    pole_count_w, zero_count_w, pole_value_w, zero_value_w = get_pz_weights(
        epoch, config, use_pz_curriculum, pz_warmup_epochs
    )

    tf_loss.pole_count_weight = pole_count_w
    tf_loss.zero_count_weight = zero_count_w
    tf_loss.pole_value_weight = pole_value_w
    tf_loss.zero_value_weight = zero_value_w

    recon_loss.set_epoch(epoch)

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            graph = batch['graph'].to(device)
            poles_list = [p.to(device) for p in batch['poles']]
            zeros_list = [z.to(device) for z in batch['zeros']]
            filter_type = batch['filter_type'].to(device)
            edge_attr_targets = graph.edge_attr
            edge_batch = graph.batch[graph.edge_index[0]]
            num_poles = batch['num_poles'].to(device)
            num_zeros = batch['num_zeros'].to(device)

            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles_list,
                zeros_list
            )

            outputs = decoder(mu, hard=False, gt_filter_type=filter_type)

            loss_recon, metrics_recon = recon_loss(
                outputs,
                filter_type,
                edge_attr_targets,
                edge_batch,
                graph.edge_index
            )

            loss_tf, metrics_tf = tf_loss(
                outputs,
                poles_list,
                zeros_list,
                num_poles,
                num_zeros
            )

            loss_kl = compute_kl_divergence(mu, logvar)
            loss = recon_weight * loss_recon + tf_weight * loss_tf + kl_weight * loss_kl

            total_loss += loss.item()
            for k, v in {**metrics_recon, **metrics_tf}.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
            metrics_sum['kl_loss'] = metrics_sum.get('kl_loss', 0) + loss_kl.item()

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    avg_metrics['total_loss'] = avg_loss

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train GraphVAE with variable-length decoder')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n" + "="*70)
    print("VARIABLE-LENGTH GRAPHVAE TRAINING")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create dataloaders
    train_loader, val_loader, dataset = create_dataloaders(config)

    # Create models
    encoder, decoder = create_models(config, device)

    # Create loss functions
    recon_loss, tf_loss, recon_weight, tf_weight, kl_weight, use_pz_curriculum, pz_warmup_epochs = create_loss_functions(config, device)

    # Create optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(
        params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )

    # Create scheduler
    if config['training'].get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('T_0', 50),
            T_mult=config['training'].get('T_mult', 2),
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []

    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")

    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer,
            recon_loss, tf_loss, recon_weight, tf_weight, kl_weight,
            epoch, config, device
        )

        # Validate
        if (epoch + 1) % config['training'].get('val_interval', 1) == 0:
            val_loss, val_metrics = validate(
                encoder, decoder, val_loader,
                recon_loss, tf_loss, recon_weight, tf_weight, kl_weight,
                epoch, config, device
            )

            # Log
            if (epoch + 1) % config['training'].get('log_interval', 5) == 0:
                print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"    Recon: {train_metrics.get('recon_loss', 0):.4f}, "
                      f"TF: {train_metrics.get('tf_loss', 0):.4f}, "
                      f"KL: {train_metrics.get('kl_loss', 0):.4f}")
                print(f"    Pole Count Acc: {train_metrics.get('pole_count_acc', 0):.2%}, "
                      f"Zero Count Acc: {train_metrics.get('zero_count_acc', 0):.2%}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"    Pole Count Acc: {val_metrics.get('pole_count_acc', 0):.2%}, "
                      f"Zero Count Acc: {val_metrics.get('zero_count_acc', 0):.2%}")

            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, checkpoint_dir / 'best.pt')

                print(f"  ✅ New best model saved! (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config['training'].get('early_stopping_patience', 30):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Step scheduler
        if scheduler:
            scheduler.step()

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': val_loss,
        'config': config
    }, checkpoint_dir / 'final.pt')

    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
