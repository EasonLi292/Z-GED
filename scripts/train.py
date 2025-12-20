#!/usr/bin/env python3
"""
Main training script for GraphVAE.

Train the hierarchical VAE model on circuit dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import argparse
from datetime import datetime

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.losses import SimplifiedCompositeLoss
from ml.data import CircuitDataset, collate_circuit_batch
from ml.training.trainer import VAETrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train, val, test dataloaders."""
    # Load dataset
    dataset = CircuitDataset(
        dataset_path=config['data']['dataset_path'],
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    # Get stratified split
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        seed=config['data']['split_seed']
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_idx)} circuits")
    print(f"  Val:   {len(val_idx)} circuits")
    print(f"  Test:  {len(test_idx)} circuits")

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_circuit_batch,
        num_workers=0  # Set to 0 for debugging, increase for performance
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, dataset


def create_models(config: dict, device: str):
    """Create encoder and decoder models."""
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout']
    )

    decoder = HybridDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout']
    )

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Print model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params

    print(f"\nModel architecture:")
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Decoder params: {decoder_params:,}")
    print(f"  Total params:   {total_params:,}")

    return encoder, decoder


def create_loss_function(config: dict, device: str):
    """Create loss function."""
    # Extract curriculum parameters (with defaults)
    use_curriculum = config['loss'].get('use_topo_curriculum', False)
    warmup_epochs = config['loss'].get('topo_curriculum_warmup_epochs', 20)
    initial_multiplier = config['loss'].get('topo_curriculum_initial_multiplier', 3.0)

    loss_fn = SimplifiedCompositeLoss(
        recon_weight=config['loss']['recon_weight'],
        tf_weight=config['loss']['tf_weight'],
        kl_weight=config['loss']['kl_weight'],
        use_topo_curriculum=use_curriculum,
        topo_curriculum_warmup_epochs=warmup_epochs,
        topo_curriculum_initial_multiplier=initial_multiplier
    )

    return loss_fn.to(device)


def create_optimizer(encoder, decoder, config: dict):
    """Create optimizer."""
    params = list(encoder.parameters()) + list(decoder.parameters())

    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")

    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler."""
    scheduler_type = config['training'].get('scheduler', None)

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('scheduler_step', 50),
            gamma=config['training'].get('scheduler_gamma', 0.5)
        )
    elif scheduler_type == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('T_0', 50),
            T_mult=config['training'].get('T_mult', 2)
        )
    else:
        scheduler = None

    return scheduler


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GraphVAE model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Determine device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n" + "="*70)
    print("GRAPHVAE TRAINING")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print("="*70)

    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(config)

    # Create models
    encoder, decoder = create_models(config, device)

    # Create loss function
    loss_fn = create_loss_function(config, device)

    # Create optimizer
    optimizer = create_optimizer(encoder, decoder, config)

    # Create scheduler
    scheduler = create_scheduler(optimizer, config)

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config to checkpoint dir
    config_save_path = checkpoint_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")

    # Create trainer
    trainer = VAETrainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=config['training'].get('log_interval', 10),
        val_interval=config['training'].get('val_interval', 1)
    )

    # Load checkpoint if provided
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train(
        num_epochs=config['training']['epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 20)
    )

    print("\n✅ Training complete!")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"Training history: {checkpoint_dir / 'training_history.json'}")


if __name__ == '__main__':
    main()
