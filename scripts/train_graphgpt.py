"""
Training script for GraphGPT-style autoregressive circuit generation.

Much simpler than diffusion training:
- No timestep sampling
- No noise injection
- Standard teacher forcing with cross-entropy
- No focal loss needed
- No gradient explosion issues

Expected improvements:
- Edge generation: 0.13 (diffusion) -> 0.45+ (GraphGPT)
- Training stability: NaN @epoch 76 -> Stable through 200 epochs
- Sampling speed: 50 steps -> 1 step
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.losses.graphgpt_loss import GraphGPTCircuitLoss
from ml.data.dataset import CircuitDataset, collate_graphgpt_batch
# from scripts.validate_edge_generation import EdgeGenerationValidator  # Disabled


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_models(config, device):
    """Create encoder and decoder models."""
    print("\n🏗️  Creating models...")

    # Load pretrained encoder
    encoder = HierarchicalEncoder(
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

    # Load pretrained weights
    if 'pretrained_encoder' in config['checkpoint']:
        checkpoint_path = config['checkpoint']['pretrained_encoder']
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading pretrained encoder from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("✅ Encoder loaded successfully")
        else:
            if checkpoint_path is None:
                print("✅ Training from scratch with random initialization (new normalization)")
            else:
                print(f"⚠️  Warning: Encoder checkpoint not found at {checkpoint_path}")
                print("   Starting with random initialization")

    # Create GraphGPT decoder
    decoder = GraphGPTDecoder(
        latent_dim=config['model']['decoder']['latent_dim'],
        conditions_dim=config['model']['decoder']['conditions_dim'],
        hidden_dim=config['model']['decoder']['hidden_dim'],
        num_heads=config['model']['decoder']['num_heads'],
        num_node_layers=config['model']['decoder']['num_node_layers'],
        max_nodes=config['model']['decoder']['max_nodes'],
        max_poles=config['model']['decoder']['max_poles'],
        max_zeros=config['model']['decoder']['max_zeros'],
        dropout=config['model']['decoder']['dropout']
    ).to(device)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())

    print("✅ Models created")
    print(f"   Encoder parameters:  {encoder_params:,}")
    print(f"   Decoder parameters:  {decoder_params:,}")
    print(f"   Total parameters:    {encoder_params + decoder_params:,}")

    return encoder, decoder


def train_epoch(
    encoder, decoder, dataloader, optimizer, loss_fn, device, config, epoch,
    freeze_encoder=True, edge_validator=None
):
    """Train for one epoch."""
    encoder.train() if not freeze_encoder else encoder.eval()
    decoder.train()

    total_loss = 0
    num_batches = 0
    metrics_sum = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        def move_to_device(v):
            if isinstance(v, torch.Tensor):
                return v.to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                return [t.to(device) for t in v]
            else:
                return v

        batch = {k: move_to_device(v) for k, v in batch.items()}

        # === Forward Pass ===

        # Encode circuit to latent
        if freeze_encoder:
            with torch.no_grad():
                z, mu, logvar = encoder(
                    batch['node_features'],
                    batch['edge_index'],
                    batch['edge_attr'],
                    batch['batch_idx'],
                    batch['poles_list'],
                    batch['zeros_list']
                )
                latent = mu  # Use mean for deterministic encoding
        else:
            z, mu, logvar = encoder(
                batch['node_features'],
                batch['edge_index'],
                batch['edge_attr'],
                batch['batch_idx'],
                batch['poles_list'],
                batch['zeros_list']
            )
            latent = z  # Use sampled latent during joint training

        # Prepare conditions (cutoff frequency, Q-factor)
        conditions = batch['specifications']  # [batch, 2]

        # Decoder forward pass with teacher forcing
        predictions = decoder(
            latent_code=latent,
            conditions=conditions,
            target_nodes=batch['node_types'],
            target_edges_exist=batch['edge_existence'],
            target_edges_values=batch['edge_values'],
            teacher_forcing=True
        )

        # === Compute Loss ===

        targets = {
            'node_types': batch['node_types'],
            'edge_existence': batch['edge_existence'],
            'edge_values': batch['edge_values'],
            'pole_count': batch['pole_count'],
            'zero_count': batch['zero_count'],
            'pole_values': batch['pole_values'],
            'zero_values': batch['zero_values']
        }

        loss, metrics = loss_fn(predictions, targets)

        # Add KL divergence if training encoder
        if not freeze_encoder:
            kl_weight = config['training'].get('kl_weight', 0.01)
            # KL divergence: KL(N(mu, var) || N(0, 1))
            # = -0.5 * sum(1 + log(var) - mu^2 - var)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            loss = loss + kl_weight * kl_div
            metrics['kl_div'] = kl_div.item()

        # === Backward Pass ===

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_clip = config['training']['grad_clip']
        if grad_clip > 0:
            if freeze_encoder:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            else:
                params = list(encoder.parameters()) + list(decoder.parameters())
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

        optimizer.step()

        # === Accumulate Metrics ===

        total_loss += loss.item()
        num_batches += 1

        for key, value in metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value

        # === Edge Generation Monitoring ===

        if edge_validator is not None:
            check_freq = config.get('edge_monitoring', {}).get('check_every_n_batches', 10)
            if batch_idx % check_freq == 0:
                edge_validator.validate_training_batch(predictions, epoch, batch_idx)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'node_acc': f"{metrics['node_type_acc']:.1f}%"
        })

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def validate(encoder, decoder, dataloader, loss_fn, device, freeze_encoder=True):
    """Validate the model."""
    encoder.eval()
    decoder.eval()

    total_loss = 0
    num_batches = 0
    metrics_sum = {}

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            def move_to_device(v):
                if isinstance(v, torch.Tensor):
                    return v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return [t.to(device) for t in v]
                else:
                    return v

            batch = {k: move_to_device(v) for k, v in batch.items()}

            # Encode
            z, mu, logvar = encoder(
                batch['node_features'],
                batch['edge_index'],
                batch['edge_attr'],
                batch['batch_idx'],
                batch['poles_list'],
                batch['zeros_list']
            )
            latent = mu  # Use mean for validation

            # Decode with teacher forcing
            predictions = decoder(
                latent_code=latent,
                conditions=batch['specifications'],
                target_nodes=batch['node_types'],
                target_edges_exist=batch['edge_existence'],
                target_edges_values=batch['edge_values'],
                teacher_forcing=True
            )

            # Compute loss
            targets = {
                'node_types': batch['node_types'],
                'edge_existence': batch['edge_existence'],
                'edge_values': batch['edge_values'],
                'pole_count': batch['pole_count'],
                'zero_count': batch['zero_count'],
                'pole_values': batch['pole_values'],
                'zero_values': batch['zero_values']
            }

            loss, metrics = loss_fn(predictions, targets)

            total_loss += loss.item()
            num_batches += 1

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0) + value

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def main(args):
    """Main training loop."""
    # Load configuration
    config = load_config(args.config)

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"GraphGPT Circuit Generation Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # Create dataset
    print("\n📂 Loading dataset...")
    dataset = CircuitDataset(config['data']['dataset_path'])

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"✅ Dataset loaded")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Validation: {len(val_dataset)}")

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_graphgpt_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphgpt_batch
    )

    # Create models
    encoder, decoder = create_models(config, device)

    # Create loss function
    loss_fn = GraphGPTCircuitLoss(
        node_type_weight=config['loss']['node_type_weight'],
        edge_exist_weight=config['loss']['edge_exist_weight'],
        edge_value_weight=config['loss']['edge_value_weight'],
        pole_count_weight=config['loss']['pole_count_weight'],
        zero_count_weight=config['loss']['zero_count_weight'],
        pole_value_weight=config['loss']['pole_value_weight'],
        zero_value_weight=config['loss']['zero_value_weight']
    )

    # Create edge validator
    edge_validator = None
    if config.get('edge_monitoring', {}).get('enabled', False):
        edge_validator = EdgeGenerationValidator(
            warning_threshold=config['edge_monitoring'].get('warning_threshold', 0.2),
            critical_threshold=config['edge_monitoring'].get('critical_threshold', 0.05)
        )
        print("\n✅ Edge generation monitoring enabled")

    # Create optimizer
    phase1_epochs = config['training']['phase1_epochs']
    phase2_epochs = config['training']['phase2_epochs']
    total_epochs = config['training']['total_epochs']

    # Training loop
    print(f"\n🚀 Starting training for {total_epochs} epochs")
    print(f"   Phase 1 (freeze encoder): Epochs 1-{phase1_epochs}")
    print(f"   Phase 2 (joint training): Epochs {phase1_epochs+1}-{total_epochs}")

    best_val_loss = float('inf')
    save_dir = config['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*70}")

        # Determine training phase
        if epoch <= phase1_epochs:
            freeze_encoder = True
            lr = config['training']['learning_rate_phase1']
            print("Training mode: Phase 1 (Freeze Encoder)")
        else:
            freeze_encoder = False
            lr = config['training']['learning_rate_phase2']
            print("Training mode: Phase 2 (Joint Training)")

        # Create optimizer for this phase
        if freeze_encoder:
            optimizer = torch.optim.AdamW(
                decoder.parameters(),
                lr=lr,
                weight_decay=config['training']['weight_decay']
            )
        else:
            params = list(encoder.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=config['training']['weight_decay']
            )

        # Train
        train_loss, train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, loss_fn, device, config, epoch,
            freeze_encoder=freeze_encoder,
            edge_validator=edge_validator
        )

        # Validate
        val_loss, val_metrics = validate(
            encoder, decoder, val_loader, loss_fn, device,
            freeze_encoder=freeze_encoder
        )

        # Print results
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"   Train Metrics:")
        print(f"      Node Type Acc:    {train_metrics['node_type_acc']:.1f}%")
        print(f"      Pole Count Acc:   {train_metrics['pole_count_acc']:.1f}%")
        print(f"      Zero Count Acc:   {train_metrics['zero_count_acc']:.1f}%")
        print(f"      Edge Exist Acc:   {train_metrics['edge_exist_acc']:.1f}%")
        print(f"   Val Metrics:")
        print(f"      Node Type Acc:    {val_metrics['node_type_acc']:.1f}%")
        print(f"      Pole Count Acc:   {val_metrics['pole_count_acc']:.1f}%")
        print(f"      Zero Count Acc:   {val_metrics['zero_count_acc']:.1f}%")
        print(f"      Edge Exist Acc:   {val_metrics['edge_exist_acc']:.1f}%")

        # Print edge generation summary
        if edge_validator is not None:
            edge_validator.print_epoch_summary(epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best.pt'))
            print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every N epochs
        if epoch % config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Saved checkpoint: epoch_{epoch}.pt")

    print(f"\n{'='*70}")
    print("✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GraphGPT circuit generation model')
    parser.add_argument('--config', type=str, default='configs/graphgpt_decoder.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')

    args = parser.parse_args()
    main(args)
