#!/usr/bin/env python3
"""
Training script for Conditional Variable-Length VAE.

This extends the standard VAE to accept conditions (specifications) as input,
enabling direct generation from desired circuit characteristics like cutoff
frequency and Q factor.

Key features:
- Conditions: [cutoff_frequency, q_factor]
- Compositional generation: Mix conditions from different circuits
- No reference circuit needed for generation
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.conditional_encoder import ConditionalHierarchicalEncoder
from ml.models.conditional_decoder import ConditionalVariableLengthDecoder
from ml.data.dataset import CircuitDataset, collate_circuit_batch
from ml.losses.variable_tf_loss import VariableLengthTransferFunctionLoss
from ml.utils.condition_utils import ConditionExtractor, compute_condition_statistics
from torch.utils.data import DataLoader, random_split


def create_models(config, condition_stats):
    """Create conditional encoder and decoder."""
    model_config = config['model']

    encoder = ConditionalHierarchicalEncoder(
        node_feature_dim=model_config.get('node_feature_dim', 4),
        edge_feature_dim=model_config.get('edge_feature_dim', 7),
        gnn_hidden_dim=model_config['gnn_hidden_dim'],
        gnn_num_layers=model_config.get('gnn_num_layers', 3),
        latent_dim=model_config['latent_dim'],
        topo_latent_dim=model_config['topo_latent_dim'],
        values_latent_dim=model_config['values_latent_dim'],
        pz_latent_dim=model_config['pz_latent_dim'],
        dropout=model_config.get('dropout', 0.1),
        # Conditional parameters
        conditions_dim=model_config.get('conditions_dim', 2),
        condition_embed_dim=model_config.get('condition_embed_dim', 64)
    )

    decoder = ConditionalVariableLengthDecoder(
        latent_dim=model_config['latent_dim'],
        hidden_dim=model_config.get('decoder_hidden_dim', 64),
        topo_latent_dim=model_config['topo_latent_dim'],
        values_latent_dim=model_config['values_latent_dim'],
        pz_latent_dim=model_config['pz_latent_dim'],
        max_nodes=model_config.get('max_nodes', 5),
        max_edges=model_config.get('max_edges', 10),
        max_poles=model_config['max_poles'],
        max_zeros=model_config['max_zeros'],
        num_filter_types=6,
        # Conditional parameters
        conditions_dim=model_config.get('conditions_dim', 2)
    )

    # Create condition extractor
    condition_extractor = ConditionExtractor(
        normalize=True,
        condition_stats=condition_stats
    )

    return encoder, decoder, condition_extractor


def extract_batch_conditions(batch, circuits_list, condition_extractor):
    """
    Extract conditions for a batch of circuits.

    Args:
        batch: Batch dictionary from DataLoader
        circuits_list: List of all circuits (to get original data)
        condition_extractor: ConditionExtractor instance

    Returns:
        conditions: Tensor [B, conditions_dim]
    """
    batch_size = len(batch['circuit_id'])
    conditions_list = []

    for i in range(batch_size):
        circuit_id = batch['circuit_id'][i]
        # Find circuit in dataset
        circuit = next(c for c in circuits_list if c['id'] == circuit_id)

        # Extract poles/zeros
        poles = circuit['label']['poles']
        zeros = circuit['label']['zeros']
        gain = circuit['label']['gain']
        filter_type = circuit['filter_type']

        # Convert to arrays
        if len(poles) > 0:
            poles_array = np.array([[p.real, p.imag] for p in poles])
        else:
            poles_array = np.zeros((0, 2))

        if len(zeros) > 0:
            zeros_array = np.array([[z.real, z.imag] for z in zeros])
        else:
            zeros_array = np.zeros((0, 2))

        # Extract and normalize conditions
        conditions_dict = condition_extractor.extract_conditions(
            poles_array, zeros_array, gain, filter_type
        )
        conditions_tensor = condition_extractor.normalize_conditions(conditions_dict)
        conditions_list.append(conditions_tensor)

    # Stack into batch tensor
    conditions = torch.stack(conditions_list)  # [B, conditions_dim]

    return conditions


def train_epoch(encoder, decoder, dataloader, optimizer, loss_fn, condition_extractor, circuits_list, device, config, kl_weight):
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    total_loss = 0
    total_metrics = {
        'recon_loss': 0,
        'kl_loss': 0,
        'topo_acc': 0,
        'pole_count_acc': 0,
        'zero_count_acc': 0
    }

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Extract conditions for this batch
        conditions = extract_batch_conditions(batch, circuits_list, condition_extractor)
        conditions = conditions.to(device)

        # Move batch to device
        batch_graph = batch['graph'].to(device)
        poles_list = batch['poles']
        zeros_list = batch['zeros']
        filter_type = batch['filter_type'].to(device)
        num_poles = batch['num_poles'].to(device)
        num_zeros = batch['num_zeros'].to(device)

        # Forward pass with conditions
        z, mu, logvar = encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr,
            batch_graph.batch,
            poles_list,
            zeros_list,
            conditions=conditions  # Pass conditions to encoder
        )

        outputs = decoder(mu, hard=False, gt_filter_type=filter_type, conditions=conditions)

        # Compute reconstruction loss
        recon_loss, metrics = loss_fn(
            outputs=outputs,
            target_poles_list=poles_list,
            target_zeros_list=zeros_list,
            target_num_poles=num_poles,
            target_num_zeros=num_zeros
        )

        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        metrics['kl_loss'] = kl_loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def validate(encoder, decoder, dataloader, loss_fn, condition_extractor, circuits_list, device, kl_weight):
    """Validate model."""
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_metrics = {
        'recon_loss': 0,
        'kl_loss': 0,
        'topo_acc': 0,
        'pole_count_acc': 0,
        'zero_count_acc': 0
    }

    for batch in dataloader:
        # Extract conditions
        conditions = extract_batch_conditions(batch, circuits_list, condition_extractor)
        conditions = conditions.to(device)

        # Move to device
        batch_graph = batch['graph'].to(device)
        poles_list = batch['poles']
        zeros_list = batch['zeros']
        filter_type = batch['filter_type'].to(device)
        num_poles = batch['num_poles'].to(device)
        num_zeros = batch['num_zeros'].to(device)

        # Forward
        z, mu, logvar = encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr,
            batch_graph.batch,
            poles_list,
            zeros_list,
            conditions=conditions
        )

        outputs = decoder(mu, hard=True, gt_filter_type=filter_type, conditions=conditions)

        # Compute reconstruction loss
        recon_loss, metrics = loss_fn(
            outputs=outputs,
            target_poles_list=poles_list,
            target_zeros_list=zeros_list,
            target_num_poles=num_poles,
            target_num_zeros=num_zeros
        )

        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        metrics['kl_loss'] = kl_loss.item()

        total_loss += loss.item()
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Conditional VAE')
    parser.add_argument('--config', type=str, default='configs/8d_conditional_vae.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/mps/cuda)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*70}")
    print("CONDITIONAL VAE TRAINING")
    print(f"{'='*70}\n")

    # Load dataset
    dataset = CircuitDataset(dataset_path='rlc_dataset/filter_dataset.pkl')

    # Compute condition statistics
    print("\n⏳ Computing condition statistics...")
    condition_stats = compute_condition_statistics(dataset)

    # Save condition stats
    output_dir = Path('checkpoints/conditional_vae')
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / 'condition_stats.json', 'w') as f:
        json.dump(condition_stats, f, indent=2)

    print(f"💾 Saved condition statistics to {output_dir}/condition_stats.json")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_circuit_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch
    )

    # Create models
    device = args.device
    encoder, decoder, condition_extractor = create_models(config, condition_stats)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Print model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Model Architecture:")
    print(f"   Encoder parameters:  {encoder_params:,}")
    print(f"   Decoder parameters:  {decoder_params:,}")
    print(f"   Total parameters:    {encoder_params + decoder_params:,}")

    # Create loss function
    loss_fn = VariableLengthTransferFunctionLoss(
        pole_count_weight=config['loss']['pole_count_weight'],
        zero_count_weight=config['loss']['zero_count_weight'],
        pole_value_weight=config['loss']['pole_value_weight'],
        zero_value_weight=config['loss']['zero_value_weight']
    )

    # Additional loss weights (to be applied manually)
    kl_weight = config['loss'].get('kl_weight', 0.001)

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config['training']['learning_rate']
    )

    # Training loop
    best_val_loss = float('inf')
    epochs = config['training']['epochs']

    print(f"\n🚀 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        # Train
        train_loss, train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer,
            loss_fn, condition_extractor, dataset.circuits, device, config, kl_weight
        )

        # Validate
        val_loss, val_metrics = validate(
            encoder, decoder, val_loader,
            loss_fn, condition_extractor, dataset.circuits, device, kl_weight
        )

        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"   Train Metrics:")
        print(f"      Topology Acc:     {train_metrics.get('topo_acc', 0):.1f}%")
        print(f"      Pole Count Acc:   {train_metrics.get('pole_count_acc', 0):.1f}%")
        print(f"      Zero Count Acc:   {train_metrics.get('zero_count_acc', 0):.1f}%")
        print(f"   Val Metrics:")
        print(f"      Topology Acc:     {val_metrics.get('topo_acc', 0):.1f}%")
        print(f"      Pole Count Acc:   {val_metrics.get('pole_count_acc', 0):.1f}%")
        print(f"      Zero Count Acc:   {val_metrics.get('zero_count_acc', 0):.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = output_dir / 'best.pt'
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'condition_stats': condition_stats,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)

            print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 Best model saved to: {output_dir}/best.pt")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
