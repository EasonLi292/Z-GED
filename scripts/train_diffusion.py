#!/usr/bin/env python3
"""
Training script for Diffusion-based Circuit Decoder.

Implements two-phase training:
1. Phase 1: Freeze encoder, train diffusion decoder only
2. Phase 2: Joint fine-tuning of encoder and decoder

Key features:
- Loads pretrained hierarchical encoder
- Trains DiffusionGraphTransformer
- Loss curriculum for transfer function loss
- Time-dependent loss weighting
- Comprehensive metrics tracking
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.encoder import HierarchicalEncoder
from ml.models.diffusion import DiffusionGraphTransformer
from ml.models.diffusion.forward_process import CircuitForwardDiffusion
from ml.losses.diffusion_loss import DiffusionCircuitLoss
from ml.losses.structure_loss import CombinedStructuralLoss
from ml.data.dataset import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader, random_split


def load_pretrained_encoder(checkpoint_path: str, config: dict, device: str):
    """
    Load pretrained encoder from checkpoint.

    Args:
        checkpoint_path: Path to encoder checkpoint
        config: Model configuration
        device: Device to load model on

    Returns:
        encoder: Loaded encoder model
    """
    print(f"\n📥 Loading pretrained encoder from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create encoder
    encoder_config = config['model']['encoder']
    encoder = HierarchicalEncoder(
        node_feature_dim=encoder_config['node_feature_dim'],
        edge_feature_dim=encoder_config['edge_feature_dim'],
        gnn_hidden_dim=encoder_config['gnn_hidden_dim'],
        gnn_num_layers=encoder_config['gnn_num_layers'],
        latent_dim=encoder_config['latent_dim'],
        topo_latent_dim=encoder_config['topo_latent_dim'],
        values_latent_dim=encoder_config['values_latent_dim'],
        pz_latent_dim=encoder_config['pz_latent_dim'],
        dropout=encoder_config.get('dropout', 0.1)
    )

    # Load state dict
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)

    print(f"✅ Loaded encoder from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")

    return encoder


def create_diffusion_decoder(config: dict, device: str):
    """
    Create diffusion decoder.

    Args:
        config: Model configuration
        device: Device

    Returns:
        decoder: DiffusionGraphTransformer
    """
    decoder_config = config['model']['decoder']

    decoder = DiffusionGraphTransformer(
        hidden_dim=decoder_config['hidden_dim'],
        num_layers=decoder_config['num_layers'],
        num_heads=decoder_config['num_heads'],
        latent_dim=decoder_config['latent_dim'],
        conditions_dim=decoder_config['conditions_dim'],
        max_nodes=decoder_config['max_nodes'],
        max_poles=decoder_config['max_poles'],
        max_zeros=decoder_config['max_zeros'],
        dropout=decoder_config['dropout'],
        timesteps=decoder_config['timesteps']
    )

    decoder = decoder.to(device)

    return decoder


def prepare_batch_for_diffusion(batch, device):
    """
    Convert batch data to format needed for diffusion training.

    Args:
        batch: Batch from DataLoader
        device: Device

    Returns:
        clean_circuit: Dictionary with clean circuit data
        graph_data: Graph data for encoder
    """
    # Move batch to device
    batch_graph = batch['graph'].to(device)
    poles_list = [p.to(device) for p in batch['poles']]
    zeros_list = [z.to(device) for z in batch['zeros']]
    num_poles = batch['num_poles'].to(device)
    num_zeros = batch['num_zeros'].to(device)
    filter_type = batch['filter_type'].to(device)

    batch_size = len(poles_list)
    max_nodes = 5
    max_poles = 4
    max_zeros = 4

    # ==================================================================
    # Prepare clean circuit for diffusion
    # ==================================================================

    # Node types (extract from graph)
    # This is a simplified version - in practice you'd extract actual node types
    node_types = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
    # For now, use a simple heuristic: GND, VIN, VOUT at positions 0,1,2
    node_types[:, 0] = 0  # GND
    node_types[:, 1] = 1  # VIN
    node_types[:, 2] = 2  # VOUT
    # Rest are INTERNAL or MASK
    node_types[:, 3:] = 3  # INTERNAL

    # Edge existence and values (extract from graph)
    edge_existence = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    edge_values = torch.zeros(batch_size, max_nodes, max_nodes, 7, device=device)

    # Simple heuristic: create basic connectivity
    # In practice, this should be extracted from batch_graph
    for b in range(batch_size):
        # Add some edges
        edge_existence[b, 0, 1] = 1  # GND-VIN
        edge_existence[b, 1, 2] = 1  # VIN-VOUT
        edge_existence[b, 2, 0] = 1  # VOUT-GND

        # Fill in edge values with mean of all edge attributes for this graph
        # Find edges belonging to this graph
        node_mask = batch_graph.batch == b
        if node_mask.any():
            # Get indices of nodes in this graph
            node_indices = torch.where(node_mask)[0]
            # Find edges where source node is in this graph
            # Use broadcasting instead of torch.isin for MPS compatibility
            edge_mask = (batch_graph.edge_index[0].unsqueeze(1) == node_indices.unsqueeze(0)).any(dim=1)
            if edge_mask.any():
                # Use mean edge attributes
                edge_values[b, 0, 1] = batch_graph.edge_attr[edge_mask].mean(dim=0)
                edge_values[b, 1, 2] = batch_graph.edge_attr[edge_mask].mean(dim=0)
                edge_values[b, 2, 0] = batch_graph.edge_attr[edge_mask].mean(dim=0)

    # Pole/zero counts and values
    pole_count = num_poles.clamp(0, max_poles)
    zero_count = num_zeros.clamp(0, max_zeros)

    # Pad poles and zeros to max length
    pole_values = torch.zeros(batch_size, max_poles, 2, device=device)
    zero_values = torch.zeros(batch_size, max_zeros, 2, device=device)

    for b in range(batch_size):
        if len(poles_list[b]) > 0:
            n_poles = min(len(poles_list[b]), max_poles)
            pole_values[b, :n_poles] = poles_list[b][:n_poles]

        if len(zeros_list[b]) > 0:
            n_zeros = min(len(zeros_list[b]), max_zeros)
            zero_values[b, :n_zeros] = zeros_list[b][:n_zeros]

    clean_circuit = {
        'node_types': node_types,
        'edge_existence': edge_existence,
        'edge_values': edge_values,
        'pole_count': pole_count,
        'zero_count': zero_count,
        'pole_values': pole_values,
        'zero_values': zero_values
    }

    graph_data = {
        'x': batch_graph.x,
        'edge_index': batch_graph.edge_index,
        'edge_attr': batch_graph.edge_attr,
        'batch': batch_graph.batch,
        'poles_list': poles_list,
        'zeros_list': zeros_list
    }

    return clean_circuit, graph_data


def train_epoch(
    encoder,
    decoder,
    forward_diffusion,
    dataloader,
    optimizer,
    loss_fn,
    structure_loss_fn,
    device,
    config,
    epoch,
    freeze_encoder=True,
    tf_weight_current=0.1
):
    """
    Train for one epoch.

    Args:
        encoder: Encoder model
        decoder: Diffusion decoder
        forward_diffusion: Forward diffusion process
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Diffusion loss function
        structure_loss_fn: Structural loss function
        device: Device
        config: Configuration
        epoch: Current epoch
        freeze_encoder: Whether to freeze encoder
        tf_weight_current: Current transfer function loss weight

    Returns:
        avg_loss: Average training loss
        avg_metrics: Average metrics
    """
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()

    decoder.train()

    total_loss = 0
    total_metrics = {
        'loss_diffusion': 0,
        'loss_structure': 0,
        'node_type_acc': 0,
        'pole_count_acc': 0,
        'zero_count_acc': 0,
        'edge_exist_acc': 0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Prepare data
        clean_circuit, graph_data = prepare_batch_for_diffusion(batch, device)
        batch_size = clean_circuit['node_types'].shape[0]

        # ==================================================================
        # 1. Encode to latent space
        # ==================================================================

        if freeze_encoder:
            with torch.no_grad():
                z, mu, logvar = encoder(
                    graph_data['x'],
                    graph_data['edge_index'],
                    graph_data['edge_attr'],
                    graph_data['batch'],
                    graph_data['poles_list'],
                    graph_data['zeros_list']
                )
                latent_code = mu  # Use mean for training
        else:
            z, mu, logvar = encoder(
                graph_data['x'],
                graph_data['edge_index'],
                graph_data['edge_attr'],
                graph_data['batch'],
                graph_data['poles_list'],
                graph_data['zeros_list']
            )
            latent_code = mu

        # ==================================================================
        # 2. Sample timesteps and add noise
        # ==================================================================

        t = forward_diffusion.sample_timesteps(batch_size)

        # Add noise to circuit
        noisy_circuit, noise_dict = forward_diffusion.add_noise_to_circuit(clean_circuit, t)

        # ==================================================================
        # 3. Predict clean circuit from noisy
        # ==================================================================

        # Dummy conditions for now (TODO: extract from batch)
        conditions = torch.randn(batch_size, 2, device=device)

        predictions = decoder(
            noisy_circuit['node_types'],
            noisy_circuit['edge_values'],
            t,
            latent_code,
            conditions
        )

        # ==================================================================
        # 4. Compute losses
        # ==================================================================

        # Diffusion loss
        diffusion_loss, metrics = loss_fn(
            predictions,
            clean_circuit,
            t,
            timesteps=config['model']['decoder']['timesteps']
        )

        # Structural loss
        structure_loss = structure_loss_fn(predictions)

        # Total loss
        loss = diffusion_loss + config['loss']['structure_weight'] * structure_loss

        # Add KL divergence if joint training
        if not freeze_encoder:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
            loss = loss + 0.01 * kl_loss  # Small KL weight

        # ==================================================================
        # 5. Backward and optimize
        # ==================================================================

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_clip = config['training']['grad_clip']
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip)
        if not freeze_encoder:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip)

        optimizer.step()

        # ==================================================================
        # 6. Accumulate metrics
        # ==================================================================

        total_loss += loss.item()
        total_metrics['loss_diffusion'] += diffusion_loss.item()
        total_metrics['loss_structure'] += structure_loss.item()

        for key in ['node_type_acc', 'pole_count_acc', 'zero_count_acc', 'edge_exist_acc']:
            if key in metrics:
                total_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'node_acc': f'{metrics.get("node_type_acc", 0):.1f}%'
        })

    # Average metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def validate(
    encoder,
    decoder,
    forward_diffusion,
    dataloader,
    loss_fn,
    structure_loss_fn,
    device,
    config
):
    """
    Validate model.

    Args:
        encoder: Encoder model
        decoder: Diffusion decoder
        forward_diffusion: Forward diffusion process
        dataloader: Validation data loader
        loss_fn: Loss function
        structure_loss_fn: Structural loss function
        device: Device
        config: Configuration

    Returns:
        avg_loss: Average validation loss
        avg_metrics: Average metrics
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_metrics = {
        'loss_diffusion': 0,
        'loss_structure': 0,
        'node_type_acc': 0,
        'pole_count_acc': 0,
        'zero_count_acc': 0,
        'edge_exist_acc': 0
    }

    for batch in dataloader:
        # Prepare data
        clean_circuit, graph_data = prepare_batch_for_diffusion(batch, device)
        batch_size = clean_circuit['node_types'].shape[0]

        # Encode
        z, mu, logvar = encoder(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['edge_attr'],
            graph_data['batch'],
            graph_data['poles_list'],
            graph_data['zeros_list']
        )
        latent_code = mu

        # Sample timesteps and add noise
        t = forward_diffusion.sample_timesteps(batch_size)
        noisy_circuit, noise_dict = forward_diffusion.add_noise_to_circuit(clean_circuit, t)

        # Predict
        conditions = torch.randn(batch_size, 2, device=device)
        predictions = decoder(
            noisy_circuit['node_types'],
            noisy_circuit['edge_values'],
            t,
            latent_code,
            conditions
        )

        # Compute losses
        diffusion_loss, metrics = loss_fn(predictions, clean_circuit, t, timesteps=config['model']['decoder']['timesteps'])
        structure_loss = structure_loss_fn(predictions)
        loss = diffusion_loss + config['loss']['structure_weight'] * structure_loss

        # Accumulate
        total_loss += loss.item()
        total_metrics['loss_diffusion'] += diffusion_loss.item()
        total_metrics['loss_structure'] += structure_loss.item()

        for key in ['node_type_acc', 'pole_count_acc', 'zero_count_acc', 'edge_exist_acc']:
            if key in metrics:
                total_metrics[key] += metrics[key]

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Diffusion Decoder')
    parser.add_argument('--config', type=str, default='configs/diffusion_decoder.yaml',
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
    print("DIFFUSION DECODER TRAINING")
    print(f"{'='*70}\n")

    device = args.device

    # ==================================================================
    # 1. Load dataset
    # ==================================================================

    print("📂 Loading dataset...")
    dataset = CircuitDataset(dataset_path=config['data']['dataset_path'])

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
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

    print(f"✅ Loaded {len(dataset)} circuits")
    print(f"   Training: {train_size}, Validation: {val_size}")

    # ==================================================================
    # 2. Load pretrained encoder
    # ==================================================================

    encoder = load_pretrained_encoder(
        config['checkpoint']['pretrained_encoder'],
        config,
        device
    )

    # ==================================================================
    # 3. Create diffusion decoder
    # ==================================================================

    print("\n🏗️  Creating diffusion decoder...")
    decoder = create_diffusion_decoder(config, device)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"✅ Models created")
    print(f"   Encoder parameters:  {encoder_params:,}")
    print(f"   Decoder parameters:  {decoder_params:,}")
    print(f"   Total parameters:    {encoder_params + decoder_params:,}")

    # ==================================================================
    # 4. Create forward diffusion process
    # ==================================================================

    forward_diffusion = CircuitForwardDiffusion(
        timesteps=config['model']['decoder']['timesteps'],
        max_nodes=config['model']['decoder']['max_nodes'],
        max_poles=config['model']['decoder']['max_poles'],
        max_zeros=config['model']['decoder']['max_zeros'],
        device=device
    )

    # ==================================================================
    # 5. Create loss functions
    # ==================================================================

    loss_fn = DiffusionCircuitLoss(
        node_type_weight=config['loss']['node_type_weight'],
        edge_exist_weight=config['loss']['edge_exist_weight'],
        edge_value_weight=config['loss']['edge_value_weight'],
        pole_count_weight=config['loss']['pole_count_weight'],
        zero_count_weight=config['loss']['zero_count_weight'],
        pole_value_weight=config['loss']['pole_value_weight'],
        zero_value_weight=config['loss']['zero_value_weight'],
        use_time_weighting=config['loss']['use_time_weighting']
    )

    structure_loss_fn = CombinedStructuralLoss()

    # ==================================================================
    # 6. Training loop
    # ==================================================================

    # Create checkpoint directory
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    total_epochs = config['training']['total_epochs']
    phase1_epochs = config['training']['phase1_epochs']

    print(f"\n🚀 Starting training for {total_epochs} epochs")
    print(f"   Phase 1 (freeze encoder): Epochs 1-{phase1_epochs}")
    print(f"   Phase 2 (joint training): Epochs {phase1_epochs+1}-{total_epochs}")

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*70}")

        # ==================================================================
        # Determine training phase
        # ==================================================================

        freeze_encoder = (epoch <= phase1_epochs)
        phase_name = "Phase 1 (Freeze Encoder)" if freeze_encoder else "Phase 2 (Joint Training)"

        print(f"Training mode: {phase_name}")

        # ==================================================================
        # Create optimizer for current phase
        # ==================================================================

        if freeze_encoder:
            lr = config['training']['learning_rate_phase1']
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=lr,
                weight_decay=config['training']['weight_decay']
            )
        else:
            lr = config['training']['learning_rate_phase2']
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=lr,
                weight_decay=config['training']['weight_decay']
            )

        # ==================================================================
        # Train
        # ==================================================================

        train_loss, train_metrics = train_epoch(
            encoder, decoder, forward_diffusion, train_loader, optimizer,
            loss_fn, structure_loss_fn, device, config, epoch,
            freeze_encoder=freeze_encoder
        )

        # ==================================================================
        # Validate
        # ==================================================================

        val_loss, val_metrics = validate(
            encoder, decoder, forward_diffusion, val_loader,
            loss_fn, structure_loss_fn, device, config
        )

        # ==================================================================
        # Print metrics
        # ==================================================================

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

        # ==================================================================
        # Save checkpoint
        # ==================================================================

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = save_dir / 'best.pt'
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, checkpoint_path)

            print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = save_dir / f'epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss
            }, checkpoint_path)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 Best model saved to: {save_dir}/best.pt")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
