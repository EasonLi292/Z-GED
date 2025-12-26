#!/usr/bin/env python3
"""
Generation script for Diffusion-based Circuit Decoder.

Generates circuits from specifications using the trained diffusion model.
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.encoder import HierarchicalEncoder
from ml.models.diffusion import DiffusionGraphTransformer
from ml.models.diffusion.reverse_process import create_sampler
from ml.models.diffusion.constraints import post_process_circuit


def load_trained_model(checkpoint_path: str, device: str):
    """
    Load trained diffusion model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device

    Returns:
        encoder: Encoder model
        decoder: Diffusion decoder
        config: Configuration dict
    """
    print(f"\n📥 Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

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

    # Create decoder
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

    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")

    return encoder, decoder, config


def generate_from_latent(
    decoder,
    latent_code: torch.Tensor,
    conditions: torch.Tensor,
    sampler_type: str = 'ddim',
    num_steps: int = 50,
    temperature: float = 1.0,
    device: str = 'cpu',
    timesteps: int = 1000
):
    """
    Generate circuit from latent code and conditions.

    Args:
        decoder: Diffusion decoder
        latent_code: Latent code [batch, latent_dim]
        conditions: Specifications [batch, conditions_dim]
        sampler_type: 'ddpm' or 'ddim'
        num_steps: Number of sampling steps
        temperature: Sampling temperature
        device: Device
        timesteps: Total number of diffusion timesteps

    Returns:
        circuit: Generated circuit
    """
    # Create sampler
    sampler = create_sampler(
        sampler_type=sampler_type,
        timesteps=timesteps,
        num_inference_steps=num_steps,
        device=device
    )

    # Generate
    circuit = sampler.sample(
        model=decoder,
        latent_code=latent_code,
        conditions=conditions,
        batch_size=latent_code.shape[0],
        temperature=temperature,
        verbose=True
    )

    # Post-process for validity
    circuit = post_process_circuit(circuit, enforce_constraints=True)

    return circuit


def generate_from_specifications(
    decoder,
    cutoff_freq: float,
    q_factor: float,
    latent_dim: int = 8,
    sampler_type: str = 'ddim',
    num_steps: int = 50,
    num_samples: int = 10,
    temperature: float = 1.0,
    device: str = 'cpu',
    timesteps: int = 1000
):
    """
    Generate circuits from specifications by sampling latent code.

    Args:
        decoder: Diffusion decoder
        cutoff_freq: Desired cutoff frequency (Hz)
        q_factor: Desired Q factor
        latent_dim: Latent code dimension
        sampler_type: Sampler type
        num_steps: Number of sampling steps
        num_samples: Number of circuits to generate
        temperature: Sampling temperature
        device: Device
        timesteps: Total diffusion timesteps

    Returns:
        circuits: List of generated circuits
    """
    print(f"\n🎨 Generating {num_samples} circuits")
    print(f"   Cutoff frequency: {cutoff_freq:.2f} Hz")
    print(f"   Q factor: {q_factor:.2f}")
    print(f"   Sampler: {sampler_type} ({num_steps} steps)")

    # Sample latent codes from prior
    latent_codes = torch.randn(num_samples, latent_dim, device=device) * temperature

    # Create conditions (need normalization - placeholder for now)
    conditions = torch.tensor(
        [[cutoff_freq, q_factor]] * num_samples,
        dtype=torch.float32,
        device=device
    )

    # TODO: Normalize conditions using condition_stats from training

    # Generate
    circuit = generate_from_latent(
        decoder,
        latent_codes,
        conditions,
        sampler_type=sampler_type,
        num_steps=num_steps,
        temperature=temperature,
        device=device,
        timesteps=timesteps
    )

    return circuit


def print_circuit(circuit: dict, idx: int):
    """
    Print circuit details.

    Args:
        circuit: Circuit dictionary
        idx: Circuit index
    """
    print(f"\n{'='*70}")
    print(f"Circuit {idx}")
    print(f"{'='*70}")

    # Node types
    node_types = circuit['node_types'][idx]
    node_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']
    print(f"\nNodes ({len(node_types)}):")
    for i, nt in enumerate(node_types):
        if nt < len(node_names):
            print(f"  Node {i}: {node_names[nt]}")

    # Edges
    edge_exist = circuit['edge_existence'][idx]
    num_edges = edge_exist.sum().item()
    print(f"\nEdges ({int(num_edges)}):")
    edges = torch.nonzero(edge_exist, as_tuple=False)
    for edge in edges[:10]:  # Show first 10
        i, j = edge
        values = circuit['edge_values'][idx, i, j, :3]  # C, G, L_inv
        print(f"  {i} → {j}: C={values[0]:.4f}, G={values[1]:.4f}, L_inv={values[2]:.4f}")

    # Poles and zeros
    pole_count = circuit['pole_count'][idx].item()
    zero_count = circuit['zero_count'][idx].item()

    print(f"\nPoles ({pole_count}):")
    for i in range(pole_count):
        pole = circuit['pole_values'][idx, i]
        print(f"  Pole {i}: {pole[0]:.4f} + {pole[1]:.4f}j")

    print(f"\nZeros ({zero_count}):")
    for i in range(zero_count):
        zero = circuit['zero_values'][idx, i]
        print(f"  Zero {i}: {zero[0]:.4f} + {zero[1]:.4f}j")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate circuits with diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='Cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Q factor')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of circuits to generate')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                       help='Sampler type')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of sampling steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/mps/cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save circuits')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("DIFFUSION-BASED CIRCUIT GENERATION")
    print(f"{'='*70}")

    # Load model
    encoder, decoder, config = load_trained_model(args.checkpoint, args.device)

    # Generate circuits
    circuits = generate_from_specifications(
        decoder,
        cutoff_freq=args.cutoff,
        q_factor=args.q_factor,
        latent_dim=config['model']['decoder']['latent_dim'],
        sampler_type=args.sampler,
        num_steps=args.steps,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
        timesteps=config['model']['decoder']['timesteps']
    )

    # Print generated circuits
    for i in range(min(args.num_samples, 3)):  # Show first 3
        print_circuit(circuits, i)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(circuits, output_path)
        print(f"\n💾 Saved circuits to: {output_path}")

    print(f"\n{'='*70}")
    print("✅ GENERATION COMPLETE")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
