#!/usr/bin/env python3
"""
Generate novel circuits using trained GraphVAE.

This script provides several generation modes:
1. Prior sampling: Generate random circuits from N(0, I)
2. Conditional: Generate specific filter types
3. Interpolation: Morph between two circuits
4. Branch modification: Change specific circuit properties

Usage:
    # Generate 10 random circuits
    python scripts/generate.py --checkpoint checkpoints/best.pt --mode prior --num-samples 10

    # Generate 5 low-pass filters
    python scripts/generate.py --checkpoint checkpoints/best.pt --mode conditional --filter-type low_pass --num-samples 5

    # Interpolate between two circuits
    python scripts/generate.py --checkpoint checkpoints/best.pt --mode interpolate --circuit1-idx 0 --circuit2-idx 10
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import json

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.generation import CircuitSampler
from ml.data import CircuitDataset
from tools.circuit_generator import create_compact_graph_representation, extract_poles_zeros_gain_analytical

# Filter type names
FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']


def load_model(checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    config_path = checkpoint_path.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Extract branch dimensions
    topo_dim = config['model'].get('topo_latent_dim', None)
    values_dim = config['model'].get('values_latent_dim', None)
    pz_dim = config['model'].get('pz_latent_dim', None)

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    decoder = HybridDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    branch_dims = (
        topo_dim if topo_dim else config['model']['latent_dim'] // 3,
        values_dim if values_dim else config['model']['latent_dim'] // 3,
        pz_dim if pz_dim else config['model']['latent_dim'] // 3
    )

    print(f"\n✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Model: {config['model']['latent_dim']}D ({branch_dims[0]}D + {branch_dims[1]}D + {branch_dims[2]}D)")
    print(f"   Epoch: {checkpoint['epoch']}, Val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return encoder, decoder, config, branch_dims


def decode_and_analyze(outputs: dict, dataset: CircuitDataset = None):
    """Decode outputs and analyze generated circuits."""
    results = []

    graphs = outputs['graphs']
    topo_probs = outputs['topo_probs']
    poles = outputs['poles']
    zeros = outputs['zeros']

    for i, graph in enumerate(graphs):
        # Get filter type
        filter_idx = topo_probs[i].argmax().item()
        filter_type = FILTER_TYPES[filter_idx]
        filter_prob = topo_probs[i, filter_idx].item()

        # Get edge features (denormalize if dataset provided)
        edge_attr = graph.edge_attr.cpu().numpy()
        if dataset is not None and dataset.normalize_features:
            # Denormalize using dataset statistics
            # edge_attr shape: [num_edges, 7] containing:
            # [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
            # Only denormalize first 3 (impedance values), keep last 4 (binary masks)
            impedance_mean = dataset.impedance_mean.numpy()
            impedance_std = dataset.impedance_std.numpy()

            # Denormalize impedance values: reverse (x - mean) / std
            edge_attr_imp = edge_attr[:, :3]  # First 3 are impedance
            edge_attr_binary = edge_attr[:, 3:]  # Last 4 are binary masks

            edge_attr_imp_denorm = edge_attr_imp * impedance_std + impedance_mean

            # If log-scaled, exp to get back original values
            if dataset.log_scale_impedance:
                edge_attr_imp = np.exp(edge_attr_imp_denorm)
            else:
                edge_attr_imp = edge_attr_imp_denorm

            # Recombine
            edge_attr = np.concatenate([edge_attr_imp, edge_attr_binary], axis=1)
        else:
            # If features are log-scaled but not normalized, still exp first 3
            if dataset is not None and dataset.log_scale_impedance:
                edge_attr_imp = np.exp(edge_attr[:, :3])
                edge_attr = np.concatenate([edge_attr_imp, edge_attr[:, 3:]], axis=1)

        # Convert to readable format
        circuit_info = {
            'filter_type': filter_type,
            'filter_confidence': float(filter_prob),
            'num_nodes': int(graph.num_nodes),
            'num_edges': int(graph.num_edges),
            'edge_features': edge_attr.tolist(),
            'predicted_poles': poles[i].cpu().numpy().tolist(),
            'predicted_zeros': zeros[i].cpu().numpy().tolist(),
            'node_types': graph.x.argmax(dim=1).cpu().numpy().tolist()
        }

        results.append(circuit_info)

    return results


def print_circuit_summary(circuits: list):
    """Print human-readable summary of generated circuits."""
    print(f"\n{'='*70}")
    print(f"GENERATED {len(circuits)} CIRCUITS")
    print(f"{'='*70}")

    for i, circuit in enumerate(circuits):
        print(f"\nCircuit {i+1}:")
        print(f"  Filter Type: {circuit['filter_type']} ({circuit['filter_confidence']:.2%} confidence)")
        print(f"  Topology: {circuit['num_nodes']} nodes, {circuit['num_edges']} edges")
        print(f"  Components: {len([e for e in circuit['edge_features'] if e is not None])} edges with features")
        print(f"  Transfer Function: {len(circuit['predicted_poles'])} poles, {len(circuit['predicted_zeros'])} zeros")


def main():
    parser = argparse.ArgumentParser(description='Generate circuits with GraphVAE')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset (for denormalization)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')

    # Generation mode
    parser.add_argument('--mode', type=str, default='prior',
                       choices=['prior', 'conditional', 'interpolate', 'modify'],
                       help='Generation mode')

    # Prior sampling arguments
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of circuits to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0.5=conservative, 2.0=exploratory)')

    # Conditional generation arguments
    parser.add_argument('--filter-type', type=str, default='low_pass',
                       choices=FILTER_TYPES,
                       help='Filter type for conditional generation')

    # Interpolation arguments
    parser.add_argument('--circuit1-idx', type=int, default=0,
                       help='Index of first circuit for interpolation')
    parser.add_argument('--circuit2-idx', type=int, default=10,
                       help='Index of second circuit for interpolation')
    parser.add_argument('--interp-steps', type=int, default=5,
                       help='Number of interpolation steps')
    parser.add_argument('--interp-type', type=str, default='linear',
                       choices=['linear', 'spherical', 'branch'],
                       help='Interpolation method')

    # Branch modification arguments
    parser.add_argument('--circuit-idx', type=int, default=0,
                       help='Index of circuit to modify')
    parser.add_argument('--modify-branch', type=str, default='values',
                       choices=['topo', 'values', 'pz'],
                       help='Which branch to modify')
    parser.add_argument('--modify-amount', type=float, default=0.5,
                       help='Amount to modify (added or multiplied)')
    parser.add_argument('--modify-op', type=str, default='add',
                       choices=['add', 'multiply'],
                       help='Modification operation')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='generated_circuits',
                       help='Directory to save generated circuits')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results as JSON')

    args = parser.parse_args()

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
    print("GRAPHVAE CIRCUIT GENERATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("="*70)

    # Load model
    encoder, decoder, config, branch_dims = load_model(args.checkpoint, device)

    # Load dataset (for denormalization)
    dataset = None
    if Path(args.dataset).exists():
        dataset = CircuitDataset(
            dataset_path=args.dataset,
            normalize_features=config['data']['normalize'],
            log_scale_impedance=config['data']['log_scale']
        )
        print(f"\n📊 Loaded dataset: {len(dataset)} circuits")

    # Create sampler
    sampler = CircuitSampler(
        encoder, decoder,
        device=device,
        latent_dim=config['model']['latent_dim'],
        branch_dims=branch_dims
    )

    # Generate based on mode
    if args.mode == 'prior':
        print(f"\n🎲 Sampling {args.num_samples} circuits from prior N(0, I)")
        print(f"   Temperature: {args.temperature}")
        outputs = sampler.sample_prior(args.num_samples, args.temperature)

    elif args.mode == 'conditional':
        print(f"\n🎯 Generating {args.num_samples} {args.filter_type} filters")
        print(f"   Temperature: {args.temperature}")
        outputs = sampler.sample_conditional(
            args.filter_type,
            args.num_samples,
            args.temperature
        )

    elif args.mode == 'interpolate':
        print(f"\n🔄 Interpolating between circuits {args.circuit1_idx} and {args.circuit2_idx}")
        print(f"   Steps: {args.interp_steps}")
        print(f"   Method: {args.interp_type}")

        if dataset is None:
            raise ValueError("Dataset required for interpolation mode")

        # Load the two circuits
        circuit1 = dataset[args.circuit1_idx]
        circuit2 = dataset[args.circuit2_idx]

        # Interpolate
        interpolated = sampler.interpolate(
            circuit1, circuit2,
            args.interp_steps,
            args.interp_type
        )

        # Combine outputs
        outputs = {
            'graphs': [out['graphs'][0] for out in interpolated],
            'topo_probs': torch.cat([out['topo_probs'] for out in interpolated]),
            'poles': torch.cat([out['poles'] for out in interpolated]),
            'zeros': torch.cat([out['zeros'] for out in interpolated])
        }

    elif args.mode == 'modify':
        print(f"\n✏️  Modifying circuit {args.circuit_idx}")
        print(f"   Branch: {args.modify_branch}")
        print(f"   Operation: {args.modify_op} {args.modify_amount}")

        if dataset is None:
            raise ValueError("Dataset required for modify mode")

        circuit = dataset[args.circuit_idx]

        outputs = sampler.modify_branch(
            circuit,
            args.modify_branch,
            args.modify_amount,
            args.modify_op
        )

    # Analyze results
    circuits = decode_and_analyze(outputs, dataset)

    # Print summary
    print_circuit_summary(circuits)

    # Save results
    if args.save_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'{args.mode}_{timestamp}.json'

        results = {
            'mode': args.mode,
            'num_circuits': len(circuits),
            'parameters': vars(args),
            'model_config': {
                'latent_dim': config['model']['latent_dim'],
                'branch_dims': branch_dims
            },
            'circuits': circuits
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Saved results to: {output_file}")

    print("\n✅ Generation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
