#!/usr/bin/env python3
"""
Generate circuits using variable-length decoder.

Usage:
    # Generate 5 low-pass filters
    python scripts/generate_variable_length.py \
        --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
        --filter-type low_pass \
        --num-samples 5

    # Sample from prior
    python scripts/generate_variable_length.py \
        --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
        --mode prior \
        --num-samples 5
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

from ml.models import HierarchicalEncoder
from ml.models.variable_decoder import VariableLengthDecoder
from ml.data import CircuitDataset


def load_model(checkpoint_path: str, device: str):
    """Load trained variable-length model from checkpoint."""
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

    # Create models
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

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    print(f"\n✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Model: {config['model']['latent_dim']}D latent space")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}, Val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    return encoder, decoder, config


def infer_filter_type_from_structure(num_poles, num_zeros):
    """Infer filter type from pole/zero structure."""
    if num_poles == 1 and num_zeros == 0:
        return 'low_pass'
    elif num_poles == 2 and num_zeros == 0:
        return 'low_pass'
    elif num_poles == 1 and num_zeros == 1:
        return 'high_pass'
    elif num_poles == 2 and num_zeros == 2:
        return 'high_pass_or_band_stop'
    elif num_poles == 2 and num_zeros == 1:
        return 'band_pass'
    else:
        return 'unknown'


def denormalize_poles_zeros(poles, zeros, dataset):
    """Denormalize poles and zeros."""
    if dataset is None:
        return poles, zeros

    # Denormalize (reverse log-scale normalization)
    poles_denorm = poles.copy()
    zeros_denorm = zeros.copy()

    if len(poles) > 0:
        # Poles are stored as [real, imag]
        # In log-scale, magnitude is normalized
        # For now, just return as is (they're already in normalized form)
        poles_denorm = poles

    if len(zeros) > 0:
        zeros_denorm = zeros

    return poles_denorm, zeros_denorm


def generate_circuits(
    decoder,
    filter_type: str,
    num_samples: int,
    temperature: float,
    device: str
):
    """
    Generate circuits conditionally.

    Args:
        decoder: VariableLengthDecoder
        filter_type: Target filter type
        num_samples: Number of circuits to generate
        temperature: Sampling temperature
        device: Device to use

    Returns:
        Generated circuits dict
    """
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    filter_idx = filter_type_names.index(filter_type)

    # Create filter type one-hot encoding
    gt_filter_type = torch.zeros(num_samples, len(filter_type_names), device=device)
    gt_filter_type[:, filter_idx] = 1.0

    # Sample latent codes from prior
    latent_dim = decoder.latent_dim
    z = torch.randn(num_samples, latent_dim, device=device) * temperature

    # Decode with teacher forcing (conditioning on filter type)
    with torch.no_grad():
        outputs = decoder(z, hard=True, gt_filter_type=gt_filter_type)

    return outputs


def generate_from_prior(
    decoder,
    num_samples: int,
    temperature: float,
    device: str
):
    """
    Generate circuits from prior (unconditional).

    Args:
        decoder: VariableLengthDecoder
        num_samples: Number of circuits to generate
        temperature: Sampling temperature
        device: Device to use

    Returns:
        Generated circuits dict
    """
    # Sample latent codes from prior
    latent_dim = decoder.latent_dim
    z = torch.randn(num_samples, latent_dim, device=device) * temperature

    # Decode without teacher forcing
    with torch.no_grad():
        outputs = decoder(z, hard=True, gt_filter_type=None)

    return outputs


def print_circuit_details(
    outputs: dict,
    dataset: CircuitDataset = None,
    mode: str = 'conditional',
    target_filter_type: str = None
):
    """Print detailed information about generated circuits."""
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Get predictions
    pred_topo = outputs['topo_logits'].argmax(dim=-1)
    topo_probs = torch.softmax(outputs['topo_logits'], dim=-1)
    pred_pole_counts = outputs['pole_count_logits'].argmax(dim=-1)
    pred_zero_counts = outputs['zero_count_logits'].argmax(dim=-1)

    poles_all = outputs['poles_all']
    zeros_all = outputs['zeros_all']

    num_samples = len(pred_topo)

    print(f"\n{'='*70}")
    print(f"GENERATED {num_samples} CIRCUITS ({mode.upper()} MODE)")
    if target_filter_type:
        print(f"Target Filter Type: {target_filter_type.upper()}")
    print(f"{'='*70}")

    results = []

    for i in range(num_samples):
        # Get predictions
        topo_idx = pred_topo[i].item()
        predicted_type = filter_type_names[topo_idx]
        confidence = topo_probs[i, topo_idx].item()

        n_poles_pred = pred_pole_counts[i].item()
        n_zeros_pred = pred_zero_counts[i].item()

        # Extract valid poles/zeros
        poles = poles_all[i, :n_poles_pred].cpu().numpy() if n_poles_pred > 0 else np.array([])
        zeros = zeros_all[i, :n_zeros_pred].cpu().numpy() if n_zeros_pred > 0 else np.array([])

        # Infer filter type from structure
        inferred_type = infer_filter_type_from_structure(n_poles_pred, n_zeros_pred)

        # Check if matches target (for conditional generation)
        match_target = (predicted_type == target_filter_type) if target_filter_type else None
        match_structure = (inferred_type == target_filter_type) if target_filter_type else None

        print(f"\n{'─'*70}")
        print(f"Circuit {i+1}:")
        print(f"{'─'*70}")
        print(f"  Predicted Topology:  {predicted_type.upper()} ({confidence:.1%} confidence)")
        print(f"  Pole/Zero Structure: {n_poles_pred} poles, {n_zeros_pred} zeros")
        print(f"  Inferred TF Type:    {inferred_type}")

        if target_filter_type:
            topo_match_str = "✅" if match_target else "❌"
            struct_match_str = "✅" if match_structure else "❌"
            print(f"  Topology Match:      {topo_match_str} ({predicted_type} vs {target_filter_type})")
            print(f"  Structure Match:     {struct_match_str} ({inferred_type} vs {target_filter_type})")

        # Print poles
        if len(poles) > 0:
            print(f"\n  Poles ({len(poles)}):")
            for j, pole in enumerate(poles):
                real, imag = pole
                print(f"    Pole {j+1}: {real:+.4f} {imag:+.4f}j")
        else:
            print(f"\n  Poles: None")

        # Print zeros
        if len(zeros) > 0:
            print(f"\n  Zeros ({len(zeros)}):")
            for j, zero in enumerate(zeros):
                real, imag = zero
                print(f"    Zero {j+1}: {real:+.4f} {imag:+.4f}j")
        else:
            print(f"\n  Zeros: None")

        # Store result
        results.append({
            'predicted_topology': predicted_type,
            'confidence': float(confidence),
            'num_poles': n_poles_pred,
            'num_zeros': n_zeros_pred,
            'inferred_type': inferred_type,
            'poles': poles.tolist() if len(poles) > 0 else [],
            'zeros': zeros.tolist() if len(zeros) > 0 else [],
            'matches_target_topology': match_target,
            'matches_target_structure': match_structure
        })

    # Summary statistics
    if target_filter_type:
        topo_matches = sum(1 for r in results if r['matches_target_topology'])
        struct_matches = sum(1 for r in results if r['matches_target_structure'])

        print(f"\n{'='*70}")
        print("GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Target: {target_filter_type.upper()}")
        print(f"Topology Match:  {topo_matches}/{num_samples} ({topo_matches/num_samples:.1%})")
        print(f"Structure Match: {struct_matches}/{num_samples} ({struct_matches/num_samples:.1%})")
        print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate circuits with variable-length decoder')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset (for denormalization)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda/mps/cpu)')

    # Generation mode
    parser.add_argument('--mode', type=str, default='conditional',
                       choices=['prior', 'conditional'],
                       help='Generation mode')

    # Generation arguments
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of circuits to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0.5=conservative, 2.0=exploratory)')

    # Conditional generation arguments
    parser.add_argument('--filter-type', type=str, default='low_pass',
                       choices=['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel'],
                       help='Filter type for conditional generation')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("VARIABLE-LENGTH DECODER CIRCUIT GENERATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    if args.mode == 'conditional':
        print(f"Filter type: {args.filter_type}")
    print(f"Num samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print("="*70)

    # Load model
    encoder, decoder, config = load_model(args.checkpoint, args.device)

    # Load dataset (for denormalization)
    dataset = None
    if Path(args.dataset).exists():
        dataset = CircuitDataset(
            dataset_path=args.dataset,
            normalize_features=config['data']['normalize'],
            log_scale_impedance=config['data']['log_scale']
        )
        print(f"\n📊 Loaded dataset: {len(dataset)} circuits")

    # Generate based on mode
    if args.mode == 'prior':
        print(f"\n🎲 Sampling {args.num_samples} circuits from prior N(0, I)")
        print(f"   Temperature: {args.temperature}")
        outputs = generate_from_prior(
            decoder,
            args.num_samples,
            args.temperature,
            args.device
        )
        results = print_circuit_details(outputs, dataset, mode='prior')

    elif args.mode == 'conditional':
        print(f"\n🎯 Generating {args.num_samples} {args.filter_type} filters")
        print(f"   Temperature: {args.temperature}")
        outputs = generate_circuits(
            decoder,
            args.filter_type,
            args.num_samples,
            args.temperature,
            args.device
        )
        results = print_circuit_details(
            outputs,
            dataset,
            mode='conditional',
            target_filter_type=args.filter_type
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'mode': args.mode,
            'filter_type': args.filter_type if args.mode == 'conditional' else None,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'checkpoint': str(args.checkpoint),
            'timestamp': datetime.now().isoformat(),
            'circuits': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Saved results to: {output_path}")

    print("\n✅ Generation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
