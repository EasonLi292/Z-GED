#!/usr/bin/env python3
"""
Generate circuits from specifications using Conditional VAE.

This script enables direct circuit generation from desired specifications
(cutoff frequency, Q factor) without requiring a reference circuit.

Key Features:
- Direct generation from specs (no reference circuit needed)
- Compositional generation (mix specs from different circuits)
- Exploration via temperature and latent space sampling

Usage:
    # Generate 10 low-pass filters with cutoff=14.3 Hz, Q=0.707
    python scripts/generate_from_specs.py \
        --checkpoint checkpoints/conditional_vae/best.pt \
        --cutoff 14.3 \
        --q-factor 0.707 \
        --filter-type low_pass \
        --num-samples 10

    # Generate high-Q band-pass filter
    python scripts/generate_from_specs.py \
        --checkpoint checkpoints/conditional_vae/best.pt \
        --cutoff 1000.0 \
        --q-factor 5.0 \
        --filter-type band_pass \
        --num-samples 5

    # Exploration mode: sample conditions from dataset ranges
    python scripts/generate_from_specs.py \
        --checkpoint checkpoints/conditional_vae/best.pt \
        --mode explore \
        --num-samples 20
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

from ml.models.conditional_encoder import ConditionalHierarchicalEncoder
from ml.models.conditional_decoder import ConditionalVariableLengthDecoder
from ml.utils.condition_utils import ConditionExtractor
from ml.data import CircuitDataset


def load_model(checkpoint_path: str, device: str):
    """Load trained conditional VAE from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config and condition stats from checkpoint
    config = checkpoint['config']
    condition_stats = checkpoint['condition_stats']

    # Create conditional encoder
    encoder = ConditionalHierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim'),
        conditions_dim=config['model'].get('conditions_dim', 2),
        condition_embed_dim=config['model'].get('condition_embed_dim', 64)
    )

    # Create conditional decoder
    decoder = ConditionalVariableLengthDecoder(
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim'),
        max_nodes=config['model'].get('max_nodes', 5),
        max_edges=config['model'].get('max_edges', 10),
        max_poles=config['model'].get('max_poles', 4),
        max_zeros=config['model'].get('max_zeros', 4),
        num_filter_types=6,
        conditions_dim=config['model'].get('conditions_dim', 2)
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Create condition extractor
    condition_extractor = ConditionExtractor(
        normalize=True,
        condition_stats=condition_stats
    )

    print(f"\n✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Model: {config['model']['latent_dim']}D latent space")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"\n📊 Condition Statistics:")
    print(f"   Cutoff range: {condition_stats['cutoff_min']:.2f} - {condition_stats['cutoff_max']:.2f} Hz")
    print(f"   Q range:      {condition_stats['q_min']:.3f} - {condition_stats['q_max']:.3f}")

    return encoder, decoder, condition_extractor, config, condition_stats


def prepare_conditions(
    cutoff_freq: float,
    q_factor: float,
    condition_extractor: ConditionExtractor,
    num_samples: int,
    device: str
) -> torch.Tensor:
    """
    Prepare normalized condition tensor.

    Args:
        cutoff_freq: Desired cutoff frequency (Hz)
        q_factor: Desired Q factor
        condition_extractor: ConditionExtractor instance
        num_samples: Number of samples (for broadcasting)
        device: Device

    Returns:
        Normalized conditions [num_samples, 2]
    """
    # Create conditions dict
    conditions_dict = {
        'cutoff_frequency': cutoff_freq,
        'q_factor': q_factor
    }

    # Normalize
    conditions_norm = condition_extractor.normalize_conditions(conditions_dict)  # [2]

    # Broadcast to batch
    conditions_batch = conditions_norm.unsqueeze(0).repeat(num_samples, 1)  # [num_samples, 2]
    conditions_batch = conditions_batch.to(device)

    return conditions_batch


def generate_from_specifications(
    decoder: ConditionalVariableLengthDecoder,
    cutoff_freq: float,
    q_factor: float,
    filter_type: str,
    condition_extractor: ConditionExtractor,
    num_samples: int,
    temperature: float,
    device: str
):
    """
    Generate circuits from specifications.

    Args:
        decoder: ConditionalVariableLengthDecoder
        cutoff_freq: Target cutoff frequency (Hz)
        q_factor: Target Q factor
        filter_type: Target filter type
        condition_extractor: ConditionExtractor
        num_samples: Number of circuits to generate
        temperature: Latent sampling temperature
        device: Device

    Returns:
        Generated circuits dict
    """
    # Prepare conditions
    conditions = prepare_conditions(
        cutoff_freq, q_factor, condition_extractor, num_samples, device
    )

    # Prepare filter type (for teacher forcing)
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    filter_idx = filter_type_names.index(filter_type)

    gt_filter_type = torch.zeros(num_samples, len(filter_type_names), device=device)
    gt_filter_type[:, filter_idx] = 1.0

    # Sample latent codes from prior
    z = torch.randn(num_samples, decoder.latent_dim, device=device) * temperature

    # Generate with conditions
    with torch.no_grad():
        outputs = decoder(z, hard=True, gt_filter_type=gt_filter_type, conditions=conditions)

    return outputs, conditions


def explore_condition_space(
    decoder: ConditionalVariableLengthDecoder,
    condition_stats: dict,
    condition_extractor: ConditionExtractor,
    num_samples: int,
    temperature: float,
    device: str
):
    """
    Generate circuits by sampling from condition space.

    Samples cutoff and Q uniformly from dataset ranges.

    Args:
        decoder: ConditionalVariableLengthDecoder
        condition_stats: Condition statistics dict
        condition_extractor: ConditionExtractor
        num_samples: Number of circuits to generate
        temperature: Latent sampling temperature
        device: Device

    Returns:
        Generated circuits dict, sampled conditions
    """
    # Sample cutoff frequencies uniformly in log-space
    log_cutoff_min = np.log10(condition_stats['cutoff_min'] + 1e-10)
    log_cutoff_max = np.log10(condition_stats['cutoff_max'] + 1e-10)
    log_cutoffs = np.random.uniform(log_cutoff_min, log_cutoff_max, num_samples)
    cutoffs = 10 ** log_cutoffs

    # Sample Q factors uniformly in log-space
    log_q_min = np.log10(condition_stats['q_min'] + 1e-10)
    log_q_max = np.log10(condition_stats['q_max'] + 1e-10)
    log_qs = np.random.uniform(log_q_min, log_q_max, num_samples)
    qs = 10 ** log_qs

    # Prepare conditions
    conditions_list = []
    for cutoff, q in zip(cutoffs, qs):
        cond_dict = {'cutoff_frequency': cutoff, 'q_factor': q}
        cond_norm = condition_extractor.normalize_conditions(cond_dict)
        conditions_list.append(cond_norm)

    conditions = torch.stack(conditions_list).to(device)  # [num_samples, 2]

    # Sample latent codes
    z = torch.randn(num_samples, decoder.latent_dim, device=device) * temperature

    # Generate (no teacher forcing - let decoder predict topology)
    with torch.no_grad():
        outputs = decoder(z, hard=True, gt_filter_type=None, conditions=conditions)

    # Return sampled specs for display
    sampled_specs = [
        {'cutoff_frequency': float(c), 'q_factor': float(q)}
        for c, q in zip(cutoffs, qs)
    ]

    return outputs, conditions, sampled_specs


def print_circuit_details(
    outputs: dict,
    conditions: torch.Tensor,
    condition_extractor: ConditionExtractor,
    mode: str = 'specs',
    target_specs: dict = None,
    sampled_specs: list = None
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
    print(f"GENERATED {num_samples} CIRCUITS (CONDITIONAL VAE - {mode.upper()} MODE)")
    if target_specs:
        print(f"Target Specifications:")
        print(f"  Cutoff frequency: {target_specs['cutoff_frequency']:.2f} Hz")
        print(f"  Q factor:         {target_specs['q_factor']:.3f}")
        if 'filter_type' in target_specs:
            print(f"  Filter type:      {target_specs['filter_type'].upper()}")
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

        # Denormalize conditions for this circuit
        cond_denorm = condition_extractor.denormalize_conditions(conditions[i].cpu())

        # If in explore mode, use sampled specs
        if sampled_specs:
            cond_denorm = sampled_specs[i]

        print(f"\n{'─'*70}")
        print(f"Circuit {i+1}:")
        print(f"{'─'*70}")
        print(f"  Target Specifications:")
        print(f"    Cutoff:  {cond_denorm['cutoff_frequency']:.2f} Hz")
        print(f"    Q:       {cond_denorm['q_factor']:.3f}")
        print(f"\n  Generated Circuit:")
        print(f"    Topology:   {predicted_type.upper()} ({confidence:.1%} confidence)")
        print(f"    Structure:  {n_poles_pred} poles, {n_zeros_pred} zeros")

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
            'target_cutoff': float(cond_denorm['cutoff_frequency']),
            'target_q': float(cond_denorm['q_factor']),
            'predicted_topology': predicted_type,
            'confidence': float(confidence),
            'num_poles': n_poles_pred,
            'num_zeros': n_zeros_pred,
            'poles': poles.tolist() if len(poles) > 0 else [],
            'zeros': zeros.tolist() if len(zeros) > 0 else []
        })

    # Summary statistics
    if target_specs and 'filter_type' in target_specs:
        target_type = target_specs['filter_type']
        topo_matches = sum(1 for r in results if r['predicted_topology'] == target_type)

        print(f"\n{'='*70}")
        print("GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Target Filter Type: {target_type.upper()}")
        print(f"Topology Match:     {topo_matches}/{num_samples} ({topo_matches/num_samples:.1%})")
        print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate circuits from specifications (Conditional VAE)')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained conditional VAE checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda/mps/cpu)')

    # Generation mode
    parser.add_argument('--mode', type=str, default='specs',
                       choices=['specs', 'explore'],
                       help='Generation mode: specs (from given specs) or explore (sample condition space)')

    # Specification arguments (for 'specs' mode)
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Target cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Target Q factor')
    parser.add_argument('--filter-type', type=str, default='low_pass',
                       choices=['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel'],
                       help='Target filter type')

    # Generation arguments
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of circuits to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Latent sampling temperature (0.5=conservative, 2.0=exploratory)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CONDITIONAL VAE CIRCUIT GENERATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    if args.mode == 'specs':
        print(f"Target Specs:")
        print(f"  Cutoff:      {args.cutoff} Hz")
        print(f"  Q factor:    {args.q_factor}")
        print(f"  Filter type: {args.filter_type}")
    print(f"Num samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print("="*70)

    # Load model
    encoder, decoder, condition_extractor, config, condition_stats = load_model(
        args.checkpoint, args.device
    )

    # Generate based on mode
    if args.mode == 'specs':
        print(f"\n🎯 Generating {args.num_samples} circuits from specifications")
        print(f"   Cutoff: {args.cutoff} Hz, Q: {args.q_factor}, Type: {args.filter_type}")
        print(f"   Temperature: {args.temperature}")

        outputs, conditions = generate_from_specifications(
            decoder,
            args.cutoff,
            args.q_factor,
            args.filter_type,
            condition_extractor,
            args.num_samples,
            args.temperature,
            args.device
        )

        target_specs = {
            'cutoff_frequency': args.cutoff,
            'q_factor': args.q_factor,
            'filter_type': args.filter_type
        }

        results = print_circuit_details(
            outputs,
            conditions,
            condition_extractor,
            mode='specs',
            target_specs=target_specs
        )

    elif args.mode == 'explore':
        print(f"\n🔍 Exploring condition space: {args.num_samples} samples")
        print(f"   Cutoff range: {condition_stats['cutoff_min']:.2f} - {condition_stats['cutoff_max']:.2f} Hz")
        print(f"   Q range:      {condition_stats['q_min']:.3f} - {condition_stats['q_max']:.3f}")
        print(f"   Temperature:  {args.temperature}")

        outputs, conditions, sampled_specs = explore_condition_space(
            decoder,
            condition_stats,
            condition_extractor,
            args.num_samples,
            args.temperature,
            args.device
        )

        results = print_circuit_details(
            outputs,
            conditions,
            condition_extractor,
            mode='explore',
            sampled_specs=sampled_specs
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'mode': args.mode,
            'specifications': {
                'cutoff_frequency': args.cutoff,
                'q_factor': args.q_factor,
                'filter_type': args.filter_type
            } if args.mode == 'specs' else None,
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
