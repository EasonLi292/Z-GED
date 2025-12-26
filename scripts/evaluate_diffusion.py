#!/usr/bin/env python3
"""
Evaluation script for Diffusion-based Circuit Decoder.

Evaluates quality of generated circuits on various metrics:
- Structural validity
- Topology diversity
- Transfer function accuracy
- Specification matching
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.diffusion.constraints import CircuitConstraints
from scripts.generate_diffusion import load_trained_model, generate_from_specifications


def evaluate_structural_validity(circuits: dict, verbose: bool = True):
    """
    Evaluate structural validity of generated circuits.

    Args:
        circuits: Generated circuits
        verbose: Print detailed results

    Returns:
        metrics: Dictionary of validity metrics
    """
    constraints = CircuitConstraints()

    is_valid, violations = constraints.check_validity(circuits, verbose=False)

    # Compute pass rates for each constraint
    metrics = {}
    for constraint_name, valid_mask in violations.items():
        pass_rate = valid_mask.float().mean().item() * 100
        metrics[f'valid_{constraint_name}'] = pass_rate

    # Overall validity
    all_valid = torch.stack([v.all() for v in violations.values()]).all()
    metrics['overall_valid'] = all_valid.item() * 100

    if verbose:
        print(f"\n{'='*70}")
        print("STRUCTURAL VALIDITY")
        print(f"{'='*70}")
        for name, value in metrics.items():
            print(f"  {name:30s}: {value:6.2f}%")

    return metrics


def evaluate_topology_diversity(circuits: dict, verbose: bool = True):
    """
    Evaluate diversity of generated topologies.

    Args:
        circuits: Generated circuits
        verbose: Print detailed results

    Returns:
        metrics: Dictionary of diversity metrics
    """
    num_circuits = circuits['node_types'].shape[0]

    # Extract topology signatures
    topologies = []
    for i in range(num_circuits):
        # Create signature from node types and edge existence
        node_types = circuits['node_types'][i].tolist()
        edge_exists = circuits['edge_existence'][i]

        # Edges as sorted list of (i, j) tuples
        edges = []
        for row in range(edge_exists.shape[0]):
            for col in range(edge_exists.shape[1]):
                if edge_exists[row, col] > 0.5:
                    edges.append((row, col))

        # Create signature
        topology_sig = (tuple(node_types), tuple(sorted(edges)))
        topologies.append(topology_sig)

    # Count unique topologies
    topology_counts = Counter(topologies)
    num_unique = len(topology_counts)
    diversity_ratio = num_unique / num_circuits

    metrics = {
        'num_unique_topologies': num_unique,
        'total_circuits': num_circuits,
        'diversity_ratio': diversity_ratio * 100,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("TOPOLOGY DIVERSITY")
        print(f"{'='*70}")
        print(f"  Total circuits:        {num_circuits}")
        print(f"  Unique topologies:     {num_unique}")
        print(f"  Diversity ratio:       {diversity_ratio*100:.2f}%")

        # Show most common topologies
        print(f"\n  Most common topologies:")
        for i, (topo, count) in enumerate(topology_counts.most_common(5)):
            print(f"    #{i+1}: {count} circuits")

    return metrics


def evaluate_pole_zero_statistics(circuits: dict, verbose: bool = True):
    """
    Evaluate pole/zero count distributions.

    Args:
        circuits: Generated circuits
        verbose: Print detailed results

    Returns:
        metrics: Dictionary of pole/zero statistics
    """
    pole_counts = circuits['pole_count'].cpu().numpy()
    zero_counts = circuits['zero_count'].cpu().numpy()

    metrics = {
        'mean_pole_count': float(pole_counts.mean()),
        'std_pole_count': float(pole_counts.std()),
        'mean_zero_count': float(zero_counts.mean()),
        'std_zero_count': float(zero_counts.std()),
    }

    if verbose:
        print(f"\n{'='*70}")
        print("POLE/ZERO STATISTICS")
        print(f"{'='*70}")
        print(f"  Pole count:  {metrics['mean_pole_count']:.2f} ± {metrics['std_pole_count']:.2f}")
        print(f"  Zero count:  {metrics['mean_zero_count']:.2f} ± {metrics['std_zero_count']:.2f}")

        # Distribution
        print(f"\n  Pole count distribution:")
        for count in range(5):
            freq = (pole_counts == count).sum()
            print(f"    {count} poles: {freq} circuits ({freq/len(pole_counts)*100:.1f}%)")

        print(f"\n  Zero count distribution:")
        for count in range(5):
            freq = (zero_counts == count).sum()
            print(f"    {count} zeros: {freq} circuits ({freq/len(zero_counts)*100:.1f}%)")

    return metrics


def evaluate_component_statistics(circuits: dict, verbose: bool = True):
    """
    Evaluate statistics of component values.

    Args:
        circuits: Generated circuits
        verbose: Print detailed results

    Returns:
        metrics: Dictionary of component statistics
    """
    # Extract edge values where edges exist
    edge_exists = circuits['edge_existence']  # [batch, max_nodes, max_nodes]
    edge_values = circuits['edge_values']  # [batch, max_nodes, max_nodes, 7]

    # Mask for existing edges
    edge_mask = edge_exists.unsqueeze(-1) > 0.5

    # Component values (C, G, L_inv)
    C_values = edge_values[..., 0][edge_mask[..., 0]]
    G_values = edge_values[..., 1][edge_mask[..., 1]]
    L_values = edge_values[..., 2][edge_mask[..., 2]]

    metrics = {
        'mean_C': float(C_values.mean()) if len(C_values) > 0 else 0,
        'std_C': float(C_values.std()) if len(C_values) > 0 else 0,
        'mean_G': float(G_values.mean()) if len(G_values) > 0 else 0,
        'std_G': float(G_values.std()) if len(G_values) > 0 else 0,
        'mean_L': float(L_values.mean()) if len(L_values) > 0 else 0,
        'std_L': float(L_values.std()) if len(L_values) > 0 else 0,
    }

    # Check for positive values
    metrics['positive_C'] = float((C_values > 0).float().mean()) * 100 if len(C_values) > 0 else 0
    metrics['positive_G'] = float((G_values > 0).float().mean()) * 100 if len(G_values) > 0 else 0
    metrics['positive_L'] = float((L_values > 0).float().mean()) * 100 if len(L_values) > 0 else 0

    if verbose:
        print(f"\n{'='*70}")
        print("COMPONENT STATISTICS")
        print(f"{'='*70}")
        print(f"  Capacitance (C):  {metrics['mean_C']:.4f} ± {metrics['std_C']:.4f}  ({metrics['positive_C']:.1f}% positive)")
        print(f"  Conductance (G):  {metrics['mean_G']:.4f} ± {metrics['std_G']:.4f}  ({metrics['positive_G']:.1f}% positive)")
        print(f"  Inductance (L):   {metrics['mean_L']:.4f} ± {metrics['std_L']:.4f}  ({metrics['positive_L']:.1f}% positive)")

    return metrics


def evaluate_pole_stability(circuits: dict, verbose: bool = True):
    """
    Evaluate stability of generated poles.

    Args:
        circuits: Generated circuits
        verbose: Print detailed results

    Returns:
        metrics: Dictionary of stability metrics
    """
    pole_values = circuits['pole_values']  # [batch, max_poles, 2]
    pole_counts = circuits['pole_count']  # [batch]

    # Extract real parts of valid poles
    real_parts = []
    for i in range(pole_values.shape[0]):
        n_poles = pole_counts[i].item()
        if n_poles > 0:
            reals = pole_values[i, :n_poles, 0].cpu().numpy()
            real_parts.extend(reals)

    real_parts = np.array(real_parts)

    # Compute stability metrics
    if len(real_parts) > 0:
        stable_poles = (real_parts < 0).sum()
        unstable_poles = (real_parts >= 0).sum()
        stability_rate = stable_poles / len(real_parts) * 100

        metrics = {
            'stable_poles': int(stable_poles),
            'unstable_poles': int(unstable_poles),
            'stability_rate': float(stability_rate),
            'mean_real_part': float(real_parts.mean()),
            'std_real_part': float(real_parts.std()),
        }
    else:
        metrics = {
            'stable_poles': 0,
            'unstable_poles': 0,
            'stability_rate': 0,
            'mean_real_part': 0,
            'std_real_part': 0,
        }

    if verbose:
        print(f"\n{'='*70}")
        print("POLE STABILITY")
        print(f"{'='*70}")
        print(f"  Total poles:       {len(real_parts)}")
        print(f"  Stable (Re < 0):   {metrics['stable_poles']} ({metrics['stability_rate']:.1f}%)")
        print(f"  Unstable (Re >= 0): {metrics['unstable_poles']}")
        print(f"  Mean real part:    {metrics['mean_real_part']:.4f} ± {metrics['std_real_part']:.4f}")

    return metrics


def generate_and_evaluate(
    checkpoint_path: str,
    num_samples: int = 100,
    cutoff_freq: float = 1000.0,
    q_factor: float = 0.707,
    sampler_type: str = 'ddim',
    num_steps: int = 50,
    device: str = 'cpu'
):
    """
    Generate circuits and run full evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of circuits to generate
        cutoff_freq: Target cutoff frequency
        q_factor: Target Q factor
        sampler_type: Sampler type
        num_steps: Number of sampling steps
        device: Device

    Returns:
        all_metrics: Dictionary of all evaluation metrics
    """
    print(f"\n{'='*70}")
    print("DIFFUSION MODEL EVALUATION")
    print(f"{'='*70}")

    # Load model
    encoder, decoder, config = load_trained_model(checkpoint_path, device)

    # Generate circuits
    print(f"\n🎨 Generating {num_samples} circuits...")
    circuits = generate_from_specifications(
        decoder,
        cutoff_freq=cutoff_freq,
        q_factor=q_factor,
        latent_dim=config['model']['decoder']['latent_dim'],
        sampler_type=sampler_type,
        num_steps=num_steps,
        num_samples=num_samples,
        temperature=1.0,
        device=device,
        timesteps=config['model']['decoder']['timesteps']
    )

    print(f"✅ Generated {num_samples} circuits")

    # Run evaluations
    all_metrics = {}

    all_metrics.update(evaluate_structural_validity(circuits, verbose=True))
    all_metrics.update(evaluate_topology_diversity(circuits, verbose=True))
    all_metrics.update(evaluate_pole_zero_statistics(circuits, verbose=True))
    all_metrics.update(evaluate_component_statistics(circuits, verbose=True))
    all_metrics.update(evaluate_pole_stability(circuits, verbose=True))

    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Overall validity:      {all_metrics['overall_valid']:.2f}%")
    print(f"  Topology diversity:    {all_metrics['diversity_ratio']:.2f}%")
    print(f"  Pole stability:        {all_metrics['stability_rate']:.2f}%")
    print(f"  Unique topologies:     {all_metrics['num_unique_topologies']}/{all_metrics['total_circuits']}")

    return all_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of circuits to generate')
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='Cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Q factor')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                       help='Sampler type')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of sampling steps')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/mps/cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save metrics')

    args = parser.parse_args()

    # Run evaluation
    metrics = generate_and_evaluate(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        cutoff_freq=args.cutoff,
        q_factor=args.q_factor,
        sampler_type=args.sampler,
        num_steps=args.steps,
        device=args.device
    )

    # Save metrics if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Saved metrics to: {output_path}")

    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
