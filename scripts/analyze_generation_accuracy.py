"""
Comprehensive accuracy analysis for GraphGPT circuit generation.

Tests generation quality across different specifications and analyzes:
- Component value accuracy
- Topology diversity
- Transfer function matching
- Physical realizability
"""

import argparse
import numpy as np
import torch
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.data.dataset import CircuitDataset


def analyze_topology_diversity(circuits, num_samples):
    """Analyze diversity of generated topologies."""
    topologies = []

    for i in range(num_samples):
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()
        node_types = circuits['node_types'][i].cpu().numpy()

        # Create topology signature
        num_nodes = sum(1 for nt in node_types if nt != 4)
        num_edges = int((edge_matrix > 0.5).sum()) // 2

        # Get node type distribution
        node_dist = tuple(sorted([int(nt) for nt in node_types if nt != 4]))

        topology = (num_nodes, num_edges, node_dist)
        topologies.append(topology)

    unique_topologies = len(set(topologies))
    topology_counts = Counter(topologies)

    return {
        'total': num_samples,
        'unique': unique_topologies,
        'diversity_ratio': unique_topologies / num_samples,
        'most_common': topology_counts.most_common(3),
        'all_topologies': topologies
    }


def analyze_component_distributions(circuits, num_samples, norm_stats):
    """Analyze distribution of component values."""
    imp_mean = norm_stats['impedance_mean']
    imp_std = norm_stats['impedance_std']

    resistors = []
    capacitors = []
    inductors = []

    for i in range(num_samples):
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()
        edge_values = circuits['edge_values'][i].cpu().numpy()

        for i_node in range(edge_matrix.shape[0]):
            for j_node in range(i_node + 1, edge_matrix.shape[1]):
                if edge_matrix[i_node, j_node] > 0.5:
                    C_log, G_log, L_inv_log = edge_values[i_node, j_node, :3]
                    has_C, has_R, has_L = edge_values[i_node, j_node, 3:6]

                    # Denormalize
                    C_denorm = np.exp((C_log * imp_std[0]) + imp_mean[0])
                    G_denorm = np.exp((G_log * imp_std[1]) + imp_mean[1])
                    L_inv_denorm = np.exp((L_inv_log * imp_std[2]) + imp_mean[2])

                    C = C_denorm
                    R = 1.0 / (G_denorm + 1e-12)
                    L = 1.0 / (L_inv_denorm + 1e-12)

                    if has_C > 0.5:
                        capacitors.append(C)
                    if has_R > 0.5:
                        resistors.append(R)
                    if has_L > 0.5:
                        inductors.append(L)

    return {
        'resistors': {
            'count': len(resistors),
            'mean': np.mean(resistors) if resistors else 0,
            'std': np.std(resistors) if resistors else 0,
            'min': np.min(resistors) if resistors else 0,
            'max': np.max(resistors) if resistors else 0,
            'practical_range': (10, 100e3),
            'in_range': sum(1 for r in resistors if 10 <= r <= 100e3)
        },
        'capacitors': {
            'count': len(capacitors),
            'mean': np.mean(capacitors) if capacitors else 0,
            'std': np.std(capacitors) if capacitors else 0,
            'min': np.min(capacitors) if capacitors else 0,
            'max': np.max(capacitors) if capacitors else 0,
            'practical_range': (1e-12, 1e-6),
            'in_range': sum(1 for c in capacitors if 1e-12 <= c <= 1e-6)
        },
        'inductors': {
            'count': len(inductors),
            'mean': np.mean(inductors) if inductors else 0,
            'std': np.std(inductors) if inductors else 0,
            'min': np.min(inductors) if inductors else 0,
            'max': np.max(inductors) if inductors else 0,
            'practical_range': (1e-9, 10e-3),
            'in_range': sum(1 for l in inductors if 1e-9 <= l <= 10e-3)
        }
    }


def analyze_pole_zero_accuracy(circuits, num_samples):
    """Analyze pole/zero generation accuracy."""
    pole_counts = circuits['pole_count'].cpu().numpy()
    zero_counts = circuits['zero_count'].cpu().numpy()

    pole_values = circuits['pole_values'].cpu().numpy()
    zero_values = circuits['zero_values'].cpu().numpy()

    analysis = {
        'pole_count': {
            'mean': np.mean(pole_counts),
            'std': np.std(pole_counts),
            'min': np.min(pole_counts),
            'max': np.max(pole_counts),
            'distribution': Counter(pole_counts)
        },
        'zero_count': {
            'mean': np.mean(zero_counts),
            'std': np.std(zero_counts),
            'min': np.min(zero_counts),
            'max': np.max(zero_counts),
            'distribution': Counter(zero_counts)
        }
    }

    # Analyze pole/zero stability (should have negative real parts)
    stable_poles = 0
    total_poles = 0

    for i in range(num_samples):
        n_poles = pole_counts[i]
        if n_poles > 0:
            poles = pole_values[i, :n_poles]
            for pole in poles:
                real_part = pole[0]
                total_poles += 1
                if real_part < 0:
                    stable_poles += 1

    analysis['stability'] = {
        'stable_poles': stable_poles,
        'total_poles': total_poles,
        'stability_ratio': stable_poles / total_poles if total_poles > 0 else 1.0
    }

    return analysis


def main(args):
    """Main analysis function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("GraphGPT Generation Accuracy Analysis")
    print(f"{'='*70}\n")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

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

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    # Get normalization stats
    dataset = CircuitDataset(config['data']['dataset_path'])
    norm_stats = {
        'impedance_mean': dataset.impedance_mean.cpu().numpy(),
        'impedance_std': dataset.impedance_std.cpu().numpy()
    }

    print("Model loaded successfully\n")

    # Test across different specifications
    test_specs = [
        (100.0, 0.5, "Low-pass (100Hz, Q=0.5)"),
        (1000.0, 0.707, "Standard (1kHz, Q=0.707)"),
        (5000.0, 1.5, "Band-pass (5kHz, Q=1.5)"),
        (10000.0, 2.0, "High-Q (10kHz, Q=2.0)")
    ]

    all_results = []

    for cutoff, q_factor, description in test_specs:
        print(f"{'='*70}")
        print(f"Testing: {description}")
        print(f"{'='*70}\n")

        # Generate circuits
        latent = torch.randn(args.num_samples, config['model']['decoder']['latent_dim'], device=device)

        log_cutoff = np.log10(cutoff) / 4.0
        log_q = np.log10(max(q_factor, 0.1)) / 2.0

        conditions = torch.tensor(
            [[log_cutoff, log_q]] * args.num_samples,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            circuits = decoder.generate(
                latent_code=latent,
                conditions=conditions,
                enforce_constraints=True,
                edge_threshold=0.5
            )

        # Analyze topology diversity
        topo_analysis = analyze_topology_diversity(circuits, args.num_samples)

        print(f"Topology Diversity:")
        print(f"  Total circuits: {topo_analysis['total']}")
        print(f"  Unique topologies: {topo_analysis['unique']}")
        print(f"  Diversity ratio: {topo_analysis['diversity_ratio']:.2%}")
        print(f"  Most common topologies:")
        for topo, count in topo_analysis['most_common']:
            nodes, edges, _ = topo
            print(f"    {nodes} nodes, {edges} edges: {count} circuits ({100*count/args.num_samples:.1f}%)")
        print()

        # Analyze component distributions
        comp_analysis = analyze_component_distributions(circuits, args.num_samples, norm_stats)

        print(f"Component Analysis:")
        for comp_type, stats in comp_analysis.items():
            if stats['count'] > 0:
                print(f"  {comp_type.capitalize()}:")
                print(f"    Count: {stats['count']}")
                print(f"    Mean: {stats['mean']:.3e}")
                print(f"    Range: [{stats['min']:.3e}, {stats['max']:.3e}]")
                print(f"    Practical: {stats['in_range']}/{stats['count']} ({100*stats['in_range']/stats['count']:.1f}%)")
        print()

        # Analyze poles/zeros
        pz_analysis = analyze_pole_zero_accuracy(circuits, args.num_samples)

        print(f"Pole/Zero Analysis:")
        print(f"  Pole count: {pz_analysis['pole_count']['mean']:.1f} ± {pz_analysis['pole_count']['std']:.1f}")
        print(f"  Zero count: {pz_analysis['zero_count']['mean']:.1f} ± {pz_analysis['zero_count']['std']:.1f}")
        print(f"  Stable poles: {pz_analysis['stability']['stable_poles']}/{pz_analysis['stability']['total_poles']} ({100*pz_analysis['stability']['stability_ratio']:.1f}%)")
        print()

        # Calculate success rate
        edge_matrices = circuits['edge_existence'].cpu().numpy()
        valid_circuits = sum(1 for i in range(args.num_samples) if (edge_matrices[i] > 0.5).sum() > 0)

        print(f"Generation Success:")
        print(f"  Valid circuits: {valid_circuits}/{args.num_samples} ({100*valid_circuits/args.num_samples:.1f}%)")
        print()

        all_results.append({
            'spec': description,
            'topology': topo_analysis,
            'components': comp_analysis,
            'poles_zeros': pz_analysis,
            'success_rate': valid_circuits / args.num_samples
        })

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")

    avg_diversity = np.mean([r['topology']['diversity_ratio'] for r in all_results])
    avg_success = np.mean([r['success_rate'] for r in all_results])

    print(f"Across all {len(test_specs)} test conditions ({args.num_samples} circuits each):")
    print(f"  Average topology diversity: {100*avg_diversity:.1f}%")
    print(f"  Average success rate: {100*avg_success:.1f}%")
    print(f"  Total circuits generated: {len(test_specs) * args.num_samples}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/graphgpt_decoder/best.pt')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='mps')

    args = parser.parse_args()
    main(args)
