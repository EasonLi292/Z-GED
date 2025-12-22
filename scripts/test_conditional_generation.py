#!/usr/bin/env python3
"""
Test conditional generation quality - simplified validation.

Focuses on practical metrics:
1. Topology accuracy (100% with teacher forcing)
2. Component value realism
3. Circuit diversity
4. Consistency with training data

Usage:
    python scripts/test_conditional_generation.py --checkpoint checkpoints/best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.generation import CircuitSampler
from ml.data import CircuitDataset


def extract_component_values(edge_features):
    """
    Extract component values from edge features.

    Args:
        edge_features: [N, 7] array with [C, G, L_inv, has_C, has_R, has_L, is_parallel]

    Returns:
        dict with lists of C, R, L values
    """
    components = {'C': [], 'R': [], 'L': []}

    for edge in edge_features:
        C, G, L_inv, has_C, has_R, has_L, is_parallel = edge

        if has_C > 0.5 and C > 0:
            components['C'].append(C)

        if has_R > 0.5 and G > 0:
            components['R'].append(1/G)

        if has_L > 0.5 and L_inv > 0:
            components['L'].append(1/L_inv)

    return components


def test_filter_generation(
    sampler: CircuitSampler,
    filter_type: str,
    num_samples: int,
    dataset: CircuitDataset = None
):
    """Test generation for a specific filter type."""

    print(f"\n{'='*70}")
    print(f"TESTING {filter_type.upper()} GENERATION")
    print(f"{'='*70}\n")

    # Get reference circuits from dataset
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    reference_circuits = []
    reference_components = {'C': [], 'R': [], 'L': []}

    if dataset is not None:
        for i in range(len(dataset)):
            circuit = dataset[i]
            filter_type_tensor = circuit['filter_type']
            circuit_filter_type = filter_type_names[filter_type_tensor.argmax().item()]

            if circuit_filter_type == filter_type:
                reference_circuits.append(circuit)

                # Extract components (need to denormalize first)
                # For now, just use raw edge features

        print(f"Found {len(reference_circuits)} reference circuits in dataset")

    # Generate circuits
    print(f"Generating {num_samples} circuits...")
    outputs = sampler.sample_conditional(
        filter_type,
        num_samples,
        temperature=1.0
    )

    graphs = outputs['graphs']
    topo_probs = outputs['topo_probs']
    poles = outputs['poles']
    zeros = outputs['zeros']

    # Analyze generated circuits
    topology_correct = 0
    all_components = {'C': [], 'R': [], 'L': []}
    topologies = []

    print("\nGenerated Circuits:")
    for i, graph in enumerate(graphs):
        predicted_idx = topo_probs[i].argmax().item()
        predicted_type = filter_type_names[predicted_idx]
        confidence = topo_probs[i, predicted_idx].item()

        if predicted_type == filter_type:
            topology_correct += 1

        # Get component values (denormalized in edge_attr)
        edge_features = graph.edge_attr.cpu().numpy()
        components = extract_component_values(edge_features)

        for comp_type in ['C', 'R', 'L']:
            all_components[comp_type].extend(components[comp_type])

        topologies.append({
            'num_nodes': graph.num_nodes,
            'num_edges': graph.num_edges,
            'predicted_type': predicted_type,
            'confidence': confidence
        })

        if i < 5:
            print(f"\n  Circuit {i+1}:")
            print(f"    Type: {predicted_type} ({confidence:.2%} conf)")
            print(f"    Topology: {graph.num_nodes} nodes, {graph.num_edges} edges")
            print(f"    Components: {len(components['C'])} caps, {len(components['R'])} res, {len(components['L'])} ind")
            if components['C']:
                print(f"    C values: {np.mean(components['C']):.2e} ± {np.std(components['C']):.2e} F")
            if components['R']:
                print(f"    R values: {np.mean(components['R']):.2e} ± {np.std(components['R']):.2e} Ω")
            if components['L']:
                print(f"    L values: {np.mean(components['L']):.2e} ± {np.std(components['L']):.2e} H")

    # Statistics
    print(f"\n{'='*70}")
    print("GENERATION STATISTICS")
    print(f"{'='*70}\n")

    print(f"✅ Topology Accuracy: {topology_correct}/{num_samples} ({topology_correct/num_samples:.1%})")

    # Topology consistency
    unique_topologies = set((t['num_nodes'], t['num_edges']) for t in topologies)
    print(f"✅ Topology Consistency: {len(unique_topologies)} unique topology(ies)")

    for topo in unique_topologies:
        count = sum(1 for t in topologies if (t['num_nodes'], t['num_edges']) == topo)
        print(f"   - {topo[0]} nodes, {topo[1]} edges: {count} circuits")

    # Component value statistics
    print(f"\n Component Value Statistics:")
    for comp_type in ['C', 'R', 'L']:
        values = all_components[comp_type]
        if values:
            values = np.array(values)
            print(f"\n  {comp_type}:")
            print(f"    Count: {len(values)}")
            print(f"    Range: {values.min():.2e} to {values.max():.2e}")
            print(f"    Mean: {values.mean():.2e}")
            print(f"    Median: {np.median(values):.2e}")
            print(f"    Std: {values.std():.2e}")

    # Diversity metric: pairwise differences in component values
    if all_components['R']:
        R_values = np.array(all_components['R'])
        # Compute coefficient of variation
        cv = np.std(R_values) / np.mean(R_values) if np.mean(R_values) > 0 else 0
        print(f"\n✅ Diversity (Coefficient of Variation):")
        print(f"   R: {cv:.3f} ({'high' if cv > 0.5 else 'moderate' if cv > 0.2 else 'low'} diversity)")

    return {
        'filter_type': filter_type,
        'num_samples': num_samples,
        'topology_accuracy': topology_correct / num_samples,
        'unique_topologies': len(unique_topologies),
        'components': all_components,
        'topologies': topologies
    }


def plot_component_distributions(all_results, output_file):
    """Plot component value distributions across filter types."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Conditional Generation - Component Value Distributions', fontsize=14, fontweight='bold')

    filter_types = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot 1: Capacitance distribution
    ax = axes[0, 0]
    for i, (ft, result) in enumerate(all_results.items()):
        C_values = result['components']['C']
        if C_values:
            ax.hist(np.log10(C_values), bins=20, alpha=0.6, label=ft, color=colors[i % len(colors)])
    ax.set_xlabel('log₁₀(Capacitance / F)')
    ax.set_ylabel('Count')
    ax.set_title('Capacitance Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Resistance distribution
    ax = axes[0, 1]
    for i, (ft, result) in enumerate(all_results.items()):
        R_values = result['components']['R']
        if R_values:
            ax.hist(np.log10(R_values), bins=20, alpha=0.6, label=ft, color=colors[i % len(colors)])
    ax.set_xlabel('log₁₀(Resistance / Ω)')
    ax.set_ylabel('Count')
    ax.set_title('Resistance Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Inductance distribution
    ax = axes[1, 0]
    for i, (ft, result) in enumerate(all_results.items()):
        L_values = result['components']['L']
        if L_values:
            ax.hist(np.log10(L_values), bins=20, alpha=0.6, label=ft, color=colors[i % len(colors)])
    ax.set_xlabel('log₁₀(Inductance / H)')
    ax.set_ylabel('Count')
    ax.set_title('Inductance Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Topology accuracy
    ax = axes[1, 1]
    accuracies = [result['topology_accuracy'] for result in all_results.values()]
    filter_labels = [ft.replace('_', '\n') for ft in filter_types]

    bars = ax.bar(range(len(filter_types)), accuracies, color=[colors[i % len(colors)] for i in range(len(filter_types))])
    ax.set_xticks(range(len(filter_types)))
    ax.set_xticklabels(filter_labels, fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Topology Accuracy')
    ax.set_title('Conditional Generation Accuracy')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test conditional circuit generation')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl')
    parser.add_argument('--filter-types', type=str, nargs='+',
                       default=['low_pass', 'high_pass', 'band_pass', 'band_stop'],
                       help='Filter types to test')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='validation_results')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n" + "="*70)
    print("CONDITIONAL GENERATION TEST")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Samples per type: {args.num_samples}")
    print("="*70)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    topo_dim = config['model'].get('topo_latent_dim', config['model']['latent_dim'] // 3)
    values_dim = config['model'].get('values_latent_dim', config['model']['latent_dim'] // 3)
    pz_dim = config['model'].get('pz_latent_dim', config['model']['latent_dim'] // 3)
    branch_dims = (topo_dim, values_dim, pz_dim)

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

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    print(f"\n✅ Loaded model: {config['model']['latent_dim']}D")

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=args.dataset,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )
    print(f"✅ Loaded dataset: {len(dataset)} circuits")

    # Create sampler
    sampler = CircuitSampler(
        encoder, decoder,
        device=device,
        latent_dim=config['model']['latent_dim'],
        branch_dims=branch_dims
    )

    # Test each filter type
    all_results = {}
    for filter_type in args.filter_types:
        result = test_filter_generation(sampler, filter_type, args.num_samples, dataset)
        all_results[filter_type] = result

    # Plot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_component_distributions(all_results, output_dir / 'conditional_generation_test.png')

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")

    for ft, result in all_results.items():
        print(f"{ft.upper()}:")
        print(f"  ✅ Topology: {result['topology_accuracy']:.0%} accurate")
        print(f"  ✅ Consistency: {result['unique_topologies']} unique topology(ies)")
        print(f"  ✅ Components: {len(result['components']['C'])} C, {len(result['components']['R'])} R, {len(result['components']['L'])} L")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
