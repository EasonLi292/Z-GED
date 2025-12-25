#!/usr/bin/env python3
"""
Analyze the high-pass filter pole count prediction issue.

This script investigates why the model predicts 2 poles instead of 1
for high-pass filters.

Usage:
    python scripts/analyze_highpass_issue.py \
        --checkpoint checkpoints/variable_length/20251222_102121/best.pt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from collections import Counter, defaultdict

from ml.models import HierarchicalEncoder
from ml.models.variable_decoder import VariableLengthDecoder
from ml.data import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def analyze_dataset_distribution(dataset_path, config):
    """Analyze pole/zero count distribution in dataset."""

    dataset = CircuitDataset(
        dataset_path=dataset_path,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Collect statistics
    filter_stats = defaultdict(lambda: {'pole_counts': [], 'zero_counts': [], 'circuits': []})

    for i in range(len(dataset)):
        circuit = dataset[i]
        filter_type_idx = circuit['filter_type'].argmax().item()
        filter_type_str = filter_type_names[filter_type_idx]

        num_poles = circuit['num_poles'].item()
        num_zeros = circuit['num_zeros'].item()

        filter_stats[filter_type_str]['pole_counts'].append(num_poles)
        filter_stats[filter_type_str]['zero_counts'].append(num_zeros)
        filter_stats[filter_type_str]['circuits'].append(i)

    print(f"\n{'='*70}")
    print("DATASET DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total circuits: {len(dataset)}\n")

    for filter_type in filter_type_names:
        stats = filter_stats[filter_type]
        if not stats['pole_counts']:
            continue

        pole_counter = Counter(stats['pole_counts'])
        zero_counter = Counter(stats['zero_counts'])

        print(f"{filter_type.upper()}:")
        print(f"  Total: {len(stats['circuits'])} circuits")
        print(f"  Pole counts: {dict(sorted(pole_counter.items()))}")
        print(f"  Zero counts: {dict(sorted(zero_counter.items()))}")
        print()

    # Detailed high-pass analysis
    print(f"{'='*70}")
    print("HIGH-PASS FILTER DETAILED ANALYSIS")
    print(f"{'='*70}")

    highpass_stats = filter_stats['high_pass']
    if highpass_stats['pole_counts']:
        print(f"\nTotal high-pass circuits: {len(highpass_stats['circuits'])}")
        print(f"Pole count distribution:")
        pole_counter = Counter(highpass_stats['pole_counts'])
        for count, num in sorted(pole_counter.items()):
            pct = num / len(highpass_stats['pole_counts']) * 100
            print(f"  {count} poles: {num} circuits ({pct:.1f}%)")

        print(f"\nZero count distribution:")
        zero_counter = Counter(highpass_stats['zero_counts'])
        for count, num in sorted(zero_counter.items()):
            pct = num / len(highpass_stats['zero_counts']) * 100
            print(f"  {count} zeros: {num} circuits ({pct:.1f}%)")

        # Show individual circuits
        print(f"\nIndividual high-pass circuits:")
        for idx in highpass_stats['circuits']:
            circuit = dataset[idx]
            num_poles = circuit['num_poles'].item()
            num_zeros = circuit['num_zeros'].item()
            poles = circuit['poles'].numpy()
            zeros = circuit['zeros'].numpy()

            print(f"\n  Circuit {idx}:")
            print(f"    Poles: {num_poles}, Zeros: {num_zeros}")
            if len(poles) > 0:
                print(f"    Pole values:")
                for i, pole in enumerate(poles):
                    print(f"      {i+1}. {pole[0]:+.4f} {pole[1]:+.4f}j")
            if len(zeros) > 0:
                print(f"    Zero values:")
                for i, zero in enumerate(zeros):
                    print(f"      {i+1}. {zero[0]:+.4f} {zero[1]:+.4f}j")

    return filter_stats


def test_model_on_all_highpass(checkpoint_path, dataset_path, config, device):
    """Test model predictions on all high-pass filters."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=dataset_path,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Find all high-pass circuits
    highpass_indices = []
    for i in range(len(dataset)):
        circuit = dataset[i]
        filter_type_idx = circuit['filter_type'].argmax().item()
        if filter_type_names[filter_type_idx] == 'high_pass':
            highpass_indices.append(i)

    print(f"\n{'='*70}")
    print("MODEL PREDICTIONS ON ALL HIGH-PASS FILTERS")
    print(f"{'='*70}")
    print(f"Testing on {len(highpass_indices)} high-pass circuits\n")

    results = {
        '1_pole_1_zero': {'correct': 0, 'total': 0, 'predictions': []},
        '2_pole_2_zero': {'correct': 0, 'total': 0, 'predictions': []},
        'other': {'correct': 0, 'total': 0, 'predictions': []}
    }

    with torch.no_grad():
        for idx in highpass_indices:
            circuit = dataset[idx]

            # Ground truth
            num_poles_gt = circuit['num_poles'].item()
            num_zeros_gt = circuit['num_zeros'].item()

            # Encode
            graph = circuit['graph'].to(device)
            poles_list = [circuit['poles'].to(device)]
            zeros_list = [circuit['zeros'].to(device)]
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles_list,
                zeros_list
            )

            # Decode
            outputs = decoder(mu, hard=False, gt_filter_type=None)

            # Predictions
            pred_pole_count = outputs['pole_count_logits'].argmax(dim=-1).item()
            pred_zero_count = outputs['zero_count_logits'].argmax(dim=-1).item()

            # Get logits for analysis
            pole_logits = outputs['pole_count_logits'][0].cpu().numpy()
            zero_logits = outputs['zero_count_logits'][0].cpu().numpy()
            pole_probs = torch.softmax(outputs['pole_count_logits'][0], dim=-1).cpu().numpy()
            zero_probs = torch.softmax(outputs['zero_count_logits'][0], dim=-1).cpu().numpy()

            # Categorize
            structure = f"{num_poles_gt}_pole_{num_zeros_gt}_zero"
            if structure not in results:
                structure = 'other'

            results[structure]['total'] += 1
            if pred_pole_count == num_poles_gt and pred_zero_count == num_zeros_gt:
                results[structure]['correct'] += 1

            results[structure]['predictions'].append({
                'idx': idx,
                'gt_poles': num_poles_gt,
                'gt_zeros': num_zeros_gt,
                'pred_poles': pred_pole_count,
                'pred_zeros': pred_zero_count,
                'pole_probs': pole_probs,
                'zero_probs': zero_probs,
                'correct': (pred_pole_count == num_poles_gt and pred_zero_count == num_zeros_gt)
            })

    # Print results by structure
    for structure, data in results.items():
        if data['total'] == 0:
            continue

        print(f"\n{structure.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {data['correct']}/{data['total']} ({data['correct']/data['total']*100:.1f}%)")

        # Show individual predictions
        print(f"  Individual predictions:")
        for pred in data['predictions']:
            status = "✅" if pred['correct'] else "❌"
            print(f"\n    Circuit {pred['idx']} {status}")
            print(f"      GT:   {pred['gt_poles']} poles, {pred['gt_zeros']} zeros")
            print(f"      Pred: {pred['pred_poles']} poles, {pred['pred_zeros']} zeros")
            print(f"      Pole probs: {' '.join([f'{i}:{p:.3f}' for i, p in enumerate(pred['pole_probs'])])}")
            print(f"      Zero probs: {' '.join([f'{i}:{p:.3f}' for i, p in enumerate(pred['zero_probs'])])}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())

    print(f"Overall high-pass accuracy: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%)")

    # Analyze the failure pattern
    pole_1_predictions = [p for r in results.values() for p in r['predictions'] if p['gt_poles'] == 1]
    if pole_1_predictions:
        pole_1_as_2 = sum(1 for p in pole_1_predictions if p['pred_poles'] == 2)
        print(f"\nKey finding:")
        print(f"  Circuits with 1 pole predicted as 2 poles: {pole_1_as_2}/{len(pole_1_predictions)} ({pole_1_as_2/len(pole_1_predictions)*100:.1f}%)")

        # Show probability distributions for 1-pole circuits
        print(f"\n  Pole count probability distribution for 1-pole circuits:")
        avg_probs = np.mean([p['pole_probs'] for p in pole_1_predictions], axis=0)
        for i, prob in enumerate(avg_probs):
            print(f"    P({i} poles) = {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze high-pass filter issue')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')

    args = parser.parse_args()

    # Load config
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*70}")
    print("HIGH-PASS FILTER POLE COUNT ISSUE INVESTIGATION")
    print(f"{'='*70}")

    # Step 1: Analyze dataset distribution
    filter_stats = analyze_dataset_distribution(args.dataset, config)

    # Step 2: Test model on all high-pass filters
    test_model_on_all_highpass(args.checkpoint, args.dataset, config, args.device)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
