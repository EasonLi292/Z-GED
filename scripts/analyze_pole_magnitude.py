#!/usr/bin/env python3
"""
Analyze pole magnitude correlation with pole count predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
from collections import defaultdict

from ml.data import CircuitDataset


def analyze_pole_magnitudes(dataset_path, config):
    """Analyze pole magnitude distribution across filter types."""

    dataset = CircuitDataset(
        dataset_path=dataset_path,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Collect pole magnitudes by filter type and pole count
    stats = defaultdict(lambda: {'magnitudes': [], 'circuits': []})

    for i in range(len(dataset)):
        circuit = dataset[i]
        filter_type_idx = circuit['filter_type'].argmax().item()
        filter_type_str = filter_type_names[filter_type_idx]

        num_poles = circuit['num_poles'].item()
        poles = circuit['poles'].numpy()

        if len(poles) > 0:
            # Compute pole magnitudes
            for pole in poles:
                real, imag = pole
                magnitude = np.abs(real)  # Magnitude of real part (for stable poles, real < 0)

                key = f"{filter_type_str}_{num_poles}_poles"
                stats[key]['magnitudes'].append(magnitude)
                stats[key]['circuits'].append(i)

    print(f"\n{'='*70}")
    print("POLE MAGNITUDE ANALYSIS")
    print(f"{'='*70}\n")

    # Compute statistics
    for key in sorted(stats.keys()):
        data = stats[key]
        mags = np.array(data['magnitudes'])

        print(f"{key.upper().replace('_', ' ')}:")
        print(f"  Count: {len(mags)} poles")
        print(f"  Mean magnitude: {mags.mean():.4f}")
        print(f"  Std: {mags.std():.4f}")
        print(f"  Min: {mags.min():.4f}")
        print(f"  Max: {mags.max():.4f}")
        print(f"  Median: {np.median(mags):.4f}")
        print()

    # Specific analysis for high-pass
    print(f"{'='*70}")
    print("HIGH-PASS POLE MAGNITUDE BREAKDOWN")
    print(f"{'='*70}\n")

    highpass_key = 'high_pass_1_poles'
    if highpass_key in stats:
        mags = np.array(stats[highpass_key]['magnitudes'])
        circuits = stats[highpass_key]['circuits']

        # Group by magnitude ranges
        ranges = [
            (0, 1, "Small (0-1)"),
            (1, 2.5, "Medium (1-2.5)"),
            (2.5, 10, "Large (2.5+)")
        ]

        for min_mag, max_mag, label in ranges:
            in_range = (mags >= min_mag) & (mags < max_mag)
            count = in_range.sum()
            if count > 0:
                print(f"{label}:")
                print(f"  Count: {count}")
                print(f"  Example circuits: {[circuits[i] for i, x in enumerate(in_range) if x][:5]}")
                print()

    # Compare with other filter types
    print(f"{'='*70}")
    print("COMPARISON: 1-POLE vs 2-POLE FILTERS")
    print(f"{'='*70}\n")

    # Get magnitudes for 1-pole and 2-pole filters
    one_pole_mags = []
    two_pole_mags = []

    for key, data in stats.items():
        if '1_poles' in key:
            one_pole_mags.extend(data['magnitudes'])
        elif '2_poles' in key:
            two_pole_mags.extend(data['magnitudes'])

    if one_pole_mags and two_pole_mags:
        one_pole_mags = np.array(one_pole_mags)
        two_pole_mags = np.array(two_pole_mags)

        print(f"1-pole filters:")
        print(f"  Mean magnitude: {one_pole_mags.mean():.4f}")
        print(f"  Std: {one_pole_mags.std():.4f}")
        print(f"  Range: [{one_pole_mags.min():.4f}, {one_pole_mags.max():.4f}]")
        print()

        print(f"2-pole filters:")
        print(f"  Mean magnitude: {two_pole_mags.mean():.4f}")
        print(f"  Std: {two_pole_mags.std():.4f}")
        print(f"  Range: [{two_pole_mags.min():.4f}, {two_pole_mags.max():.4f}]")
        print()

        # Check overlap
        overlap_threshold = one_pole_mags.max()
        two_pole_in_overlap = (two_pole_mags <= overlap_threshold).sum()
        total_two_pole = len(two_pole_mags)

        print(f"Overlap analysis:")
        print(f"  Max 1-pole magnitude: {one_pole_mags.max():.4f}")
        print(f"  2-pole filters with magnitude <= {one_pole_mags.max():.4f}: {two_pole_in_overlap}/{total_two_pole} ({two_pole_in_overlap/total_two_pole*100:.1f}%)")
        print()

        # Identify the problematic high-pass circuits
        print(f"High-pass circuits with large poles (>= 2.5):")
        highpass_mags = np.array(stats['high_pass_1_poles']['magnitudes'])
        highpass_circuits = stats['high_pass_1_poles']['circuits']

        large_pole_mask = highpass_mags >= 2.5
        large_pole_circuits = [highpass_circuits[i] for i, x in enumerate(large_pole_mask) if x]
        large_pole_values = [highpass_mags[i] for i, x in enumerate(large_pole_mask) if x]

        for circuit_idx, mag in zip(large_pole_circuits, large_pole_values):
            print(f"  Circuit {circuit_idx}: magnitude = {mag:.4f}")
            circuit = dataset[circuit_idx]
            pole = circuit['poles'].numpy()[0]
            print(f"    Pole: {pole[0]:+.4f} {pole[1]:+.4f}j")

    print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl')
    args = parser.parse_args()

    # Load config
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    analyze_pole_magnitudes(args.dataset, config)


if __name__ == '__main__':
    main()
