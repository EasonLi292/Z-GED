#!/usr/bin/env python3
"""Explore behavioral specifications in the dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
from collections import defaultdict

from ml.data import CircuitDataset


def compute_transfer_function_from_poles_zeros(poles, zeros, gain=1.0):
    """Compute transfer function from poles and zeros."""
    poles_complex = poles[:, 0] + 1j * poles[:, 1] if len(poles) > 0 else np.array([])
    zeros_complex = zeros[:, 0] + 1j * zeros[:, 1] if len(zeros) > 0 else np.array([])

    def H(s):
        numerator = complex(gain)
        for z in zeros_complex:
            numerator *= (s - z)

        denominator = complex(1.0)
        for p in poles_complex:
            denominator *= (s - p)

        if abs(denominator) < 1e-10:
            return complex(0.0)

        return numerator / denominator

    return H


def analyze_filter(poles, zeros, gain=1.0):
    """Analyze filter characteristics."""
    if len(poles) == 0:
        return None

    H = compute_transfer_function_from_poles_zeros(poles, zeros, gain)

    freqs = np.logspace(1, 8, 1000)
    omega = 2 * np.pi * freqs
    s_vals = 1j * omega

    response = np.array([abs(H(s)) for s in s_vals])
    response_db = 20 * np.log10(response + 1e-10)

    dc_gain_db = response_db[0]
    hf_gain_db = response_db[-1]

    cutoff_freq = None

    if dc_gain_db > hf_gain_db + 10:
        # Low-pass
        threshold = dc_gain_db - 3
        idx = np.where(response_db < threshold)[0]
        if len(idx) > 0:
            cutoff_freq = freqs[idx[0]]
    elif hf_gain_db > dc_gain_db + 10:
        # High-pass
        threshold = hf_gain_db - 3
        idx = np.where(response_db > threshold)[0]
        if len(idx) > 0:
            cutoff_freq = freqs[idx[0]]

    # Q factor
    q_factor = None
    for pole in poles:
        real, imag = pole
        omega_n = np.sqrt(real**2 + imag**2)
        if abs(omega_n) > 1e-6:
            zeta = -real / omega_n
            q = 1 / (2 * zeta) if abs(zeta) > 1e-6 else float('inf')
            if q_factor is None or (q > 0.5 and q < 100):
                q_factor = q

    return {
        'cutoff_freq': cutoff_freq,
        'q_factor': q_factor
    }


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

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=args.dataset,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Collect specs by filter type
    specs_by_type = defaultdict(list)

    for i in range(len(dataset)):
        circuit = dataset[i]
        filter_idx = circuit['filter_type'].argmax().item()
        filter_type = filter_type_names[filter_idx]

        poles = circuit['poles'].numpy()
        zeros = circuit['zeros'].numpy()

        chars = analyze_filter(poles, zeros, gain=1.0)

        if chars:
            specs_by_type[filter_type].append({
                'idx': i,
                'cutoff': chars['cutoff_freq'],
                'q_factor': chars['q_factor']
            })

    # Print summary
    print(f"\n{'='*70}")
    print("DATASET BEHAVIORAL SPECIFICATIONS")
    print(f"{'='*70}\n")

    for filter_type in filter_type_names:
        specs = specs_by_type[filter_type]
        if not specs:
            continue

        print(f"{filter_type.upper()}:")
        print(f"  Total: {len(specs)} circuits")

        cutoffs = [s['cutoff'] for s in specs if s['cutoff'] is not None]
        if cutoffs:
            cutoffs = np.array(cutoffs)
            print(f"  Cutoff frequencies:")
            print(f"    Range: [{cutoffs.min():.2e}, {cutoffs.max():.2e}] Hz")
            print(f"    Mean: {cutoffs.mean():.2e} Hz")
            print(f"    Median: {np.median(cutoffs):.2e} Hz")

            # Show some examples
            print(f"  Examples:")
            for s in sorted(specs, key=lambda x: x['cutoff'] if x['cutoff'] else 0)[:5]:
                if s['cutoff']:
                    q_str = f", Q={s['q_factor']:.2f}" if s['q_factor'] else ""
                    print(f"    Circuit {s['idx']}: fc={s['cutoff']:.2e} Hz{q_str}")

        print()

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
