#!/usr/bin/env python3
import pickle
import numpy as np
from scipy import signal
from collections import defaultdict

# Load dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Comprehensive Pole-Zero Verification")
print(f"=" * 70)
print(f"Total circuits: {len(dataset)}\n")

# Analyze by filter type
stats_by_type = defaultdict(lambda: {'mse': [], 'num_poles': [], 'num_zeros': [], 'gains': []})

for data in dataset:
    filter_type = data['filter_type']
    poles = data['label']['poles']
    zeros = data['label']['zeros']
    gain = data['label']['gain']

    # Compute MSE
    freqs = data['frequency_response']['freqs']
    H_spice = data['frequency_response']['H_complex']

    w = 2 * np.pi * freqs
    s = 1j * w

    H_analytical = np.ones_like(s, dtype=complex) * gain
    for z in zeros:
        H_analytical *= (s - z)
    for p in poles:
        H_analytical /= (s - p)

    mse = np.mean(np.abs(H_spice - H_analytical) ** 2)

    stats_by_type[filter_type]['mse'].append(mse)
    stats_by_type[filter_type]['num_poles'].append(len(poles))
    stats_by_type[filter_type]['num_zeros'].append(len(zeros))
    stats_by_type[filter_type]['gains'].append(gain)

# Print statistics
for filter_type in sorted(stats_by_type.keys()):
    stats = stats_by_type[filter_type]
    print(f"{filter_type.upper()}")
    print(f"  Samples: {len(stats['mse'])}")
    print(f"  Poles: {stats['num_poles'][0]} (consistent: {len(set(stats['num_poles'])) == 1})")
    print(f"  Zeros: {stats['num_zeros'][0]} (consistent: {len(set(stats['num_zeros'])) == 1})")
    print(f"  MSE: min={min(stats['mse']):.2e}, max={max(stats['mse']):.2e}, avg={np.mean(stats['mse']):.2e}")

    # Check if any have excellent MSE (< 1e-10)
    excellent = sum(1 for mse in stats['mse'] if mse < 1e-10)
    good = sum(1 for mse in stats['mse'] if mse < 0.01)
    fair = sum(1 for mse in stats['mse'] if 0.01 <= mse < 0.5)
    poor = sum(1 for mse in stats['mse'] if mse >= 0.5)

    print(f"  Quality: Excellent(<1e-10)={excellent}, Good(<0.01)={good}, Fair(0.01-0.5)={fair}, Poor(>=0.5)={poor}")
    print()

print("=" * 70)
print("SUMMARY:")
all_mse = [mse for stats in stats_by_type.values() for mse in stats['mse']]
print(f"Overall MSE: min={min(all_mse):.2e}, max={max(all_mse):.2e}, avg={np.mean(all_mse):.2e}")

excellent_total = sum(1 for mse in all_mse if mse < 1e-10)
good_total = sum(1 for mse in all_mse if mse < 0.01)
fair_total = sum(1 for mse in all_mse if 0.01 <= mse < 0.5)
poor_total = sum(1 for mse in all_mse if mse >= 0.5)

print(f"Quality distribution:")
print(f"  Excellent (<1e-10): {excellent_total}/{len(all_mse)} ({100*excellent_total/len(all_mse):.1f}%)")
print(f"  Good      (<0.01):  {good_total}/{len(all_mse)} ({100*good_total/len(all_mse):.1f}%)")
print(f"  Fair   (0.01-0.5):  {fair_total}/{len(all_mse)} ({100*fair_total/len(all_mse):.1f}%)")
print(f"  Poor     (>=0.5):   {poor_total}/{len(all_mse)} ({100*poor_total/len(all_mse):.1f}%)")
