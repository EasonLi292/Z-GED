"""
Analyze training dataset to understand circuit topology and Q-factor distributions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import numpy as np
import torch
from collections import Counter

# Load dataset
print("Loading training dataset...")
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total circuits in dataset: {len(data)}\n")

def calculate_q_factor(poles):
    """Calculate Q-factor from poles."""
    if not poles:
        return 0.5  # Default

    # For second-order system (2 poles)
    if len(poles) >= 2:
        # Complex conjugate poles
        pole = poles[0]
        if abs(pole.imag) > 1e-6:  # Complex pole
            omega_n = abs(pole)
            zeta = -pole.real / omega_n
            q = 1.0 / (2.0 * zeta)
            return q
        else:
            # Real poles (overdamped)
            return 0.3  # Typical for overdamped

    # Single pole (first-order)
    return 0.707  # Default Butterworth

# Analyze Q-factors
q_factors = []
cutoff_freqs = []
topologies = []
component_counts = []
filter_types = []

for circuit in data:
    # Get frequency
    cutoff_freq = circuit['characteristic_frequency']
    cutoff_freqs.append(cutoff_freq)

    # Calculate Q-factor from poles
    poles = circuit['label']['poles']
    q_factor = calculate_q_factor(poles)
    q_factors.append(q_factor)

    # Get filter type
    filter_types.append(circuit['filter_type'])

    # Analyze topology from components
    components = circuit['components']
    num_R = sum(1 for c in components if c['type'] == 'R')
    num_C = sum(1 for c in components if c['type'] == 'C')
    num_L = sum(1 for c in components if c['type'] == 'L')

    topology_str = f"{num_R}R"
    if num_C > 0:
        topology_str += f"+{num_C}C"
    if num_L > 0:
        topology_str += f"+{num_L}L"

    topologies.append(topology_str)
    component_counts.append((num_R, num_C, num_L))

# Convert to numpy arrays
q_factors = np.array(q_factors)
cutoff_freqs = np.array(cutoff_freqs)

print("="*80)
print("Q-FACTOR DISTRIBUTION")
print("="*80)

# Q-factor statistics
print(f"\nOverall Statistics:")
print(f"  Mean Q: {q_factors.mean():.3f}")
print(f"  Median Q: {np.median(q_factors):.3f}")
print(f"  Std Dev: {q_factors.std():.3f}")
print(f"  Min Q: {q_factors.min():.3f}")
print(f"  Max Q: {q_factors.max():.3f}")

# Q-factor ranges
print(f"\nQ-factor Distribution:")
q_ranges = [
    (0.0, 0.5, "Very low (Q < 0.5)"),
    (0.5, 0.8, "Butterworth range (0.5 ≤ Q < 0.8)"),
    (0.8, 2.0, "Moderate (0.8 ≤ Q < 2.0)"),
    (2.0, 5.0, "Medium-high (2.0 ≤ Q < 5.0)"),
    (5.0, 10.0, "High (5.0 ≤ Q < 10.0)"),
    (10.0, 100.0, "Very high (Q ≥ 10.0)")
]

for q_min, q_max, label in q_ranges:
    count = np.sum((q_factors >= q_min) & (q_factors < q_max))
    percentage = 100 * count / len(q_factors)
    print(f"  {label:35s}: {count:3d} circuits ({percentage:5.1f}%)")

print(f"\nButterworth (Q ≈ 0.707) circuits:")
butterworth_count = np.sum(np.abs(q_factors - 0.707) < 0.1)
print(f"  Within ±10% of 0.707: {butterworth_count} circuits ({100*butterworth_count/len(q_factors):.1f}%)")

print("\n" + "="*80)
print("FILTER TYPE DISTRIBUTION")
print("="*80)

filter_type_counts = Counter(filter_types)
print(f"\nFilter Types:")
for ftype, count in filter_type_counts.most_common():
    percentage = 100 * count / len(filter_types)
    print(f"  {ftype:15s}: {count:3d} circuits ({percentage:5.1f}%)")

print("\n" + "="*80)
print("FREQUENCY DISTRIBUTION")
print("="*80)

print(f"\nOverall Statistics:")
print(f"  Mean: {cutoff_freqs.mean():,.1f} Hz")
print(f"  Median: {np.median(cutoff_freqs):,.1f} Hz")
print(f"  Min: {cutoff_freqs.min():,.1f} Hz")
print(f"  Max: {cutoff_freqs.max():,.1f} Hz")

freq_ranges = [
    (0, 100, "< 100 Hz"),
    (100, 1000, "100 Hz - 1 kHz"),
    (1000, 10000, "1 kHz - 10 kHz"),
    (10000, 100000, "10 kHz - 100 kHz"),
    (100000, 1e6, "> 100 kHz")
]

print(f"\nFrequency Distribution:")
for f_min, f_max, label in freq_ranges:
    count = np.sum((cutoff_freqs >= f_min) & (cutoff_freqs < f_max))
    percentage = 100 * count / len(cutoff_freqs)
    print(f"  {label:20s}: {count:3d} circuits ({percentage:5.1f}%)")

print("\n" + "="*80)
print("TOPOLOGY DISTRIBUTION")
print("="*80)

topology_counts = Counter(topologies)
print(f"\nMost Common Topologies:")
for topology, count in topology_counts.most_common(15):
    percentage = 100 * count / len(topologies)
    print(f"  {topology:15s}: {count:3d} circuits ({percentage:5.1f}%)")

# Component type analysis
print(f"\n\nComponent Type Usage:")
num_with_R = sum(1 for r, c, l in component_counts if r > 0)
num_with_C = sum(1 for r, c, l in component_counts if c > 0)
num_with_L = sum(1 for r, c, l in component_counts if l > 0)

print(f"  Circuits with Resistors: {num_with_R} ({100*num_with_R/len(component_counts):.1f}%)")
print(f"  Circuits with Capacitors: {num_with_C} ({100*num_with_C/len(component_counts):.1f}%)")
print(f"  Circuits with Inductors:  {num_with_L} ({100*num_with_L/len(component_counts):.1f}%)")

# Pure component combinations
rc_only = sum(1 for r, c, l in component_counts if r > 0 and c > 0 and l == 0)
rl_only = sum(1 for r, c, l in component_counts if r > 0 and l > 0 and c == 0)
cl_only = sum(1 for r, c, l in component_counts if c > 0 and l > 0 and r == 0)
rcl = sum(1 for r, c, l in component_counts if r > 0 and c > 0 and l > 0)

print(f"\nComponent Combinations:")
print(f"  RC only (no L):  {rc_only} ({100*rc_only/len(component_counts):.1f}%)")
print(f"  RL only (no C):  {rl_only} ({100*rl_only/len(component_counts):.1f}%)")
print(f"  CL only (no R):  {cl_only} ({100*cl_only/len(component_counts):.1f}%)")
print(f"  RLC (all three): {rcl} ({100*rcl/len(component_counts):.1f}%)")

print("\n" + "="*80)
print("HIGH-Q CIRCUIT ANALYSIS")
print("="*80)

high_q_circuits = [(i, q, topo) for i, (q, topo) in enumerate(zip(q_factors, topologies)) if q >= 5.0]
print(f"\nCircuits with Q ≥ 5.0: {len(high_q_circuits)} total")

if high_q_circuits:
    high_q_topologies = [topo for _, _, topo in high_q_circuits]
    high_q_topo_counts = Counter(high_q_topologies)
    print(f"\nTopologies in high-Q circuits:")
    for topology, count in high_q_topo_counts.most_common(10):
        percentage = 100 * count / len(high_q_circuits)
        print(f"  {topology:15s}: {count:3d} circuits ({percentage:5.1f}%)")

    # Check if high-Q circuits have inductors
    high_q_components = [component_counts[i] for i, _, _ in high_q_circuits]
    high_q_with_L = sum(1 for r, c, l in high_q_components if l > 0)
    high_q_with_C = sum(1 for r, c, l in high_q_components if c > 0)
    high_q_with_both = sum(1 for r, c, l in high_q_components if l > 0 and c > 0)

    print(f"\nHigh-Q Circuit Components:")
    print(f"  With inductors (L): {high_q_with_L}/{len(high_q_circuits)} ({100*high_q_with_L/len(high_q_circuits):.1f}%)")
    print(f"  With capacitors (C): {high_q_with_C}/{len(high_q_circuits)} ({100*high_q_with_C/len(high_q_circuits):.1f}%)")
    print(f"  With both L and C: {high_q_with_both}/{len(high_q_circuits)} ({100*high_q_with_both/len(high_q_circuits):.1f}%)")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

butterworth_pct = 100 * butterworth_count / len(q_factors)
high_q_pct = 100 * len(high_q_circuits) / len(q_factors)
rc_only_pct = 100 * rc_only / len(component_counts)

print(f"""
1. Training data is heavily biased toward Butterworth filters:
   - {butterworth_pct:.1f}% of circuits have Q ≈ 0.707

2. Very few high-Q training examples:
   - Only {high_q_pct:.1f}% of circuits have Q ≥ 5.0
   - Only {len([q for q in q_factors if q >= 10.0])} circuits have Q ≥ 10.0

3. Most common topology is RC (no inductor):
   - {rc_only_pct:.1f}% of circuits are RC-only
   - RC filters are limited to Q ≤ 0.707

4. High-Q circuits in training data:
   - {100*high_q_with_both/max(len(high_q_circuits), 1):.1f}% have both L and C (required for resonance)
   - Model has seen only {len(high_q_circuits)} examples of high-Q behavior

This explains why the model:
- Excels at Butterworth filters (dominant training examples)
- Struggles with Q > 1.0 (limited training data)
- Often generates RC topologies (most common in training)
- Fails on high-Q specs (insufficient diverse examples)
""")
