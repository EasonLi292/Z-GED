"""
Analyze what filter types are in the training dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np

# Load dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    circuits = pickle.load(f)

print("="*80)
print("Filter Type Analysis")
print("="*80)
print(f"\nTotal circuits: {len(circuits)}")

# Count filter types
filter_types = {}
for circuit in circuits:
    ftype = circuit['filter_type']
    if ftype not in filter_types:
        filter_types[ftype] = []
    filter_types[ftype].append(circuit)

print(f"\nFilter types found:")
for ftype, circuits_list in sorted(filter_types.items()):
    print(f"  {ftype}: {len(circuits_list)} circuits")

# Analyze specifications by filter type
print("\n" + "="*80)
print("Specification Ranges by Filter Type")
print("="*80)

for ftype, circuits_list in sorted(filter_types.items()):
    cutoffs = [c['characteristic_frequency'] for c in circuits_list]
    print(f"\n{ftype} ({len(circuits_list)} circuits):")
    print(f"  Cutoff range: {min(cutoffs):.1f} - {max(cutoffs):.1f} Hz")

    # Show a few examples
    print(f"  Examples:")
    for i, c in enumerate(circuits_list[:3]):
        edges = sum(len(neighbors) for neighbors in c['graph_adj']['adjacency']) // 2
        print(f"    {i+1}. {c['characteristic_frequency']:.1f} Hz, {edges} edges")
