#!/usr/bin/env python3
"""Investigate why low-pass and high-pass have similar GED."""

import pickle
import numpy as np
from tools.graph_edit_distance import CircuitGED, load_graph_from_dataset


def load_dataset(path='rlc_dataset/filter_dataset.pkl'):
    """Load circuit dataset from pickle file."""
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def analyze_lowpass_highpass():
    """Compare low-pass and high-pass circuit structures."""
    dataset = load_dataset()

    # Get low-pass and high-pass filters
    lowpass = [s for s in dataset if s['filter_type'] == 'low_pass'][:3]
    highpass = [s for s in dataset if s['filter_type'] == 'high_pass'][:3]

    print("="*80)
    print("LOW-PASS vs HIGH-PASS CIRCUIT ANALYSIS")
    print("="*80)

    # Analyze structure
    print("\nLOW-PASS FILTERS:")
    print("-" * 80)
    for i, circuit in enumerate(lowpass):
        G = load_graph_from_dataset(circuit['graph_adj'])
        print(f"\nCircuit {i+1}:")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Frequency: {circuit['characteristic_frequency']:.2f} Hz")

        # Analyze node types
        node_types = {}
        for node_id, attrs in G.nodes(data=True):
            features = attrs['features']
            if features == (1, 0, 0, 0):
                node_type = 'GND'
            elif features == (0, 1, 0, 0):
                node_type = 'VIN'
            elif features == (0, 0, 1, 0):
                node_type = 'VOUT'
            elif features == (0, 0, 0, 1):
                node_type = 'INTERNAL'
            else:
                node_type = 'UNKNOWN'
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f"  Node types: {node_types}")

        # Analyze edge components
        edge_info = []
        for u, v, attrs in G.edges(data=True):
            den = attrs['impedance_den']
            C, G_cond, L_inv = den[0], den[1], den[2]

            components = []
            if C > 0:
                components.append(f"C={C:.2e}")
            if G_cond > 0:
                components.append(f"G={G_cond:.2e}")
            if L_inv > 0:
                components.append(f"L_inv={L_inv:.2e}")

            edge_info.append(f"({u},{v}): {', '.join(components)}")

        print(f"  Edges:")
        for info in edge_info:
            print(f"    {info}")

    print("\n" + "="*80)
    print("HIGH-PASS FILTERS:")
    print("-" * 80)
    for i, circuit in enumerate(highpass):
        G = load_graph_from_dataset(circuit['graph_adj'])
        print(f"\nCircuit {i+1}:")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Frequency: {circuit['characteristic_frequency']:.2f} Hz")

        # Analyze node types
        node_types = {}
        for node_id, attrs in G.nodes(data=True):
            features = attrs['features']
            if features == (1, 0, 0, 0):
                node_type = 'GND'
            elif features == (0, 1, 0, 0):
                node_type = 'VIN'
            elif features == (0, 0, 1, 0):
                node_type = 'VOUT'
            elif features == (0, 0, 0, 1):
                node_type = 'INTERNAL'
            else:
                node_type = 'UNKNOWN'
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f"  Node types: {node_types}")

        # Analyze edge components
        edge_info = []
        for u, v, attrs in G.edges(data=True):
            den = attrs['impedance_den']
            C, G_cond, L_inv = den[0], den[1], den[2]

            components = []
            if C > 0:
                components.append(f"C={C:.2e}")
            if G_cond > 0:
                components.append(f"G={G_cond:.2e}")
            if L_inv > 0:
                components.append(f"L_inv={L_inv:.2e}")

            edge_info.append(f"({u},{v}): {', '.join(components)}")

        print(f"  Edges:")
        for info in edge_info:
            print(f"    {info}")

    # Compute pairwise GEDs
    print("\n" + "="*80)
    print("PAIRWISE GED COMPARISON")
    print("="*80)

    ged_calc = CircuitGED()

    print("\nLow-pass to Low-pass GEDs:")
    for i in range(len(lowpass)):
        for j in range(i+1, len(lowpass)):
            G1 = load_graph_from_dataset(lowpass[i]['graph_adj'])
            G2 = load_graph_from_dataset(lowpass[j]['graph_adj'])
            ged = ged_calc.compute_ged(G1, G2, timeout=10)
            print(f"  LP{i+1} <-> LP{j+1}: {ged:.4f}")

    print("\nHigh-pass to High-pass GEDs:")
    for i in range(len(highpass)):
        for j in range(i+1, len(highpass)):
            G1 = load_graph_from_dataset(highpass[i]['graph_adj'])
            G2 = load_graph_from_dataset(highpass[j]['graph_adj'])
            ged = ged_calc.compute_ged(G1, G2, timeout=10)
            print(f"  HP{i+1} <-> HP{j+1}: {ged:.4f}")

    print("\nLow-pass to High-pass GEDs:")
    for i in range(len(lowpass)):
        for j in range(len(highpass)):
            G1 = load_graph_from_dataset(lowpass[i]['graph_adj'])
            G2 = load_graph_from_dataset(highpass[j]['graph_adj'])
            ged = ged_calc.compute_ged(G1, G2, timeout=10)
            print(f"  LP{i+1} <-> HP{j+1}: {ged:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("""
The low GED between low-pass and high-pass filters indicates they have very
similar GRAPH STRUCTURES (same topology, number of nodes/edges).

The difference between low-pass and high-pass is typically in the COMPONENT
VALUES (R, L, C), which our GED metric considers via impedance distance.

However, if the impedance distances are very small, it suggests:
1. The component values are similar in magnitude
2. The weighting parameters (w_C, w_G, w_L_inv) may need adjustment
3. The graph structure alone is insufficient to distinguish filter types

For ML to work, you may need to:
- Include additional features beyond graph structure (e.g., frequency response)
- Adjust GED weighting to better separate impedance differences
- Use a hybrid approach: GED + transfer function characteristics
""")


if __name__ == '__main__':
    analyze_lowpass_highpass()
