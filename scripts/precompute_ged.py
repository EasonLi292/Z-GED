#!/usr/bin/env python3
"""
Precompute Graph Edit Distance (GED) matrix for all circuit pairs.

This script computes the 120×120 pairwise GED distances between all circuits
in the dataset. The resulting matrix is used for metric learning in the GraphVAE.

Expected runtime: ~2-3 hours for 7,140 unique pairs (120 choose 2).
"""

import sys
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import time

# Add tools to path
sys.path.insert(0, 'tools')
from graph_edit_distance import CircuitGED


def load_dataset(dataset_path='rlc_dataset/filter_dataset.pkl'):
    """Load the circuit dataset."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} circuits from {dataset_path}")
    return data


def convert_to_networkx(graph_adj_dict):
    """Convert graph_adj dictionary to NetworkX graph."""
    # The graph_adj dict uses 'adjacency' key, but NetworkX expects 'links'
    # We need to manually build the graph from the adjacency list format

    G = nx.DiGraph() if graph_adj_dict['directed'] else nx.Graph()

    # Add nodes with their attributes
    for node_data in graph_adj_dict['nodes']:
        node_id = node_data['id']
        node_attrs = {k: v for k, v in node_data.items() if k != 'id'}
        G.add_node(node_id, **node_attrs)

    # Add edges from adjacency list
    for source_id, neighbors in enumerate(graph_adj_dict['adjacency']):
        for neighbor in neighbors:
            target_id = neighbor['id']
            edge_attrs = {k: v for k, v in neighbor.items() if k != 'id'}
            G.add_edge(source_id, target_id, **edge_attrs)

    return G


def precompute_ged_matrix(dataset, output_path='ged_matrix.npy', checkpoint_path='ged_checkpoint.npz'):
    """
    Compute all pairwise GED distances and save to file.

    Args:
        dataset: List of circuit dictionaries
        output_path: Where to save the final GED matrix
        checkpoint_path: Where to save checkpoints during computation

    Returns:
        ged_matrix: 120×120 numpy array of pairwise distances
    """
    n_circuits = len(dataset)
    ged_matrix = np.zeros((n_circuits, n_circuits), dtype=np.float32)

    # Initialize GED calculator
    ged_calc = CircuitGED()

    # Convert all graphs upfront
    print("\nConverting circuits to NetworkX graphs...")
    graphs = []
    for circuit in tqdm(dataset):
        graph = convert_to_networkx(circuit['graph_adj'])
        graphs.append(graph)

    # Check if checkpoint exists
    start_i, start_j = 0, 1
    if Path(checkpoint_path).exists():
        print(f"\nFound checkpoint at {checkpoint_path}")
        checkpoint = np.load(checkpoint_path)
        ged_matrix = checkpoint['ged_matrix']
        start_i = int(checkpoint['last_i'])
        start_j = int(checkpoint['last_j'])
        print(f"Resuming from position ({start_i}, {start_j})")

    # Compute pairwise distances
    total_pairs = n_circuits * (n_circuits - 1) // 2
    computed_pairs = 0

    # Count already computed pairs
    for i in range(start_i):
        for j in range(i + 1, n_circuits):
            computed_pairs += 1
    for j in range(start_i + 1, start_j):
        computed_pairs += 1

    print(f"\nComputing GED for {total_pairs} unique pairs...")
    print(f"Already computed: {computed_pairs}/{total_pairs}")

    start_time = time.time()
    checkpoint_interval = 100  # Save checkpoint every 100 pairs
    pairs_since_checkpoint = 0

    with tqdm(total=total_pairs, initial=computed_pairs) as pbar:
        for i in range(start_i, n_circuits):
            # Start from correct j position
            j_start = start_j if i == start_i else i + 1

            for j in range(j_start, n_circuits):
                # Compute GED
                distance = ged_calc.compute_ged(graphs[i], graphs[j])
                ged_matrix[i, j] = distance
                ged_matrix[j, i] = distance  # Symmetric

                pbar.update(1)
                computed_pairs += 1
                pairs_since_checkpoint += 1

                # Update progress bar with ETA
                elapsed = time.time() - start_time
                if computed_pairs > 0:
                    pairs_per_sec = computed_pairs / elapsed
                    remaining_pairs = total_pairs - computed_pairs
                    eta_seconds = remaining_pairs / pairs_per_sec if pairs_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600
                    pbar.set_postfix({
                        'pairs/s': f'{pairs_per_sec:.2f}',
                        'ETA': f'{eta_hours:.1f}h'
                    })

                # Save checkpoint periodically
                if pairs_since_checkpoint >= checkpoint_interval:
                    np.savez(checkpoint_path,
                            ged_matrix=ged_matrix,
                            last_i=i,
                            last_j=j + 1)
                    pairs_since_checkpoint = 0

    # Save final matrix
    np.save(output_path, ged_matrix)
    print(f"\n✅ Saved GED matrix to {output_path}")

    # Remove checkpoint file
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"   Removed checkpoint file {checkpoint_path}")

    return ged_matrix


def print_statistics(ged_matrix, dataset):
    """Print statistics about the computed GED matrix."""
    print("\n" + "="*70)
    print("GED MATRIX STATISTICS")
    print("="*70)

    # Overall statistics
    # Exclude diagonal (zeros)
    off_diagonal = ged_matrix[np.triu_indices_from(ged_matrix, k=1)]

    print(f"\nOverall distances:")
    print(f"  Min:    {off_diagonal.min():.4f}")
    print(f"  Max:    {off_diagonal.max():.4f}")
    print(f"  Mean:   {off_diagonal.mean():.4f}")
    print(f"  Median: {np.median(off_diagonal):.4f}")
    print(f"  Std:    {off_diagonal.std():.4f}")

    # Per filter type statistics
    filter_types = {}
    for idx, circuit in enumerate(dataset):
        ftype = circuit['filter_type']
        if ftype not in filter_types:
            filter_types[ftype] = []
        filter_types[ftype].append(idx)

    print(f"\nFilter type distribution:")
    for ftype, indices in sorted(filter_types.items()):
        print(f"  {ftype:<15}: {len(indices)} circuits")

    # Within-type vs between-type distances
    print(f"\nWithin-type vs Between-type distances:")
    for ftype, indices in sorted(filter_types.items()):
        # Within-type distances
        within_distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_distances.append(ged_matrix[indices[i], indices[j]])

        # Between-type distances
        between_distances = []
        other_indices = [idx for idx in range(len(dataset)) if idx not in indices]
        for i in indices:
            for j in other_indices:
                between_distances.append(ged_matrix[i, j])

        if within_distances and between_distances:
            within_mean = np.mean(within_distances)
            between_mean = np.mean(between_distances)
            separation = between_mean / within_mean if within_mean > 0 else float('inf')

            print(f"\n  {ftype}:")
            print(f"    Within-type:  {within_mean:.4f} (± {np.std(within_distances):.4f})")
            print(f"    Between-type: {between_mean:.4f} (± {np.std(between_distances):.4f})")
            print(f"    Separation:   {separation:.2f}x")

    print()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Precompute GED matrix for circuit dataset')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--output', type=str, default='ged_matrix.npy',
                       help='Path to save GED matrix')
    parser.add_argument('--checkpoint', type=str, default='ged_checkpoint.npz',
                       help='Path for checkpoint file')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics from existing GED matrix')

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)

    if args.stats_only:
        # Load existing matrix and print stats
        if not Path(args.output).exists():
            print(f"❌ GED matrix not found at {args.output}")
            return 1
        ged_matrix = np.load(args.output)
        print(f"Loaded GED matrix from {args.output}")
    else:
        # Compute GED matrix
        ged_matrix = precompute_ged_matrix(dataset, args.output, args.checkpoint)

    # Print statistics
    print_statistics(ged_matrix, dataset)

    return 0


if __name__ == '__main__':
    sys.exit(main())
