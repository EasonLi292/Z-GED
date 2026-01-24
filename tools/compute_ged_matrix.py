#!/usr/bin/env python3
"""
Compute GED Matrix for Circuit Dataset.

Computes pairwise Graph Edit Distance matrix for all circuits in the dataset.
This matrix is used by the generation pipeline for GED-weighted K-NN interpolation.

Usage:
    python tools/compute_ged_matrix.py [--timeout 10] [--output analysis_results/ged_matrix_120.npy]
"""

import argparse
import pickle
import numpy as np
from graph_edit_distance import CircuitGED, load_graph_from_dataset

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"{desc}...")
        return iterable


def compute_ged_matrix(dataset_path, output_path, timeout=10):
    """
    Compute and save GED matrix for entire dataset.

    Args:
        dataset_path: Path to pickle dataset
        output_path: Path to save numpy matrix
        timeout: Timeout per GED computation in seconds
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    n = len(dataset)
    print(f"Dataset has {n} circuits")
    print(f"Computing {n*(n-1)//2} pairwise GED values...")
    print(f"Timeout per computation: {timeout}s")

    # Convert all graphs
    print("\nConverting graphs...")
    graphs = []
    for sample in tqdm(dataset, desc="Loading graphs"):
        G = load_graph_from_dataset(sample['graph_adj'])
        graphs.append(G)

    # Initialize GED calculator
    ged_calc = CircuitGED()

    # Compute matrix
    matrix = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    print(f"\nComputing GED matrix ({len(pairs)} pairs)...")

    if TQDM_AVAILABLE:
        iterator = tqdm(pairs, desc="GED computation")
    else:
        iterator = pairs
        print("Progress: ", end="", flush=True)

    for idx, (i, j) in enumerate(iterator):
        ged = ged_calc.compute_ged(graphs[i], graphs[j], timeout=timeout)
        matrix[i, j] = ged
        matrix[j, i] = ged  # Symmetric

        if not TQDM_AVAILABLE and idx % 500 == 0:
            print(f"{idx}/{len(pairs)} ", end="", flush=True)

    if not TQDM_AVAILABLE:
        print("Done!")

    # Save matrix
    np.save(output_path, matrix)
    print(f"\nSaved GED matrix to {output_path}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"GED statistics:")
    print(f"  Min (non-zero): {matrix[matrix > 0].min():.3f}")
    print(f"  Max: {matrix.max():.3f}")
    print(f"  Mean: {matrix[matrix > 0].mean():.3f}")
    print(f"  Std: {matrix[matrix > 0].std():.3f}")

    return matrix


def main():
    parser = argparse.ArgumentParser(description='Compute GED matrix for circuit dataset')
    parser.add_argument('--dataset', default='rlc_dataset/filter_dataset.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--output', default='analysis_results/ged_matrix_360.npy',
                        help='Path to output numpy file')
    parser.add_argument('--timeout', type=float, default=10,
                        help='Timeout per GED computation in seconds (default: 10)')

    args = parser.parse_args()

    print("=" * 70)
    print("GED Matrix Computation")
    print("=" * 70)

    compute_ged_matrix(args.dataset, args.output, args.timeout)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
