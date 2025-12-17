#!/usr/bin/env python3
"""
Graph Edit Distance Examples and Validation

Demonstrates usage of CircuitGED for circuit similarity analysis.
Includes validation tests, clustering, and nearest neighbor search.
"""

import pickle
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
from graph_edit_distance import CircuitGED, load_graph_from_dataset, validate_ged_properties


def load_dataset(path='rlc_dataset/filter_dataset.pkl'):
    """Load circuit dataset from pickle file."""
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def example_basic_ged():
    """Example 1: Compute GED between two circuits."""
    print("\n" + "="*70)
    print("Example 1: Basic GED Computation")
    print("="*70)

    # Load dataset
    dataset = load_dataset()

    # Create GED calculator with default parameters
    ged_calc = CircuitGED()

    # Load two circuits
    circuit1 = dataset[0]
    circuit2 = dataset[1]

    G1 = load_graph_from_dataset(circuit1['graph_adj'])
    G2 = load_graph_from_dataset(circuit2['graph_adj'])

    print(f"Circuit 1: {circuit1['filter_type']}")
    print(f"  Nodes: {G1.number_of_nodes()}, Edges: {G1.number_of_edges()}")
    print(f"  Frequency: {circuit1['characteristic_frequency']:.2f} Hz")

    print(f"\nCircuit 2: {circuit2['filter_type']}")
    print(f"  Nodes: {G2.number_of_nodes()}, Edges: {G2.number_of_edges()}")
    print(f"  Frequency: {circuit2['characteristic_frequency']:.2f} Hz")

    # Compute GED
    print(f"\nComputing GED (timeout=10s)...")
    ged = ged_calc.compute_ged(G1, G2, timeout=10)

    print(f"Graph Edit Distance: {ged:.3f}")


def example_identical_circuits():
    """Example 2: Verify GED of identical circuits is zero."""
    print("\n" + "="*70)
    print("Example 2: Identical Circuits (Should have GED=0)")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    circuit = dataset[0]
    G = load_graph_from_dataset(circuit['graph_adj'])

    print(f"Circuit: {circuit['filter_type']}")
    print(f"Computing GED(G, G)...")

    ged = ged_calc.compute_ged(G, G, timeout=10)

    print(f"GED(G, G) = {ged:.6f}")

    if abs(ged) < 1e-5:
        print("✅ PASS: Identical circuits have GED ≈ 0")
    else:
        print(f"⚠️  FAIL: Expected GED=0, got {ged}")


def example_same_filter_type():
    """Example 3: Compare circuits of same type vs different types."""
    print("\n" + "="*70)
    print("Example 3: Same Type vs Different Type Comparison")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Get two low-pass filters
    lowpass = [s for s in dataset if s['filter_type'] == 'low_pass']
    G1 = load_graph_from_dataset(lowpass[0]['graph_adj'])
    G2 = load_graph_from_dataset(lowpass[1]['graph_adj'])

    # Get one high-pass filter
    highpass = [s for s in dataset if s['filter_type'] == 'high_pass']
    G3 = load_graph_from_dataset(highpass[0]['graph_adj'])

    print("Computing GED between two low-pass filters...")
    ged_same = ged_calc.compute_ged(G1, G2, timeout=10)

    print("Computing GED between low-pass and high-pass filters...")
    ged_diff = ged_calc.compute_ged(G1, G3, timeout=10)

    print(f"\nGED (low-pass, low-pass):  {ged_same:.3f}")
    print(f"GED (low-pass, high-pass): {ged_diff:.3f}")

    if ged_same >= 0 and ged_diff >= 0:
        print("✅ Both GEDs are non-negative")
    else:
        print("⚠️  WARNING: Negative GED detected")


def example_validate_properties():
    """Example 4: Validate mathematical properties of GED."""
    print("\n" + "="*70)
    print("Example 4: Validate GED Mathematical Properties")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Load three different circuits
    G1 = load_graph_from_dataset(dataset[0]['graph_adj'])
    G2 = load_graph_from_dataset(dataset[1]['graph_adj'])
    G3 = load_graph_from_dataset(dataset[2]['graph_adj'])

    print("Testing GED properties...")
    results = validate_ged_properties(ged_calc, G1, G2, G3, tol=1e-3)

    print("\nResults:")
    for prop, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {prop:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All properties validated!")
    else:
        print("\n⚠️  Some properties failed validation")


def example_ged_by_filter_type():
    """Example 5: Compute average GED within and between filter types."""
    print("\n" + "="*70)
    print("Example 5: GED Analysis by Filter Type")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Group by filter type
    by_type = {}
    for sample in dataset:
        ftype = sample['filter_type']
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append(sample)

    # Sample 5 circuits per type for efficiency
    filter_types = list(by_type.keys())
    sampled = {ft: by_type[ft][:5] for ft in filter_types}

    print(f"Analyzing {len(filter_types)} filter types...")
    print(f"Samples per type: 5")

    # Compute within-type and between-type GEDs
    results = {}

    for ft in filter_types:
        graphs = [load_graph_from_dataset(s['graph_adj']) for s in sampled[ft]]

        # Within-type GED (average pairwise)
        within_geds = []
        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                ged = ged_calc.compute_ged(graphs[i], graphs[j], timeout=10)
                within_geds.append(ged)

        avg_within = np.mean(within_geds) if within_geds else 0

        results[ft] = {'within': avg_within}

    # Display results
    print(f"\n{'Filter Type':<20} {'Avg Within-Type GED':<20}")
    print("-" * 40)
    for ft in sorted(filter_types):
        print(f"{ft:<20} {results[ft]['within']:<20.3f}")


def example_ged_matrix_small():
    """Example 6: Compute and visualize GED matrix for small subset."""
    print("\n" + "="*70)
    print("Example 6: GED Matrix Computation and Visualization")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Use small subset (10 circuits)
    subset = dataset[:10]
    graphs = [load_graph_from_dataset(s['graph_adj']) for s in subset]
    labels = [s['filter_type'] for s in subset]

    print(f"Computing 10×10 GED matrix...")
    matrix = ged_calc.compute_ged_matrix(graphs, show_progress=True)

    print(f"\nGED Matrix shape: {matrix.shape}")
    print(f"Mean GED: {matrix[matrix > 0].mean():.3f}")
    print(f"Max GED: {matrix.max():.3f}")
    print(f"Min GED (non-zero): {matrix[matrix > 0].min():.3f}")

    # Visualize
    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis_r',
                    square=True, xticklabels=range(10), yticklabels=range(10))
        plt.title('Circuit Similarity Matrix (Graph Edit Distance)')
        plt.xlabel('Circuit Index')
        plt.ylabel('Circuit Index')
        plt.tight_layout()
        plt.savefig('ged_matrix_small.png', dpi=150)
        print(f"\nVisualization saved to: ged_matrix_small.png")
    else:
        print(f"\n⚠️  Matplotlib/Seaborn not available. Skipping visualization.")


def example_nearest_neighbors():
    """Example 7: Find nearest neighbors for a query circuit."""
    print("\n" + "="*70)
    print("Example 7: Nearest Neighbor Search")
    print("="*70)

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Query circuit (use a band-stop filter)
    bandstop = [s for i, s in enumerate(dataset) if s['filter_type'] == 'band_stop']
    query_idx = dataset.index(bandstop[0])
    query_circuit = bandstop[0]
    query_graph = load_graph_from_dataset(query_circuit['graph_adj'])

    # Database (all other circuits, use subset for speed)
    database_subset = dataset[:30]  # Use first 30 for demo
    database = [load_graph_from_dataset(s['graph_adj'])
                for i, s in enumerate(database_subset) if i != query_idx]
    database_meta = [s for i, s in enumerate(database_subset) if i != query_idx]

    print(f"Query circuit: {query_circuit['filter_type']}")
    print(f"  Frequency: {query_circuit['characteristic_frequency']:.2f} Hz")
    print(f"  ID: {query_circuit['id'][:16]}...")

    print(f"\nSearching {len(database)} circuits for 5 nearest neighbors...")
    neighbors = ged_calc.find_nearest_neighbors(query_graph, database, k=5)

    print(f"\nNearest neighbors:")
    print(f"{'Rank':<6} {'Filter Type':<15} {'Frequency (Hz)':<15} {'GED':<10}")
    print("-" * 55)

    for rank, (idx, distance) in enumerate(neighbors, 1):
        neighbor = database_meta[idx]
        print(f"{rank:<6} {neighbor['filter_type']:<15} "
              f"{neighbor['characteristic_frequency']:<15.2f} {distance:<10.3f}")


def example_clustering():
    """Example 8: Cluster circuits by GED similarity."""
    print("\n" + "="*70)
    print("Example 8: Clustering Circuits by GED")
    print("="*70)

    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("⚠️  Scikit-learn not installed. Skipping clustering example.")
        print("   Install with: pip install scikit-learn")
        return

    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Use subset for efficiency (30 circuits, 5 per type)
    by_type = {}
    for sample in dataset:
        ftype = sample['filter_type']
        if ftype not in by_type:
            by_type[ftype] = []
        if len(by_type[ftype]) < 5:
            by_type[ftype].append(sample)

    subset = []
    for ftype in sorted(by_type.keys()):
        subset.extend(by_type[ftype])

    graphs = [load_graph_from_dataset(s['graph_adj']) for s in subset]
    true_labels = [s['filter_type'] for s in subset]

    print(f"Computing GED matrix for {len(graphs)} circuits...")
    ged_matrix = ged_calc.compute_ged_matrix(graphs, show_progress=True)

    # Hierarchical clustering
    print(f"\nPerforming hierarchical clustering (k=6)...")
    clustering = AgglomerativeClustering(
        n_clusters=6,  # One per filter type
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(ged_matrix)

    # Evaluate clustering quality
    ari = adjusted_rand_score(true_labels, cluster_labels)

    print(f"\nClustering Results:")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  (1.0 = perfect clustering, 0.0 = random)")

    # Show cluster composition
    print(f"\nCluster composition:")
    for cluster_id in range(6):
        indices = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
        types = [true_labels[i] for i in indices]
        type_counts = {t: types.count(t) for t in set(types)}
        print(f"  Cluster {cluster_id}: {type_counts}")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*70)
    print("CIRCUIT GRAPH EDIT DISTANCE - EXAMPLES AND VALIDATION")
    print("="*70)

    examples = [
        example_basic_ged,
        example_identical_circuits,
        example_same_filter_type,
        example_validate_properties,
        example_ged_by_filter_type,
        example_ged_matrix_small,
        example_nearest_neighbors,
        example_clustering
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example {i} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1]
        if example_name == 'all':
            run_all_examples()
        else:
            example_func = globals().get(f'example_{example_name}')
            if example_func:
                example_func()
            else:
                print(f"Unknown example: {example_name}")
                print("Available examples:")
                print("  basic_ged, identical_circuits, same_filter_type,")
                print("  validate_properties, ged_by_filter_type,")
                print("  ged_matrix_small, nearest_neighbors, clustering, all")
    else:
        # Run all examples by default
        run_all_examples()
