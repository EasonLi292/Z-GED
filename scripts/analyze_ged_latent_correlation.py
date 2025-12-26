#!/usr/bin/env python3
"""
GED-Latent Space Correlation Analysis

This script performs comprehensive analysis of the relationship between
Graph Edit Distance (GED) and the learned latent space of the circuit VAE.

Outputs:
- GED matrix and statistics
- Latent space encodings
- Correlation analysis between GED and latent distances
- Clustering metrics and separability analysis
- Comprehensive visualizations

Expected runtime: ~3-4 hours total
  - GED computation: 2-3 hours
  - Encoding: 5-10 minutes
  - Analysis: 10-20 minutes
"""

import sys
import pickle
import numpy as np
import networkx as nx
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.graph_edit_distance import CircuitGED
from ml.models import HierarchicalEncoder, VariableLengthDecoder
from ml.data.dataset import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path):
    """Load model checkpoint and configuration."""
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract configuration
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    print(f"  Model config: {model_config}")

    return checkpoint, config


def create_models(config):
    """Create encoder and decoder from config."""
    model_config = config.get('model', {})

    # Extract dimensions
    latent_dim = model_config.get('latent_dim', 8)
    topo_latent_dim = model_config.get('topo_latent_dim', 2)
    values_latent_dim = model_config.get('values_latent_dim', 2)
    pz_latent_dim = model_config.get('pz_latent_dim', 4)

    print(f"\nCreating models:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Topology: {topo_latent_dim}, Values: {values_latent_dim}, P&Z: {pz_latent_dim}")

    # Create encoder
    encoder = HierarchicalEncoder(
        node_feature_dim=4,  # [GND, VIN, VOUT, internal]
        edge_feature_dim=7,  # [log_C, log_G, log_L_inv, has_C, has_R, has_L, is_parallel]
        gnn_hidden_dim=64,
        latent_dim=latent_dim,
        topo_latent_dim=topo_latent_dim,
        values_latent_dim=values_latent_dim,
        pz_latent_dim=pz_latent_dim
    )

    # Create decoder
    decoder_hidden_dim = model_config.get('decoder_hidden_dim', 64)
    decoder = VariableLengthDecoder(
        latent_dim=latent_dim,
        topo_latent_dim=topo_latent_dim,
        values_latent_dim=values_latent_dim,
        pz_latent_dim=pz_latent_dim,
        hidden_dim=decoder_hidden_dim,
        max_poles=4,
        max_zeros=4
    )

    return encoder, decoder


def load_models(checkpoint_path):
    """Load trained models from checkpoint."""
    checkpoint, config = load_checkpoint(checkpoint_path)
    encoder, decoder = create_models(config)

    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    print(f"✅ Models loaded successfully")

    return encoder, decoder, config


def load_dataset(dataset_path='rlc_dataset/filter_dataset.pkl'):
    """Load the circuit dataset."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"\nLoaded {len(data)} circuits from {dataset_path}")
    return data


def convert_to_networkx(graph_adj_dict):
    """Convert graph_adj dictionary to NetworkX graph."""
    G = nx.DiGraph() if graph_adj_dict['directed'] else nx.Graph()

    # Add nodes
    for node_data in graph_adj_dict['nodes']:
        node_id = node_data['id']
        node_attrs = {k: v for k, v in node_data.items() if k != 'id'}
        G.add_node(node_id, **node_attrs)

    # Add edges
    for source_id, neighbors in enumerate(graph_adj_dict['adjacency']):
        for neighbor in neighbors:
            target_id = neighbor['id']
            edge_attrs = {k: v for k, v in neighbor.items() if k != 'id'}
            G.add_edge(source_id, target_id, **edge_attrs)

    return G


def compute_or_load_ged_matrix(raw_dataset, output_path, checkpoint_path):
    """Compute GED matrix if not exists, otherwise load."""
    if Path(output_path).exists():
        print(f"\n✅ Found existing GED matrix at {output_path}")
        ged_matrix = np.load(output_path)
        print(f"   Shape: {ged_matrix.shape}")
        return ged_matrix

    print(f"\n⏳ Computing GED matrix (this will take 2-3 hours)...")
    print(f"   Tip: You can monitor progress and stop/resume using checkpoints")

    n_circuits = len(raw_dataset)
    ged_matrix = np.zeros((n_circuits, n_circuits), dtype=np.float32)

    # Initialize GED calculator
    ged_calc = CircuitGED()

    # Convert all graphs
    print("\n  Converting circuits to NetworkX graphs...")
    graphs = []
    for circuit in tqdm(raw_dataset):
        graph = convert_to_networkx(circuit['graph_adj'])
        graphs.append(graph)

    # Check for checkpoint
    start_i, start_j = 0, 1
    if Path(checkpoint_path).exists():
        print(f"\n  Found checkpoint at {checkpoint_path}")
        checkpoint = np.load(checkpoint_path)
        ged_matrix = checkpoint['ged_matrix']
        start_i = int(checkpoint['last_i'])
        start_j = int(checkpoint['last_j'])
        print(f"  Resuming from position ({start_i}, {start_j})")

    # Compute pairwise distances
    total_pairs = n_circuits * (n_circuits - 1) // 2
    computed_pairs = sum(n_circuits - i - 1 for i in range(start_i)) + (start_j - start_i - 1)

    print(f"\n  Computing {total_pairs} unique pairs...")
    print(f"  Already computed: {computed_pairs}/{total_pairs}")

    start_time = time.time()
    checkpoint_interval = 100
    pairs_since_checkpoint = 0

    with tqdm(total=total_pairs, initial=computed_pairs) as pbar:
        for i in range(start_i, n_circuits):
            j_start = start_j if i == start_i else i + 1

            for j in range(j_start, n_circuits):
                distance = ged_calc.compute_ged(graphs[i], graphs[j])
                ged_matrix[i, j] = distance
                ged_matrix[j, i] = distance

                pbar.update(1)
                computed_pairs += 1
                pairs_since_checkpoint += 1

                # Update ETA
                elapsed = time.time() - start_time
                if computed_pairs > 0:
                    pairs_per_sec = computed_pairs / elapsed
                    remaining_pairs = total_pairs - computed_pairs
                    eta_hours = (remaining_pairs / pairs_per_sec) / 3600 if pairs_per_sec > 0 else 0
                    pbar.set_postfix({'pairs/s': f'{pairs_per_sec:.2f}', 'ETA': f'{eta_hours:.1f}h'})

                # Checkpoint
                if pairs_since_checkpoint >= checkpoint_interval:
                    np.savez(checkpoint_path, ged_matrix=ged_matrix, last_i=i, last_j=j + 1)
                    pairs_since_checkpoint = 0

    # Save final matrix
    np.save(output_path, ged_matrix)
    print(f"\n✅ Saved GED matrix to {output_path}")

    # Clean up checkpoint
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return ged_matrix


def encode_dataset(encoder, dataset_loader, device='cpu'):
    """
    Encode all circuits to latent space.

    Returns:
        latent_full: [N, latent_dim] full latent vectors
        latent_topo: [N, topo_latent_dim] topology latent
        latent_values: [N, values_latent_dim] values latent
        latent_pz: [N, pz_latent_dim] poles/zeros latent
        filter_types: [N] filter type indices
        circuit_ids: List of circuit IDs
    """
    print("\n⏳ Encoding all circuits to latent space...")

    encoder.to(device)
    encoder.eval()

    latent_full_list = []
    latent_mu_list = []
    latent_logvar_list = []
    filter_types = []
    circuit_ids = []

    with torch.no_grad():
        for batch in tqdm(dataset_loader):
            # Move to device
            batch_graph = batch['graph'].to(device)
            poles_list = batch['poles']
            zeros_list = batch['zeros']

            # Encode (deterministic mode - use μ instead of sampling)
            z, mu, logvar = encoder(
                batch_graph.x,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.batch,
                poles_list,
                zeros_list
            )

            # Use mu (deterministic) instead of sampled z
            latent_mu_list.append(mu.cpu().numpy())
            latent_full_list.append(z.cpu().numpy())
            latent_logvar_list.append(logvar.cpu().numpy())

            # Store metadata
            filter_type = batch['filter_type'].argmax(dim=-1).cpu().numpy()
            filter_types.append(filter_type)
            circuit_ids.extend(batch['circuit_id'])

    # Concatenate
    latent_full = np.vstack(latent_full_list)  # Use sampled z
    latent_mu = np.vstack(latent_mu_list)
    latent_logvar = np.vstack(latent_logvar_list)
    filter_types = np.concatenate(filter_types)

    print(f"\n✅ Encoded {len(latent_full)} circuits")
    print(f"   Latent shape: {latent_full.shape}")
    print(f"   Using μ (deterministic) for analysis")

    # Split into branches (using mu for analysis)
    topo_dim = 2
    values_dim = 2
    pz_dim = 4

    latent_topo = latent_mu[:, :topo_dim]
    latent_values = latent_mu[:, topo_dim:topo_dim+values_dim]
    latent_pz = latent_mu[:, -pz_dim:]

    return {
        'latent_mu': latent_mu,  # Use this for analysis (deterministic)
        'latent_sampled': latent_full,
        'latent_logvar': latent_logvar,
        'latent_topo': latent_topo,
        'latent_values': latent_values,
        'latent_pz': latent_pz,
        'filter_types': filter_types,
        'circuit_ids': circuit_ids
    }


def compute_pairwise_distances(latent_vectors):
    """Compute pairwise Euclidean distances between latent vectors."""
    # Use pdist for efficiency
    distances_flat = pdist(latent_vectors, metric='euclidean')
    distances_matrix = squareform(distances_flat)
    return distances_matrix


def analyze_correlation(ged_matrix, latent_distances, filter_types, output_dir):
    """
    Compute correlation between GED and latent distances.

    Returns correlation statistics and creates visualizations.
    """
    print("\n" + "="*70)
    print("GED-LATENT CORRELATION ANALYSIS")
    print("="*70)

    # Exclude diagonal (self-distances are 0)
    n = ged_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)

    ged_flat = ged_matrix[upper_tri_indices]
    latent_flat = latent_distances[upper_tri_indices]

    # Overall correlation
    pearson_r, pearson_p = pearsonr(ged_flat, latent_flat)
    spearman_r, spearman_p = spearmanr(ged_flat, latent_flat)

    print(f"\n📊 Overall Correlation:")
    print(f"   Pearson:  r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"   Spearman: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Interpretation
    if pearson_r > 0.7:
        print(f"   ✅ Strong positive correlation - latent space well preserves GED structure")
    elif pearson_r > 0.4:
        print(f"   ⚠️  Moderate correlation - some GED structure preserved")
    else:
        print(f"   ❌ Weak correlation - latent space organized differently")

    # Per filter type pair analysis
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    n_types = len(filter_type_names)

    pair_correlations = np.zeros((n_types, n_types))
    pair_counts = np.zeros((n_types, n_types), dtype=int)

    print(f"\n📊 Per Filter-Type-Pair Correlations:")

    for type_i in range(n_types):
        for type_j in range(type_i, n_types):
            # Get indices of circuits with these types
            idx_i = np.where(filter_types == type_i)[0]
            idx_j = np.where(filter_types == type_j)[0]

            # Extract GED and latent distances for this pair
            if type_i == type_j:
                # Within same type - use upper triangle
                pairs = [(i, j) for i in idx_i for j in idx_i if i < j]
            else:
                # Between types - all pairs
                pairs = [(i, j) for i in idx_i for j in idx_j]

            if len(pairs) < 2:
                continue

            ged_pairs = [ged_matrix[i, j] for i, j in pairs]
            latent_pairs = [latent_distances[i, j] for i, j in pairs]

            corr, _ = pearsonr(ged_pairs, latent_pairs)
            pair_correlations[type_i, type_j] = corr
            pair_correlations[type_j, type_i] = corr
            pair_counts[type_i, type_j] = len(pairs)
            pair_counts[type_j, type_i] = len(pairs)

            if type_i == type_j:
                print(f"   {filter_type_names[type_i]:15} (within):  r = {corr:6.3f}  ({len(pairs)} pairs)")
            else:
                print(f"   {filter_type_names[type_i]:15} vs {filter_type_names[type_j]:15}: r = {corr:6.3f}  ({len(pairs)} pairs)")

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Sample points for visualization (all pairs would be too many)
    sample_size = min(5000, len(ged_flat))
    sample_indices = np.random.choice(len(ged_flat), sample_size, replace=False)

    plt.scatter(ged_flat[sample_indices], latent_flat[sample_indices],
               alpha=0.3, s=10, c='steelblue', edgecolors='none')

    # Add trend line
    z = np.polyfit(ged_flat, latent_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ged_flat.min(), ged_flat.max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    plt.xlabel('GED Distance', fontsize=12)
    plt.ylabel('Latent Space Distance', fontsize=12)
    plt.title(f'GED vs Latent Distance\nPearson r = {pearson_r:.4f}, Spearman ρ = {spearman_r:.4f}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'ged_vs_latent_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved scatter plot to {output_path}")
    plt.close()

    # Create heatmap of pair-type correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(pair_correlations, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
               xticklabels=filter_type_names, yticklabels=filter_type_names,
               vmin=-0.5, vmax=1.0, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('GED-Latent Correlation by Filter Type Pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'pair_type_correlations_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved pair-type correlations heatmap to {output_path}")
    plt.close()

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pair_correlations': pair_correlations,
        'pair_counts': pair_counts
    }


def analyze_clustering(latent_vectors, filter_types, output_dir):
    """
    Analyze clustering quality and separability of filter types in latent space.
    """
    print("\n" + "="*70)
    print("LATENT SPACE CLUSTERING ANALYSIS")
    print("="*70)

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    n_types = len(filter_type_names)

    # Compute silhouette scores
    silhouette_avg = silhouette_score(latent_vectors, filter_types)
    silhouette_samples_all = silhouette_samples(latent_vectors, filter_types)

    print(f"\n📊 Clustering Metrics:")
    print(f"   Silhouette Score:       {silhouette_avg:.4f}  (range: [-1, 1], higher is better)")

    # Davies-Bouldin index (lower is better)
    db_index = davies_bouldin_score(latent_vectors, filter_types)
    print(f"   Davies-Bouldin Index:   {db_index:.4f}  (lower is better)")

    # Calinski-Harabasz index (higher is better)
    ch_index = calinski_harabasz_score(latent_vectors, filter_types)
    print(f"   Calinski-Harabasz:      {ch_index:.2f}  (higher is better)")

    # Per-type analysis
    print(f"\n📊 Per-Type Silhouette Scores:")
    per_type_silhouette = {}
    for type_idx in range(n_types):
        mask = filter_types == type_idx
        silhouette_type = silhouette_samples_all[mask].mean()
        per_type_silhouette[filter_type_names[type_idx]] = silhouette_type
        print(f"   {filter_type_names[type_idx]:15}: {silhouette_type:7.4f}")

    # Separation ratios
    print(f"\n📊 Within-Type vs Between-Type Distances:")
    separation_matrix = np.zeros((n_types, n_types))

    for type_i in range(n_types):
        idx_i = np.where(filter_types == type_i)[0]

        # Within-type distance
        if len(idx_i) > 1:
            within_distances = pdist(latent_vectors[idx_i])
            within_mean = within_distances.mean()
            within_std = within_distances.std()
        else:
            within_mean = 0
            within_std = 0

        # Between-type distances
        for type_j in range(n_types):
            if type_i == type_j:
                continue

            idx_j = np.where(filter_types == type_j)[0]
            between_distances = []
            for i in idx_i:
                for j in idx_j:
                    between_distances.append(np.linalg.norm(latent_vectors[i] - latent_vectors[j]))

            between_mean = np.mean(between_distances)
            separation = between_mean / within_mean if within_mean > 0 else float('inf')
            separation_matrix[type_i, type_j] = separation

        # Average separation
        avg_separation = separation_matrix[type_i, :].mean()

        print(f"   {filter_type_names[type_i]:15}: within = {within_mean:.4f} (±{within_std:.4f}), "
              f"avg separation = {avg_separation:.2f}x")

    # 2D visualization using PCA
    print(f"\n⏳ Creating 2D PCA visualization...")
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(12, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, n_types))

    for type_idx in range(n_types):
        mask = filter_types == type_idx
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                   c=[colors[type_idx]], label=filter_type_names[type_idx],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title(f'Latent Space Projection (PCA)\nSilhouette = {silhouette_avg:.3f}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'latent_space_pca_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved PCA visualization to {output_path}")
    plt.close()

    # t-SNE visualization (slower but better for non-linear structure)
    print(f"\n⏳ Creating t-SNE visualization (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(12, 9))

    for type_idx in range(n_types):
        mask = filter_types == type_idx
        plt.scatter(latent_tsne[mask, 0], latent_tsne[mask, 1],
                   c=[colors[type_idx]], label=filter_type_names[type_idx],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f'Latent Space Projection (t-SNE)\nSilhouette = {silhouette_avg:.3f}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'latent_space_tsne_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved t-SNE visualization to {output_path}")
    plt.close()

    return {
        'silhouette_avg': silhouette_avg,
        'silhouette_per_type': per_type_silhouette,
        'davies_bouldin': db_index,
        'calinski_harabasz': ch_index,
        'separation_matrix': separation_matrix,
        'pca_2d': latent_2d,
        'tsne_2d': latent_tsne
    }


def main():
    """Main analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='GED-Latent Space Correlation Analysis')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/variable_length/20251222_102121/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str,
                       default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                       default='analysis_results/ged_latent_analysis',
                       help='Output directory for results')
    parser.add_argument('--ged-matrix', type=str,
                       default='analysis_results/ged_matrix_120.npy',
                       help='Path to save/load GED matrix')
    parser.add_argument('--ged-checkpoint', type=str,
                       default='analysis_results/ged_checkpoint.npz',
                       help='Path for GED checkpoint')
    parser.add_argument('--skip-ged', action='store_true',
                       help='Skip GED computation (load existing matrix)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for encoding (cpu/mps/cuda)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Load raw dataset
    raw_dataset = load_dataset(args.dataset)

    # Step 1: Compute or load GED matrix
    if args.skip_ged and Path(args.ged_matrix).exists():
        print(f"\n⏭️  Skipping GED computation, loading existing matrix...")
        ged_matrix = np.load(args.ged_matrix)
    else:
        ged_matrix = compute_or_load_ged_matrix(raw_dataset, args.ged_matrix, args.ged_checkpoint)

    # Save GED matrix
    np.save(output_dir / 'ged_matrix.npy', ged_matrix)

    # Step 2: Load models and encode dataset
    encoder, decoder, config = load_models(args.checkpoint)

    # Create PyTorch Geometric dataset
    pyg_dataset = CircuitDataset(dataset_path=args.dataset)
    dataloader = DataLoader(pyg_dataset, batch_size=16, shuffle=False, collate_fn=collate_circuit_batch)

    # Encode all circuits
    encoding_results = encode_dataset(encoder, dataloader, device=args.device)

    # Save latent vectors
    np.save(output_dir / 'latent_mu.npy', encoding_results['latent_mu'])
    np.save(output_dir / 'latent_topo.npy', encoding_results['latent_topo'])
    np.save(output_dir / 'latent_values.npy', encoding_results['latent_values'])
    np.save(output_dir / 'latent_pz.npy', encoding_results['latent_pz'])
    np.save(output_dir / 'filter_types.npy', encoding_results['filter_types'])

    with open(output_dir / 'circuit_ids.txt', 'w') as f:
        f.write('\n'.join(encoding_results['circuit_ids']))

    print(f"\n💾 Saved latent encodings to {output_dir}/")

    # Step 3: Compute latent distances
    print(f"\n⏳ Computing pairwise latent distances...")
    latent_distances = compute_pairwise_distances(encoding_results['latent_mu'])
    np.save(output_dir / 'latent_distances.npy', latent_distances)

    # Step 4: Correlation analysis
    correlation_results = analyze_correlation(
        ged_matrix, latent_distances, encoding_results['filter_types'], output_dir
    )

    # Step 5: Clustering analysis
    clustering_results = analyze_clustering(
        encoding_results['latent_mu'], encoding_results['filter_types'], output_dir
    )

    # Save summary
    summary = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'n_circuits': len(raw_dataset),
        'latent_dim': encoding_results['latent_mu'].shape[1],
        'correlation': {
            'pearson_r': float(correlation_results['pearson_r']),
            'spearman_r': float(correlation_results['spearman_r'])
        },
        'clustering': {
            'silhouette': float(clustering_results['silhouette_avg']),
            'davies_bouldin': float(clustering_results['davies_bouldin']),
            'calinski_harabasz': float(clustering_results['calinski_harabasz'])
        }
    }

    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📊 Results saved to: {output_dir}/")
    print(f"\n📈 Key Findings:")
    print(f"   GED-Latent Correlation:  {correlation_results['pearson_r']:.4f}")
    print(f"   Clustering Quality:      {clustering_results['silhouette_avg']:.4f}")
    print(f"\nSee visualizations in output directory.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
