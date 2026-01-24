"""
Generate circuits from user specifications (cutoff frequency, Q-factor).

This script uses the trained encoder to build a specification → latent mapping,
then generates circuits matching user-provided specifications.

Usage:
    python scripts/generate_from_specs.py --cutoff 10000 --q-factor 0.707
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import SimplifiedCircuitDecoder
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def collate_circuit_batch(batch_list):
    """Custom collate function."""
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    specifications = torch.stack([item['specifications'] for item in batch_list])

    batched_graph = Batch.from_data_list(graphs)
    return {
        'graph': batched_graph,
        'poles': poles,
        'zeros': zeros,
        'specifications': specifications
    }


def build_specification_database(encoder, dataset, device='cpu'):
    """
    Build a database mapping specifications to latent codes.

    Args:
        encoder: Trained encoder
        dataset: CircuitDataset
        device: Device to use

    Returns:
        specs: [N, 2] tensor of [cutoff_freq, q_factor]
        latents: [N, 8] tensor of latent codes
        indices: [N] list of dataset indices
    """
    encoder.eval()

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_circuit_batch)

    all_specs = []
    all_latents = []
    all_indices = []

    print("Building specification database...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            poles = batch['poles']
            zeros = batch['zeros']

            # Encode circuit
            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles,
                zeros
            )

            # Store specifications and latent code
            specs = batch['specifications'][0]  # [2]
            all_specs.append(specs)
            all_latents.append(mu[0])
            all_indices.append(idx)

    specs_tensor = torch.stack(all_specs)  # [N, 2]
    latents_tensor = torch.stack(all_latents)  # [N, 8]
    indices_list = all_indices  # [N]

    print(f"Built database with {len(specs_tensor)} circuits")
    print(f"  Cutoff range: {specs_tensor[:, 0].min():.1f} - {specs_tensor[:, 0].max():.1f} Hz")
    print(f"  Q-factor range: {specs_tensor[:, 1].min():.3f} - {specs_tensor[:, 1].max():.3f}")

    return specs_tensor, latents_tensor, indices_list


def find_nearest_latent(target_cutoff, target_q, specs_db, latents_db):
    """
    Find latent code for circuit with nearest specifications.

    Args:
        target_cutoff: Target cutoff frequency (Hz)
        target_q: Target Q-factor
        specs_db: [N, 2] database of specifications
        latents_db: [N, 8] database of latent codes

    Returns:
        latent: [8] latent code for nearest circuit
        actual_specs: [2] actual specs of nearest circuit
        distance: Distance to target
    """
    # Normalize specifications (log scale for frequency, linear for Q)
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    # Compute distances (weighted)
    freq_weight = 1.0  # Frequency matching weight
    q_weight = 2.0     # Q-factor matching weight

    distances = (
        freq_weight * (db_normalized[:, 0] - target[0])**2 +
        q_weight * (db_normalized[:, 1] - target[1])**2
    ).sqrt()

    # Find nearest
    nearest_idx = distances.argmin()
    nearest_latent = latents_db[nearest_idx]
    nearest_specs = specs_db[nearest_idx]
    nearest_dist = distances[nearest_idx].item()

    return nearest_latent, nearest_specs, nearest_dist


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, indices_db,
                       ged_matrix=None, k=5, ged_weight=0.5):
    """
    Interpolate between k nearest latent codes using spec distance + GED.

    Args:
        target_cutoff: Target cutoff frequency
        target_q: Target Q-factor
        specs_db: [N, 2] specifications database
        latents_db: [N, 8] latents database
        indices_db: [N] dataset indices
        ged_matrix: [N, N] Graph Edit Distance matrix (optional)
        k: Number of neighbors to interpolate
        ged_weight: Weight for GED vs spec distance (0=only specs, 1=only GED)

    Returns:
        latent: [8] interpolated latent code
        info: dict with neighbor information
    """
    # Normalize target
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    # Compute specification distances
    spec_distances = ((db_normalized - target)**2).sum(dim=1).sqrt()

    # Find k nearest by specification
    spec_nearest_indices = spec_distances.argsort()[:k]

    if ged_matrix is not None and ged_weight > 0:
        # Use GED to refine weights among k nearest
        # Get GED distances from each candidate to all others in the k-set
        k_indices = indices_db[spec_nearest_indices.numpy()]

        # For each candidate, compute average GED to other candidates
        # Lower GED = more similar structure = higher weight
        ged_avg = []
        for i, idx in enumerate(k_indices):
            # Get GEDs to other candidates (excluding self)
            other_indices = [k_indices[j] for j in range(k) if j != i]
            if len(other_indices) > 0:
                geds_to_others = ged_matrix[idx, other_indices]
                avg_ged = np.mean(geds_to_others)
            else:
                avg_ged = 0.0
            ged_avg.append(avg_ged)

        ged_avg = torch.tensor(ged_avg, dtype=torch.float32)

        # Combine: lower distance = higher weight
        # Normalize both distances to [0, 1] range
        spec_dists_k = spec_distances[spec_nearest_indices]
        spec_dists_norm = spec_dists_k / (spec_dists_k.max() + 1e-6)
        ged_dists_norm = ged_avg / (ged_avg.max() + 1e-6)

        # Combined distance
        combined_distances = (1 - ged_weight) * spec_dists_norm + ged_weight * ged_dists_norm

        # Inverse distance weighting
        weights = 1.0 / (combined_distances + 1e-6)
        weights = weights / weights.sum()
    else:
        # Only use specification distances
        spec_dists_k = spec_distances[spec_nearest_indices]
        weights = 1.0 / (spec_dists_k + 1e-6)
        weights = weights / weights.sum()

    # Weighted average of latents
    interpolated = (latents_db[spec_nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

    # Gather info for reporting
    info = {
        'neighbor_indices': spec_nearest_indices.numpy(),
        'neighbor_specs': specs_db[spec_nearest_indices].numpy(),
        'weights': weights.numpy(),
        'spec_distances': spec_distances[spec_nearest_indices].numpy()
    }

    return interpolated, info


def main():
    parser = argparse.ArgumentParser(description='Generate circuits from specifications')
    parser.add_argument('--cutoff', type=float, required=True,
                        help='Target cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                        help='Target Q-factor (default: 0.707 for Butterworth)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of circuits to generate')
    parser.add_argument('--method', choices=['nearest', 'interpolate'], default='interpolate',
                        help='Generation method')
    parser.add_argument('--ged-weight', type=float, default=0.5,
                        help='Weight for GED vs spec distance (0=only specs, 1=only GED, default=0.5)')
    parser.add_argument('--ged-matrix', default='analysis_results/ged_matrix_360.npy',
                        help='Path to GED matrix')
    parser.add_argument('--checkpoint', default='checkpoints/production/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', default='cpu', help='Device to use')

    args = parser.parse_args()

    print("="*70)
    print("Specification-Driven Circuit Generation")
    print("="*70)
    print(f"\nTarget specifications:")
    print(f"  Cutoff frequency: {args.cutoff:.1f} Hz")
    print(f"  Q-factor: {args.q_factor:.3f}")
    print(f"  Generation method: {args.method}")
    if args.method == 'interpolate':
        print(f"  GED weight: {args.ged_weight:.2f} (0=spec only, 1=GED only)")
    print()

    device = torch.device(args.device)

    # Load models
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=7,
        gnn_hidden_dim=64,
        gnn_num_layers=3,
        latent_dim=8,
        topo_latent_dim=2,
        values_latent_dim=2,
        pz_latent_dim=4,
        dropout=0.1
    ).to(device)

    decoder = SimplifiedCircuitDecoder(
        latent_dim=8,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=10  # Must match training config
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load dataset and build specification database
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db, indices_db = build_specification_database(encoder, dataset, device)
    indices_db = np.array(indices_db)  # Convert to numpy array for indexing

    # Load GED matrix if using interpolation
    ged_matrix = None
    if args.method == 'interpolate' and args.ged_weight > 0:
        try:
            ged_matrix = np.load(args.ged_matrix)
            # Check if GED matrix dimensions match dataset size
            if ged_matrix.shape[0] != len(dataset):
                print(f"\nGED matrix size ({ged_matrix.shape[0]}) doesn't match dataset ({len(dataset)})")
                print("  Falling back to spec-only weighting")
                ged_matrix = None
                args.ged_weight = 0
            else:
                print(f"\nLoaded GED matrix: {ged_matrix.shape}")
                print(f"  GED range: {ged_matrix[ged_matrix > 0].min():.2f} - {ged_matrix.max():.2f}")
        except FileNotFoundError:
            print(f"\nGED matrix not found at {args.ged_matrix}")
            print("  Falling back to spec-only weighting")
            args.ged_weight = 0

    # Generate circuits
    print(f"\nGenerating {args.num_samples} circuits...")
    print("="*70)

    for i in range(args.num_samples):
        if args.method == 'nearest':
            latent, actual_specs, dist = find_nearest_latent(
                args.cutoff, args.q_factor, specs_db, latents_db
            )
            print(f"\nSample {i+1}: Nearest neighbor (distance={dist:.3f})")
        else:
            latent, info = interpolate_latents(
                args.cutoff, args.q_factor, specs_db, latents_db, indices_db,
                ged_matrix=ged_matrix, k=5, ged_weight=args.ged_weight
            )
            # Find closest match for reporting
            _, actual_specs, dist = find_nearest_latent(
                args.cutoff, args.q_factor, specs_db, latents_db
            )
            print(f"\nSample {i+1}: Interpolated from 5 nearest")
            if ged_matrix is not None and args.ged_weight > 0:
                print(f"  Neighbor weights (GED-adjusted): {info['weights']}")
                print(f"  Top neighbor: cutoff={info['neighbor_specs'][0][0]:.1f} Hz, "
                      f"Q={info['neighbor_specs'][0][1]:.3f}, weight={info['weights'][0]:.3f}")

        # Add small random variation for diversity
        if i > 0:
            latent = latent + torch.randn_like(latent) * 0.1

        # Generate circuit from latent (no conditions needed)
        latent = latent.unsqueeze(0).to(device).float()  # Ensure float32

        with torch.no_grad():
            circuit = decoder.generate(latent, verbose=False)

        # Analyze generated circuit
        edge_exist = circuit['edge_existence'][0]
        comp_types = circuit['component_types'][0]
        node_types = circuit['node_types'][0]
        num_nodes = node_types.shape[0]
        num_edges = (edge_exist > 0.5).sum().item() // 2

        # Build circuit description
        # Map node types to base names, then number internal nodes
        BASE_NAMES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INT', 4: 'INT'}
        COMP_NAMES = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']

        # Build unique names for each node (numbering internal nodes)
        node_names = []
        int_counter = 1
        for idx in range(num_nodes):
            nt = node_types[idx].item()
            if nt >= 3:  # Internal node
                node_names.append(f'INT{int_counter}')
                int_counter += 1
            else:
                node_names.append(BASE_NAMES[nt])

        edges = []
        for ni in range(num_nodes):
            for nj in range(ni):
                if edge_exist[ni, nj] > 0.5:
                    n1 = node_names[nj]
                    n2 = node_names[ni]
                    comp = COMP_NAMES[comp_types[ni, nj].item()]
                    edges.append(f"{n1}--{comp}--{n2}")

        circuit_str = ', '.join(edges) if edges else '(no edges)'

        print(f"  Nearest match: cutoff={actual_specs[0]:.1f} Hz, Q={actual_specs[1]:.3f}")
        print(f"  Generated: {circuit_str}")

        # Check VIN/VOUT connectivity
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()

        status = "Valid" if (vin_connected and vout_connected) else "Invalid"
        print(f"  Status: {status} ({num_nodes} nodes, {num_edges} edges)")

    print("\n" + "="*70)
    print("Generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
