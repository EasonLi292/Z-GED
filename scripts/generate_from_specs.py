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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
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
    """
    encoder.eval()

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_circuit_batch)

    all_specs = []
    all_latents = []

    print("Building specification database...")
    with torch.no_grad():
        for batch in loader:
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

    specs_tensor = torch.stack(all_specs)  # [N, 2]
    latents_tensor = torch.stack(all_latents)  # [N, 8]

    print(f"Built database with {len(specs_tensor)} circuits")
    print(f"  Cutoff range: {specs_tensor[:, 0].min():.1f} - {specs_tensor[:, 0].max():.1f} Hz")
    print(f"  Q-factor range: {specs_tensor[:, 1].min():.3f} - {specs_tensor[:, 1].max():.3f}")

    return specs_tensor, latents_tensor


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


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5):
    """
    Interpolate between k nearest latent codes.

    Args:
        target_cutoff: Target cutoff frequency
        target_q: Target Q-factor
        specs_db: [N, 2] specifications database
        latents_db: [N, 8] latents database
        k: Number of neighbors to interpolate

    Returns:
        latent: [8] interpolated latent code
    """
    # Normalize target
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    # Find k nearest neighbors
    distances = ((db_normalized - target)**2).sum(dim=1).sqrt()
    nearest_indices = distances.argsort()[:k]

    # Inverse distance weighting
    nearest_dists = distances[nearest_indices]
    weights = 1.0 / (nearest_dists + 1e-6)
    weights = weights / weights.sum()

    # Weighted average of latents
    interpolated = (latents_db[nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

    return interpolated


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

    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=5
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load dataset and build specification database
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

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
            latent = interpolate_latents(
                args.cutoff, args.q_factor, specs_db, latents_db, k=5
            )
            # Find closest match for reporting
            _, actual_specs, dist = find_nearest_latent(
                args.cutoff, args.q_factor, specs_db, latents_db
            )
            print(f"\nSample {i+1}: Interpolated from 5 nearest")

        # Add small random variation for diversity
        if i > 0:
            latent = latent + torch.randn_like(latent) * 0.1

        # Generate circuit
        latent = latent.unsqueeze(0).to(device).float()  # Ensure float32
        conditions = torch.randn(1, 2, device=device)

        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Analyze generated circuit
        edge_exist = circuit['edge_existence'][0]
        num_edges = (edge_exist > 0.5).sum().item() // 2

        print(f"  Reference circuit: cutoff={actual_specs[0]:.1f} Hz, Q={actual_specs[1]:.3f}")
        print(f"  Generated: {num_edges} edges")

        # Check VIN/VOUT connectivity
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()

        status = "✅" if (vin_connected and vout_connected) else "❌"
        print(f"  Valid circuit: {status}")

    print("\n" + "="*70)
    print("Generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
