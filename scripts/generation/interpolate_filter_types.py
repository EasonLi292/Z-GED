"""
Interpolate between filter types in latent space.

This script computes latent centroids for each filter type, then generates
circuits at various interpolation points (alpha) between two filter types.

Usage:
    python scripts/generation/interpolate_filter_types.py --from low_pass --to high_pass
    python scripts/generation/interpolate_filter_types.py --from band_pass --to rlc_parallel --steps 5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import SimplifiedCircuitDecoder
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']


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


def build_filter_type_centroids(encoder, dataset, raw_dataset, device='cpu'):
    """
    Build latent centroids for each filter type.

    Args:
        encoder: Trained encoder
        dataset: CircuitDataset (for graph data)
        raw_dataset: Raw pickle data (for filter_type labels)
        device: Device to use

    Returns:
        centroids: dict mapping filter_type -> latent centroid tensor
        all_latents: dict mapping filter_type -> list of all latent codes
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_circuit_batch)

    # Group latents by filter type
    latents_by_type = {ft: [] for ft in FILTER_TYPES}

    print("Encoding circuits and grouping by filter type...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            poles = batch['poles']
            zeros = batch['zeros']

            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles,
                zeros
            )

            # Get filter type from raw dataset
            filter_type = raw_dataset[idx]['filter_type']
            latents_by_type[filter_type].append(mu[0])

    # Compute centroids
    centroids = {}
    for ft in FILTER_TYPES:
        if latents_by_type[ft]:
            stacked = torch.stack(latents_by_type[ft])
            centroids[ft] = stacked.mean(dim=0)
            print(f"  {ft}: {len(latents_by_type[ft])} circuits, centroid z[0:2]=[{centroids[ft][0]:.2f}, {centroids[ft][1]:.2f}]")
        else:
            print(f"  {ft}: no circuits found")

    return centroids, latents_by_type


def circuit_to_string(circuit):
    """Convert generated circuit to string representation."""
    edge_exist = circuit['edge_existence'][0]
    comp_types = circuit['component_types'][0]
    node_types = circuit['node_types'][0]
    num_nodes = node_types.shape[0]

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

    return ', '.join(edges) if edges else '(no edges)'


def check_validity(circuit):
    """Check if circuit has VIN and VOUT connected."""
    edge_exist = circuit['edge_existence'][0]
    vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
    vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
    return vin_connected and vout_connected


def interpolate_and_generate(decoder, z1, z2, alpha, device):
    """Interpolate between two latent codes and generate circuit."""
    z_interp = (1 - alpha) * z1 + alpha * z2
    z_interp = z_interp.unsqueeze(0).to(device).float()

    with torch.no_grad():
        circuit = decoder.generate(z_interp, verbose=False)

    return circuit, z_interp[0]


def main():
    parser = argparse.ArgumentParser(description='Interpolate between filter types in latent space')
    parser.add_argument('--from', dest='from_type', type=str, required=True,
                        choices=FILTER_TYPES, help='Source filter type')
    parser.add_argument('--to', dest='to_type', type=str, required=True,
                        choices=FILTER_TYPES, help='Target filter type')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of interpolation steps (default: 5)')
    parser.add_argument('--checkpoint', default='checkpoints/production/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', default='rlc_dataset/filter_dataset.pkl',
                        help='Path to dataset')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--show-centroids', action='store_true',
                        help='Show all filter type centroids')

    args = parser.parse_args()

    print("=" * 70)
    print("Latent Space Interpolation Between Filter Types")
    print("=" * 70)

    device = torch.device(args.device)

    # Load models
    print("\nLoading models...")
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
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
        max_nodes=10
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load dataset
    print("\nLoading dataset...")
    dataset = CircuitDataset(args.dataset)
    with open(args.dataset, 'rb') as f:
        raw_dataset = pickle.load(f)

    # Build centroids
    print()
    centroids, all_latents = build_filter_type_centroids(encoder, dataset, raw_dataset, device)

    if args.show_centroids:
        print("\n" + "=" * 70)
        print("Filter Type Centroids (8D latent)")
        print("=" * 70)
        for ft in FILTER_TYPES:
            if ft in centroids:
                z = centroids[ft].numpy()
                print(f"\n{ft}:")
                print(f"  z = [{', '.join(f'{v:.3f}' for v in z)}]")

    # Validate filter types exist
    if args.from_type not in centroids:
        print(f"\nError: No circuits found for filter type '{args.from_type}'")
        return
    if args.to_type not in centroids:
        print(f"\nError: No circuits found for filter type '{args.to_type}'")
        return

    z1 = centroids[args.from_type]
    z2 = centroids[args.to_type]

    # Generate interpolation
    print("\n" + "=" * 70)
    print(f"Interpolation: {args.from_type} -> {args.to_type}")
    print("=" * 70)
    print(f"\nSource ({args.from_type}) z[0:2] = [{z1[0]:.2f}, {z1[1]:.2f}]")
    print(f"Target ({args.to_type}) z[0:2] = [{z2[0]:.2f}, {z2[1]:.2f}]")

    alphas = np.linspace(0, 1, args.steps)

    print(f"\nGenerating {args.steps} interpolation steps:\n")
    print("-" * 70)
    print(f"{'alpha':<8} {'Circuit':<50} {'Valid'}")
    print("-" * 70)

    results = []
    for alpha in alphas:
        circuit, z_interp = interpolate_and_generate(decoder, z1, z2, alpha, device)
        circuit_str = circuit_to_string(circuit)
        valid = check_validity(circuit)

        label = ""
        if alpha == 0:
            label = f" ({args.from_type})"
        elif alpha == 1:
            label = f" ({args.to_type})"
        elif 0.4 <= alpha <= 0.6:
            label = " (transition)"

        print(f"{alpha:<8.2f} `{circuit_str}`{label:<20} {'Yes' if valid else 'No'}")

        results.append({
            'alpha': alpha,
            'circuit': circuit_str,
            'valid': valid,
            'z': z_interp.numpy()
        })

    print("-" * 70)

    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\nSummary: {valid_count}/{len(results)} valid circuits")

    # Find transition point
    prev_circuit = None
    for r in results:
        if prev_circuit is not None and r['circuit'] != prev_circuit:
            print(f"Topology change detected at alpha = {r['alpha']:.2f}")
        prev_circuit = r['circuit']

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
