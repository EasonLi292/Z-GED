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
from ml.models.constants import FILTER_TYPES
from ml.utils.circuit_ops import walk_to_string, is_valid_walk, generate_walk
from ml.utils.runtime import load_encoder_decoder, make_collate_fn
from torch.utils.data import DataLoader


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
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=make_collate_fn(include_specifications=True),
    )

    # Group latents by filter type
    latents_by_type = {ft: [] for ft in FILTER_TYPES}

    print("Encoding circuits and grouping by filter type...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)

            _, mu, _ = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            )

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
    encoder, decoder, vocab, _ = load_encoder_decoder(
        checkpoint_path=args.checkpoint,
        device=str(device),
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = CircuitDataset(args.dataset)
    with open(args.dataset, 'rb') as f:
        raw_dataset = pickle.load(f)

    # Build centroids
    print()
    centroids, _ = build_filter_type_centroids(encoder, dataset, raw_dataset, device)

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
        z_interp = (1 - alpha) * z1 + alpha * z2
        tokens = generate_walk(decoder, z_interp.to(device), vocab)
        circuit_str = walk_to_string(tokens, vocab)
        valid = is_valid_walk(tokens, vocab)

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
