"""
Test circuit generation on a single unseen specification.

This script generates a circuit for one target specification and provides
detailed analysis of the result.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.utils.circuit_ops import walk_to_string, is_valid_walk, generate_walk
from ml.utils.runtime import load_encoder_decoder, make_collate_fn
from torch.utils.data import DataLoader


def build_specification_database(encoder, dataset, device='cpu'):
    """Build database of specifications -> latent codes."""
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=make_collate_fn(include_specifications=True),
    )

    all_specs = []
    all_latents = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)

            _, mu, _ = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch
            )

            specs = batch['specifications'][0]
            all_specs.append(specs)
            all_latents.append(mu[0])

    specs_tensor = torch.stack(all_specs)
    latents_tensor = torch.stack(all_latents)

    return specs_tensor, latents_tensor


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5):
    """Interpolate latent codes from k-nearest neighbors."""
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    spec_distances = ((db_normalized - target)**2).sum(dim=1).sqrt()
    spec_nearest_indices = spec_distances.argsort()[:k]
    spec_dists_k = spec_distances[spec_nearest_indices]
    weights = 1.0 / (spec_dists_k + 1e-6)
    weights = weights / weights.sum()

    interpolated = (latents_db[spec_nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

    info = {
        'neighbor_indices': spec_nearest_indices.numpy(),
        'neighbor_specs': specs_db[spec_nearest_indices].numpy(),
        'weights': weights.numpy(),
        'spec_distances': spec_distances[spec_nearest_indices].numpy()
    }

    return interpolated, info


def main():
    device = 'cpu'

    # Get target specification from command line or use default
    if len(sys.argv) >= 3:
        target_cutoff = float(sys.argv[1])
        target_q = float(sys.argv[2])
    else:
        target_cutoff = 15000.0  # 15 kHz
        target_q = 1.5

    print("="*80)
    print("Single Specification Test")
    print("="*80)
    print(f"\nTarget: {target_cutoff:.1f} Hz, Q={target_q:.3f}")
    print()

    # Load models
    print("Loading models...")
    encoder, decoder, vocab, _ = load_encoder_decoder(
        checkpoint_path='checkpoints/production/best.pt',
        device=device,
    )

    # Load dataset
    print("Loading dataset and building specification database...")
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

    print(f"Dataset: {len(specs_db)} circuits")
    print(f"  Cutoff range: {specs_db[:, 0].min():.1f} - {specs_db[:, 0].max():.1f} Hz")
    print(f"  Q range: {specs_db[:, 1].min():.3f} - {specs_db[:, 1].max():.3f}")

    # Check if target is in training data
    target_tensor = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([torch.log10(specs_db[:, 0]), specs_db[:, 1]], dim=1)
    distances = ((db_normalized - target_tensor)**2).sum(dim=1).sqrt()
    min_dist = distances.min().item()

    print(f"\nTarget distance from training data: {min_dist:.4f}")
    if min_dist < 0.05:
        print("  WARNING: Very close to training data")
    else:
        print("  Truly unseen specification")

    # Find k-NN neighbors and interpolate
    print("\n" + "="*80)
    print("K-NN Interpolation (k=5)")
    print("="*80)

    latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

    print("\nNearest 5 training circuits:")
    for i in range(5):
        neighbor_specs = info['neighbor_specs'][i]
        weight = info['weights'][i]
        dist = info['spec_distances'][i]
        print(f"  {i+1}. {neighbor_specs[0]:>8.1f} Hz, Q={neighbor_specs[1]:>6.3f}  "
              f"(weight={weight:.3f}, dist={dist:.3f})")

    print(f"\nInterpolated latent code: {latent.numpy()}")

    # Generate circuit
    print("\n" + "="*80)
    print("Circuit Generation")
    print("="*80)

    tokens = generate_walk(decoder, latent.to(device), vocab)
    circuit_str = walk_to_string(tokens, vocab)
    valid = is_valid_walk(tokens, vocab)

    print(f"\nGenerated walk: {tokens}")
    print(f"Circuit: `{circuit_str}`")

    # Analyze structure
    print("\n" + "="*80)
    print("Circuit Structure Analysis")
    print("="*80)

    # Extract components and their connections
    from collections import defaultdict
    comp_nets = defaultdict(set)
    for i, tok in enumerate(tokens):
        if vocab.token_type(tok) == 'component':
            if i > 0 and vocab.token_type(tokens[i - 1]) == 'net':
                comp_nets[tok].add(tokens[i - 1])
            if i < len(tokens) - 1 and vocab.token_type(tokens[i + 1]) == 'net':
                comp_nets[tok].add(tokens[i + 1])

    # Nets used
    nets = set(t for t in tokens if vocab.token_type(t) == 'net')
    print(f"\nNets ({len(nets)}): {sorted(nets)}")

    # Components and connections
    print(f"\nComponents ({len(comp_nets)}):")
    for comp in sorted(comp_nets.keys()):
        ctype = vocab.component_type(comp)
        net_list = sorted(comp_nets[comp])
        if len(net_list) == 2:
            print(f"  {comp} ({ctype}): {net_list[0]} -- {net_list[1]}")
        else:
            print(f"  {comp} ({ctype}): {net_list}")

    # Validity
    print(f"\nValidity check:")
    has_vin = 'VIN' in nets
    has_vout = 'VOUT' in nets
    print(f"  VIN present:  {'yes' if has_vin else 'no'}")
    print(f"  VOUT present: {'yes' if has_vout else 'no'}")
    print(f"  Overall: {'VALID' if valid else 'INVALID'}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nGenerated topology: `{circuit_str}`")
    print(f"Circuit valid: {'yes' if valid else 'no'}")


if __name__ == '__main__':
    main()
