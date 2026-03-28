"""
Comprehensive test of various input specifications.

Tests different combinations of cutoff frequency and Q-factor to document
what topologies are generated for different specifications.
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
    }

    return interpolated, info


def main():
    device = 'cpu'

    # Load models
    print("Loading models...", flush=True)
    encoder, decoder, vocab, _ = load_encoder_decoder(
        checkpoint_path='checkpoints/production/best.pt',
        device=device,
    )

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

    # Test specifications
    test_cases = [
        # Low-pass filters (Q ~ 0.707)
        (100, 0.707, "100 Hz, Q=0.707"),
        (10000, 0.707, "10 kHz, Q=0.707"),
        (100000, 0.707, "100 kHz, Q=0.707"),

        # High-pass filters
        (500, 0.707, "500 Hz, Q=0.707"),
        (50000, 0.707, "50 kHz, Q=0.707"),

        # Band-pass filters (moderate Q)
        (1000, 1.5, "1 kHz, Q=1.5"),
        (5000, 2.0, "5 kHz, Q=2.0"),
        (15000, 3.0, "15 kHz, Q=3.0"),
        (50000, 2.5, "50 kHz, Q=2.5"),

        # High-Q resonators
        (1000, 5.0, "1 kHz, Q=5.0"),
        (10000, 10.0, "10 kHz, Q=10.0"),
        (5000, 20.0, "5 kHz, Q=20.0"),

        # Overdamped filters (low Q)
        (1000, 0.3, "1 kHz, Q=0.3"),
        (50000, 0.1, "50 kHz, Q=0.1"),

        # Edge cases
        (50, 0.707, "50 Hz, Q=0.707"),
        (500000, 0.707, "500 kHz, Q=0.707"),
        (10000, 0.05, "10 kHz, Q=0.05"),
        (5000, 30.0, "5 kHz, Q=30.0"),
    ]

    results = []

    print("\n" + "="*80)
    print("Comprehensive Specification Test")
    print("="*80)
    print(f"Testing {len(test_cases)} different specifications\n")

    for idx, (target_cutoff, target_q, description) in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] {description}", flush=True)
        print(f"  Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}", flush=True)

        # K-NN interpolation to find latent code
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

        # Generate circuit from latent
        tokens = generate_walk(decoder, latent.to(device), vocab)
        circuit_str = walk_to_string(tokens, vocab)
        valid = is_valid_walk(tokens, vocab)

        # Analyze components
        components = set()
        for t in tokens:
            ctype = vocab.component_type(t)
            if ctype:
                for c in ['R', 'C', 'L']:
                    if c in ctype:
                        components.add(c)

        num_comps = sum(1 for t in tokens if vocab.token_type(t) == 'component')
        comp_str = '+'.join(sorted(components)) if components else 'None'

        print(f"  Generated: `{circuit_str}` ({comp_str}), valid={valid}", flush=True)
        print(f"  Nearest neighbor: {info['neighbor_specs'][0][0]:.1f} Hz, Q={info['neighbor_specs'][0][1]:.3f}\n", flush=True)

        results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'circuit': circuit_str,
            'components': comp_str,
            'num_components': num_comps,
            'valid': valid,
            'nearest_neighbor': info['neighbor_specs'][0]
        })

    # Generate summary report
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print("\nOverall Statistics:")
    valid_count = sum(1 for r in results if r['valid'])
    print(f"  Valid circuits: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")

    # Topology distribution
    print("\nTopology Distribution:")
    topo_counts = {}
    for r in results:
        key = r['circuit']
        topo_counts[key] = topo_counts.get(key, 0) + 1

    for topo, count in sorted(topo_counts.items(), key=lambda x: -x[1]):
        print(f"  `{topo}`: {count} circuits")

    # Validity by category
    categories = {
        'Low-pass (Q~0.707)': [r for r in results if 0.5 < r['target_q'] < 1.0 and r['target_cutoff'] < 150000],
        'Band-pass (1<Q<5)': [r for r in results if 1.0 < r['target_q'] < 5.0],
        'High-Q (Q>=5)': [r for r in results if r['target_q'] >= 5.0],
        'Overdamped (Q<0.5)': [r for r in results if r['target_q'] < 0.5],
    }

    print("\nValidity by Category:")
    for cat_name, cat_results in categories.items():
        valid_in_cat = sum(1 for r in cat_results if r['valid'])
        if cat_results:
            print(f"  {cat_name}: {valid_in_cat}/{len(cat_results)} valid ({100*valid_in_cat/len(cat_results):.1f}%)")

    # Save detailed results
    print("\nSaving detailed results...")
    os.makedirs('docs', exist_ok=True)
    with open('docs/GENERATION_TEST_RESULTS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Comprehensive Topology Generation Test Results (Sequence Decoder)\n")
        f.write("="*80 + "\n\n")

        for r in results:
            f.write(f"\n{r['description']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target: {r['target_cutoff']:.1f} Hz, Q={r['target_q']:.3f}\n")
            f.write(f"Circuit: {r['circuit']}\n")
            f.write(f"Components: {r['components']}\n")
            f.write(f"Valid: {'Yes' if r['valid'] else 'No'}\n")
            f.write(f"Nearest neighbor: {r['nearest_neighbor'][0]:.1f} Hz, Q={r['nearest_neighbor'][1]:.3f}\n")

    print("Results saved to docs/GENERATION_TEST_RESULTS.txt")


if __name__ == '__main__':
    main()
