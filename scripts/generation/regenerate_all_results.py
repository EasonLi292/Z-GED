"""
Regenerate all results for GENERATION_RESULTS.md and NOVEL_TOPOLOGY_GENERATED.md.
Runs spec-based generation, centroids, interpolation, reconstruction, and novel topology exploration.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.constants import FILTER_TYPES, PZ_LOG_SCALE
from ml.utils.circuit_ops import walk_to_string, is_valid_walk, generate_walk
from ml.utils.runtime import load_encoder_decoder, make_collate_fn
from ml.utils.evaluate import (
    sequence_to_topology_key, get_training_topology_keys,
)
from torch.utils.data import DataLoader


def main():
    device = 'cpu'

    # Load models
    print("Loading models...")
    encoder, decoder, vocab, _ = load_encoder_decoder(
        checkpoint_path='checkpoints/production/best.pt',
        device=device,
    )

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # Build spec database and centroids
    print("Building spec database and centroids...")
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=make_collate_fn(include_specifications=True, include_pz_target=True),
    )

    all_specs, all_latents = [], []
    latents_by_type = {ft: [] for ft in FILTER_TYPES}

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            _, mu, _ = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            all_specs.append(batch['specifications'][0])
            all_latents.append(mu[0])
            latents_by_type[raw_data[idx]['filter_type']].append(mu[0])

    specs_db = torch.stack(all_specs)
    latents_db = torch.stack(all_latents)
    centroids = {ft: torch.stack(v).mean(0) for ft, v in latents_by_type.items() if v}

    print(f"  Cutoff range: {specs_db[:, 0].min():.2f} - {specs_db[:, 0].max():.1f} Hz")
    print(f"  Q range: {specs_db[:, 1].min():.3f} - {specs_db[:, 1].max():.3f}")

    # Helper: interpolate latent from specs
    def get_latent_for_specs(cutoff, q, k=5):
        target = torch.tensor([np.log10(cutoff), q])
        db_norm = torch.stack([torch.log10(specs_db[:, 0]), specs_db[:, 1]], dim=1)
        dists = ((db_norm - target)**2).sum(1).sqrt()
        top_k = dists.argsort()[:k]
        w = 1.0 / (dists[top_k] + 1e-6)
        w = w / w.sum()
        return (latents_db[top_k] * w.unsqueeze(1)).sum(0)

    # Helper: generate from latent
    def gen(z_vec):
        tokens = generate_walk(decoder, z_vec, vocab)
        return tokens

    # =========================================================================
    # 1. SPEC-BASED GENERATION
    # =========================================================================
    print("\n" + "="*70)
    print("1. SPEC-BASED GENERATION")
    print("="*70)

    spec_tests = [
        (1000, 0.707, "Standard"),
        (10000, 0.707, "Standard"),
        (100000, 0.707, "Standard"),
        (10000, 5.0, "Standard"),
        (1, 0.707, "Edge case"),
        (1000000, 0.707, "Edge case"),
        (10000, 0.01, "Edge case"),
        (10000, 0.1, "Edge case"),
        (10000, 2.0, "Edge case"),
        (50, 5.0, "Edge case"),
        (500000, 0.1, "Edge case"),
    ]

    for cutoff, q, category in spec_tests:
        z = get_latent_for_specs(cutoff, q)
        tokens = gen(z)
        cstr = walk_to_string(tokens, vocab)
        valid = is_valid_walk(tokens, vocab)
        print(f"  {cutoff:>10.0f} Hz, Q={q:<6.3f} -> `{cstr}` [{category}, {'Valid' if valid else 'INVALID'}]")

    # =========================================================================
    # 2. FILTER TYPE CENTROIDS
    # =========================================================================
    print("\n" + "="*70)
    print("2. FILTER TYPE CENTROIDS")
    print("="*70)

    for ft in FILTER_TYPES:
        z = centroids[ft]
        tokens = gen(z)
        cstr = walk_to_string(tokens, vocab)
        z_vals = z.numpy()
        print(f"  {ft:<15} z[0:4]=[{z_vals[0]:+.2f}, {z_vals[1]:+.2f}, {z_vals[2]:+.2f}, {z_vals[3]:+.2f}]  ->  `{cstr}`")

    # =========================================================================
    # 3. INTERPOLATION
    # =========================================================================
    print("\n" + "="*70)
    print("3. INTERPOLATION")
    print("="*70)

    interp_pairs = [
        ('low_pass', 'high_pass'),
        ('band_pass', 'rlc_parallel'),
    ]

    for from_ft, to_ft in interp_pairs:
        print(f"\n  {from_ft} -> {to_ft}:")
        z1, z2 = centroids[from_ft], centroids[to_ft]
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            z_interp = (1 - alpha) * z1 + alpha * z2
            tokens = gen(z_interp)
            cstr = walk_to_string(tokens, vocab)
            label = f" ({from_ft})" if alpha == 0 else f" ({to_ft})" if alpha == 1 else " (transition)" if alpha == 0.5 else ""
            print(f"    alpha={alpha:.2f}  `{cstr}`{label}")

    # =========================================================================
    # 4. RECONSTRUCTION ACCURACY
    # =========================================================================
    print("\n" + "="*70)
    print("4. RECONSTRUCTION ACCURACY")
    print("="*70)

    type_correct = {ft: 0 for ft in FILTER_TYPES}
    type_total = {ft: 0 for ft in FILTER_TYPES}
    type_examples = {}

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            graph = item['graph']
            ft = raw_data[idx]['filter_type']

            batch_idx = torch.zeros(graph.x.shape[0], dtype=torch.long)
            _, mu, _ = encoder(graph.x, graph.edge_index, graph.edge_attr, batch_idx)

            tokens = gen(mu[0])
            valid = is_valid_walk(tokens, vocab)
            type_total[ft] += 1
            if valid:
                type_correct[ft] += 1
            if ft not in type_examples:
                type_examples[ft] = walk_to_string(tokens, vocab)

    total_correct = sum(type_correct.values())
    total_total = sum(type_total.values())
    print(f"\n  Total: {total_correct}/{total_total} valid reconstructions")
    for ft in FILTER_TYPES:
        print(f"    {ft:<15} {type_correct[ft]}/{type_total[ft]}  example: `{type_examples.get(ft, 'N/A')}`")

    # =========================================================================
    # 5. NOVEL TOPOLOGY EXPLORATION (500 random samples)
    # =========================================================================
    print("\n" + "="*70)
    print("5. NOVEL TOPOLOGY EXPLORATION")
    print("="*70)

    # Get training topologies
    training_keys = get_training_topology_keys('rlc_dataset/filter_dataset.pkl', vocab)
    print(f"\n  Training topology keys ({len(training_keys)}):")
    for k in sorted(training_keys):
        print(f"    `{k}`")

    # Random sampling
    torch.manual_seed(42)
    z_random = torch.randn(500, 8)

    known_count = 0
    novel_valid = {}
    invalid_count = 0
    known_topo_counts = {}

    print(f"\n  Sampling 500 random latent codes...")
    with torch.no_grad():
        for i in range(500):
            tokens = gen(z_random[i])
            key = sequence_to_topology_key(tokens, vocab)

            if key is None:
                invalid_count += 1
            elif key in training_keys:
                known_count += 1
                known_topo_counts[key] = known_topo_counts.get(key, 0) + 1
            else:
                novel_valid[key] = novel_valid.get(key, 0) + 1

    novel_total = sum(novel_valid.values())
    print(f"\n  Results:")
    print(f"    Known topology samples: {known_count} ({100*known_count/500:.1f}%)")
    print(f"    Valid novel samples:    {novel_total} ({100*novel_total/500:.1f}%)")
    print(f"    Invalid samples:        {invalid_count} ({100*invalid_count/500:.1f}%)")
    print(f"    Unique novel topologies: {len(novel_valid)}")

    print(f"\n  Known topology breakdown:")
    for topo, count in sorted(known_topo_counts.items(), key=lambda x: -x[1]):
        print(f"    `{topo}` -> {count} samples")

    print(f"\n  Novel topologies:")
    for topo, count in sorted(novel_valid.items(), key=lambda x: -x[1]):
        print(f"    `{topo}` -> {count} samples")

    # =========================================================================
    # 6. POLE/ZERO-DRIVEN GENERATION
    # =========================================================================
    print("\n" + "="*70)
    print("6. POLE/ZERO-DRIVEN GENERATION")
    print("="*70)

    def signed_log_normalize(x, scale=PZ_LOG_SCALE):
        if abs(x) < 1e-30:
            return 0.0
        sign = 1.0 if x >= 0 else -1.0
        return sign * np.log10(abs(x) + 1.0) / scale

    def pz_to_latent(pole_real, pole_imag, zero_real, zero_imag):
        return torch.tensor([
            signed_log_normalize(pole_real),
            signed_log_normalize(abs(pole_imag)),
            signed_log_normalize(zero_real),
            signed_log_normalize(abs(zero_imag)),
        ], dtype=torch.float32)

    pz_tests = [
        (-6283, 0, 0, 0,       "RC low-pass ~1kHz (real pole, no zero)"),
        (-62832, 0, 0, 0,      "RC low-pass ~10kHz"),
        (-628318, 0, 0, 0,     "RC low-pass ~100kHz"),
        (0, 0, -6283, 0,       "RC high-pass ~1kHz (no pole, real zero)"),
        (-3142, 49348, 0, 0,   "Band-pass ~50kHz (conjugate pole)"),
        (-3142, 49348, 0, 49348, "Band-stop ~50kHz (pole+zero)"),
        (-100, 0, 0, 0,        "Very low freq pole"),
        (-1e6, 0, 0, 0,        "Very high freq pole"),
    ]

    torch.manual_seed(42)

    for pole_r, pole_i, zero_r, zero_i, desc in pz_tests:
        pz_latent = pz_to_latent(pole_r, pole_i, zero_r, zero_i)
        print(f"\n  {desc}")
        print(f"    Pole: {pole_r:+.0f} + {pole_i:.0f}j, Zero: {zero_r:+.0f} + {zero_i:.0f}j")
        print(f"    z[4:8] = [{pz_latent[0]:+.3f}, {pz_latent[1]:+.3f}, {pz_latent[2]:+.3f}, {pz_latent[3]:+.3f}]")

        for s in range(3):
            z_topo = torch.randn(4)
            z = torch.cat([z_topo, pz_latent])
            tokens = gen(z)
            cstr = walk_to_string(tokens, vocab)
            valid = is_valid_walk(tokens, vocab)
            print(f"    Sample {s+1}: `{cstr}` [{'Valid' if valid else 'INVALID'}]")

    print("\n" + "="*70)
    print("ALL RESULTS GENERATED")
    print("="*70)


if __name__ == '__main__':
    main()
