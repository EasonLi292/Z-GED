"""
Probe the sequence decoder's latent space for novel topologies.

Strategies:
1. Dense random sampling (10,000 samples at various scales)
2. Boundary probing — interpolate between centroids of different filter types
3. Extrapolation — push beyond training latent range
4. Targeted exploration — sample near decision boundaries
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from collections import Counter, defaultdict

from ml.utils.runtime import build_encoder
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.evaluate import sequence_to_topology_key, get_training_topology_keys


def walk_to_description(tokens, vocab):
    comp_nets = defaultdict(set)
    for i, tok in enumerate(tokens):
        if vocab.token_type(tok) == 'component':
            if i > 0 and vocab.token_type(tokens[i - 1]) == 'net':
                comp_nets[tok].add(tokens[i - 1])
            if i < len(tokens) - 1 and vocab.token_type(tokens[i + 1]) == 'net':
                comp_nets[tok].add(tokens[i + 1])
    if not comp_nets:
        return '(no components)'
    parts = []
    for comp in sorted(comp_nets.keys()):
        nets = sorted(comp_nets[comp])
        ctype = vocab.component_type(comp)
        if len(nets) == 2:
            parts.append(f"{nets[0]}--{ctype}--{nets[1]}")
        elif len(nets) == 1:
            parts.append(f"{nets[0]}--{ctype}--{nets[0]}")
        else:
            parts.append(f"{ctype}({','.join(nets)})")
    return ', '.join(parts)


def generate_batch(decoder, z_batch, vocab, max_length=32):
    """Generate and classify a batch of latent codes."""
    with torch.no_grad():
        generated = decoder.generate(
            z_batch.float(), max_length=max_length, greedy=True, eos_id=vocab.eos_id,
        )
    results = []
    for gen_ids in generated:
        tokens = vocab.decode(gen_ids)
        key = sequence_to_topology_key(tokens, vocab)
        desc = walk_to_description(tokens, vocab)
        results.append((key, desc, tokens))
    return results


def main():
    device = 'cpu'

    vocab = CircuitVocabulary(max_internal=10, max_components=10)
    seq_ckpt = torch.load(
        os.path.join(os.path.dirname(__file__), 'best.pt'),
        map_location=device, weights_only=False,
    )

    seq_encoder = build_encoder(device=device)
    seq_encoder.load_state_dict(seq_ckpt['encoder_state_dict'])
    seq_encoder.eval()

    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size, latent_dim=8, d_model=256,
        n_heads=4, n_layers=4, max_seq_len=33, dropout=0.0, pad_id=vocab.pad_id,
    ).to(device)
    decoder.load_state_dict(seq_ckpt['decoder_state_dict'])
    decoder.eval()

    training_keys = get_training_topology_keys('rlc_dataset/filter_dataset.pkl', vocab)
    print(f"Training topology keys: {len(training_keys)}")
    for k in sorted(training_keys):
        print(f"  {k}")

    # Build centroids per filter type
    import pickle
    from ml.data.dataset import CircuitDataset
    from ml.utils.runtime import make_collate_fn
    from torch.utils.data import DataLoader

    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    loader = DataLoader(dataset, batch_size=1, collate_fn=make_collate_fn(include_specifications=True))
    latents_by_type = defaultdict(list)
    all_latents = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            _, mu, _ = seq_encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            latents_by_type[raw_data[idx]['filter_type']].append(mu[0])
            all_latents.append(mu[0])

    all_latents_t = torch.stack(all_latents)
    centroids = {ft: torch.stack(v).mean(0) for ft, v in latents_by_type.items()}

    # Compute latent space statistics
    lat_mean = all_latents_t.mean(0)
    lat_std = all_latents_t.std(0)
    lat_min = all_latents_t.min(0).values
    lat_max = all_latents_t.max(0).values

    print(f"\nLatent space statistics:")
    for d in range(8):
        print(f"  z[{d}]: mean={lat_mean[d]:.3f}, std={lat_std[d]:.3f}, "
              f"range=[{lat_min[d]:.3f}, {lat_max[d]:.3f}]")

    all_novel = Counter()
    all_known = Counter()
    all_invalid = 0
    total_samples = 0

    def process_results(results, label=""):
        nonlocal all_invalid, total_samples
        novel = Counter()
        known = Counter()
        invalid = 0
        for key, desc, tokens in results:
            total_samples += 1
            if key is None:
                invalid += 1
                all_invalid += 1
            elif key in training_keys:
                known[desc] += 1
                all_known[desc] += 1
            else:
                novel[desc] += 1
                all_novel[desc] += 1
        return novel, known, invalid

    # =================================================================
    # 1. DENSE RANDOM SAMPLING (10,000 at standard scale)
    # =================================================================
    print("\n" + "=" * 80)
    print("1. DENSE RANDOM SAMPLING (N=10,000, z ~ N(0,1))")
    print("=" * 80)

    torch.manual_seed(123)
    N = 10000
    batch_size = 64
    results_all = []
    for i in range(0, N, batch_size):
        bs = min(batch_size, N - i)
        z = torch.randn(bs, 8)
        results_all.extend(generate_batch(decoder, z, vocab))

    novel, known, invalid = process_results(results_all)
    print(f"  Valid: {N - invalid}/{N}, Novel: {sum(novel.values())}, Known: {sum(known.values())}")
    if novel:
        print(f"  Novel topologies:")
        for desc, count in novel.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 2. WIDER RANDOM SAMPLING (various sigma)
    # =================================================================
    print("\n" + "=" * 80)
    print("2. WIDER RANDOM SAMPLING (push latent scale)")
    print("=" * 80)

    for sigma in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        torch.manual_seed(456)
        results_s = []
        for i in range(0, 2000, batch_size):
            bs = min(batch_size, 2000 - i)
            z = torch.randn(bs, 8) * sigma
            results_s.extend(generate_batch(decoder, z, vocab))

        novel_s, known_s, invalid_s = process_results(results_s)
        n_novel = sum(novel_s.values())
        n_known = sum(known_s.values())
        print(f"  sigma={sigma:.1f}: valid={2000-invalid_s}/2000, "
              f"known={n_known}, novel={n_novel}, invalid={invalid_s}")
        if novel_s:
            for desc, count in novel_s.most_common(5):
                print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 3. PAIRWISE CENTROID INTERPOLATION (all pairs, dense alpha)
    # =================================================================
    print("\n" + "=" * 80)
    print("3. PAIRWISE CENTROID INTERPOLATION (all pairs, 21 alpha steps)")
    print("=" * 80)

    filter_types = sorted(centroids.keys())
    interp_novel = Counter()
    interp_total = 0

    for i, ft1 in enumerate(filter_types):
        for ft2 in filter_types[i+1:]:
            z1 = centroids[ft1]
            z2 = centroids[ft2]
            alphas = torch.linspace(0, 1, 21)
            z_batch = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
            results_interp = generate_batch(decoder, z_batch, vocab)
            interp_total += len(results_interp)
            for key, desc, tokens in results_interp:
                if key is not None and key not in training_keys:
                    interp_novel[desc] += 1
                    all_novel[desc] += 1
                    total_samples += 1

    print(f"  Total interpolation points: {interp_total}")
    print(f"  Novel topologies found: {sum(interp_novel.values())}")
    if interp_novel:
        for desc, count in interp_novel.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 4. CENTROID EXTRAPOLATION (push beyond each centroid)
    # =================================================================
    print("\n" + "=" * 80)
    print("4. CENTROID EXTRAPOLATION (scale centroids by 1.5x to 3x)")
    print("=" * 80)

    extrap_novel = Counter()
    extrap_total = 0

    for ft in filter_types:
        z_c = centroids[ft]
        direction = z_c - lat_mean  # direction from mean to centroid
        for scale in [1.5, 2.0, 2.5, 3.0]:
            z = lat_mean + direction * scale
            results_e = generate_batch(decoder, z.unsqueeze(0), vocab)
            extrap_total += 1
            for key, desc, tokens in results_e:
                if key is not None and key not in training_keys:
                    extrap_novel[desc] += 1
                    all_novel[desc] += 1
                    total_samples += 1

    print(f"  Total extrapolation points: {extrap_total}")
    print(f"  Novel topologies found: {sum(extrap_novel.values())}")
    if extrap_novel:
        for desc, count in extrap_novel.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 5. RANDOM WALKS BETWEEN PAIRS OF TRAINING LATENTS
    # =================================================================
    print("\n" + "=" * 80)
    print("5. RANDOM MIDPOINTS BETWEEN TRAINING LATENTS (5000 pairs)")
    print("=" * 80)

    torch.manual_seed(789)
    N_pairs = 5000
    idx1 = torch.randint(0, len(all_latents), (N_pairs,))
    idx2 = torch.randint(0, len(all_latents), (N_pairs,))
    alphas = torch.rand(N_pairs)

    midpoint_novel = Counter()
    for i in range(0, N_pairs, batch_size):
        bs = min(batch_size, N_pairs - i)
        z_batch = torch.stack([
            (1 - alphas[i+j]) * all_latents[idx1[i+j]] + alphas[i+j] * all_latents[idx2[i+j]]
            for j in range(bs)
        ])
        results_m = generate_batch(decoder, z_batch, vocab)
        for key, desc, tokens in results_m:
            total_samples += 1
            if key is None:
                all_invalid += 1
            elif key in training_keys:
                all_known[desc] += 1
            else:
                midpoint_novel[desc] += 1
                all_novel[desc] += 1

    print(f"  Novel topologies found: {sum(midpoint_novel.values())}")
    if midpoint_novel:
        for desc, count in midpoint_novel.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 6. GRID SEARCH IN TOP-2 PCA DIRECTIONS
    # =================================================================
    print("\n" + "=" * 80)
    print("6. GRID SEARCH IN TOP-2 PCA DIRECTIONS (41x41 grid)")
    print("=" * 80)

    # Simple PCA via SVD
    centered = all_latents_t - lat_mean.unsqueeze(0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]

    # Project training data to get range
    proj1 = (centered @ pc1).squeeze()
    proj2 = (centered @ pc2).squeeze()
    range1 = (proj1.min().item(), proj1.max().item())
    range2 = (proj2.min().item(), proj2.max().item())

    # Expand range by 50%
    margin1 = (range1[1] - range1[0]) * 0.25
    margin2 = (range2[1] - range2[0]) * 0.25
    grid1 = torch.linspace(range1[0] - margin1, range1[1] + margin1, 41)
    grid2 = torch.linspace(range2[0] - margin2, range2[1] + margin2, 41)

    pca_novel = Counter()
    pca_total = 0
    z_grid = []
    for g1 in grid1:
        for g2 in grid2:
            z = lat_mean + g1 * pc1 + g2 * pc2
            z_grid.append(z)

    z_grid_t = torch.stack(z_grid)
    for i in range(0, len(z_grid_t), batch_size):
        bs = min(batch_size, len(z_grid_t) - i)
        results_pca = generate_batch(decoder, z_grid_t[i:i+bs], vocab)
        pca_total += bs
        for key, desc, tokens in results_pca:
            total_samples += 1
            if key is None:
                all_invalid += 1
            elif key in training_keys:
                all_known[desc] += 1
            else:
                pca_novel[desc] += 1
                all_novel[desc] += 1

    print(f"  Grid points: {pca_total}")
    print(f"  Novel topologies found: {sum(pca_novel.values())}")
    if pca_novel:
        for desc, count in pca_novel.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =================================================================
    # 7. TEMPERATURE SAMPLING (non-greedy)
    # =================================================================
    print("\n" + "=" * 80)
    print("7. TEMPERATURE SAMPLING (non-greedy, T=0.5, 0.7, 1.0, 1.5)")
    print("=" * 80)

    for temp in [0.5, 0.7, 1.0, 1.5]:
        torch.manual_seed(999)
        temp_novel = Counter()
        temp_invalid = 0
        temp_known = 0
        N_temp = 2000
        for i in range(0, N_temp, batch_size):
            bs = min(batch_size, N_temp - i)
            z = torch.randn(bs, 8)
            with torch.no_grad():
                generated = decoder.generate(
                    z.float(), max_length=32, greedy=False,
                    temperature=temp, eos_id=vocab.eos_id,
                )
            for gen_ids in generated:
                tokens = vocab.decode(gen_ids)
                key = sequence_to_topology_key(tokens, vocab)
                desc = walk_to_description(tokens, vocab)
                total_samples += 1
                if key is None:
                    temp_invalid += 1
                    all_invalid += 1
                elif key in training_keys:
                    temp_known += 1
                    all_known[desc] += 1
                else:
                    temp_novel[desc] += 1
                    all_novel[desc] += 1

        n_novel = sum(temp_novel.values())
        print(f"  T={temp}: valid={N_temp-temp_invalid}/{N_temp}, "
              f"known={temp_known}, novel={n_novel}, invalid={temp_invalid}")
        if temp_novel:
            for desc, count in temp_novel.most_common(10):
                print(f"    {count:>4}x  {desc}")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — ALL NOVEL TOPOLOGIES FOUND")
    print("=" * 80)
    print(f"\nTotal samples probed: {total_samples}")
    print(f"Total invalid: {all_invalid}")
    print(f"Total known: {sum(all_known.values())}")
    print(f"Total novel: {sum(all_novel.values())}")
    print(f"Unique novel topologies: {len(all_novel)}")

    if all_novel:
        print(f"\nAll novel topologies (sorted by count):")
        for desc, count in all_novel.most_common():
            print(f"  {count:>5}x  {desc}")
    else:
        print("\nNo novel topologies found across all probing strategies.")

    print(f"\nKnown topology coverage:")
    for desc, count in all_known.most_common():
        print(f"  {count:>5}x  {desc}")


if __name__ == '__main__':
    main()
