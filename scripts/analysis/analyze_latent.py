"""
Deep analysis of the v2 5D structured latent space.

Sections:
  1. Per-type centroid statistics & inter/intra-class separation
  2. Dimension-by-dimension analysis (what each of the 5 dims encodes)
  3. Attribute prediction quality on full dataset
  4. Pairwise type distances & nearest-neighbour confusion
  5. Interpolation smoothness (walk validity along centroid-to-centroid paths)
  6. Centroid-conditioned generation quality
  7. Perturbation robustness (how far can you push from centroid?)

Run:  .venv/bin/python scripts/analysis/analyze_latent.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import random
from collections import defaultdict, Counter
import itertools

import numpy as np
import torch
from torch_geometric.data import Batch

from ml.models.vocabulary import CircuitVocabulary
from ml.models.decoder import SequenceDecoder
from ml.data.graph_signature import (
    well_formed, is_electrically_valid, walk_topology_signature,
)
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.attribute_heads import FreqHead, GainHead, TypeHead
from ml.models.constants import G_REF, C_REF, L_INV_REF, TYPE_TO_IDX
from ml.data.cross_topo_dataset import CrossTopoSequenceDataset


ALL_TYPES = sorted([
    'low_pass', 'high_pass',
    'rl_lowpass', 'rl_highpass',
    'lc_lowpass', 'cl_highpass',
    'band_pass', 'band_stop',
    'rlc_series', 'rlc_parallel',
])


def load_everything(ckpt_path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    vocab = CircuitVocabulary(max_internal=10, max_components=10)
    latent_dim = ckpt.get('latent_dim', 5)

    encoder = AdmittanceEncoder(
        node_feature_dim=4, hidden_dim=64, latent_dim=latent_dim,
        num_layers=3, dropout=0.0, vae=True,
    ).to(device)
    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size, latent_dim=latent_dim,
        d_model=128, n_heads=4, n_layers=2, max_seq_len=33,
        dropout=0.0, pad_id=vocab.pad_id,
    ).to(device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    encoder.eval(); decoder.eval()

    # Load attribute heads
    freq_head = FreqHead(latent_dim).to(device)
    gain_head = GainHead(latent_dim).to(device)
    type_head = TypeHead(latent_dim, len(TYPE_TO_IDX)).to(device)
    freq_head.load_state_dict(ckpt['freq_head_state_dict'])
    gain_head.load_state_dict(ckpt['gain_head_state_dict'])
    type_head.load_state_dict(ckpt['type_head_state_dict'])
    freq_head.eval(); gain_head.eval(); type_head.eval()

    return encoder, decoder, vocab, freq_head, gain_head, type_head, latent_dim


@torch.no_grad()
def encode_dataset(encoder, ds, device):
    """Encode all circuits, return per-type mu, logvar, and metadata."""
    by_type = defaultdict(lambda: {'mu': [], 'logvar': [], 'idx': [],
                                    'fc': [], 'gain': [], 'walk_sig': []})
    all_mu, all_logvar, all_types = [], [], []
    all_fc, all_gain = [], []

    for i in range(len(ds)):
        circ = ds.circuits[i]
        ft = circ['filter_type']
        g = ds.pyg_graphs[i].to(device)
        batch_idx = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
        z, mu, logvar = encoder(g.x, g.edge_index, g.edge_attr, batch_idx)
        mu_cpu = mu[0].cpu()
        lv_cpu = logvar[0].cpu()

        # Ground-truth behavior targets (same as training)
        fc = circ['characteristic_frequency']
        gain_1k = CrossTopoSequenceDataset._compute_gain_at_freq(circ, 1000.0)

        by_type[ft]['mu'].append(mu_cpu)
        by_type[ft]['logvar'].append(lv_cpu)
        by_type[ft]['idx'].append(i)
        by_type[ft]['fc'].append(fc)
        by_type[ft]['gain'].append(gain_1k)

        sig = walk_topology_signature(ds.all_walks[i][0])
        by_type[ft]['walk_sig'].append(sig)

        all_mu.append(mu_cpu)
        all_logvar.append(lv_cpu)
        all_types.append(ft)
        all_fc.append(np.log10(max(fc, 1.0)))  # same as training target
        all_gain.append(gain_1k)

    # Stack
    for ft in by_type:
        by_type[ft]['mu'] = torch.stack(by_type[ft]['mu'])
        by_type[ft]['logvar'] = torch.stack(by_type[ft]['logvar'])
        by_type[ft]['fc'] = np.array(by_type[ft]['fc'])
        by_type[ft]['gain'] = np.array(by_type[ft]['gain'])

    all_mu = torch.stack(all_mu)
    all_logvar = torch.stack(all_logvar)
    all_fc = np.array(all_fc)
    all_gain = np.array(all_gain)
    return by_type, all_mu, all_logvar, all_types, all_fc, all_gain


@torch.no_grad()
def sample_walks(decoder, vocab, latent, n_samples, device, temperature=1.0):
    """Generate walks from a latent vector."""
    latent = latent.to(device)
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)
    latents = latent.expand(n_samples, -1)
    gen = decoder.generate(latents, max_length=32, temperature=temperature,
                           greedy=False, eos_id=vocab.eos_id)
    walks = []
    for ids in gen:
        toks = tuple(t for t in vocab.decode(ids) if t not in ('BOS', 'EOS', 'PAD'))
        walks.append(toks)
    return walks


def walk_stats(walks):
    """Return (n_wellformed, n_valid, sig_counts, sig_walks)."""
    wf = [w for w in walks if well_formed(w)]
    elec = [w for w in wf if is_electrically_valid(w)]
    sig_counts = Counter()
    sig_walks = {}
    for w in elec:
        sig = walk_topology_signature(w)
        if sig:
            sig_counts[sig] += 1
            sig_walks.setdefault(sig, w)
    return len(wf), len(elec), sig_counts, sig_walks


# ─────────────────────────────────────────────────────────────────
# Section 1: Centroid statistics
# ─────────────────────────────────────────────────────────────────
def section_centroids(by_type, latent_dim):
    print("\n" + "=" * 80)
    print("SECTION 1: PER-TYPE CENTROID STATISTICS")
    print("=" * 80)
    DIM_NAMES = ['z_topo₁', 'z_topo₂', 'z_VIN', 'z_VOUT', 'z_GND']

    centroids = {}
    print(f"\n{'Type':>14s}  {'N':>4s}  ", end='')
    for d in range(latent_dim):
        print(f"  {DIM_NAMES[d]:>8s}", end='')
    print(f"  {'intra-σ':>7s}")
    print("-" * (22 + 10 * latent_dim + 10))

    for ft in ALL_TYPES:
        data = by_type[ft]
        mu = data['mu']
        centroid = mu.mean(0)
        centroids[ft] = centroid
        intra_std = (mu - centroid).norm(dim=1).mean().item()
        print(f"{ft:>14s}  {len(mu):4d}  ", end='')
        for d in range(latent_dim):
            print(f"  {centroid[d].item():+8.3f}", end='')
        print(f"  {intra_std:7.3f}")

    # Intra/inter class separation
    print("\n  Intra-class avg L2 distance from centroid:")
    intra_dists = {}
    for ft in ALL_TYPES:
        mu = by_type[ft]['mu']
        c = centroids[ft]
        d = (mu - c).norm(dim=1).mean().item()
        intra_dists[ft] = d
        print(f"    {ft:>14s}: {d:.4f}")

    print(f"\n  Overall intra-class mean: {np.mean(list(intra_dists.values())):.4f}")

    print("\n  Inter-class centroid distances:")
    inter_dists = []
    pairs = list(itertools.combinations(ALL_TYPES, 2))
    for ta, tb in pairs:
        d = (centroids[ta] - centroids[tb]).norm().item()
        inter_dists.append(d)
    print(f"    min: {min(inter_dists):.4f}  max: {max(inter_dists):.4f}  "
          f"mean: {np.mean(inter_dists):.4f}")

    sep_ratio = np.mean(inter_dists) / np.mean(list(intra_dists.values()))
    print(f"\n  Separation ratio (inter/intra): {sep_ratio:.1f}×")

    return centroids


# ─────────────────────────────────────────────────────────────────
# Section 2: Dimension-by-dimension analysis
# ─────────────────────────────────────────────────────────────────
def section_dimensions(by_type, latent_dim):
    print("\n" + "=" * 80)
    print("SECTION 2: DIMENSION-BY-DIMENSION ANALYSIS")
    print("=" * 80)
    DIM_NAMES = ['z_topo₁', 'z_topo₂', 'z_VIN', 'z_VOUT', 'z_GND']
    DIM_DESC = [
        'Topology (global terminal concat)',
        'Topology (global terminal concat)',
        'VIN local admittance view',
        'VOUT local admittance view',
        'GND local admittance view',
    ]

    for d in range(latent_dim):
        print(f"\n  --- Dimension {d}: {DIM_NAMES[d]} ({DIM_DESC[d]}) ---")
        vals = []
        for ft in ALL_TYPES:
            mu_d = by_type[ft]['mu'][:, d]
            mean = mu_d.mean().item()
            std = mu_d.std().item()
            vals.append((ft, mean, std))

        # Sort by mean value to show the axis ordering
        vals.sort(key=lambda x: x[1])
        print(f"  {'Type':>14s}  {'mean':>7s}  {'std':>6s}  {'range':>12s}")
        for ft, m, s in vals:
            lo = m - 2*s
            hi = m + 2*s
            print(f"  {ft:>14s}  {m:+7.3f}  {s:6.3f}  [{lo:+6.2f}, {hi:+6.2f}]")

        # Check discrimination: how well does this dim alone separate types?
        all_vals = []
        all_labels = []
        for ft in ALL_TYPES:
            mu_d = by_type[ft]['mu'][:, d].numpy()
            all_vals.extend(mu_d)
            all_labels.extend([ft] * len(mu_d))
        all_vals = np.array(all_vals)
        all_labels = np.array(all_labels)

        # 1-D linear discriminant score: between-class var / within-class var
        overall_mean = all_vals.mean()
        between_var = sum(
            len(by_type[ft]['mu']) * (by_type[ft]['mu'][:, d].mean().item() - overall_mean) ** 2
            for ft in ALL_TYPES
        ) / len(all_vals)
        within_var = sum(
            by_type[ft]['mu'][:, d].var().item() * len(by_type[ft]['mu'])
            for ft in ALL_TYPES
        ) / len(all_vals)
        fisher = between_var / (within_var + 1e-9)
        print(f"  Fisher discriminant ratio: {fisher:.2f}")

def section_dim_correlations(by_type, all_mu, all_fc_gt, all_gain_gt, latent_dim):
    """Correlations between each latent dimension and circuit attributes."""
    print(f"\n  --- Dimension-attribute correlations ---")
    DIM_NAMES_short = ['topo₁', 'topo₂', 'VIN', 'VOUT', 'GND']
    print(f"  {'Dim':>8s}  {'corr(log fc)':>12s}  {'corr(gain)':>10s}")
    for d in range(latent_dim):
        z_d = all_mu[:, d].numpy()
        corr_fc = np.corrcoef(z_d, all_fc_gt)[0, 1]
        corr_gain = np.corrcoef(z_d, all_gain_gt)[0, 1]
        print(f"  {DIM_NAMES_short[d]:>8s}  {corr_fc:+12.3f}  {corr_gain:+10.3f}")


# ─────────────────────────────────────────────────────────────────
# Section 3: Attribute prediction quality
# ─────────────────────────────────────────────────────────────────
def section_attributes(by_type, all_mu, all_types, all_fc_gt, all_gain_gt,
                       freq_head, gain_head, type_head, device):
    print("\n" + "=" * 80)
    print("SECTION 3: ATTRIBUTE PREDICTION FROM μ")
    print("=" * 80)

    mu_dev = all_mu.to(device)
    with torch.no_grad():
        freq_pred = freq_head(mu_dev).cpu().numpy().flatten()
        gain_pred = gain_head(mu_dev).cpu().numpy().flatten()
        type_logits = type_head(mu_dev).cpu()
        type_pred = type_logits.argmax(dim=1).numpy()

    # Gather ground truth
    idx_to_type = {v: k for k, v in TYPE_TO_IDX.items()}
    type_gt = np.array([TYPE_TO_IDX[ft] for ft in all_types])
    type_correct = (type_pred == type_gt).sum()
    type_acc = type_correct / len(type_gt)

    print(f"\n  Type classification accuracy: {type_correct}/{len(type_gt)} = {type_acc:.1%}")

    # Per-type accuracy
    print(f"\n  {'Type':>14s}  {'correct':>7s}  {'total':>5s}  {'acc':>6s}  {'predicted as':>30s}")
    for ft in ALL_TYPES:
        mask = np.array(all_types) == ft
        gt_idx = TYPE_TO_IDX[ft]
        pred_sub = type_pred[mask]
        correct = (pred_sub == gt_idx).sum()
        total = mask.sum()
        wrong_mask = pred_sub != gt_idx
        if wrong_mask.any():
            confused = Counter(idx_to_type[p] for p in pred_sub[wrong_mask])
            conf_str = ', '.join(f'{k}({v})' for k, v in confused.most_common(3))
        else:
            conf_str = '--'
        print(f"  {ft:>14s}  {correct:7d}  {total:5d}  {correct/total:6.1%}  {conf_str:>30s}")

    # Frequency prediction (target = log10(characteristic_frequency))
    freq_mse = np.mean((freq_pred - all_fc_gt) ** 2)
    freq_corr = np.corrcoef(freq_pred, all_fc_gt)[0, 1]
    print(f"\n  Frequency prediction (log10 Hz):")
    print(f"    MSE: {freq_mse:.4f}  |  correlation: {freq_corr:.4f}")
    print(f"    GT range: [{all_fc_gt.min():.2f}, {all_fc_gt.max():.2f}]")
    print(f"    Pred range: [{freq_pred.min():.2f}, {freq_pred.max():.2f}]")

    # Per-type freq error
    print(f"\n    {'Type':>14s}  {'mean_gt':>8s}  {'mean_pred':>9s}  {'MSE':>7s}  {'corr':>6s}")
    offset = 0
    for ft in ALL_TYPES:
        n = len(by_type[ft]['fc'])
        gt_sub = all_fc_gt[offset:offset+n]
        pred_sub = freq_pred[offset:offset+n]
        mse = np.mean((pred_sub - gt_sub) ** 2)
        c = np.corrcoef(pred_sub, gt_sub)[0, 1] if np.std(gt_sub) > 1e-6 else float('nan')
        print(f"    {ft:>14s}  {gt_sub.mean():8.3f}  {pred_sub.mean():9.3f}  "
              f"{mse:7.4f}  {c:6.3f}")
        offset += n

    # Gain prediction (target = |H(1kHz)|)
    gain_mse = np.mean((gain_pred - all_gain_gt) ** 2)
    gain_corr = np.corrcoef(gain_pred, all_gain_gt)[0, 1] if np.std(all_gain_gt) > 0 else 0
    print(f"\n  Gain prediction (|H(1kHz)|):")
    print(f"    MSE: {gain_mse:.6f}  |  correlation: {gain_corr:.4f}")
    print(f"    GT range: [{all_gain_gt.min():.3f}, {all_gain_gt.max():.3f}]")
    print(f"    Pred range: [{gain_pred.min():.3f}, {gain_pred.max():.3f}]")

    # Per-type gain error
    print(f"\n    {'Type':>14s}  {'mean_gt':>8s}  {'mean_pred':>9s}  {'MSE':>8s}  {'corr':>6s}")
    offset = 0
    for ft in ALL_TYPES:
        n = len(by_type[ft]['gain'])
        gt_sub = all_gain_gt[offset:offset+n]
        pred_sub = gain_pred[offset:offset+n]
        mse = np.mean((pred_sub - gt_sub) ** 2)
        c = np.corrcoef(pred_sub, gt_sub)[0, 1] if np.std(gt_sub) > 1e-6 else float('nan')
        print(f"    {ft:>14s}  {gt_sub.mean():8.4f}  {pred_sub.mean():9.4f}  "
              f"{mse:8.6f}  {c:6.3f}")
        offset += n


# ─────────────────────────────────────────────────────────────────
# Section 4: Pairwise type distances & confusion
# ─────────────────────────────────────────────────────────────────
def section_pairwise(by_type, centroids):
    print("\n" + "=" * 80)
    print("SECTION 4: PAIRWISE TYPE DISTANCES & NEAREST-NEIGHBOUR CONFUSION")
    print("=" * 80)

    # Distance matrix
    types = ALL_TYPES
    n = len(types)
    dist_mat = np.zeros((n, n))
    for i, ta in enumerate(types):
        for j, tb in enumerate(types):
            dist_mat[i, j] = (centroids[ta] - centroids[tb]).norm().item()

    print(f"\n  Centroid L2 distance matrix:")
    print(f"  {'':>14s}", end='')
    for ft in types:
        print(f"  {ft[:6]:>6s}", end='')
    print()
    for i, ta in enumerate(types):
        print(f"  {ta:>14s}", end='')
        for j in range(n):
            d = dist_mat[i, j]
            print(f"  {d:6.2f}", end='')
        print()

    # Nearest neighbour for each type
    print(f"\n  Nearest / farthest neighbour per type:")
    for i, ft in enumerate(types):
        dists = [(dist_mat[i, j], types[j]) for j in range(n) if i != j]
        dists.sort()
        nearest = dists[0]
        farthest = dists[-1]
        print(f"    {ft:>14s}  nearest={nearest[1]:>14s} ({nearest[0]:.3f})  "
              f"farthest={farthest[1]:>14s} ({farthest[0]:.3f})")

    # Sample-level 1-NN classification accuracy
    print(f"\n  Sample-level 1-NN classification (using centroids):")
    correct = 0
    total = 0
    per_type_correct = defaultdict(int)
    per_type_total = defaultdict(int)
    for ft in types:
        for mu_i in by_type[ft]['mu']:
            dists = {t: (mu_i - centroids[t]).norm().item() for t in types}
            pred = min(dists, key=dists.get)
            per_type_total[ft] += 1
            total += 1
            if pred == ft:
                correct += 1
                per_type_correct[ft] += 1
    print(f"    Overall: {correct}/{total} = {correct/total:.1%}")
    for ft in types:
        c = per_type_correct[ft]
        t = per_type_total[ft]
        print(f"      {ft:>14s}: {c}/{t} = {c/t:.1%}")

    return dist_mat


# ─────────────────────────────────────────────────────────────────
# Section 5: Logvar analysis — posterior uncertainty
# ─────────────────────────────────────────────────────────────────
def section_logvar(by_type, latent_dim):
    print("\n" + "=" * 80)
    print("SECTION 5: POSTERIOR UNCERTAINTY (logvar → std)")
    print("=" * 80)
    DIM_NAMES = ['topo₁', 'topo₂', 'VIN', 'VOUT', 'GND']

    print(f"\n  Mean std (exp(0.5*logvar)) per type per dimension:")
    print(f"  {'Type':>14s}", end='')
    for d in range(latent_dim):
        print(f"  {DIM_NAMES[d]:>7s}", end='')
    print(f"  {'avg':>7s}")
    print("  " + "-" * (16 + 9 * (latent_dim + 1)))

    for ft in ALL_TYPES:
        lv = by_type[ft]['logvar']
        stds = torch.exp(0.5 * lv).mean(0)
        avg_std = stds.mean().item()
        print(f"  {ft:>14s}", end='')
        for d in range(latent_dim):
            print(f"  {stds[d].item():7.3f}", end='')
        print(f"  {avg_std:7.3f}")

    # Which dimensions are most/least certain?
    all_lv = torch.cat([by_type[ft]['logvar'] for ft in ALL_TYPES])
    global_std = torch.exp(0.5 * all_lv).mean(0)
    print(f"\n  Global mean std per dimension:")
    for d in range(latent_dim):
        print(f"    {DIM_NAMES[d]:>7s}: {global_std[d].item():.4f}")
    print(f"  Most certain dim:  {DIM_NAMES[global_std.argmin().item()]} "
          f"(std={global_std.min().item():.4f})")
    print(f"  Least certain dim: {DIM_NAMES[global_std.argmax().item()]} "
          f"(std={global_std.max().item():.4f})")


# ─────────────────────────────────────────────────────────────────
# Section 6: Interpolation smoothness
# ─────────────────────────────────────────────────────────────────
def section_interpolation(centroids, decoder, vocab, known_sigs, sig_to_type,
                          device, n_samples=500, temperature=1.0):
    print("\n" + "=" * 80)
    print("SECTION 6: INTERPOLATION SMOOTHNESS")
    print("=" * 80)
    print(f"  (n_samples={n_samples}, temperature={temperature})")

    # Select interesting pairs
    pairs = [
        ('low_pass', 'high_pass'),
        ('band_pass', 'band_stop'),
        ('rl_lowpass', 'rl_highpass'),
        ('lc_lowpass', 'cl_highpass'),
        ('band_pass', 'rlc_parallel'),
        ('low_pass', 'band_pass'),
        ('high_pass', 'rlc_series'),
    ]

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ta, tb in pairs:
        print(f"\n  --- {ta} → {tb} ---")
        print(f"  {'alpha':>6s}  {'valid%':>6s}  {'#topo':>5s}  {'novel':>5s}  "
              f"{'dominant topology':>40s}  {'count':>5s}")
        ca, cb = centroids[ta], centroids[tb]
        for alpha in alphas:
            z = (1 - alpha) * ca + alpha * cb
            walks = sample_walks(decoder, vocab, z, n_samples, device,
                                 temperature=temperature)
            n_wf, n_elec, sig_counts, sig_walks = walk_stats(walks)
            novel = sum(1 for s in sig_counts if s not in known_sigs)
            if sig_counts:
                top_sig, top_cnt = sig_counts.most_common(1)[0]
                top_label = sig_to_type.get(top_sig, 'NOVEL')
            else:
                top_label, top_cnt = '--', 0
            valid_pct = 100 * n_elec / len(walks) if walks else 0
            print(f"  {alpha:6.2f}  {valid_pct:5.1f}%  {len(sig_counts):5d}  "
                  f"{novel:5d}  {top_label:>40s}  {top_cnt:5d}")


# ─────────────────────────────────────────────────────────────────
# Section 7: Centroid-conditioned generation
# ─────────────────────────────────────────────────────────────────
def section_centroid_generation(centroids, decoder, vocab, known_sigs,
                                sig_to_type, device, n_samples=1000,
                                temperature=1.0):
    print("\n" + "=" * 80)
    print("SECTION 7: CENTROID-CONDITIONED GENERATION")
    print("=" * 80)
    print(f"  (n_samples={n_samples}, temperature={temperature})")

    print(f"\n  {'Type':>14s}  {'valid%':>6s}  {'#topo':>5s}  {'self%':>5s}  "
          f"{'novel':>5s}  {'top walk (if not self)':>50s}")
    for ft in ALL_TYPES:
        z = centroids[ft]
        walks = sample_walks(decoder, vocab, z, n_samples, device,
                             temperature=temperature)
        n_wf, n_elec, sig_counts, sig_walks = walk_stats(walks)
        valid_pct = 100 * n_elec / len(walks) if walks else 0
        novel = sum(1 for s in sig_counts if s not in known_sigs)

        # Find what fraction match self-type
        self_sigs = set()
        for s, lab in sig_to_type.items():
            if lab == ft:
                self_sigs.add(s)
        self_count = sum(sig_counts[s] for s in self_sigs if s in sig_counts)
        self_pct = 100 * self_count / n_elec if n_elec > 0 else 0

        # Second-most-common that isn't self
        other_label = '--'
        for sig, cnt in sig_counts.most_common(5):
            lab = sig_to_type.get(sig, 'NOVEL')
            if lab != ft:
                other_label = f"{lab} (×{cnt})"
                break

        print(f"  {ft:>14s}  {valid_pct:5.1f}%  {len(sig_counts):5d}  "
              f"{self_pct:4.1f}%  {novel:5d}  {other_label:>50s}")


# ─────────────────────────────────────────────────────────────────
# Section 8: Perturbation robustness
# ─────────────────────────────────────────────────────────────────
def section_perturbation(centroids, decoder, vocab, known_sigs, sig_to_type,
                         device, n_samples=500, temperature=1.0):
    print("\n" + "=" * 80)
    print("SECTION 8: PERTURBATION ROBUSTNESS (centroid + noise)")
    print("=" * 80)
    print("  How far can you push from centroid before self-type recovery drops?")

    noise_scales = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    print(f"\n  {'Type':>14s}", end='')
    for ns in noise_scales:
        print(f"  {'σ='+str(ns):>7s}", end='')
    print("    (self-type recovery %)")

    for ft in ALL_TYPES:
        c = centroids[ft]
        self_sigs = set(s for s, lab in sig_to_type.items() if lab == ft)
        print(f"  {ft:>14s}", end='')
        for ns in noise_scales:
            torch.manual_seed(42)
            z = c + ns * torch.randn_like(c) if ns > 0 else c
            walks = sample_walks(decoder, vocab, z, n_samples, device,
                                 temperature=temperature)
            _, n_elec, sig_counts, _ = walk_stats(walks)
            self_count = sum(sig_counts[s] for s in self_sigs if s in sig_counts)
            pct = 100 * self_count / n_elec if n_elec > 0 else 0
            print(f"  {pct:6.1f}%", end='')
        print()


# ─────────────────────────────────────────────────────────────────
# Section 9: Topology signature census
# ─────────────────────────────────────────────────────────────────
def section_topology_census(by_type):
    print("\n" + "=" * 80)
    print("SECTION 9: TOPOLOGY SIGNATURE CENSUS (training data)")
    print("=" * 80)
    print("  How many distinct topology signatures per type?")

    sig_to_type = {}
    for ft in ALL_TYPES:
        sigs = set(by_type[ft]['walk_sig'])
        sigs.discard(None)
        for s in sigs:
            sig_to_type.setdefault(s, ft)
        print(f"    {ft:>14s}: {len(sigs)} unique signatures")

    all_sigs = set()
    for ft in ALL_TYPES:
        all_sigs.update(s for s in by_type[ft]['walk_sig'] if s is not None)
    print(f"    {'TOTAL':>14s}: {len(all_sigs)} unique signatures across all types")

    return sig_to_type, all_sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/production/best_v2.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gen-samples', type=int, default=1000,
                    help='samples per generation experiment')
    ap.add_argument('--temperature', type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("DEEP ANALYSIS: v2 5D Structured Latent Space")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  temperature: {args.temperature}")
    print(f"  gen-samples: {args.gen_samples}")
    print("=" * 80)

    (encoder, decoder, vocab,
     freq_head, gain_head, type_head, latent_dim) = load_everything(args.ckpt, device)
    print(f"  Loaded encoder ({latent_dim}D VAE) + decoder + attribute heads")

    ds = CrossTopoSequenceDataset(
        ['rlc_dataset/filter_dataset.pkl', 'rlc_dataset/rl_dataset.pkl'],
        set(ALL_TYPES), vocab, augment=False, max_seq_len=32,
        edge_feature_mode='polynomial')
    print(f"  Dataset: {len(ds)} circuits")

    by_type, all_mu, all_logvar, all_types, all_fc_gt, all_gain_gt = \
        encode_dataset(encoder, ds, device)

    # Section 1
    centroids = section_centroids(by_type, latent_dim)

    # Section 2
    section_dimensions(by_type, latent_dim)
    section_dim_correlations(by_type, all_mu, all_fc_gt, all_gain_gt, latent_dim)

    # Section 3
    section_attributes(by_type, all_mu, all_types, all_fc_gt, all_gain_gt,
                       freq_head, gain_head, type_head, device)

    # Section 4
    dist_mat = section_pairwise(by_type, centroids)

    # Section 5
    section_logvar(by_type, latent_dim)

    # Section 9 (topology census — needed by later sections)
    sig_to_type, known_sigs = section_topology_census(by_type)

    # Section 6
    section_interpolation(centroids, decoder, vocab, known_sigs, sig_to_type,
                          device, n_samples=args.gen_samples,
                          temperature=args.temperature)

    # Section 7
    section_centroid_generation(centroids, decoder, vocab, known_sigs,
                                sig_to_type, device, n_samples=args.gen_samples,
                                temperature=args.temperature)

    # Section 8
    section_perturbation(centroids, decoder, vocab, known_sigs, sig_to_type,
                         device, n_samples=args.gen_samples // 2,
                         temperature=args.temperature)


if __name__ == '__main__':
    main()
