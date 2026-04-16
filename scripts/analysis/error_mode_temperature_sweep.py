"""
Fine-grained temperature sweep with error-mode breakdown.

For each sample at each temperature, classify as:
  (a) valid + known     - passes all checks, matches a training topology
  (b) valid + novel     - passes all checks, new topology
  (c) invalid_self_loop - well-formed but has a component with same terminals
  (d) invalid_dangling  - well-formed but has internal net incident to < 2 comps
  (e) invalid_missing_terminal - well-formed but VIN/VOUT/VSS not connected
  (f) ill_formed_seq    - wrong alternation, odd length, or doesn't start/end at VSS
  (g) ill_formed_comp_count - some component appears != 2 times

This breakdown shows *how* the decoder fails as temperature increases,
not just the aggregate validity rate.

Usage:
  .venv/bin/python scripts/analysis/error_mode_temperature_sweep.py \
    --fc 10000 --gain 0.5 --samples 2000 --seeds 0 1 2 \
    --temperatures 0.1 0.3 0.5 0.7 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.7 2.0 \
    --out analysis_results/error_mode_sweep.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.data.cross_topo_dataset import CrossTopoSequenceDataset
from ml.data.graph_signature import (
    well_formed, walk_to_graph, walk_topology_signature,
)
from ml.utils.runtime import load_v2_model
from scripts.analysis.fixed_spec_temperature_sweep import (
    _known_signatures, build_reference, knn_interpolate, optimise_mu, DATASETS,
)


def categorize_walk(walk, known_sigs):
    """Return one of: valid_known, valid_novel, invalid_self_loop,
    invalid_dangling, invalid_missing_terminal, ill_formed_seq,
    ill_formed_comp_count.
    """
    walk = list(walk)

    # Structural well-formedness
    if len(walk) < 3 or len(walk) % 2 == 0:
        return 'ill_formed_seq'
    if walk[0] != 'VSS' or walk[-1] != 'VSS':
        return 'ill_formed_seq'
    # Token alternation
    from ml.data.graph_signature import _is_net_token
    for i in range(0, len(walk), 2):
        if not _is_net_token(walk[i]):
            return 'ill_formed_seq'
    for i in range(1, len(walk), 2):
        if _is_net_token(walk[i]):
            return 'ill_formed_seq'
    # Component count
    comp_counts = Counter(walk[1::2])
    if any(c != 2 for c in comp_counts.values()):
        return 'ill_formed_comp_count'

    # It's well-formed; now electrical checks
    g = walk_to_graph(walk)
    if g is None:
        return 'ill_formed_seq'

    # Self-loop check
    for comp, (a, b) in g.comp_terminals.items():
        if a == b:
            return 'invalid_self_loop'

    # Terminal connectivity
    incident = defaultdict(int)
    for comp, (a, b) in g.comp_terminals.items():
        incident[a] += 1
        incident[b] += 1
    for required in ('VSS', 'VIN', 'VOUT'):
        if incident.get(required, 0) < 1:
            return 'invalid_missing_terminal'

    # Dangling internals
    for net in g.net_nodes:
        if net.startswith('INTERNAL_') and incident[net] < 2:
            return 'invalid_dangling'

    # Electrically valid — classify novelty
    sig = walk_topology_signature(tuple(walk))
    if sig is None:
        return 'ill_formed_seq'
    return 'valid_known' if sig in known_sigs else 'valid_novel'


ALL_CATEGORIES = [
    'valid_known', 'valid_novel',
    'invalid_self_loop', 'invalid_dangling', 'invalid_missing_terminal',
    'ill_formed_seq', 'ill_formed_comp_count',
]


def decode_and_categorize(decoder, vocab, mu, known_sigs, n_samples,
                          temperature, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    latents = mu.to(device).unsqueeze(0).expand(n_samples, -1)
    gen = decoder.generate(
        latents, max_length=32, temperature=temperature,
        greedy=False, eos_id=vocab.eos_id)

    categories = Counter()
    novel_sigs = Counter()
    for ids in gen:
        toks = tuple(t for t in vocab.decode(ids)
                     if t not in ('BOS', 'EOS', 'PAD'))
        cat = categorize_walk(toks, known_sigs)
        categories[cat] += 1
        if cat == 'valid_novel':
            novel_sigs[walk_topology_signature(toks)] += 1

    return categories, novel_sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fc', type=float, default=10000.0)
    ap.add_argument('--gain', type=float, default=0.5)
    ap.add_argument('--type', default=None, choices=FILTER_TYPES_V2)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--temperatures', type=float, nargs='+',
                    default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2,
                             1.3, 1.4, 1.5, 1.7, 2.0])
    ap.add_argument('--ckpt', default='checkpoints/production/best_v2.pt')
    ap.add_argument('--out', default='analysis_results/error_mode_sweep.json')
    args = ap.parse_args()

    device = 'cpu'
    encoder, decoder, vocab, heads, _ = load_v2_model(args.ckpt, device)
    ds = CrossTopoSequenceDataset(
        DATASETS, set(FILTER_TYPES_V2), vocab,
        augment=False, max_seq_len=32, edge_feature_mode='polynomial')
    known_sigs = _known_signatures(ds)

    ref_mus, ref_log_fcs, ref_gains, ref_types = build_reference(
        encoder, ds, device)

    target_log_fc = float(np.log10(args.fc))
    target_type_idx = TYPE_TO_IDX[args.type] if args.type else None
    mu_init, _ = knn_interpolate(
        ref_mus, ref_log_fcs, ref_gains, ref_types,
        target_log_fc, args.gain, args.type, k=10)
    mu_opt = optimise_mu(
        mu_init, heads['freq'], heads['gain'], heads['type'],
        target_log_fc, args.gain, target_type_idx, device, n_steps=200)

    print(f"Target: fc={args.fc}Hz, gain={args.gain}, type={args.type}")
    print(f"Samples per (T, seed): {args.samples}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Seeds: {args.seeds}")
    print(f"Known topology count: {len(known_sigs)}\n")

    results = []
    all_novel_sigs = Counter()

    header = (
        f"{'T':>5s}  "
        f"{'known':>7s} {'novel':>6s}  "
        f"{'self':>5s} {'dang':>5s} {'term':>5s}  "
        f"{'seq':>5s} {'ccnt':>5s}"
    )
    print(header)
    print('-' * len(header))

    for temp in args.temperatures:
        per_seed_cats = []
        total_novel = 0
        for seed in args.seeds:
            cats, novel = decode_and_categorize(
                decoder, vocab, mu_opt, known_sigs,
                args.samples, temp, seed, device)
            per_seed_cats.append(cats)
            total_novel += sum(novel.values())
            for sig, cnt in novel.items():
                all_novel_sigs[sig] += cnt

        # Aggregate over seeds
        totals = Counter()
        for cats in per_seed_cats:
            totals.update(cats)
        n_total = args.samples * len(args.seeds)

        print(
            f"{temp:5.2f}  "
            f"{totals['valid_known']:>7d} {totals['valid_novel']:>6d}  "
            f"{totals['invalid_self_loop']:>5d} "
            f"{totals['invalid_dangling']:>5d} "
            f"{totals['invalid_missing_terminal']:>5d}  "
            f"{totals['ill_formed_seq']:>5d} "
            f"{totals['ill_formed_comp_count']:>5d}"
        )

        results.append({
            'temperature': temp,
            'total_samples': n_total,
            'counts': dict(totals),
            'per_seed_counts': [dict(c) for c in per_seed_cats],
        })

    print(f"\nTotal novel unique: {len(all_novel_sigs)}, "
          f"total novel samples: {sum(all_novel_sigs.values())}")

    out = {
        'target': {'fc_hz': args.fc, 'gain': args.gain, 'type': args.type},
        'checkpoint': args.ckpt,
        'samples_per_config': args.samples,
        'seeds': args.seeds,
        'known_topology_count': len(known_sigs),
        'categories': ALL_CATEGORIES,
        'results': results,
        'novel_topologies': [
            {'signature': [list(fs) for fs in sig], 'count': cnt}
            for sig, cnt in all_novel_sigs.most_common()
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
