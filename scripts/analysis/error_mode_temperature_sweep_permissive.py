"""
Permissive re-analysis of the error-mode sweep.

Same sampling protocol as error_mode_temperature_sweep.py, but relaxes the
"component must appear exactly twice" rule. A walk is re-parsed by collecting
the set of unique net-neighbors per component; if every component ends up
with exactly 2 distinct terminals, the walk describes a valid 2-terminal
circuit regardless of how many times the component token appeared in the
raw Euler sequence.

This separates two things that the strict parser conflates:

  - sequence-level Euler-traversal conformance (count == 2)
  - graph-level validity                      (|unique terminals| == 2,
                                               electrical checks pass)

Usage:
  .venv/bin/python scripts/analysis/error_mode_temperature_sweep_permissive.py \
    --fc 10000 --gain 0.5 --samples 2000 --seeds 0 1 2 \
    --out analysis_results/error_mode_sweep_permissive.json
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
from ml.data.graph_signature import _is_net_token, _comp_type
from ml.utils.runtime import load_v2_model
from scripts.analysis.fixed_spec_temperature_sweep import (
    _known_signatures, build_reference, knn_interpolate, optimise_mu, DATASETS,
)


def permissive_parse(walk):
    """Permissive walk parser.

    Accepts any alternating net-component-net sequence that starts and ends
    at VSS, with any component-occurrence count. Returns (comp_terminals,
    incident_count) if every component has exactly 2 distinct terminals,
    else None.

    comp_terminals: dict[comp] -> frozenset({net_a, net_b})
    incident_count: dict[net] -> int (how many components touch this net)
    """
    walk = list(walk)
    if len(walk) < 3 or len(walk) % 2 == 0:
        return None
    if walk[0] != 'VSS' or walk[-1] != 'VSS':
        return None
    for i in range(0, len(walk), 2):
        if not _is_net_token(walk[i]):
            return None
    for i in range(1, len(walk), 2):
        if _is_net_token(walk[i]):
            return None

    comp_nets: dict = defaultdict(set)
    for i in range(1, len(walk), 2):
        comp = walk[i]
        comp_nets[comp].add(walk[i - 1])
        comp_nets[comp].add(walk[i + 1])

    comp_terminals = {}
    for comp, nets in comp_nets.items():
        if len(nets) != 2:
            return None  # self-loop (1 net) or multi-terminal (3+)
        comp_terminals[comp] = frozenset(nets)

    incident: dict = defaultdict(int)
    for comp, nets in comp_terminals.items():
        for n in nets:
            incident[n] += 1

    return comp_terminals, incident


def permissive_signature(comp_terminals):
    """Canonical frozenset signature, invariant under INTERNAL relabeling
    and component-instance renaming within a type."""
    # Incident component types per net
    incident_types: dict = defaultdict(list)
    for comp, nets in comp_terminals.items():
        ct = _comp_type(comp)
        for n in nets:
            incident_types[n].append(ct)

    internals = [n for n in incident_types if n.startswith('INTERNAL_')]
    keyed = [(tuple(sorted(incident_types[n])), n) for n in internals]
    keyed.sort()
    relabel = {old: f'_INT{i}' for i, (_, old) in enumerate(keyed)}
    for n in incident_types:
        if n not in relabel:
            relabel[n] = n

    edges = Counter()
    for comp, nets in comp_terminals.items():
        ct = _comp_type(comp)
        a, b = tuple(nets)
        pair = frozenset({relabel[a], relabel[b]})
        edges[(ct, pair)] += 1
    return frozenset(edges.items())


def categorize_permissive(walk, known_sigs):
    walk = list(walk)
    # Sequence-level alternation still required
    if len(walk) < 3 or len(walk) % 2 == 0:
        return 'ill_formed_seq', None, None
    if walk[0] != 'VSS' or walk[-1] != 'VSS':
        return 'ill_formed_seq', None, None
    for i in range(0, len(walk), 2):
        if not _is_net_token(walk[i]):
            return 'ill_formed_seq', None, None
    for i in range(1, len(walk), 2):
        if _is_net_token(walk[i]):
            return 'ill_formed_seq', None, None

    # Track strict counting result for comparison
    comp_counts = Counter(walk[1::2])
    non_euler = any(c != 2 for c in comp_counts.values())

    # Permissive graph-level parse
    parsed = permissive_parse(walk)
    if parsed is None:
        # A component had 1 terminal (self-loop) or 3+ terminals
        # (multi-terminal, not supported by vocab)
        comp_nets: dict = defaultdict(set)
        for i in range(1, len(walk), 2):
            comp_nets[walk[i]].add(walk[i - 1])
            comp_nets[walk[i]].add(walk[i + 1])
        has_self_loop = any(len(n) == 1 for n in comp_nets.values())
        if has_self_loop:
            return 'invalid_self_loop', None, non_euler
        return 'invalid_multi_terminal', None, non_euler

    comp_terminals, incident = parsed

    for required in ('VSS', 'VIN', 'VOUT'):
        if incident.get(required, 0) < 1:
            return 'invalid_missing_terminal', None, non_euler
    for net, cnt in incident.items():
        if net.startswith('INTERNAL_') and cnt < 2:
            return 'invalid_dangling', None, non_euler

    sig = permissive_signature(comp_terminals)
    cat = 'valid_known' if sig in known_sigs else 'valid_novel'
    return cat, sig, non_euler


ALL_CATEGORIES = [
    'valid_known', 'valid_novel',
    'invalid_self_loop', 'invalid_multi_terminal',
    'invalid_dangling', 'invalid_missing_terminal',
    'ill_formed_seq',
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
    non_euler_valid = 0           # valid walks whose token counts ≠ 2
    non_euler_valid_novel = 0
    novel_sigs = Counter()
    for ids in gen:
        toks = tuple(t for t in vocab.decode(ids)
                     if t not in ('BOS', 'EOS', 'PAD'))
        cat, sig, non_euler = categorize_permissive(toks, known_sigs)
        categories[cat] += 1
        if cat in ('valid_known', 'valid_novel') and non_euler:
            non_euler_valid += 1
            if cat == 'valid_novel':
                non_euler_valid_novel += 1
        if cat == 'valid_novel':
            novel_sigs[sig] += 1

    return categories, novel_sigs, non_euler_valid, non_euler_valid_novel


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
    ap.add_argument('--out',
                    default='analysis_results/error_mode_sweep_permissive.json')
    args = ap.parse_args()

    device = 'cpu'
    encoder, decoder, vocab, heads, _ = load_v2_model(args.ckpt, device)
    ds = CrossTopoSequenceDataset(
        DATASETS, set(FILTER_TYPES_V2), vocab,
        augment=False, max_seq_len=32, edge_feature_mode='polynomial')

    # Permissive "known" signatures: use same permissive parser on training
    known_sigs = set()
    from ml.data.bipartite_graph import from_pickle_circuit
    from ml.data.traversal import hierholzer
    import pickle
    for pkl in DATASETS:
        with open(pkl, 'rb') as f:
            circuits = pickle.load(f)
        for c in circuits:
            bg = from_pickle_circuit(c)
            walk = hierholzer(bg, start='VSS', rng=None)
            parsed = permissive_parse(walk)
            if parsed is not None:
                sig = permissive_signature(parsed[0])
                known_sigs.add(sig)

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
        f"{'self':>5s} {'mulT':>5s} {'dang':>5s} {'term':>5s}  "
        f"{'seq':>5s}  "
        f"{'ne_v':>5s} {'ne_vn':>5s}"
    )
    print(header)
    print('-' * len(header))

    for temp in args.temperatures:
        per_seed_cats = []
        per_seed_ne_valid = []
        per_seed_ne_valid_novel = []
        for seed in args.seeds:
            cats, novel, ne_v, ne_vn = decode_and_categorize(
                decoder, vocab, mu_opt, known_sigs,
                args.samples, temp, seed, device)
            per_seed_cats.append(cats)
            per_seed_ne_valid.append(ne_v)
            per_seed_ne_valid_novel.append(ne_vn)
            for sig, cnt in novel.items():
                all_novel_sigs[sig] += cnt

        totals = Counter()
        for cats in per_seed_cats:
            totals.update(cats)
        n_total = args.samples * len(args.seeds)
        ne_valid_total = sum(per_seed_ne_valid)
        ne_valid_novel_total = sum(per_seed_ne_valid_novel)

        print(
            f"{temp:5.2f}  "
            f"{totals['valid_known']:>7d} {totals['valid_novel']:>6d}  "
            f"{totals['invalid_self_loop']:>5d} "
            f"{totals['invalid_multi_terminal']:>5d} "
            f"{totals['invalid_dangling']:>5d} "
            f"{totals['invalid_missing_terminal']:>5d}  "
            f"{totals['ill_formed_seq']:>5d}  "
            f"{ne_valid_total:>5d} {ne_valid_novel_total:>5d}"
        )

        results.append({
            'temperature': temp,
            'total_samples': n_total,
            'counts': dict(totals),
            'non_euler_valid': ne_valid_total,
            'non_euler_valid_novel': ne_valid_novel_total,
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
        'note': ('permissive parser: component-count != 2 is not a rejection; '
                 'ne_v and ne_vn count how many valid (resp. valid-novel) '
                 'walks had at least one component with count != 2'),
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
