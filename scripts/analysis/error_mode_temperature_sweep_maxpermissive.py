"""Max-permissive re-analysis: extend the permissive parser to also
accept walks where a component claims 3+ distinct terminals.

Relative to `error_mode_temperature_sweep_permissive.py`, two additions:

1. **Even-length salvage**: trim trailing component token when len(walk)
   is even (EOS cutoff artifact), then re-check grammar.

2. **Multi-edge interpretation of 3+ terminals.** The permissive parser
   rejected `|comp_nets[comp]| >= 3` as `invalid_multi_terminal`, on the
   grounds that Z-GED's vocabulary is 2-terminal. This parser instead
   interprets each distinct adjacency pair `(net_a, net_b)` (where
   net_a != net_b) as a separate 2-terminal edge that happens to share
   a label with sibling edges — the charitable reading is that the
   decoder made two resistors and gave them the same instance ID. In
   particular the |union| in {1, 2} branches are **unchanged** from the
   permissive parser, so we preserve its generous self-loop repair.

Concretely the classification becomes:

  |union of neighbors of comp|:
    1   → self-loop (reject, same as permissive)
    2   → one 2-terminal edge (same as permissive, even if all raw
          adjacencies were self-loop visits — union-based recovery)
    >=3 → multiple 2-terminal edges, one per distinct adjacency pair
          (NEW; was invalid_multi_terminal under permissive)

Graph-level checks and the category names are otherwise unchanged.
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
    build_reference, knn_interpolate, optimise_mu, DATASETS,
)


def _grammar_ok(walk):
    if len(walk) < 3 or len(walk) % 2 == 0:
        return False
    if walk[0] != 'VSS' or walk[-1] != 'VSS':
        return False
    for i in range(0, len(walk), 2):
        if not _is_net_token(walk[i]):
            return False
    for i in range(1, len(walk), 2):
        if _is_net_token(walk[i]):
            return False
    return True


def max_permissive_parse(walk):
    """Return ({comp: [pair, ...]}, incident) or None."""
    walk = list(walk)
    trimmed = False
    if len(walk) >= 4 and len(walk) % 2 == 0:
        last = walk[-1]
        if not _is_net_token(last):
            walk = walk[:-1]
            trimmed = True

    if not _grammar_ok(walk):
        return None, trimmed

    # union of all adjacent nets per comp, plus set of distinct pairs
    comp_nets = defaultdict(set)
    comp_pairs = defaultdict(set)
    for i in range(1, len(walk), 2):
        comp = walk[i]
        a, b = walk[i - 1], walk[i + 1]
        comp_nets[comp].add(a)
        comp_nets[comp].add(b)
        if a != b:
            comp_pairs[comp].add(frozenset({a, b}))

    comp_edges: dict = {}  # comp -> list of frozenset pairs
    for comp, nets in comp_nets.items():
        if len(nets) == 1:
            return None, trimmed  # pure self-loop
        elif len(nets) == 2:
            comp_edges[comp] = [frozenset(nets)]
        else:  # |nets| >= 3
            pairs = list(comp_pairs[comp])
            if not pairs:
                return None, trimmed  # pathological
            comp_edges[comp] = pairs

    incident = defaultdict(int)
    for comp, pairs in comp_edges.items():
        for pair in pairs:
            for n in pair:
                incident[n] += 1

    return (comp_edges, incident), trimmed


def max_permissive_signature(comp_edges):
    """Canonical signature: multiset of (comp_type, pair-with-relabeled-internals)."""
    incident_types = defaultdict(list)
    for comp, pairs in comp_edges.items():
        ct = _comp_type(comp)
        for pair in pairs:
            for n in pair:
                incident_types[n].append(ct)
    internals = [n for n in incident_types if n.startswith('INTERNAL_')]
    keyed = [(tuple(sorted(incident_types[n])), n) for n in internals]
    keyed.sort()
    relabel = {old: f'_INT{i}' for i, (_, old) in enumerate(keyed)}
    for n in incident_types:
        if n not in relabel:
            relabel[n] = n

    bag = Counter()
    for comp, pairs in comp_edges.items():
        ct = _comp_type(comp)
        for pair in pairs:
            a, b = tuple(pair)
            bag[(ct, frozenset({relabel[a], relabel[b]}))] += 1
    return frozenset(bag.items())


def categorize_max(walk, known_sigs):
    walk = list(walk)
    # salvage
    trimmed = False
    if len(walk) >= 4 and len(walk) % 2 == 0:
        last = walk[-1]
        if not _is_net_token(last):
            walk = walk[:-1]
            trimmed = True

    if not _grammar_ok(walk):
        return 'ill_formed_seq', None, trimmed

    parsed, _ = max_permissive_parse(walk)
    if parsed is None:
        return 'invalid_self_loop', None, trimmed

    comp_edges, incident = parsed

    for required in ('VSS', 'VIN', 'VOUT'):
        if incident.get(required, 0) < 1:
            return 'invalid_missing_terminal', None, trimmed
    for net, cnt in incident.items():
        if net.startswith('INTERNAL_') and cnt < 2:
            return 'invalid_dangling', None, trimmed

    sig = max_permissive_signature(comp_edges)
    return ('valid_known' if sig in known_sigs else 'valid_novel'), sig, trimmed


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
                    default='analysis_results/error_mode_sweep_maxpermissive.json')
    args = ap.parse_args()

    device = 'cpu'
    encoder, decoder, vocab, heads, _ = load_v2_model(args.ckpt, device)
    ds = CrossTopoSequenceDataset(
        DATASETS, set(FILTER_TYPES_V2), vocab,
        augment=False, max_seq_len=32, edge_feature_mode='polynomial')

    # Build known sigs from training walks
    from ml.data.bipartite_graph import from_pickle_circuit
    from ml.data.traversal import hierholzer
    import pickle
    known_sigs = set()
    for pkl in DATASETS:
        with open(pkl, 'rb') as f:
            circuits = pickle.load(f)
        for c in circuits:
            bg = from_pickle_circuit(c)
            walk = hierholzer(bg, start='VSS', rng=None)
            parsed, _ = max_permissive_parse(walk)
            if parsed is not None:
                known_sigs.add(max_permissive_signature(parsed[0]))

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
    print(f"Known topology count (max-perm): {len(known_sigs)}\n")

    header = (f"{'T':>5s}  {'known':>7s} {'novel':>6s}  "
              f"{'self':>5s} {'dang':>5s} {'term':>5s}  {'seq':>6s}  "
              f"{'trim':>5s}")
    print(header)
    print('-' * len(header))

    results = []
    all_novel = Counter()

    for temp in args.temperatures:
        totals = Counter()
        trim_count = 0
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            latents = mu_opt.to(device).unsqueeze(0).expand(args.samples, -1)
            gen = decoder.generate(
                latents, max_length=32, temperature=temp,
                greedy=False, eos_id=vocab.eos_id)
            for ids in gen:
                toks = tuple(t for t in vocab.decode(ids)
                             if t not in ('BOS', 'EOS', 'PAD'))
                cat, sig, trimmed = categorize_max(toks, known_sigs)
                totals[cat] += 1
                if trimmed:
                    trim_count += 1
                if cat == 'valid_novel':
                    all_novel[sig] += 1

        print(f"{temp:5.2f}  "
              f"{totals['valid_known']:>7d} {totals['valid_novel']:>6d}  "
              f"{totals['invalid_self_loop']:>5d} "
              f"{totals['invalid_dangling']:>5d} "
              f"{totals['invalid_missing_terminal']:>5d}  "
              f"{totals['ill_formed_seq']:>6d}  "
              f"{trim_count:>5d}")

        results.append({
            'temperature': temp,
            'total_samples': args.samples * len(args.seeds),
            'counts': dict(totals),
            'trimmed_even_length': trim_count,
        })

    print(f"\nTotal max-permissive novel unique: {len(all_novel)}, "
          f"samples: {sum(all_novel.values())}")

    out = {
        'target': {'fc_hz': args.fc, 'gain': args.gain, 'type': args.type},
        'checkpoint': args.ckpt,
        'known_topology_count': len(known_sigs),
        'note': ('max-permissive: permissive + multi-edge interpretation of '
                 '|union|>=3 + even-length salvage via trailing-comp trim'),
        'results': results,
        'novel_topologies': [
            {'signature': [list(fs) for fs in sig], 'count': cnt}
            for sig, cnt in all_novel.most_common()
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
