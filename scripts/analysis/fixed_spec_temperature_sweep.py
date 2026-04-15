"""
Run a V2 fixed-spec decoder temperature sweep.

This script computes one optimized latent code from a fixed cutoff/gain
target, then reuses that exact latent for every temperature. That isolates
temperature as the only changed generation parameter.

Example:
  .venv/bin/python scripts/analysis/fixed_spec_temperature_sweep.py \
    --fc 10000 --gain 0.5 --samples 20 \
    --temperatures 0.1 0.3 0.7 1.0 1.3 1.7 \
    --out analysis_results/fixed_spec_temperature_sweep_10khz_gain0p5.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.data.cross_topo_dataset import CrossTopoSequenceDataset
from ml.data.graph_signature import (
    is_electrically_valid,
    walk_topology_signature,
    well_formed,
)
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.utils.runtime import load_v2_model
from scripts.generation.generate_inverse_design import (
    IDX_TO_TYPE,
    _format_netlist,
    build_reference,
    knn_interpolate,
    optimise_mu,
)


DATASETS = ['rlc_dataset/filter_dataset.pkl', 'rlc_dataset/rl_dataset.pkl']


def _signature_to_string(sig: frozenset | None) -> str | None:
    """Convert canonical signature to a compact stable string for reports."""
    if sig is None:
        return None
    parts = []
    for (ctype, nets), count in sorted(sig, key=lambda item: (item[0][0], sorted(item[0][1]))):
        net_text = '-'.join(sorted(nets))
        suffix = f"x{count}" if count > 1 else ""
        parts.append(f"{ctype}({net_text}){suffix}")
    return "|".join(parts)


def _clean_walk(ids: Iterable[int], vocab) -> tuple[str, ...]:
    return tuple(t for t in vocab.decode(ids) if t not in ('BOS', 'EOS', 'PAD'))


def _walk_report(walk: tuple[str, ...], known_sigs: set[frozenset]) -> dict:
    is_wf = well_formed(walk)
    is_valid = is_wf and is_electrically_valid(walk)
    sig = walk_topology_signature(walk) if is_valid else None
    comp_summary, connections = _format_netlist(walk) if is_valid else (None, None)
    return {
        'walk': list(walk),
        'well_formed': is_wf,
        'valid': is_valid,
        'known_topology': bool(sig in known_sigs) if sig is not None else False,
        'signature': _signature_to_string(sig),
        'component_summary': comp_summary,
        'netlist': connections,
    }


def _known_signatures(ds: CrossTopoSequenceDataset) -> set[frozenset]:
    known = set()
    for walks in ds.all_walks:
        sig = walk_topology_signature(walks[0])
        if sig is not None:
            known.add(sig)
    return known


@torch.no_grad()
def _latent_attributes(mu, heads):
    freq_log10 = heads['freq'](mu.unsqueeze(0)).item()
    gain = heads['gain'](mu.unsqueeze(0)).item()
    type_probs = F.softmax(heads['type'](mu.unsqueeze(0)), dim=-1)[0]
    top_idx = int(type_probs.argmax().item())
    return {
        'predicted_fc_hz': float(10 ** freq_log10),
        'predicted_log10_fc': float(freq_log10),
        'predicted_gain_1khz': float(gain),
        'predicted_type': IDX_TO_TYPE[top_idx],
        'predicted_type_probability': float(type_probs[top_idx].item()),
        'type_probabilities': {
            IDX_TO_TYPE[i]: float(type_probs[i].item())
            for i in range(len(type_probs))
        },
    }


def _sample_temperature(decoder, vocab, mu, known_sigs, samples, temperature, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    latents = mu.to(device).unsqueeze(0).expand(samples, -1)
    generated = decoder.generate(
        latents,
        max_length=32,
        temperature=temperature,
        greedy=False,
        eos_id=vocab.eos_id,
    )
    reports = [_walk_report(_clean_walk(ids, vocab), known_sigs) for ids in generated]
    valid = [r for r in reports if r['valid']]
    novel = [r for r in valid if not r['known_topology']]

    sig_counts = Counter(r['signature'] for r in valid if r['signature'])
    examples_by_sig = {}
    for r in valid:
        if r['signature'] and r['signature'] not in examples_by_sig:
            examples_by_sig[r['signature']] = r

    topologies = []
    for sig, count in sig_counts.most_common(5):
        ex = examples_by_sig[sig]
        topologies.append({
            'signature': sig,
            'count': count,
            'known_topology': ex['known_topology'],
            'component_summary': ex['component_summary'],
            'netlist': ex['netlist'],
        })

    return {
        'temperature': temperature,
        'samples': samples,
        'valid': len(valid),
        'invalid': samples - len(valid),
        'valid_novel': len(novel),
        'novel_fraction_of_valid': float(len(novel) / len(valid)) if valid else 0.0,
        'unique_valid_topologies': len(sig_counts),
        'topologies': topologies,
        'examples': reports[: min(8, len(reports))],
    }


def main():
    ap = argparse.ArgumentParser(description='V2 fixed-spec temperature sweep.')
    ap.add_argument('--fc', type=float, required=True, help='Target cutoff frequency in Hz')
    ap.add_argument('--gain', type=float, required=True, help='Target |H(1kHz)| gain')
    ap.add_argument('--type', choices=FILTER_TYPES_V2, default=None,
                    help='Optional target filter type. Omit to target cutoff/gain only.')
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--temperatures', type=float, nargs='+',
                    default=[0.1, 0.3, 0.7, 1.0, 1.3, 1.7])
    ap.add_argument('--ckpt', default='checkpoints/production/best_v2.pt')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--opt-steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder, decoder, vocab, heads, _ = load_v2_model(args.ckpt, device)
    ds = CrossTopoSequenceDataset(
        DATASETS,
        set(FILTER_TYPES_V2),
        vocab,
        augment=False,
        max_seq_len=32,
        edge_feature_mode='polynomial',
    )
    known_sigs = _known_signatures(ds)

    ref_mus, ref_log_fcs, ref_gains, ref_types = build_reference(encoder, ds, device)

    target_log_fc = float(np.log10(args.fc))
    target_type_idx = TYPE_TO_IDX[args.type] if args.type else None
    mu_init, topk_idx = knn_interpolate(
        ref_mus,
        ref_log_fcs,
        ref_gains,
        ref_types,
        target_log_fc,
        args.gain,
        args.type,
        k=args.k,
    )
    mu_opt = optimise_mu(
        mu_init,
        heads['freq'],
        heads['gain'],
        heads['type'],
        target_log_fc,
        args.gain,
        target_type_idx,
        device,
        n_steps=args.opt_steps,
    )

    nearest = []
    for i in topk_idx:
        nearest.append({
            'filter_type': str(ref_types[i]),
            'fc_hz': float(10 ** ref_log_fcs[i]),
            'gain_1khz': float(ref_gains[i]),
        })

    result = {
        'checkpoint': args.ckpt,
        'datasets': DATASETS,
        'target': {
            'fc_hz': args.fc,
            'log10_fc': target_log_fc,
            'gain_1khz': args.gain,
            'filter_type': args.type,
        },
        'method': {
            'description': 'One optimized V2 latent is reused for every row; only decoder temperature changes.',
            'k': args.k,
            'opt_steps': args.opt_steps,
            'seed': args.seed,
            'samples_per_temperature': args.samples,
        },
        'latent': {
            'mu_init': [float(v) for v in mu_init.cpu().numpy()],
            'mu_opt': [float(v) for v in mu_opt.cpu().numpy()],
            'delta_l2': float((mu_opt.cpu() - mu_init).norm().item()),
            'attributes': _latent_attributes(mu_opt.to(device), heads),
            'nearest_training_points': nearest,
        },
        'known_topology_count': len(known_sigs),
        'results': [
            _sample_temperature(
                decoder,
                vocab,
                mu_opt,
                known_sigs,
                args.samples,
                temp,
                args.seed,
                device,
            )
            for temp in args.temperatures
        ],
    }

    text = json.dumps(result, indent=2)
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text)
            f.write('\n')
    print(text)


if __name__ == '__main__':
    main()
