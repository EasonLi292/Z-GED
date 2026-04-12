"""
Generate circuits from target specifications using the v2 admittance encoder.

Usage:
  # Generate a band_pass filter at 10 kHz
  .venv/bin/python scripts/generation/generate_inverse_design.py --type band_pass --fc 10000

  # Generate with gain target too
  .venv/bin/python scripts/generation/generate_inverse_design.py --type band_pass --fc 10000 --gain 0.5

  # More samples, higher temperature
  .venv/bin/python scripts/generation/generate_inverse_design.py --type low_pass --fc 1000 --samples 20 --temperature 1.5

  # Frequency only (no type constraint)
  .venv/bin/python scripts/generation/generate_inverse_design.py --fc 50000

Pipeline:
  1. Encode all training circuits -> (mu, log10_fc, gain, type)
  2. K-NN interpolation: find K nearest by target attributes, average mu
  3. Gradient descent on mu to minimize attribute head losses
  4. Decode walks from optimised mu
  5. Validate and report
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from ml.models.vocabulary import CircuitVocabulary
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.data.cross_topo_dataset import CrossTopoSequenceDataset
from ml.data.graph_signature import (
    well_formed, is_electrically_valid, walk_topology_signature, walk_to_graph,
)
from ml.utils.runtime import load_v2_model


IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}


@torch.no_grad()
def build_reference(encoder, ds, device):
    """Encode training set. Returns (mus, log_fcs, gains, types)."""
    mus, log_fcs, gains, types = [], [], [], []
    for i in range(len(ds)):
        circ = ds.circuits[i]
        g = ds.pyg_graphs[i].to(device)
        batch_idx = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
        _, mu, _ = encoder(g.x, g.edge_index, g.edge_attr, batch_idx)
        mus.append(mu[0].cpu())
        fc = circ['characteristic_frequency']
        log_fcs.append(np.log10(max(fc, 1.0)))
        gains.append(CrossTopoSequenceDataset._compute_gain_at_freq(circ, 1000.0))
        types.append(circ['filter_type'])
    return (torch.stack(mus), np.array(log_fcs),
            np.array(gains), np.array(types))


def knn_interpolate(ref_mus, ref_log_fcs, ref_gains, ref_types,
                    target_log_fc, target_gain, target_type, k=10):
    """Find K nearest training circuits by attribute distance, return mean mu."""
    n = len(ref_mus)
    scores = np.zeros(n)

    # Frequency distance (normalised by dataset range)
    fc_range = ref_log_fcs.max() - ref_log_fcs.min() + 1e-9
    if target_log_fc is not None:
        scores += ((ref_log_fcs - target_log_fc) / fc_range) ** 2

    # Gain distance
    gain_range = ref_gains.max() - ref_gains.min() + 1e-9
    if target_gain is not None:
        scores += ((ref_gains - target_gain) / gain_range) ** 2

    # Type match bonus (large penalty for wrong type)
    if target_type is not None:
        type_mismatch = (ref_types != target_type).astype(float)
        scores += 10.0 * type_mismatch

    topk_idx = np.argsort(scores)[:k]
    # Distance-weighted average (inverse distance weighting)
    topk_scores = scores[topk_idx]
    weights = 1.0 / (topk_scores + 1e-9)
    weights /= weights.sum()
    mu_init = sum(w * ref_mus[i] for i, w in zip(topk_idx, weights))
    return mu_init, topk_idx


def optimise_mu(mu_init, freq_head, gain_head, type_head,
                target_log_fc, target_gain, target_type_idx,
                device, n_steps=200, lr=0.05,
                w_freq=1.0, w_gain=1.0, w_type=1.0, w_prior=0.01):
    """Gradient descent on mu to hit target attributes.

    Also includes a mild N(0,1) prior penalty to keep mu in-distribution.
    """
    mu = mu_init.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([mu], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = 0.0

        if target_log_fc is not None:
            freq_pred = freq_head(mu.unsqueeze(0)).squeeze()
            loss = loss + w_freq * (freq_pred - target_log_fc) ** 2

        if target_gain is not None:
            gain_pred = gain_head(mu.unsqueeze(0)).squeeze()
            loss = loss + w_gain * (gain_pred - target_gain) ** 2

        if target_type_idx is not None:
            type_logits = type_head(mu.unsqueeze(0))
            loss = loss + w_type * F.cross_entropy(
                type_logits, torch.tensor([target_type_idx], device=device))

        # Mild prior: keep mu close to N(0,1)
        loss = loss + w_prior * mu.pow(2).sum()

        loss.backward()
        optimizer.step()

    return mu.detach()


def _format_freq(hz):
    """Human-readable frequency string."""
    if hz >= 1e6:
        return f"{hz/1e6:.1f} MHz"
    if hz >= 1e3:
        return f"{hz/1e3:.1f} kHz"
    return f"{hz:.1f} Hz"


def _format_netlist(walk):
    """Convert walk to human-readable netlist: 'R1(VIN--VOUT), C1(VOUT--VSS)'."""
    g = walk_to_graph(walk)
    if g is None:
        return None, None
    connections = []
    for comp in sorted(g.comp_nodes):
        a, b = g.comp_terminals[comp]
        ct = g.comp_types[comp]
        connections.append(f"{comp}({a}--{b})")
    comp_types = Counter(g.comp_types[c] for c in g.comp_nodes)
    comp_summary = '+'.join(f"{n}{t}" for t, n in sorted(comp_types.items()))
    return comp_summary, connections


def _format_schematic(walk):
    """Build an ASCII block diagram from the walk's graph."""
    g = walk_to_graph(walk)
    if g is None:
        return []
    lines = []
    for comp in sorted(g.comp_nodes):
        a, b = g.comp_terminals[comp]
        ct = g.comp_types[comp]
        lines.append(f"    {a:>12s} ---[{ct} {comp}]--- {b}")
    return lines


@torch.no_grad()
def decode_and_report(decoder, vocab, mu, n_samples, device, temperature,
                      freq_head, gain_head, type_head):
    """Generate walks, validate, and print results."""
    mu_dev = mu.to(device)

    # Predict attributes at the optimised mu
    freq_pred = freq_head(mu_dev.unsqueeze(0)).item()
    gain_pred = gain_head(mu_dev.unsqueeze(0)).item()
    type_logits = type_head(mu_dev.unsqueeze(0))
    type_probs = F.softmax(type_logits, dim=-1)[0]
    top_type_idx = type_probs.argmax().item()
    top_type_prob = type_probs[top_type_idx].item()

    print(f"\n  Predicted attributes at optimised mu:")
    print(f"    frequency : {_format_freq(10**freq_pred)} (10^{freq_pred:.2f} Hz)")
    print(f"    gain      : |H(1kHz)| = {gain_pred:.3f}")
    print(f"    type      : {IDX_TO_TYPE[top_type_idx]} ({top_type_prob:.1%})")
    print(f"    mu        : [{', '.join(f'{v:+.3f}' for v in mu_dev.cpu().numpy())}]")

    # Decode
    latents = mu_dev.unsqueeze(0).expand(n_samples, -1)
    gen = decoder.generate(latents, max_length=32, temperature=temperature,
                           greedy=False, eos_id=vocab.eos_id)
    walks = []
    for ids in gen:
        toks = tuple(t for t in vocab.decode(ids)
                     if t not in ('BOS', 'EOS', 'PAD'))
        walks.append(toks)

    wf = [w for w in walks if well_formed(w)]
    elec = [w for w in wf if is_electrically_valid(w)]

    print(f"\n  Generated {n_samples} samples: "
          f"{len(wf)} well-formed, {len(elec)} valid "
          f"({100*len(elec)/n_samples:.0f}%)")

    if not elec:
        print("  No valid circuits generated.")
        return

    # Group by topology
    sig_counts = Counter()
    sig_walks = {}
    for w in elec:
        sig = walk_topology_signature(w)
        if sig:
            sig_counts[sig] += 1
            sig_walks.setdefault(sig, w)

    print(f"  {len(sig_counts)} unique topology(ies)\n")

    for rank, (sig, cnt) in enumerate(sig_counts.most_common(10), 1):
        walk = sig_walks[sig]
        comp_summary, connections = _format_netlist(walk)
        pct = 100 * cnt / len(elec)
        if comp_summary is None:
            continue

        print(f"  Topology #{rank}  [{comp_summary}]  "
              f"({cnt}/{len(elec)} = {pct:.0f}%)")
        print(f"  {'':>4s}Netlist: {', '.join(connections)}")
        schematic = _format_schematic(walk)
        if schematic:
            print(f"  {'':>4s}Schematic:")
            for line in schematic:
                print(f"  {line}")
        print()


def main():
    ap = argparse.ArgumentParser(
        description='Generate circuits from target specifications (v2 model).')
    ap.add_argument('--type', choices=FILTER_TYPES_V2, default=None,
                    help='Target filter type')
    ap.add_argument('--fc', type=float, default=None,
                    help='Target cutoff frequency in Hz')
    ap.add_argument('--gain', type=float, default=None,
                    help='Target |H(1kHz)| gain')
    ap.add_argument('--samples', type=int, default=10,
                    help='Number of circuits to generate')
    ap.add_argument('--temperature', type=float, default=0.1,
                    help='Decoder sampling temperature (default: 0.1, near-greedy)')
    ap.add_argument('--ckpt', default='checkpoints/production/best_v2.pt')
    ap.add_argument('--k', type=int, default=10,
                    help='K for K-NN interpolation')
    ap.add_argument('--opt-steps', type=int, default=200,
                    help='Gradient descent steps')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.fc is None and args.type is None and args.gain is None:
        ap.error('Specify at least one of --type, --fc, --gain')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load
    encoder, decoder, vocab, heads, _ = load_v2_model(args.ckpt, device)
    freq_head = heads['freq']
    gain_head = heads['gain']
    type_head = heads['type']

    ds = CrossTopoSequenceDataset(
        ['rlc_dataset/filter_dataset.pkl', 'rlc_dataset/rl_dataset.pkl'],
        set(FILTER_TYPES_V2), vocab,
        augment=False, max_seq_len=32, edge_feature_mode='polynomial')

    ref_mus, ref_log_fcs, ref_gains, ref_types = build_reference(
        encoder, ds, device)

    # Targets
    target_log_fc = np.log10(args.fc) if args.fc is not None else None
    target_gain = args.gain
    target_type = args.type
    target_type_idx = TYPE_TO_IDX[args.type] if args.type else None

    print("=" * 70)
    print("CIRCUIT GENERATION")
    print("=" * 70)
    specs = []
    if target_type:
        specs.append(f"type = {target_type}")
    if args.fc is not None:
        specs.append(f"fc = {args.fc:.0f} Hz (log10 = {target_log_fc:.2f})")
    if target_gain is not None:
        specs.append(f"|H(1kHz)| = {target_gain:.3f}")
    print(f"  Target: {', '.join(specs)}")
    print(f"  Samples: {args.samples}, temperature: {args.temperature}")

    # Step 1: K-NN interpolation
    mu_init, topk_idx = knn_interpolate(
        ref_mus, ref_log_fcs, ref_gains, ref_types,
        target_log_fc, target_gain, target_type, k=args.k)

    print(f"\n  K-NN init (k={args.k}):")
    print(f"    mu_init = [{', '.join(f'{v:+.3f}' for v in mu_init.numpy())}]")
    nn_types = Counter(ref_types[topk_idx])
    print(f"    neighbours: {dict(nn_types)}")
    print(f"    neighbour fc range: [{10**ref_log_fcs[topk_idx].min():.0f}, "
          f"{10**ref_log_fcs[topk_idx].max():.0f}] Hz")

    # Step 2: Gradient descent
    has_target = (target_log_fc is not None or target_gain is not None
                  or target_type_idx is not None)
    if has_target and args.opt_steps > 0:
        print(f"\n  Optimising mu ({args.opt_steps} steps)...")
        mu_opt = optimise_mu(
            mu_init, freq_head, gain_head, type_head,
            target_log_fc, target_gain, target_type_idx,
            device, n_steps=args.opt_steps)
        delta = (mu_opt.cpu() - mu_init).norm().item()
        print(f"    ||delta_mu|| = {delta:.4f}")
    else:
        mu_opt = mu_init.to(device)

    # Step 3: Decode and report
    decode_and_report(decoder, vocab, mu_opt, args.samples, device,
                      args.temperature, freq_head, gain_head, type_head)


if __name__ == '__main__':
    main()
