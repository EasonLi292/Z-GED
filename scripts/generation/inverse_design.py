"""Physics-conditioned inverse design harness.

Given a target behavioural signature `h*` (latent vector), sample walks from the
trained decoder, validate them, group by topology signature, and report
which topologies were proposed.

Target sources:
  1. Held-out training-circuit `h`s -- does the decoder rediscover the
     source topology from behaviour alone?
  2. Pairwise centroid blends `(h_a + h_b)/2` -- in-distribution blends
     of behaviour, hopefully forcing the decoder to mix templates.
  3. Centroid + small Gaussian perturbation -- robustness probe.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from ml.models.vocabulary import CircuitVocabulary
from ml.models.decoder import SequenceDecoder
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.constants import FILTER_TYPES_V2, G_REF, C_REF, L_INV_REF
from ml.data.cross_topo_dataset import CrossTopoSequenceDataset
from ml.data.graph_signature import (
    well_formed, is_electrically_valid,
    walk_to_graph, walk_topology_signature,
)
from ml.utils.runtime import load_v2_model


ALL_TYPES = set(FILTER_TYPES_V2)

_UNIT_EDGE = {
    'R':   [1.0, 0.0, 0.0],
    'C':   [0.0, 1.0, 0.0],
    'L':   [0.0, 0.0, 1.0],
    'RC':  [1.0, 1.0, 0.0],
    'RL':  [1.0, 0.0, 1.0],
    'CL':  [0.0, 1.0, 1.0],
    'RCL': [1.0, 1.0, 1.0],
}


def walk_to_pyg_graph(walk):
    """Build a PyG `Data` graph from a generated walk for re-encoding."""
    g = walk_to_graph(walk)
    if g is None:
        return None
    nets = g.net_nodes
    net_idx = {n: i for i, n in enumerate(nets)}

    x = []
    for n in nets:
        if n == 'VSS':
            x.append([1.0, 0.0, 0.0, 0.0])
        elif n == 'VIN':
            x.append([0.0, 1.0, 0.0, 0.0])
        elif n == 'VOUT':
            x.append([0.0, 0.0, 1.0, 0.0])
        else:
            x.append([0.0, 0.0, 0.0, 1.0])

    edge_idx, edge_attr = [], []
    for comp, (a, b) in g.comp_terminals.items():
        ct = g.comp_types[comp]
        if ct not in _UNIT_EDGE:
            return None
        feat = _UNIT_EDGE[ct]
        ai, bi = net_idx[a], net_idx[b]
        edge_idx.append([ai, bi]); edge_attr.append(feat)
        edge_idx.append([bi, ai]); edge_attr.append(feat)

    if not edge_idx:
        return None
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_idx, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )


def _get_mu(encoder_output):
    """Extract mu from encoder output (handles both VAE and deterministic)."""
    if isinstance(encoder_output, tuple):
        return encoder_output[1]
    return encoder_output


@torch.no_grad()
def encode_walks(encoder, walks, device, batch_size=64):
    """Re-encode walks at unit values. Returns (h, valid_mask)."""
    graphs, mask = [], []
    for w in walks:
        g = walk_to_pyg_graph(w)
        if g is None:
            mask.append(False)
        else:
            mask.append(True)
            graphs.append(g)
    if not graphs:
        return torch.zeros(0, encoder.h_dim, device=device), mask
    hs = []
    for i in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[i:i + batch_size]).to(device)
        out = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        hs.append(_get_mu(out))
    return torch.cat(hs, 0), mask


@torch.no_grad()
def sample_decoder(decoder, vocab, target_h, n_samples, device,
                   temperature=1.0, max_length=32):
    """Draw `n_samples` walks from the decoder conditioned on `target_h`."""
    target_h = target_h.to(device)
    if target_h.dim() == 1:
        target_h = target_h.unsqueeze(0)
    latents = target_h.expand(n_samples, -1)
    gen = decoder.generate(latents, max_length=max_length,
                           temperature=temperature, greedy=False,
                           eos_id=vocab.eos_id)
    walks = []
    for ids in gen:
        toks = tuple(t for t in vocab.decode(ids)
                     if t not in ('BOS', 'EOS', 'PAD'))
        walks.append(toks)
    return walks


def _format_walk(walk, max_len=120):
    s = ' '.join(walk)
    return s if len(s) <= max_len else s[:max_len] + '...'


def _component_summary(walk):
    g = walk_to_graph(walk)
    if g is None:
        return '?'
    types = Counter(g.comp_types[c] for c in g.comp_nodes)
    return ', '.join(f'{t}x{n}' for t, n in sorted(types.items()))


def inverse_design(decoder, encoder, vocab, target_h, *,
                   n_samples=1000, temperature=1.0, device='cpu'):
    """Sample, validate, group by topology signature."""
    walks = sample_decoder(decoder, vocab, target_h, n_samples,
                           device, temperature=temperature)
    n_total = len(walks)
    wf = [w for w in walks if well_formed(w)]
    n_wf = len(wf)
    elec = [w for w in wf if is_electrically_valid(w)]
    n_elec = len(elec)

    sig_counts = Counter()
    sig_walks = {}
    for w in elec:
        sig = walk_topology_signature(w)
        if sig is None:
            continue
        sig_counts[sig] += 1
        sig_walks.setdefault(sig, w)

    sig_h_err = {}
    if sig_walks:
        rep_walks = list(sig_walks.values())
        rep_sigs = list(sig_walks.keys())
        h_gen, _ = encode_walks(encoder, rep_walks, device)
        target_norm = target_h.norm().item() + 1e-9
        diffs = (h_gen - target_h.to(device).unsqueeze(0)).norm(dim=-1)
        for sig, d in zip(rep_sigs, diffs.cpu().tolist()):
            sig_h_err[sig] = d / target_norm

    return {
        'n_total': n_total,
        'n_wf': n_wf,
        'n_elec': n_elec,
        'sig_counts': sig_counts,
        'sig_walks': sig_walks,
        'sig_h_err': sig_h_err,
    }


def build_reference_set(encoder, vocab, device):
    """Encode every training circuit and build reference data."""
    ds = CrossTopoSequenceDataset(
        ['rlc_dataset/filter_dataset.pkl', 'rlc_dataset/rl_dataset.pkl'],
        ALL_TYPES, vocab, augment=False, max_seq_len=32,
        edge_feature_mode='polynomial')

    by_type_h = defaultdict(list)
    by_type_circ = defaultdict(list)
    by_type_sig = defaultdict(list)
    known_sigs = set()
    sig_to_type = {}

    with torch.no_grad():
        for i in range(len(ds)):
            ft = ds.circuits[i]['filter_type']
            g = ds.pyg_graphs[i].to(device)
            batch_idx = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
            out = encoder(g.x, g.edge_index, g.edge_attr, batch_idx)
            h = _get_mu(out)
            by_type_h[ft].append(h[0].cpu())
            by_type_circ[ft].append(i)
            sig = walk_topology_signature(ds.all_walks[i][0])
            by_type_sig[ft].append(sig)
            if sig is not None:
                known_sigs.add(sig)
                sig_to_type.setdefault(sig, ft)

    centroids = {ft: torch.stack(v).mean(0) for ft, v in by_type_h.items()}
    return (ds, by_type_h, by_type_circ, by_type_sig,
            centroids, known_sigs, sig_to_type)


def report(label, res, known_sigs, sig_to_type, source_sig=None, top_k=5):
    """Print summary for one target."""
    print(f"\n-- {label} --")
    print(f"  drew {res['n_total']}  ->  well-formed {res['n_wf']}  "
          f"->  electrically valid {res['n_elec']}  "
          f"->  unique topologies {len(res['sig_counts'])}")

    if not res['sig_counts']:
        print("  (no valid walks)")
        return 0, 0

    novel_sigs = [s for s in res['sig_counts'] if s not in known_sigs]
    if source_sig is not None:
        hit = source_sig in res['sig_counts']
        n_hits = res['sig_counts'].get(source_sig, 0)
        src_label = sig_to_type.get(source_sig, '?')
        marker = 'Y' if hit else 'N'
        print(f"  source-topology recovery: {marker}  "
              f"({n_hits}/{res['n_elec']} samples = {src_label})")

    print(f"  novel topologies (not in training): {len(novel_sigs)}")
    print(f"  top {top_k} produced topologies:")
    ranked = sorted(res['sig_counts'].items(), key=lambda x: -x[1])
    for sig, cnt in ranked[:top_k]:
        ft_label = sig_to_type.get(sig, 'NOVEL')
        marker = '*' if sig not in known_sigs else ' '
        err = res['sig_h_err'].get(sig, float('nan'))
        walk = res['sig_walks'][sig]
        print(f"   {marker} x{cnt:4d}  rel-h={err:.2f}  "
              f"[{ft_label:>13s}]  [{_component_summary(walk)}]  "
              f"{_format_walk(walk, 80)}")
    return len(res['sig_counts']), len(novel_sigs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/production/best_v2.pt')
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 72)
    print("INVERSE DESIGN -- sampling walks conditioned on target h")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  n_samples   : {args.n_samples}")
    print(f"  temperature : {args.temperature}")
    print("=" * 72)

    encoder, decoder, vocab, _, _ = load_v2_model(args.ckpt, device)
    h_dim = encoder.latent_dim
    print(f"  loaded encoder ({h_dim}-D) + decoder")

    (ds, by_type_h, by_type_circ, by_type_sig,
     centroids, known_sigs, sig_to_type) = \
        build_reference_set(encoder, vocab, device)
    print(f"  reference set: {len(ds)} circuits, "
          f"{len(known_sigs)} known topology signatures")

    # -- Target 1: hold-out training circuits --
    print("\n" + "=" * 72)
    print("TARGET TYPE 1 -- hold-out: real `h` of one training circuit per type")
    print("=" * 72)
    src_hits = 0
    for ft in sorted(by_type_h.keys()):
        h_target = by_type_h[ft][0]
        source_sig = by_type_sig[ft][0]
        res = inverse_design(decoder, encoder, vocab, h_target,
                             n_samples=args.n_samples,
                             temperature=args.temperature, device=device)
        report(f"target = {ft}[0]", res, known_sigs, sig_to_type,
               source_sig=source_sig)
        if source_sig in res['sig_counts']:
            src_hits += 1
    print(f"\n  hold-out recovery: {src_hits}/{len(by_type_h)} types")

    novel_db = {}

    def absorb(label, res):
        for sig, cnt in res['sig_counts'].items():
            if sig in known_sigs:
                continue
            err = res['sig_h_err'].get(sig, float('nan'))
            walk = res['sig_walks'][sig]
            if sig not in novel_db or cnt > novel_db[sig][2]:
                novel_db[sig] = (label, walk, cnt, err)

    # -- Target 2: pairwise centroid midpoints --
    print("\n" + "=" * 72)
    print("TARGET TYPE 2 -- pairwise centroid midpoints")
    print("=" * 72)
    types = sorted(centroids.keys())
    for i, ta in enumerate(types):
        for tb in types[i + 1:]:
            mid = 0.5 * (centroids[ta] + centroids[tb])
            res = inverse_design(decoder, encoder, vocab, mid,
                                 n_samples=args.n_samples,
                                 temperature=args.temperature, device=device)
            report(f"0.5({ta} + {tb})", res, known_sigs, sig_to_type)
            absorb(f"interp 0.5({ta}+{tb})", res)
    n_interp_novel = len(novel_db)

    # -- Target 3: centroid + Gaussian noise --
    print("\n" + "=" * 72)
    print("TARGET TYPE 3 -- centroid + N(0, 0.5)")
    print("=" * 72)
    for ft in sorted(centroids.keys()):
        h_target = centroids[ft] + 0.5 * torch.randn_like(centroids[ft])
        res = inverse_design(decoder, encoder, vocab, h_target,
                             n_samples=args.n_samples,
                             temperature=args.temperature, device=device)
        report(f"{ft} + N(0, 0.5)", res, known_sigs, sig_to_type)
        absorb(f"perturb {ft}", res)

    # -- Summary --
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  hold-out source-topology recovery: {src_hits}/{len(by_type_h)}")
    print(f"  novel signatures (cumulative): {len(novel_db)}")
    if novel_db:
        print("\n  novel topologies discovered:")
        for i, (sig, (origin, walk, cnt, err)) in enumerate(
                sorted(novel_db.items(), key=lambda x: -x[1][2])):
            print(f"    #{i+1}  x{cnt:3d}  rel-h={err:.2f}  "
                  f"[{_component_summary(walk)}]  origin={origin}")
            print(f"         walk: {_format_walk(walk, 200)}")


if __name__ == '__main__':
    main()
