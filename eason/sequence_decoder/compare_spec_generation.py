"""
Comprehensive comparison: adjacency-matrix decoder vs sequence decoder.

Tests:
1. Random latent space sampling (1000 samples) — validity, diversity, mode coverage
2. Spec-based generation (dense grid) — topology correctness, consistency
3. Local stability — how much topology changes with small spec perturbations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import torch
import numpy as np
from collections import Counter, defaultdict

from ml.data.dataset import CircuitDataset
from ml.models.constants import FILTER_TYPES
from ml.utils.circuit_ops import walk_to_string, is_valid_walk
from ml.utils.runtime import load_encoder_decoder, build_encoder, make_collate_fn
from torch.utils.data import DataLoader

from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.evaluate import sequence_to_topology_key


def walk_to_description(tokens, vocab):
    """Convert walk tokens to a compact component connection string."""
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


def build_spec_database(encoder, dataset, device):
    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=make_collate_fn(include_specifications=True),
    )
    all_specs, all_latents = [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch['graph'].to(device)
            _, mu, _ = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            all_specs.append(batch['specifications'][0])
            all_latents.append(mu[0])
    return torch.stack(all_specs), torch.stack(all_latents)


def get_latent_for_specs(specs_db, latents_db, cutoff, q, k=5):
    target = torch.tensor([np.log10(cutoff), q])
    db_norm = torch.stack([torch.log10(specs_db[:, 0]), specs_db[:, 1]], dim=1)
    dists = ((db_norm - target)**2).sum(1).sqrt()
    top_k = dists.argsort()[:k]
    w = 1.0 / (dists[top_k] + 1e-6)
    w = w / w.sum()
    return (latents_db[top_k] * w.unsqueeze(1)).sum(0)


def main():
    device = 'cpu'

    # =====================================================================
    # Load models
    # =====================================================================
    print("Loading models...")
    adj_encoder, adj_decoder, _ = load_encoder_decoder(
        checkpoint_path='checkpoints/production/best.pt',
        device=device, decoder_overrides={'max_nodes': 10},
    )

    vocab = CircuitVocabulary(max_internal=10, max_components=10)
    seq_ckpt = torch.load(
        os.path.join(os.path.dirname(__file__), 'best.pt'),
        map_location=device, weights_only=False,
    )
    seq_encoder = build_encoder(device=device)
    seq_encoder.load_state_dict(seq_ckpt['encoder_state_dict'])
    seq_encoder.eval()

    seq_decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size, latent_dim=8, d_model=256,
        n_heads=4, n_layers=4, max_seq_len=33, dropout=0.0, pad_id=vocab.pad_id,
    ).to(device)
    seq_decoder.load_state_dict(seq_ckpt['decoder_state_dict'])
    seq_decoder.eval()

    # Build spec databases
    print("Building spec databases...")
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    adj_specs, adj_latents = build_spec_database(adj_encoder, dataset, device)
    seq_specs, seq_latents = build_spec_database(seq_encoder, dataset, device)

    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # Get training topology sets for both decoders
    print("Building training topology sets...")
    adj_training_topos = set()
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            graph = item['graph']
            batch_idx = torch.zeros(graph.x.shape[0], dtype=torch.long)
            _, mu, _ = adj_encoder(graph.x, graph.edge_index, graph.edge_attr, batch_idx)
            circuit = adj_decoder.generate(mu[0].unsqueeze(0).float(), verbose=False)
            adj_training_topos.add(circuit_to_string(circuit))

    from ml.utils.evaluate import get_training_topology_keys
    seq_training_keys = get_training_topology_keys('rlc_dataset/filter_dataset.pkl', vocab)

    # =====================================================================
    # 1. RANDOM LATENT SPACE SAMPLING (1000 samples)
    # =====================================================================
    print("\n" + "=" * 100)
    print("1. RANDOM LATENT SPACE SAMPLING (1000 samples)")
    print("=" * 100)

    N = 1000
    torch.manual_seed(42)
    z_random = torch.randn(N, 8)

    # Adjacency decoder
    adj_valid = 0
    adj_known = 0
    adj_novel_valid = 0
    adj_invalid = 0
    adj_topo_counter = Counter()
    adj_novel_topos = Counter()

    with torch.no_grad():
        for i in range(N):
            circuit = adj_decoder.generate(z_random[i].unsqueeze(0).float(), verbose=False)
            cstr = circuit_to_string(circuit)
            valid = is_valid_circuit(circuit)
            if not valid:
                adj_invalid += 1
            elif cstr in adj_training_topos:
                adj_known += 1
                adj_topo_counter[cstr] += 1
            else:
                adj_novel_valid += 1
                adj_novel_topos[cstr] += 1
                adj_topo_counter[cstr] += 1

    # Sequence decoder
    seq_valid_count = 0
    seq_known = 0
    seq_novel_valid = 0
    seq_invalid = 0
    seq_topo_counter = Counter()
    seq_novel_topos = Counter()

    with torch.no_grad():
        for i in range(0, N, 32):
            batch_size = min(32, N - i)
            z_batch = z_random[i:i+batch_size]
            generated = seq_decoder.generate(
                z_batch, max_length=32, greedy=True, eos_id=vocab.eos_id,
            )
            for gen_ids in generated:
                tokens = vocab.decode(gen_ids)
                key = sequence_to_topology_key(tokens, vocab)
                desc = walk_to_description(tokens, vocab)
                if key is None:
                    seq_invalid += 1
                elif key in seq_training_keys:
                    seq_known += 1
                    seq_topo_counter[desc] += 1
                else:
                    seq_novel_valid += 1
                    seq_novel_topos[desc] += 1
                    seq_topo_counter[desc] += 1

    adj_valid_total = adj_known + adj_novel_valid
    seq_valid_total = seq_known + seq_novel_valid

    print(f"\n  {'Metric':<35} | {'Adjacency':>12} | {'Sequence':>12}")
    print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Valid samples':<35} | {adj_valid_total:>8} ({adj_valid_total/N*100:>4.1f}%) | {seq_valid_total:>8} ({seq_valid_total/N*100:>4.1f}%)")
    print(f"  {'Invalid samples':<35} | {adj_invalid:>8} ({adj_invalid/N*100:>4.1f}%) | {seq_invalid:>8} ({seq_invalid/N*100:>4.1f}%)")
    print(f"  {'Known training topologies':<35} | {adj_known:>8} ({adj_known/N*100:>4.1f}%) | {seq_known:>8} ({seq_known/N*100:>4.1f}%)")
    print(f"  {'Novel valid topologies':<35} | {adj_novel_valid:>8} ({adj_novel_valid/N*100:>4.1f}%) | {seq_novel_valid:>8} ({seq_novel_valid/N*100:>4.1f}%)")
    print(f"  {'Unique topologies generated':<35} | {len(adj_topo_counter):>12} | {len(seq_topo_counter):>12}")
    print(f"  {'Unique novel topologies':<35} | {len(adj_novel_topos):>12} | {len(seq_novel_topos):>12}")

    print(f"\n  Adjacency — topology distribution (top 10):")
    for topo, count in adj_topo_counter.most_common(10):
        tag = "known" if topo in adj_training_topos else "NOVEL"
        print(f"    {count:>4}x  [{tag:<5}]  {topo}")

    print(f"\n  Sequence — topology distribution (top 10):")
    for desc, count in seq_topo_counter.most_common(10):
        print(f"    {count:>4}x  {desc}")

    if adj_novel_topos:
        print(f"\n  Adjacency — novel topologies:")
        for topo, count in adj_novel_topos.most_common(20):
            print(f"    {count:>4}x  {topo}")

    if seq_novel_topos:
        print(f"\n  Sequence — novel topologies:")
        for desc, count in seq_novel_topos.most_common(20):
            print(f"    {count:>4}x  {desc}")

    # =====================================================================
    # 2. SPEC-BASED GENERATION (dense grid)
    # =====================================================================
    print("\n" + "=" * 100)
    print("2. SPEC-BASED GENERATION (dense grid)")
    print("   Cutoff: 10 Hz to 1 MHz (log-spaced, 15 points)")
    print("   Q: 0.01, 0.1, 0.707, 2.0, 5.0")
    print("=" * 100)

    cutoffs = np.logspace(1, 6, 15)  # 10 Hz to 1 MHz
    qs = [0.01, 0.1, 0.707, 2.0, 5.0]

    adj_spec_valid = 0
    seq_spec_valid = 0
    adj_spec_total = 0
    seq_spec_total = 0
    adj_spec_topos = Counter()
    seq_spec_topos = Counter()

    print(f"\n  {'Cutoff':>10}  {'Q':>6}  {'Adjacency-Matrix':<55}  {'Sequence'}")
    print("  " + "-" * 140)

    for q in qs:
        for cutoff in cutoffs:
            # Adjacency
            z_adj = get_latent_for_specs(adj_specs, adj_latents, cutoff, q)
            with torch.no_grad():
                adj_circuit = adj_decoder.generate(z_adj.unsqueeze(0).float(), verbose=False)
            adj_str = circuit_to_string(adj_circuit)
            adj_v = is_valid_circuit(adj_circuit)
            adj_spec_total += 1
            if adj_v:
                adj_spec_valid += 1
            adj_spec_topos[adj_str] += 1

            # Sequence
            z_seq = get_latent_for_specs(seq_specs, seq_latents, cutoff, q)
            with torch.no_grad():
                seq_result = seq_decoder.generate(
                    z_seq.unsqueeze(0).float(), max_length=32, greedy=True, eos_id=vocab.eos_id,
                )
            seq_tokens = vocab.decode(seq_result[0])
            seq_str = walk_to_description(seq_tokens, vocab)
            seq_key = sequence_to_topology_key(seq_tokens, vocab)
            seq_v = seq_key is not None
            seq_spec_total += 1
            if seq_v:
                seq_spec_valid += 1
            seq_spec_topos[seq_str] += 1

            av = "V" if adj_v else "X"
            sv = "V" if seq_v else "X"
            print(f"  {cutoff:>10.0f}  {q:>6.3f}  [{av}] {adj_str:<52}  [{sv}] {seq_str}")
        print()

    print(f"  Spec-based generation summary:")
    print(f"    {'Metric':<35} | {'Adjacency':>12} | {'Sequence':>12}")
    print(f"    {'-'*35}-+-{'-'*12}-+-{'-'*12}")
    print(f"    {'Valid':<35} | {adj_spec_valid:>8}/{adj_spec_total:<3} | {seq_spec_valid:>8}/{seq_spec_total:<3}")
    print(f"    {'Unique topologies':<35} | {len(adj_spec_topos):>12} | {len(seq_spec_topos):>12}")

    print(f"\n    Adjacency — spec topologies:")
    for topo, count in adj_spec_topos.most_common():
        print(f"      {count:>3}x  {topo}")
    print(f"\n    Sequence — spec topologies:")
    for desc, count in seq_spec_topos.most_common():
        print(f"      {count:>3}x  {desc}")

    # =====================================================================
    # 3. LOCAL STABILITY (how often topology changes with ±1% cutoff)
    # =====================================================================
    print("\n" + "=" * 100)
    print("3. LOCAL STABILITY (±1% cutoff perturbation, 50 base points)")
    print("=" * 100)

    base_cutoffs = np.logspace(1.5, 5.5, 50)
    base_q = 0.707

    adj_changes = 0
    seq_changes = 0
    total_pairs = 0

    for base_f in base_cutoffs:
        adj_topos_local = []
        seq_topos_local = []
        for delta in [-0.01, 0, 0.01]:
            cutoff = base_f * (1 + delta)

            z_adj = get_latent_for_specs(adj_specs, adj_latents, cutoff, base_q)
            with torch.no_grad():
                adj_circuit = adj_decoder.generate(z_adj.unsqueeze(0).float(), verbose=False)
            adj_topos_local.append(circuit_to_string(adj_circuit))

            z_seq = get_latent_for_specs(seq_specs, seq_latents, cutoff, base_q)
            with torch.no_grad():
                seq_result = seq_decoder.generate(
                    z_seq.unsqueeze(0).float(), max_length=32, greedy=True, eos_id=vocab.eos_id,
                )
            seq_tokens = vocab.decode(seq_result[0])
            seq_topos_local.append(walk_to_description(seq_tokens, vocab))

        # Count changes: compare -1% vs center, and +1% vs center
        if adj_topos_local[0] != adj_topos_local[1]:
            adj_changes += 1
        if adj_topos_local[2] != adj_topos_local[1]:
            adj_changes += 1
        if seq_topos_local[0] != seq_topos_local[1]:
            seq_changes += 1
        if seq_topos_local[2] != seq_topos_local[1]:
            seq_changes += 1
        total_pairs += 2

    adj_stability = (total_pairs - adj_changes) / total_pairs * 100
    seq_stability = (total_pairs - seq_changes) / total_pairs * 100

    print(f"\n  {'Metric':<35} | {'Adjacency':>12} | {'Sequence':>12}")
    print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Topology changes (of {total_pairs})':<35} | {adj_changes:>12} | {seq_changes:>12}")
    print(f"  {'Local stability':<35} | {adj_stability:>11.1f}% | {seq_stability:>11.1f}%")

    # =====================================================================
    # 4. Q-DEPENDENT TOPOLOGY DIFFERENTIATION
    #    Does the decoder change topology as Q changes? (it should)
    # =====================================================================
    print("\n" + "=" * 100)
    print("4. Q-DEPENDENT TOPOLOGY DIFFERENTIATION")
    print("   At cutoff=10kHz, how many distinct topologies across Q=[0.01..6.5]?")
    print("=" * 100)

    q_test = [0.01, 0.05, 0.1, 0.3, 0.5, 0.707, 1.0, 1.5, 2.0, 3.0, 5.0, 6.5]
    adj_q_topos = set()
    seq_q_topos = set()

    for q in q_test:
        z_adj = get_latent_for_specs(adj_specs, adj_latents, 10000, q)
        with torch.no_grad():
            adj_circuit = adj_decoder.generate(z_adj.unsqueeze(0).float(), verbose=False)
        adj_q_topos.add(circuit_to_string(adj_circuit))

        z_seq = get_latent_for_specs(seq_specs, seq_latents, 10000, q)
        with torch.no_grad():
            seq_result = seq_decoder.generate(
                z_seq.unsqueeze(0).float(), max_length=32, greedy=True, eos_id=vocab.eos_id,
            )
        seq_tokens = vocab.decode(seq_result[0])
        seq_q_topos.add(walk_to_description(seq_tokens, vocab))

    print(f"\n  Adjacency distinct topologies: {len(adj_q_topos)}")
    for t in sorted(adj_q_topos):
        print(f"    {t}")
    print(f"\n  Sequence distinct topologies: {len(seq_q_topos)}")
    for t in sorted(seq_q_topos):
        print(f"    {t}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"\n  {'Metric':<40} | {'Adjacency':>12} | {'Sequence':>12}")
    print(f"  {'-'*40}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Random: valid rate (N=1000)':<40} | {adj_valid_total/N*100:>11.1f}% | {seq_valid_total/N*100:>11.1f}%")
    print(f"  {'Random: known topologies':<40} | {adj_known/N*100:>11.1f}% | {seq_known/N*100:>11.1f}%")
    print(f"  {'Random: novel valid':<40} | {adj_novel_valid/N*100:>11.1f}% | {seq_novel_valid/N*100:>11.1f}%")
    print(f"  {'Random: unique topologies':<40} | {len(adj_topo_counter):>12} | {len(seq_topo_counter):>12}")
    print(f"  {'Spec-based: valid rate':<40} | {adj_spec_valid}/{adj_spec_total} ({adj_spec_valid/adj_spec_total*100:.0f}%){'':<3} | {seq_spec_valid}/{seq_spec_total} ({seq_spec_valid/seq_spec_total*100:.0f}%)")
    print(f"  {'Spec-based: unique topologies':<40} | {len(adj_spec_topos):>12} | {len(seq_spec_topos):>12}")
    print(f"  {'Local stability (±1%)':<40} | {adj_stability:>11.1f}% | {seq_stability:>11.1f}%")
    print(f"  {'Q differentiation (distinct topos)':<40} | {len(adj_q_topos):>12} | {len(seq_q_topos):>12}")


if __name__ == '__main__':
    main()
