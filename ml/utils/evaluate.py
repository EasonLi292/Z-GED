"""
Evaluation and novel topology generation for the sequence decoder.

Reports:
    - Reconstruction accuracy (encode → decode → compare)
    - Novel topology analysis (sample from latent → generate → classify)
    - Comparison with NOVEL_TOPOLOGY_GENERATED.md results
"""

import pickle
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader

from ml.data.bipartite_graph import from_pickle_circuit
from ml.data.traversal import hierholzer, walk_to_circuit
from ml.data.sequence_dataset import SequenceDataset, collate_sequence_batch
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.runtime import build_encoder


def sequence_to_topology_key(
    walk: List[str],
    vocab: CircuitVocabulary,
) -> Optional[str]:
    """
    Convert a generated walk to a canonical topology string.

    Returns None if the walk is invalid (doesn't start/end at VSS,
    or has unparseable structure).
    """
    if not walk or walk[0] != 'VSS' or walk[-1] != 'VSS':
        return None

    # Collect all neighboring nets for each component across all appearances.
    # Each component appears twice in the Euler walk; its true terminals are
    # the union of net neighbors across both appearances.
    from collections import defaultdict
    comp_nets: Dict[str, set] = defaultdict(set)

    for i, tok in enumerate(walk):
        if vocab.token_type(tok) == 'component':
            if i > 0 and vocab.token_type(walk[i - 1]) == 'net':
                comp_nets[tok].add(walk[i - 1])
            if i < len(walk) - 1 and vocab.token_type(walk[i + 1]) == 'net':
                comp_nets[tok].add(walk[i + 1])

    if not comp_nets:
        return None

    # Build canonical key from (component_type, sorted_net_pair)
    # Sort by (type, nets) for canonical ordering
    parts = []
    for comp in sorted(comp_nets.keys(), key=lambda c: (vocab.component_type(c), c)):
        nets = sorted(comp_nets[comp])
        ctype = vocab.component_type(comp)
        if len(nets) == 2:
            parts.append(f'{ctype}({nets[0]}-{nets[1]})')
        elif len(nets) == 1:
            # Self-loop: both terminals on same net
            parts.append(f'{ctype}({nets[0]}-{nets[0]})')
        else:
            return None  # invalid

    return '|'.join(parts)


def get_training_topology_keys(
    dataset_path: str,
    vocab: CircuitVocabulary,
) -> Set[str]:
    """Get the set of all topology keys present in the training data."""
    with open(dataset_path, 'rb') as f:
        all_circuits = pickle.load(f)

    keys = set()
    for circuit in all_circuits:
        bg = from_pickle_circuit(circuit)
        walk = hierholzer(bg, start='VSS', rng=None)
        tokens = walk  # already string tokens
        key = sequence_to_topology_key(tokens, vocab)
        if key:
            keys.add(key)

    return keys


def evaluate_reconstruction(
    encoder,
    decoder,
    dataloader,
    vocab,
    device,
) -> Dict:
    """Evaluate reconstruction: encode circuit → generate walk → compare topology."""
    encoder.eval()
    decoder.eval()

    total = 0
    exact_match = 0
    topology_match = 0
    valid_walks = 0
    per_type_results: Dict[str, Dict] = {}

    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        all_circuits = pickle.load(f)

    split = torch.load('rlc_dataset/stratified_split.pt', weights_only=False)
    val_indices = split['val_indices']

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            graph = batch['graph'].to(device)
            seq = batch['seq']
            seq_len = batch['seq_len']

            B = seq.shape[0]

            # Encode
            z, mu, logvar = encoder(
                graph.x, graph.edge_index, graph.edge_attr, graph.batch
            )

            # Generate
            generated = decoder.generate(
                mu, max_length=32, greedy=True, eos_id=vocab.eos_id,
            )

            for b in range(B):
                total += 1
                idx_in_val = batch_idx * dataloader.batch_size + b
                if idx_in_val >= len(val_indices):
                    continue
                circuit_idx = val_indices[idx_in_val]
                circuit = all_circuits[circuit_idx]
                ft = circuit['filter_type']

                if ft not in per_type_results:
                    per_type_results[ft] = {'total': 0, 'valid': 0, 'topo_match': 0}
                per_type_results[ft]['total'] += 1

                # Ground truth walk (strip EOS token for topology comparison)
                gt_tokens = vocab.decode(seq[b, :seq_len[b]].tolist())
                gt_tokens = [t for t in gt_tokens if t != 'EOS']
                gt_key = sequence_to_topology_key(gt_tokens, vocab)

                # Generated walk
                gen_tokens = vocab.decode(generated[b])
                gen_key = sequence_to_topology_key(gen_tokens, vocab)

                if gen_key is not None:
                    valid_walks += 1
                    per_type_results[ft]['valid'] += 1

                    if gen_key == gt_key:
                        topology_match += 1
                        per_type_results[ft]['topo_match'] += 1

                if gen_tokens == gt_tokens:
                    exact_match += 1

    return {
        'total': total,
        'valid_walks': valid_walks,
        'valid_rate': valid_walks / max(total, 1) * 100,
        'topology_match': topology_match,
        'topology_accuracy': topology_match / max(total, 1) * 100,
        'exact_match': exact_match,
        'exact_accuracy': exact_match / max(total, 1) * 100,
        'per_type': per_type_results,
    }


def generate_novel_topologies(
    decoder,
    vocab,
    device,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict:
    """Generate topologies from random latent samples and classify them."""
    decoder.eval()
    torch.manual_seed(seed)

    training_keys = get_training_topology_keys('rlc_dataset/filter_dataset.pkl', vocab)
    print(f"Training topology keys: {len(training_keys)}")

    known_count = 0
    novel_count = 0
    invalid_count = 0
    novel_topologies: Counter = Counter()
    all_keys: Counter = Counter()

    with torch.no_grad():
        for i in range(0, n_samples, 32):
            batch_size = min(32, n_samples - i)
            z = torch.randn(batch_size, 8, device=device)

            generated = decoder.generate(
                z, max_length=32, greedy=True, eos_id=vocab.eos_id,
            )

            for gen_ids in generated:
                tokens = vocab.decode(gen_ids)
                key = sequence_to_topology_key(tokens, vocab)

                if key is None:
                    invalid_count += 1
                elif key in training_keys:
                    known_count += 1
                    all_keys[key] += 1
                else:
                    novel_count += 1
                    novel_topologies[key] += 1
                    all_keys[key] += 1

    return {
        'n_samples': n_samples,
        'known': known_count,
        'novel': novel_count,
        'invalid': invalid_count,
        'valid_rate': (known_count + novel_count) / max(n_samples, 1) * 100,
        'novel_rate': novel_count / max(known_count + novel_count, 1) * 100,
        'unique_novel': len(novel_topologies),
        'novel_topologies': novel_topologies.most_common(20),
        'all_keys': all_keys.most_common(20),
    }


def main():
    device = 'cpu'

    vocab = CircuitVocabulary(max_internal=10, max_components=10)

    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first with train.py.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Val loss: {ckpt['val_loss']:.4f}")
    print(f"Val accuracy: {ckpt.get('val_accuracy', 'N/A')}")

    # Build models
    encoder = build_encoder(device=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])

    max_seq_len = 32
    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size,
        latent_dim=8,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=max_seq_len + 1,  # +1 for latent prefix
        dropout=0.0,
        pad_id=vocab.pad_id,
    ).to(device)
    decoder.load_state_dict(ckpt['decoder_state_dict'])

    # Validation dataset
    split = torch.load('rlc_dataset/stratified_split.pt', weights_only=False)
    val_dataset = SequenceDataset(
        'rlc_dataset/filter_dataset.pkl', split['val_indices'], vocab,
        augment=False, max_seq_len=max_seq_len,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    # --- Reconstruction ---
    print("\n" + "=" * 60)
    print("RECONSTRUCTION EVALUATION")
    print("=" * 60)

    recon = evaluate_reconstruction(encoder, decoder, val_loader, vocab, device)
    print(f"Total:            {recon['total']}")
    print(f"Valid walks:       {recon['valid_walks']} ({recon['valid_rate']:.1f}%)")
    print(f"Topology match:   {recon['topology_match']} ({recon['topology_accuracy']:.1f}%)")
    print(f"Exact match:      {recon['exact_match']} ({recon['exact_accuracy']:.1f}%)")

    print(f"\nPer filter type:")
    for ft, stats in sorted(recon['per_type'].items()):
        topo_pct = stats['topo_match'] / max(stats['total'], 1) * 100
        valid_pct = stats['valid'] / max(stats['total'], 1) * 100
        print(f"  {ft:<15}: {stats['total']:3d} samples, "
              f"valid={valid_pct:.0f}%, topo_match={topo_pct:.0f}%")

    # --- Novel Topology Generation ---
    print("\n" + "=" * 60)
    print("NOVEL TOPOLOGY GENERATION (500 random latent samples)")
    print("=" * 60)

    novel = generate_novel_topologies(decoder, vocab, device, n_samples=500)
    print(f"Known topologies:  {novel['known']} ({novel['known']/5:.1f}%)")
    print(f"Novel topologies:  {novel['novel']} ({novel['novel_rate']:.1f}% of valid)")
    print(f"Invalid samples:   {novel['invalid']} ({novel['invalid']/5:.1f}%)")
    print(f"Valid rate:         {novel['valid_rate']:.1f}%")
    print(f"Unique novel:      {novel['unique_novel']}")

    if novel['novel_topologies']:
        print(f"\nTop novel topologies:")
        for key, count in novel['novel_topologies']:
            print(f"  {count:3d}x  {key}")

    # --- Comparison with existing NOVEL_TOPOLOGY_GENERATED.md ---
    print("\n" + "=" * 60)
    print("COMPARISON WITH ADJACENCY-MATRIX DECODER")
    print("=" * 60)
    print("  Metric                    | Adj-Matrix Decoder | Sequence Decoder")
    print("  --------------------------+--------------------+-----------------")
    print(f"  Valid rate                 | 89.2%              | {novel['valid_rate']:.1f}%")
    print(f"  Known training topologies | 66.8%              | {novel['known']/5:.1f}%")
    print(f"  Novel valid topologies    | 22.4%              | {novel['novel']/5:.1f}%")
    print(f"  Unique novel structures   | 9                  | {novel['unique_novel']}")


if __name__ == '__main__':
    main()
