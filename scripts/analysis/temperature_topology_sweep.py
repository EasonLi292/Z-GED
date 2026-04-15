"""Temperature sweep for random-latent topology generation.

Samples the production sequence decoder at several temperatures and reports
validity, novelty, and the most frequent generated topology strings.

Example:
    .venv/bin/python scripts/analysis/temperature_topology_sweep.py \
        --samples 500 \
        --out analysis_results/temperature_topology_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch

from ml.data.bipartite_graph import from_pickle_circuit
from ml.data.graph_signature import is_electrically_valid, walk_topology_signature
from ml.data.traversal import hierholzer
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.evaluate import get_training_topology_keys, sequence_to_topology_key
from ml.utils.runtime import load_decoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_signatures(
    dataset_path: str,
) -> Tuple[set, Dict[str, str]]:
    with open(dataset_path, 'rb') as f:
        circuits = pickle.load(f)

    known_sigs = set()
    sig_labels: Dict[str, str] = {}
    for circuit in circuits:
        graph = from_pickle_circuit(circuit)
        walk = hierholzer(graph, start='VSS', rng=None)
        sig = walk_topology_signature(walk)
        if sig is None:
            continue
        known_sigs.add(sig)
        sig_labels[signature_id(sig)] = sequence_to_topology_key(walk, CircuitVocabulary())
    return known_sigs, sig_labels


def signature_id(sig) -> str:
    parts = []
    rows = sorted(
        sig,
        key=lambda item: (item[0][0], tuple(sorted(item[0][1])), item[1]),
    )
    for (comp_type, nets), count in rows:
        parts.append(f'{comp_type}:{",".join(sorted(nets))}:{count}')
    return '|'.join(parts)


def sample_temperature(
    decoder,
    vocab,
    known_keys: set,
    known_sigs: set,
    temperature: float,
    samples: int,
    batch_size: int,
    device: str,
) -> Dict:
    valid_count = 0
    invalid_count = 0
    known_count = 0
    novel_count = 0
    topology_counts: Counter = Counter()
    novel_counts: Counter = Counter()
    key_valid_count = 0
    key_invalid_count = 0
    key_known_count = 0
    key_novel_count = 0
    key_topology_counts: Counter = Counter()
    key_novel_counts: Counter = Counter()

    with torch.no_grad():
        for start in range(0, samples, batch_size):
            current = min(batch_size, samples - start)
            z = torch.randn(current, 8, device=device)
            generated = decoder.generate(
                z,
                max_length=32,
                temperature=temperature,
                greedy=False,
                eos_id=vocab.eos_id,
            )

            for token_ids in generated:
                walk = vocab.decode(token_ids)
                key = sequence_to_topology_key(walk, vocab)
                if key is None:
                    key_invalid_count += 1
                else:
                    key_valid_count += 1
                    key_topology_counts[key] += 1
                    if key in known_keys:
                        key_known_count += 1
                    else:
                        key_novel_count += 1
                        key_novel_counts[key] += 1

                sig = walk_topology_signature(walk)
                if sig is None or not is_electrically_valid(walk):
                    invalid_count += 1
                    continue

                valid_count += 1
                label = key or signature_id(sig)
                topology_counts[label] += 1

                if sig in known_sigs:
                    known_count += 1
                else:
                    novel_count += 1
                    novel_counts[label] += 1

    return {
        'temperature': temperature,
        'samples': samples,
        'valid': valid_count,
        'invalid': invalid_count,
        'known_valid': known_count,
        'novel_valid': novel_count,
        'valid_rate': 100 * valid_count / max(samples, 1),
        'invalid_rate': 100 * invalid_count / max(samples, 1),
        'novel_rate_of_all': 100 * novel_count / max(samples, 1),
        'novel_rate_of_valid': 100 * novel_count / max(valid_count, 1),
        'unique_valid_topologies': len(topology_counts),
        'unique_novel_topologies': len(novel_counts),
        'top_topologies': topology_counts.most_common(10),
        'top_novel_topologies': novel_counts.most_common(10),
        'topology_key_metric': {
            'valid': key_valid_count,
            'invalid': key_invalid_count,
            'known_valid': key_known_count,
            'novel_valid': key_novel_count,
            'valid_rate': 100 * key_valid_count / max(samples, 1),
            'invalid_rate': 100 * key_invalid_count / max(samples, 1),
            'novel_rate_of_all': 100 * key_novel_count / max(samples, 1),
            'novel_rate_of_valid': 100 * key_novel_count / max(key_valid_count, 1),
            'unique_valid_topologies': len(key_topology_counts),
            'unique_novel_topologies': len(key_novel_counts),
            'top_topologies': key_topology_counts.most_common(10),
            'top_novel_topologies': key_novel_counts.most_common(10),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/production/best.pt')
    parser.add_argument('--dataset', default='rlc_dataset/filter_dataset.pkl')
    parser.add_argument('--out', default='analysis_results/temperature_topology_sweep.json')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--temperatures',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    decoder, vocab, checkpoint = load_decoder(args.checkpoint, device=device)
    known_keys = get_training_topology_keys(args.dataset, vocab)
    known_sigs, _ = load_training_signatures(args.dataset)

    results = []
    for temperature in args.temperatures:
        set_seed(args.seed)
        print(f'Temperature {temperature:g}: sampling {args.samples}...')
        row = sample_temperature(
            decoder,
            vocab,
            known_keys,
            known_sigs,
            temperature,
            args.samples,
            args.batch_size,
            device,
        )
        results.append(row)
        print(
            f"  valid={row['valid_rate']:.1f}% "
            f"novel_valid={row['novel_rate_of_valid']:.1f}% "
            f"unique={row['unique_valid_topologies']}"
        )

    payload = {
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_val_loss': checkpoint.get('val_loss'),
        'dataset': args.dataset,
        'seed': args.seed,
        'samples_per_temperature': args.samples,
        'known_training_topologies': len(known_sigs),
        'known_training_topology_keys': len(known_keys),
        'validity_note': (
            'Top-level metrics use strict electrical-walk validity. '
            'topology_key_metric preserves the older sequence_to_topology_key '
            'definition, which can count self-loops or dangling internals as '
            'valid topology strings.'
        ),
        'results': results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
