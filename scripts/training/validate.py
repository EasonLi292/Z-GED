"""Validation of the sequence decoder circuit generation model."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import torch
from torch.utils.data import DataLoader

from ml.data.sequence_dataset import SequenceDataset, collate_sequence_batch
from ml.utils.evaluate import (
    sequence_to_topology_key, get_training_topology_keys,
)
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.runtime import load_encoder_decoder

print("=" * 70)
print("Circuit Generation Model Validation (Sequence Decoder)")
print("=" * 70)

device = 'cpu'

# Load models
encoder, decoder, vocab, checkpoint = load_encoder_decoder(
    checkpoint_path='checkpoints/production/best.pt',
    device=device,
)

print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
if 'val_accuracy' in checkpoint:
    print(f"Token accuracy: {checkpoint['val_accuracy']:.1f}%")

# Load validation data
split_data = torch.load('rlc_dataset/stratified_split.pt', weights_only=False)
val_indices = split_data['val_indices']

max_seq_len = 32
val_dataset = SequenceDataset(
    'rlc_dataset/filter_dataset.pkl', val_indices, vocab,
    augment=False, max_seq_len=max_seq_len,
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=collate_sequence_batch,
)

# Load raw data for filter type labels
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    raw_data = pickle.load(f)

print(f"\n{'=' * 70}")
print("Validating on Full Validation Set")
print(f"{'=' * 70}\n")

# Per filter-type tracking
type_total = {}
type_valid = {}
type_topo_match = {}
total = 0
valid_count = 0
topo_match_count = 0

encoder.eval()
decoder.eval()

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        graph = batch['graph'].to(device)
        seq = batch['seq']
        seq_len = batch['seq_len']

        circuit_idx = val_indices[batch_idx]
        ft = raw_data[circuit_idx]['filter_type']

        if ft not in type_total:
            type_total[ft] = 0
            type_valid[ft] = 0
            type_topo_match[ft] = 0

        type_total[ft] += 1
        total += 1

        # Encode
        _, mu, _ = encoder(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )

        # Ground truth topology key
        gt_tokens = vocab.decode(seq[0, :seq_len[0]].tolist())
        gt_tokens = [t for t in gt_tokens if t != 'EOS']
        gt_key = sequence_to_topology_key(gt_tokens, vocab)

        # Generate walk
        generated = decoder.generate(
            mu, max_length=max_seq_len, greedy=True, eos_id=vocab.eos_id,
        )
        gen_tokens = vocab.decode(generated[0])
        gen_key = sequence_to_topology_key(gen_tokens, vocab)

        if gen_key is not None:
            valid_count += 1
            type_valid[ft] += 1

            if gen_key == gt_key:
                topo_match_count += 1
                type_topo_match[ft] += 1

# Print results
print(f"Overall Results:")
print(f"  Total circuits:     {total}")
print(f"  Valid walks:        {valid_count}/{total} ({100 * valid_count / total:.1f}%)")
print(f"  Topology match:    {topo_match_count}/{total} ({100 * topo_match_count / total:.1f}%)")

print(f"\nPer Filter Type:")
print("-" * 60)
print(f"{'Filter Type':<18} {'Total':>6} {'Valid':>6} {'Match':>6} {'Accuracy':>10}")
print("-" * 60)
for ft in sorted(type_total.keys()):
    t = type_total[ft]
    v = type_valid[ft]
    m = type_topo_match[ft]
    acc = 100 * m / t if t > 0 else 0
    print(f"{ft:<18} {t:>6} {v:>6} {m:>6} {acc:>9.1f}%")

print(f"\n{'=' * 70}")
print("Validation Complete!")
print("=" * 70)
