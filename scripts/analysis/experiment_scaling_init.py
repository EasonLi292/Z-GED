"""Experiment: initialize AdmittanceConv alpha/beta scalars to 0.5.

This is intentionally isolated from production training. It reuses the V2
inverse-design training loop but does not overwrite production checkpoints.

Example:
    .venv/bin/python scripts/analysis/experiment_scaling_init.py --epochs 10
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.cross_topo_dataset import CrossTopoSequenceDataset, collate_fn
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.attribute_heads import FreqHead, GainHead, TypeHead
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from scripts.training.train_inverse_design import train_epoch, validate


ALL_TYPES = set(FILTER_TYPES_V2)


def set_scaling_init(encoder, value):
    """Set all AdmittanceConv alpha/beta scalars to the same value."""
    with torch.no_grad():
        for conv in encoder.convs:
            for ch in ('G', 'C', 'L'):
                getattr(conv, f'alpha_{ch}').fill_(value)
                getattr(conv, f'beta_{ch}').fill_(value)


def collect_scalars(encoder):
    """Return alpha/beta scalars for every layer/channel."""
    rows = []
    for layer, conv in enumerate(encoder.convs):
        for channel in ('G', 'C', 'L'):
            alpha = float(getattr(conv, f'alpha_{channel}').detach().cpu())
            beta = float(getattr(conv, f'beta_{channel}').detach().cpu())
            rows.append({
                'layer': layer,
                'channel': channel,
                'alpha': alpha,
                'beta': beta,
                'beta_over_alpha': beta / alpha if abs(alpha) > 1e-12 else None,
                'alpha_delta_from_0p5': alpha - 0.5,
                'beta_delta_from_0p5': beta - 0.5,
            })
    return rows


def print_scalars(rows):
    print("layer  channel   alpha        beta         beta/alpha")
    for r in rows:
        ratio = r['beta_over_alpha']
        ratio_s = f"{ratio:+.6f}" if ratio is not None else "nan"
        print(
            f"{r['layer']:>5}  {r['channel']:>7}  "
            f"{r['alpha']:+.6f}  {r['beta']:+.6f}  {ratio_s}"
        )


def build_loaders(seed, batch_size, max_seq_len, n_augment_walks):
    vocab = CircuitVocabulary(max_internal=10, max_components=10)
    pkl_main = 'rlc_dataset/filter_dataset.pkl'
    pkl_rl = 'rlc_dataset/rl_dataset.pkl'

    full_ds = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        augment=False, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial')

    type_indices = {}
    for i, circ in enumerate(full_ds.circuits):
        type_indices.setdefault(circ['filter_type'], []).append(i)

    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for _, idxs in sorted(type_indices.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_idx.extend(idxs[:split].tolist())
        val_idx.extend(idxs[split:].tolist())
    del full_ds

    train_dataset = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        indices=train_idx, augment=True, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial', n_augment_walks=n_augment_walks)
    val_dataset = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        indices=val_idx, augment=False, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn)
    return vocab, train_dataset, val_dataset, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-value', type=float, default=0.5)
    parser.add_argument('--out', default='analysis_results/scaling_init_ab_0p5.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    lr = 3e-4
    max_seq_len = 32
    latent_dim = 5
    n_augment_walks = 32

    beta_topo = 0.1
    beta_term = 0.02
    beta_warmup = 20
    alpha_freq = 0.5
    alpha_gain = 0.5
    alpha_type = 0.5

    print("=" * 72)
    print("SCALING INIT EXPERIMENT: alpha=beta=0.5")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"output={args.out}")
    print("=" * 72)

    vocab, train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        args.seed, args.batch_size, max_seq_len, n_augment_walks)
    print(f"Train: {len(train_dataset)} circuits")
    print(f"Val:   {len(val_dataset)} circuits")

    encoder = AdmittanceEncoder(
        node_feature_dim=4, hidden_dim=64, latent_dim=latent_dim,
        num_layers=3, dropout=0.1, vae=True,
    ).to(device)
    set_scaling_init(encoder, args.init_value)

    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size, latent_dim=latent_dim,
        d_model=128, n_heads=4, n_layers=2,
        max_seq_len=max_seq_len + 1, dropout=0.15, pad_id=vocab.pad_id,
    ).to(device)
    heads = {
        'freq': FreqHead(latent_dim).to(device),
        'gain': GainHead(latent_dim).to(device),
        'type': TypeHead(latent_dim).to(device),
    }

    all_params = (list(encoder.parameters()) + list(decoder.parameters())
                  + [p for h in heads.values() for p in h.parameters()])
    optimizer = torch.optim.AdamW(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8)

    history = [{
        'epoch': 0,
        'scalars': collect_scalars(encoder),
    }]
    print("\nInitial scalars:")
    print_scalars(history[-1]['scalars'])

    for epoch in range(1, args.epochs + 1):
        t = train_epoch(
            encoder, decoder, heads, train_loader, optimizer, device, epoch,
            beta_topo=beta_topo, beta_term=beta_term, beta_warmup=beta_warmup,
            alpha_freq=alpha_freq, alpha_gain=alpha_gain, alpha_type=alpha_type)
        v = validate(
            encoder, decoder, heads, val_loader, device,
            beta_topo=beta_topo, beta_term=beta_term,
            alpha_freq=alpha_freq, alpha_gain=alpha_gain, alpha_type=alpha_type)
        scheduler.step(v['ce'])

        scalars = collect_scalars(encoder)
        history.append({
            'epoch': epoch,
            'train': t,
            'val': v,
            'scalars': scalars,
        })

        print(f"\nEpoch {epoch}: v_CE={v['ce']:.4f}, v_acc={v['acc']:.2f}%, "
              f"freq={v['freq_mse']:.4f}, gain={v['gain_mse']:.5f}, "
              f"type_acc={v['type_acc']:.1f}%")
        print_scalars(scalars)

    result = {
        'experiment': 'admittance_scaling_alpha_beta_init_0p5',
        'init_value': args.init_value,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'history': history,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
