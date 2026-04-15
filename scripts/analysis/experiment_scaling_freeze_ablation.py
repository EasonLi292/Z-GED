"""Ablate admittance scaling init under late-encoder freezing.

This experiment compares whether the linear or logarithmic coefficient branch
matters more when early admittance layers have to carry more of the work.

It intentionally does not write production checkpoints.

Example:
    .venv/bin/python scripts/analysis/experiment_scaling_freeze_ablation.py \
        --epochs 10 \
        --out analysis_results/scaling_freeze_ablation_epochs10.json
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.cross_topo_dataset import CrossTopoSequenceDataset, collate_fn
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.attribute_heads import FreqHead, GainHead
from ml.models.attribute_heads import TypeHead
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from scripts.training.train_inverse_design import train_epoch, validate


ALL_TYPES = set(FILTER_TYPES_V2)


@dataclass(frozen=True)
class RunConfig:
    name: str
    alpha_init: float
    beta_init: float
    train_scalars: bool
    freeze_regime: str


RUNS = [
    # Normal training: checks whether symmetric blend init changes convergence.
    RunConfig('linear_learned_none', 1.0, 0.0, True, 'none'),
    RunConfig('blend_learned_none', 0.5, 0.5, True, 'none'),

    # Front-heavy stress: later encoder layers are fixed, so layer 0 and the
    # trainable downstream heads/decoder must adapt around a restricted encoder.
    RunConfig('linear_learned_late_frozen', 1.0, 0.0, True, 'late_encoder_frozen'),
    RunConfig('blend_learned_late_frozen', 0.5, 0.5, True, 'late_encoder_frozen'),
    RunConfig('linear_fixed_late_frozen', 1.0, 0.0, False, 'late_encoder_frozen'),
    RunConfig('log_fixed_late_frozen', 0.0, 1.0, False, 'late_encoder_frozen'),
    RunConfig('blend_fixed_late_frozen', 0.5, 0.5, False, 'late_encoder_frozen'),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_scaling_init(encoder: AdmittanceEncoder, alpha: float, beta: float) -> None:
    with torch.no_grad():
        for conv in encoder.convs:
            for ch in ('G', 'C', 'L'):
                getattr(conv, f'alpha_{ch}').fill_(alpha)
                getattr(conv, f'beta_{ch}').fill_(beta)


def set_scalars_trainable(encoder: AdmittanceEncoder, trainable: bool) -> None:
    for conv in encoder.convs:
        for ch in ('G', 'C', 'L'):
            getattr(conv, f'alpha_{ch}').requires_grad_(trainable)
            getattr(conv, f'beta_{ch}').requires_grad_(trainable)


def freeze_late_encoder(encoder: AdmittanceEncoder) -> None:
    """Freeze encoder layers after the first message-passing layer.

    This leaves conv/norm/residual layer 0 trainable. Decoder, VAE readout, and
    attribute heads remain trainable so the output map is not a random fixed
    bottleneck.
    """
    for modules in (encoder.convs, encoder.norms, encoder.residual_projs):
        for module in modules[1:]:
            for param in module.parameters():
                param.requires_grad_(False)


def apply_freeze_regime(encoder: AdmittanceEncoder, regime: str) -> None:
    if regime == 'none':
        return
    if regime == 'late_encoder_frozen':
        freeze_late_encoder(encoder)
        return
    raise ValueError(f'Unknown freeze regime: {regime}')


def collect_scalars(encoder: AdmittanceEncoder) -> List[Dict[str, float]]:
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
                'alpha_trainable': bool(getattr(conv, f'alpha_{channel}').requires_grad),
                'beta_trainable': bool(getattr(conv, f'beta_{channel}').requires_grad),
            })
    return rows


def count_trainable(modules: Iterable[torch.nn.Module]) -> Dict[str, int]:
    total = 0
    trainable = 0
    for module in modules:
        for param in module.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def make_models(vocab: CircuitVocabulary, device: str):
    latent_dim = 5
    max_seq_len = 32
    encoder = AdmittanceEncoder(
        node_feature_dim=4, hidden_dim=64, latent_dim=latent_dim,
        num_layers=3, dropout=0.1, vae=True,
    ).to(device)
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
    return encoder, decoder, heads


def build_loaders(seed: int, batch_size: int, max_seq_len: int, n_augment_walks: int):
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


def trainable_params(encoder, decoder, heads):
    params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + [p for h in heads.values() for p in h.parameters()]
    )
    return [p for p in params if p.requires_grad]


def run_one(cfg: RunConfig, args, vocab, train_loader, val_loader, device: str):
    set_seed(args.seed)
    encoder, decoder, heads = make_models(vocab, device)
    set_scaling_init(encoder, cfg.alpha_init, cfg.beta_init)
    apply_freeze_regime(encoder, cfg.freeze_regime)
    set_scalars_trainable(encoder, cfg.train_scalars)

    params = trainable_params(encoder, decoder, heads)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8)

    modules = [encoder, decoder, *heads.values()]
    result = {
        'config': asdict(cfg),
        'param_count': count_trainable(modules),
        'initial_scalars': collect_scalars(encoder),
        'history': [],
    }

    print('\n' + '=' * 80)
    print(f"RUN {cfg.name}")
    print(f"init alpha={cfg.alpha_init}, beta={cfg.beta_init}, "
          f"train_scalars={cfg.train_scalars}, freeze={cfg.freeze_regime}")
    print(f"trainable params: {result['param_count']['trainable']:,} / "
          f"{result['param_count']['total']:,}")
    print('=' * 80)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        t = train_epoch(
            encoder, decoder, heads, train_loader, optimizer, device, epoch,
            beta_topo=args.beta_topo, beta_term=args.beta_term,
            beta_warmup=args.beta_warmup,
            alpha_freq=args.alpha_freq, alpha_gain=args.alpha_gain,
            alpha_type=args.alpha_type)
        v = validate(
            encoder, decoder, heads, val_loader, device,
            beta_topo=args.beta_topo, beta_term=args.beta_term,
            alpha_freq=args.alpha_freq, alpha_gain=args.alpha_gain,
            alpha_type=args.alpha_type)
        scheduler.step(v['ce'])

        result['history'].append({
            'epoch': epoch,
            'train': t,
            'val': v,
            'scalars': collect_scalars(encoder),
        })

        print(f"{cfg.name} epoch {epoch:>2d}: "
              f"v_CE={v['ce']:.4f}, v_acc={v['acc']:.2f}%, "
              f"freq={v['freq_mse']:.4f}, gain={v['gain_mse']:.5f}, "
              f"type_acc={v['type_acc']:.1f}%")

    result['runtime_sec'] = time.time() - started
    return result


def summarize(results):
    rows = []
    for run in results:
        final = run['history'][-1]
        val = final['val']
        scalars = final['scalars']
        ratios = [
            r['beta_over_alpha'] for r in scalars
            if r['beta_over_alpha'] is not None
        ]
        rows.append({
            'name': run['config']['name'],
            'freeze': run['config']['freeze_regime'],
            'train_scalars': run['config']['train_scalars'],
            'val_ce': val['ce'],
            'val_acc': val['acc'],
            'freq_mse': val['freq_mse'],
            'gain_mse': val['gain_mse'],
            'type_acc': val['type_acc'],
            'ratio_min': min(ratios) if ratios else None,
            'ratio_max': max(ratios) if ratios else None,
        })
    rows.sort(key=lambda r: (r['freeze'], r['val_ce']))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--out', default='analysis_results/scaling_freeze_ablation.json')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='Optional subset of run names to execute.')
    parser.add_argument('--beta-topo', type=float, default=0.1)
    parser.add_argument('--beta-term', type=float, default=0.02)
    parser.add_argument('--beta-warmup', type=int, default=20)
    parser.add_argument('--alpha-freq', type=float, default=0.5)
    parser.add_argument('--alpha-gain', type=float, default=0.5)
    parser.add_argument('--alpha-type', type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    max_seq_len = 32
    n_augment_walks = 32
    selected = RUNS
    if args.runs:
        wanted = set(args.runs)
        selected = [run for run in RUNS if run.name in wanted]
        missing = wanted - {run.name for run in selected}
        if missing:
            raise ValueError(f'Unknown runs: {sorted(missing)}')

    print('=' * 80)
    print('SCALING/FREEZE ABLATION')
    print(f'device={device}, epochs={args.epochs}, batch_size={args.batch_size}')
    print(f'output={args.out}')
    print(f'runs={", ".join(run.name for run in selected)}')
    print('=' * 80)

    vocab, train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        args.seed, args.batch_size, max_seq_len, n_augment_walks)
    print(f'Train: {len(train_dataset)} circuits')
    print(f'Val:   {len(val_dataset)} circuits')
    print(f'Vocab: {vocab.vocab_size} tokens')

    results = [
        run_one(cfg, args, vocab, train_loader, val_loader, device)
        for cfg in selected
    ]
    summary = summarize(results)

    payload = {
        'experiment': 'admittance_scaling_freeze_ablation',
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'freeze_note': (
            'late_encoder_frozen freezes encoder layers 1..N-1 only; '
            'layer 0, VAE readout, decoder, and attribute heads remain trainable.'
        ),
        'summary': summary,
        'runs': results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print('\nSUMMARY sorted by freeze regime then val CE')
    print('name                            freeze                CE      acc    type   freq     gain    ratio_min ratio_max')
    for row in summary:
        print(f"{row['name']:<31} {row['freeze']:<21} "
              f"{row['val_ce']:<7.4f} {row['val_acc']:<6.2f} "
              f"{row['type_acc']:<6.1f} {row['freq_mse']:<8.4f} "
              f"{row['gain_mse']:<7.5f} "
              f"{row['ratio_min']!s:<9} {row['ratio_max']!s:<9}")
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
