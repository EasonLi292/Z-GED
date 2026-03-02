"""
Evaluate pole/zero prediction quality.

Loads the trained encoder, encodes all circuits (train + val split),
and compares mu[:, 4:] against ground-truth pz_target.

Reports per-dimension MSE, R², and Pearson correlation,
broken down by filter type.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy import stats
import pickle

from ml.data.dataset import CircuitDataset
from ml.models.constants import FILTER_TYPES
from ml.utils.runtime import build_encoder, make_collate_fn

DIM_NAMES = ['sigma_p', 'omega_p', 'sigma_z', 'omega_z']


def encode_all(encoder, dataloader, device):
    """Encode all circuits and collect mu[:, 4:], pz_target, and dataset indices."""
    encoder.eval()
    all_mu_pz = []
    all_target = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            graph = batch['graph'].to(device)
            pz_target = batch['pz_target']

            _, mu, _ = encoder(
                graph.x, graph.edge_index, graph.edge_attr,
                graph.batch
            )

            all_mu_pz.append(mu[:, 4:].cpu())
            all_target.append(pz_target)
            all_indices.extend(batch['indices'])

    return torch.cat(all_mu_pz, dim=0), torch.cat(all_target, dim=0), all_indices


def compute_dim_metrics(p, t):
    """Compute MSE, MAE, R², Pearson r for a single dimension."""
    mse = np.mean((p - t) ** 2)
    mae = np.mean(np.abs(p - t))

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')

    if np.std(t) > 1e-10 and np.std(p) > 1e-10:
        r, _ = stats.pearsonr(p, t)
    else:
        r = float('nan')

    return mse, mae, r2, r


def report_metrics(name, pred, target):
    """Print per-dimension and overall metrics."""
    n = pred.shape[0]

    print(f"\n{'='*60}")
    print(f"  {name}  (N={n})")
    print(f"{'='*60}")

    mse_overall = ((pred - target) ** 2).mean().item()
    mae_overall = (pred - target).abs().mean().item()
    print(f"  Overall MSE: {mse_overall:.6f}    MAE: {mae_overall:.6f}")

    print(f"\n  {'Dim':<10} {'MSE':>10} {'MAE':>10} {'R²':>10} {'Pearson r':>10} {'pred σ':>10} {'tgt σ':>10}")
    print(f"  {'-'*72}")

    for d in range(4):
        p = pred[:, d].numpy()
        t = target[:, d].numpy()
        mse, mae, r2, r = compute_dim_metrics(p, t)
        print(f"  {DIM_NAMES[d]:<10} {mse:>10.6f} {mae:>10.6f} {r2:>10.4f} {r:>10.4f} {p.std():>10.4f} {t.std():>10.4f}")

    errors = (pred - target).abs()
    max_err = errors.max(dim=0).values
    print(f"\n  Max |error| per dim: {[f'{v:.4f}' for v in max_err.tolist()]}")

    return mse_overall


def report_by_filter_type(pred, target, indices, raw_data, split_name):
    """Break down metrics by filter type."""
    # Map each sample to its filter type
    filter_labels = []
    for idx in indices:
        filter_labels.append(raw_data[idx]['filter_type'])

    filter_labels = np.array(filter_labels)

    print(f"\n{'='*60}")
    print(f"  {split_name} — Breakdown by filter type")
    print(f"{'='*60}")

    # Header
    print(f"\n  {'Filter':<14} {'N':>4}  {'MSE':>8}  {'sigma_p':>8} {'omega_p':>8} {'sigma_z':>8} {'omega_z':>8}  {'R²_σp':>7} {'R²_ωp':>7} {'R²_σz':>7} {'R²_ωz':>7}")
    print(f"  {'-'*108}")

    for ft in FILTER_TYPES:
        mask = filter_labels == ft
        n = mask.sum()
        if n == 0:
            continue

        p = pred[mask].numpy()
        t = target[mask].numpy()

        mse_overall = np.mean((p - t) ** 2)

        row = f"  {ft:<14} {n:>4}  {mse_overall:>8.6f}"

        r2_vals = []
        for d in range(4):
            mse_d, _, r2_d, _ = compute_dim_metrics(p[:, d], t[:, d])
            row += f"  {mse_d:>8.6f}"
            r2_vals.append(r2_d)

        for r2_d in r2_vals:
            if np.isnan(r2_d):
                row += f"  {'N/A':>7}"
            else:
                row += f"  {r2_d:>7.3f}"

        print(row)

    # Also show which filter types have non-zero sigma_z targets
    print(f"\n  Target sigma_z stats by filter type:")
    for ft in FILTER_TYPES:
        mask = filter_labels == ft
        if mask.sum() == 0:
            continue
        t = target[mask, 2].numpy()
        nonzero = np.sum(np.abs(t) > 1e-6)
        print(f"    {ft:<14}  nonzero: {nonzero}/{mask.sum()}  mean: {t.mean():.6f}  std: {t.std():.6f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load raw data for filter type labels
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    print(f"Dataset size: {len(dataset)}")

    # Load train/val split
    split_data = torch.load('rlc_dataset/stratified_split.pt', weights_only=True)
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False,
        collate_fn=make_collate_fn(include_pz_target=True, include_indices=True),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=make_collate_fn(include_pz_target=True, include_indices=True),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build encoder
    encoder = build_encoder(device=device)

    # Load best checkpoint
    ckpt_path = 'checkpoints/production/best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # Encode
    train_pred, train_target, train_idx = encode_all(encoder, train_loader, device)
    val_pred, val_target, val_idx = encode_all(encoder, val_loader, device)

    # Overall metrics
    report_metrics("Train", train_pred, train_target)
    report_metrics("Val", val_pred, val_target)

    # Per-filter-type breakdown
    report_by_filter_type(train_pred, train_target, train_idx, raw_data, "Train")
    report_by_filter_type(val_pred, val_target, val_idx, raw_data, "Val")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    train_mse = ((train_pred - train_target) ** 2).mean().item()
    val_mse = ((val_pred - val_target) ** 2).mean().item()
    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Val   MSE: {val_mse:.6f}")
    print(f"  Val/Train ratio: {val_mse / train_mse:.2f}x")


if __name__ == '__main__':
    main()
