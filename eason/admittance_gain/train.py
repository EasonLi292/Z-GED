"""
Training script for admittance-based gain prediction experiment.

Loss: MSE(gain_pred, gain_target) + kl_weight * KL_loss
KL warmup over 10 epochs to target 0.001.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from eason.admittance_gain.dataset import AdmittanceDataset, collate_admittance
from eason.admittance_gain.gain_encoder import GainEncoder


def kl_divergence(mu, logvar):
    """KL(q(z|x) || p(z)) where p(z) = N(0, I)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, loader, optimizer, device, kl_weight):
    model.train()
    total_mse = 0.0
    total_kl = 0.0
    n = 0

    for graph, targets, _ in tqdm(loader, desc='  train', leave=False):
        graph = graph.to(device)
        targets = targets.to(device)

        z, mu, logvar, gain_pred = model(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )

        mse = F.mse_loss(gain_pred, targets)
        kl = kl_divergence(mu, logvar)
        loss = mse + kl_weight * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = targets.size(0)
        total_mse += mse.item() * bs
        total_kl += kl.item() * bs
        n += bs

    return total_mse / n, total_kl / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_kl = 0.0
    n = 0

    # Per-filter-type tracking
    filter_mse = {}
    filter_n = {}

    for graph, targets, ftypes in loader:
        graph = graph.to(device)
        targets = targets.to(device)

        z, mu, logvar, gain_pred = model(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )

        mse = F.mse_loss(gain_pred, targets, reduction='none')
        kl = kl_divergence(mu, logvar)

        bs = targets.size(0)
        total_mse += mse.sum().item()
        total_kl += kl.item() * bs
        n += bs

        # Per-filter-type
        for ft_idx in ftypes.unique():
            ft = ft_idx.item()
            mask = ftypes == ft_idx
            if ft not in filter_mse:
                filter_mse[ft] = 0.0
                filter_n[ft] = 0
            filter_mse[ft] += mse[mask].sum().item()
            filter_n[ft] += mask.sum().item()

    per_filter = {ft: filter_mse[ft] / filter_n[ft] for ft in filter_mse}
    return total_mse / n, total_kl / n, per_filter


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    lr = 1e-3
    num_epochs = 50
    kl_target = 0.001
    kl_warmup_epochs = 10

    print("=" * 60)
    print("Admittance-Based Gain Prediction Experiment")
    print("=" * 60)
    print(f"Device: {device}")

    # Load split (circuit-level indices)
    split = torch.load('rlc_dataset/stratified_split.pt')
    train_indices = split['train_indices']
    val_indices = split['val_indices']

    dataset_path = 'rlc_dataset/filter_dataset.pkl'
    train_dataset = AdmittanceDataset(dataset_path, train_indices)
    val_dataset = AdmittanceDataset(dataset_path, val_indices)

    print(f"Train samples: {len(train_dataset)} ({len(train_indices)} circuits x {train_dataset.num_freqs} freqs)")
    print(f"Val samples:   {len(val_dataset)} ({len(val_indices)} circuits x {val_dataset.num_freqs} freqs)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_admittance, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_admittance, num_workers=0
    )

    # Compute mean-prediction baseline
    print("\nComputing mean-prediction baseline...")
    all_targets = []
    for _, targets, _ in DataLoader(val_dataset, batch_size=256, collate_fn=collate_admittance):
        all_targets.append(targets)
    all_targets = torch.cat(all_targets)
    mean_target = all_targets.mean()
    baseline_mse = F.mse_loss(mean_target.expand_as(all_targets), all_targets).item()
    print(f"Baseline MSE (predict mean): {baseline_mse:.4f}")

    # Model
    model = GainEncoder().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_mse = float('inf')
    ckpt_path = os.path.join(os.path.dirname(__file__), 'best.pt')

    filter_names = AdmittanceDataset.FILTER_TYPES

    print(f"\n{'Epoch':>5} {'Train MSE':>10} {'Val MSE':>10} {'KL':>8} {'kl_w':>6} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        kl_weight = min(kl_target, kl_target * epoch / kl_warmup_epochs)

        train_mse, train_kl = train_epoch(model, train_loader, optimizer, device, kl_weight)
        val_mse, val_kl, per_filter = validate(model, val_loader, device)

        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:5d} {train_mse:10.4f} {val_mse:10.4f} {val_kl:8.4f} {kl_weight:6.4f} {current_lr:10.6f}", end="")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_mse,
                'baseline_mse': baseline_mse,
            }, ckpt_path)
            print(" *", end="")

        print()

        # Per-filter breakdown every 10 epochs
        if epoch % 10 == 0:
            print("  Per-filter MSE:")
            for ft_idx in sorted(per_filter.keys()):
                name = filter_names[ft_idx]
                print(f"    {name:<15}: {per_filter[ft_idx]:.4f}")

    print(f"\nDone. Best val MSE: {best_val_mse:.4f} (baseline: {baseline_mse:.4f})")
    print(f"Ratio: {best_val_mse / baseline_mse:.3f}x baseline")


if __name__ == '__main__':
    main()
