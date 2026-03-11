"""
Training script for auxiliary pole prediction + filter classification.

Architecture:
    HierarchicalEncoder (frozen backbone optional) -> z [B, 8]
        -> RegressionMLP    -> [pole_real, pole_imag]   (MSE loss)
        -> ClassificationMLP -> filter type logits [B,8] (CrossEntropy loss)

Loss:
    total_loss = kl_loss * kl_weight + regression_loss + classification_loss

KL warmup over 10 epochs to target weight 0.001.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from ml.models.encoder import HierarchicalEncoder
from yubo.auxiliary_heads import RegressionMLP, ClassificationMLP
from yubo.dataset import YuboDataset, collate_yubo


def kl_divergence(mu, logvar):
    """KL(q(z|x) || p(z)) where p(z) = N(0, I).
    Sum over latent dims, mean over batch — standard VAE formulation.
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def train_epoch(encoder, reg_head, cls_head, loader, optimizer, device, kl_weight):
    encoder.train()
    reg_head.train()
    cls_head.train()

    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    total_kl = 0.0
    n = 0

    for graph, pole_real, pole_imag, labels in tqdm(loader, desc='  train', leave=False):
        graph = graph.to(device)
        pole_real = pole_real.to(device)
        pole_imag = pole_imag.to(device)
        labels = labels.to(device)

        z, mu, logvar = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        pole_pred = reg_head(mu)                  # [B, 2] — use mean, not noisy sample
        pole_target = torch.stack([pole_real, pole_imag], dim=1)  # [B, 2]
        reg_loss = F.mse_loss(pole_pred, pole_target)

        cls_logits = cls_head(mu)                 # [B, 8] — use mean, not noisy sample
        cls_loss = F.cross_entropy(cls_logits, labels)

        kl = kl_divergence(mu, logvar)
        loss = kl_weight * kl + reg_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(reg_head.parameters()) + list(cls_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        bs = pole_real.size(0)
        total_loss += loss.item() * bs
        total_reg += reg_loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_kl += kl.item() * bs
        n += bs

    return total_loss / n, total_reg / n, total_cls / n, total_kl / n


@torch.no_grad()
def validate(encoder, reg_head, cls_head, loader, device, kl_weight):
    encoder.eval()
    reg_head.eval()
    cls_head.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    total_kl = 0.0
    n = 0

    for graph, pole_real, pole_imag, labels in loader:
        graph = graph.to(device)
        pole_real = pole_real.to(device)
        pole_imag = pole_imag.to(device)
        labels = labels.to(device)

        z, mu, logvar = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        pole_pred = reg_head(mu)                  # [B, 2] — use mean for stable eval
        pole_target = torch.stack([pole_real, pole_imag], dim=1)
        reg_loss = F.mse_loss(pole_pred, pole_target)

        cls_logits = cls_head(mu)                 # [B, 8] — use mean for stable eval
        cls_loss = F.cross_entropy(cls_logits, labels)

        kl = kl_divergence(mu, logvar)
        loss = kl_weight * kl + reg_loss + cls_loss

        bs = pole_real.size(0)
        total_loss += loss.item() * bs
        total_reg += reg_loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_kl += kl.item() * bs
        n += bs

    return total_loss / n, total_reg / n, total_cls / n, total_kl / n


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
    print("Yubo: Auxiliary Pole Prediction + Filter Classification")
    print("=" * 60)
    print(f"Device: {device}")

    dataset_path = 'rlc_dataset/filter_dataset.pkl'

    # Single dataset load — build full dataset, split, then use Subset views
    # (avoids the 3× pkl reload that separate YuboDataset instances would cause)
    full_dataset = YuboDataset(dataset_path, [])
    full_dataset.circuit_indices = list(range(len(full_dataset._base)))
    train_indices, val_indices = full_dataset.get_train_val_split(train_ratio=0.8, seed=seed)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    print(f"Train circuits: {len(train_dataset)}")
    print(f"Val circuits:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_yubo, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_yubo, num_workers=0
    )

    # Models
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        gnn_hidden_dim=64,
        gnn_num_layers=3,
        latent_dim=8,
        dropout=0.1
    ).to(device)

    reg_head = RegressionMLP(latent_dim=8).to(device)
    cls_head = ClassificationMLP(latent_dim=8, num_classes=8).to(device)

    num_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in reg_head.parameters()) +
        sum(p.numel() for p in cls_head.parameters())
    )
    print(f"Total parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(reg_head.parameters()) + list(cls_head.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    ckpt_path = os.path.join(os.path.dirname(__file__), 'best.pt')

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Reg Loss':>10} {'Cls Loss':>10} {'KL':>8} {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        kl_weight = min(kl_target, kl_target * epoch / kl_warmup_epochs)

        train_loss, train_reg, train_cls, train_kl = train_epoch(
            encoder, reg_head, cls_head, train_loader, optimizer, device, kl_weight
        )
        val_loss, val_reg, val_cls, val_kl = validate(
            encoder, reg_head, cls_head, val_loader, device, kl_weight
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"{epoch:5d} {train_loss:11.4f} {val_loss:10.4f} "
            f"{val_reg:10.4f} {val_cls:10.4f} {val_kl:8.4f} {current_lr:10.6f}",
            end=""
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'reg_head_state_dict': reg_head.state_dict(),
                'cls_head_state_dict': cls_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'regression_loss': val_reg,
                'classification_loss': val_cls,
            }, ckpt_path)
            print(" *", end="")

        print()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
