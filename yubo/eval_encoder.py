"""
Encoder quality evaluation: fit a RegressionMLP probe on frozen encoder
embeddings and measure how well Z predicts dominant pole location.

Steps
-----
1. Load HierarchicalEncoder from checkpoints/production/best.pt (frozen).
2. Run all 1920 circuits through encoder (deterministic: z = mu).
3. Fit RegressionMLP on the Z embeddings using an 80/20 train/val split.
4. Compute R² for pole_real and pole_imag on the val set.
5. Per filter-type mean absolute error on the full dataset.
6. Print a clean summary table.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np

from ml.models.encoder import HierarchicalEncoder
from ml.data.dataset import CircuitDataset
from ml.models.constants import FILTER_TYPES
from yubo.auxiliary_heads import RegressionMLP


# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = 'checkpoints/production/best.pt'
DATASET_PATH = 'rlc_dataset/filter_dataset.pkl'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE   = 64
PROBE_EPOCHS = 200
PROBE_LR     = 1e-3
SEED         = 42


# ── 1. Load encoder ─────────────────────────────────────────────────────────────
def load_encoder(checkpoint_path: str) -> HierarchicalEncoder:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder = HierarchicalEncoder()
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ── 2. Encode all circuits (z = mu, deterministic) ─────────────────────────────
@torch.no_grad()
def encode_dataset(encoder: HierarchicalEncoder, dataset: CircuitDataset) -> tuple:
    """
    Returns
    -------
    Z           : np.ndarray [N, 8]
    pole_real   : np.ndarray [N]   — signed-log dominant pole real part
    pole_imag   : np.ndarray [N]   — signed-log dominant pole imag part
    filter_ids  : np.ndarray [N]   — integer filter type index
    """
    encoder = encoder.to(DEVICE)

    all_z, all_pr, all_pi, all_ft = [], [], [], []

    # Process in batches using PyG batching
    indices = list(range(len(dataset)))
    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        items = [dataset[i] for i in batch_idx]

        graphs     = Batch.from_data_list([it['graph'] for it in items]).to(DEVICE)
        pz_targets = torch.stack([it['pz_target'] for it in items])          # [B, 4]
        ft_labels  = torch.stack([it['filter_type'] for it in items]).argmax(dim=1)  # [B]

        mu = encoder.encode_deterministic(
            graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        )  # [B, 8]

        all_z.append(mu.cpu().numpy())
        all_pr.append(pz_targets[:, 0].numpy())   # sigma_p (pole_real)
        all_pi.append(pz_targets[:, 1].numpy())   # omega_p (pole_imag)
        all_ft.append(ft_labels.numpy())

    Z          = np.concatenate(all_z,  axis=0)
    pole_real  = np.concatenate(all_pr, axis=0)
    pole_imag  = np.concatenate(all_pi, axis=0)
    filter_ids = np.concatenate(all_ft, axis=0)

    return Z, pole_real, pole_imag, filter_ids


# ── 3. Fit probe ────────────────────────────────────────────────────────────────
def fit_probe(
    Z_train: np.ndarray,
    y_train: np.ndarray,
) -> RegressionMLP:
    """Train RegressionMLP on (Z_train → y_train [B, 2])."""
    probe = RegressionMLP(latent_dim=8).to(DEVICE)
    opt   = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    X = torch.tensor(Z_train, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = F.mse_loss(probe(X), Y)
        loss.backward()
        opt.step()

    probe.eval()
    return probe


# ── 4. Metrics ──────────────────────────────────────────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    # 1. Load encoder
    print("Loading encoder …")
    encoder = load_encoder(CHECKPOINT)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params: {n_params:,}")

    # 2. Load dataset and encode
    print("\nLoading dataset …")
    dataset = CircuitDataset(DATASET_PATH, normalize_features=False)

    print(f"\nEncoding {len(dataset)} circuits …")
    Z, pole_real, pole_imag, filter_ids = encode_dataset(encoder, dataset)
    print(f"  Z shape: {Z.shape}")

    # 80/20 stratified split (same seed as training)
    rng = np.random.default_rng(SEED)
    train_mask = np.zeros(len(dataset), dtype=bool)
    for ft_idx in range(len(FILTER_TYPES)):
        idxs = np.where(filter_ids == ft_idx)[0]
        idxs = rng.permutation(idxs)
        n_train = int(len(idxs) * 0.8)
        train_mask[idxs[:n_train]] = True
    val_mask = ~train_mask

    Z_train, Z_val         = Z[train_mask], Z[val_mask]
    pr_train, pr_val       = pole_real[train_mask], pole_real[val_mask]
    pi_train, pi_val       = pole_imag[train_mask], pole_imag[val_mask]
    ft_train, ft_val       = filter_ids[train_mask], filter_ids[val_mask]

    print(f"  Train: {train_mask.sum()}   Val: {val_mask.sum()}")

    # 3. Fit probe
    print(f"\nFitting RegressionMLP probe ({PROBE_EPOCHS} epochs, lr={PROBE_LR}) …")
    y_train = np.stack([pr_train, pi_train], axis=1)  # [N_train, 2]
    probe   = fit_probe(Z_train, y_train)

    # 4. Predict on val set
    with torch.no_grad():
        Z_val_t  = torch.tensor(Z_val, dtype=torch.float32, device=DEVICE)
        pred_val = probe(Z_val_t).cpu().numpy()  # [N_val, 2]

    pred_pr_val = pred_val[:, 0]
    pred_pi_val = pred_val[:, 1]

    r2_pr = r2_score(pr_val, pred_pr_val)
    r2_pi = r2_score(pi_val, pred_pi_val)

    # Per filter-type MAE (full dataset, probe infers from Z)
    with torch.no_grad():
        Z_all_t  = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
        pred_all = probe(Z_all_t).cpu().numpy()  # [N, 2]

    # ── Print summary ──────────────────────────────────────────────────────────
    SEP = "─" * 62

    print(f"\n{'═' * 62}")
    print(f"  ENCODER QUALITY EVALUATION")
    print(f"{'═' * 62}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Dataset    : {len(dataset)} circuits   Val set: {val_mask.sum()}")
    print(f"  Probe      : RegressionMLP  ({PROBE_EPOCHS} epochs)")
    print(f"{'═' * 62}\n")

    print(f"  OVERALL R^2 (val set, n={val_mask.sum()})")
    print(f"  {SEP}")
    print(f"  {'Metric':<30} {'R^2':>8}")
    print(f"  {SEP}")
    print(f"  {'pole_real (σ_p, signed-log)':<30} {r2_pr:>8.4f}")
    print(f"  {'pole_imag (ω_p, signed-log)':<30} {r2_pi:>8.4f}")
    print(f"  {SEP}\n")

    print(f"  PER FILTER-TYPE MAE  (full dataset, signed-log scale)")
    print(f"  {SEP}")
    print(f"  {'Filter Type':<18} {'N':>5}  {'MAE pole_real':>14}  {'MAE pole_imag':>14}")
    print(f"  {SEP}")
    for ft_idx, ft_name in enumerate(FILTER_TYPES):
        mask = filter_ids == ft_idx
        if mask.sum() == 0:
            continue
        mae_pr = np.mean(np.abs(pred_all[mask, 0] - pole_real[mask]))
        mae_pi = np.mean(np.abs(pred_all[mask, 1] - pole_imag[mask]))
        print(f"  {ft_name:<18} {mask.sum():>5}  {mae_pr:>14.4f}  {mae_pi:>14.4f}")
    print(f"  {SEP}")

    # Overall MAE on full dataset
    mae_all_pr = np.mean(np.abs(pred_all[:, 0] - pole_real))
    mae_all_pi = np.mean(np.abs(pred_all[:, 1] - pole_imag))
    print(f"  {'ALL':<18} {len(dataset):>5}  {mae_all_pr:>14.4f}  {mae_all_pi:>14.4f}")
    print(f"  {SEP}\n")

    # Z-space statistics (sanity check)
    print(f"  LATENT Z STATISTICS (mu across all {len(dataset)} circuits)")
    print(f"  {SEP}")
    print(f"  {'Dim':<6} {'Branch':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {SEP}")
    branch_labels = ['topo'] * 2 + ['struct'] * 2 + ['pz'] * 4
    for d in range(Z.shape[1]):
        print(f"  {d:<6} {branch_labels[d]:<12} {Z[:, d].mean():>8.4f} "
              f"{Z[:, d].std():>8.4f} {Z[:, d].min():>8.4f} {Z[:, d].max():>8.4f}")
    print(f"  {SEP}")
    print()


if __name__ == '__main__':
    main()
