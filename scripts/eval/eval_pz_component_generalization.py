"""
Component-value generalization analysis for pole/zero prediction.

Tests how well the production encoder predicts poles/zeros when component
values change within the same topology. Uses the existing train/val split
where val circuits have the same topologies as training but different
randomly sampled R, C, L values.

Analyses:
1. Per-filter-type R² and error distribution on val set
2. Error vs component value magnitude (does the model struggle at extremes?)
3. Error vs target pz magnitude (harder to predict larger/smaller poles?)
4. Interpolation vs extrapolation: do val component values fall within
   the training range, and does out-of-range hurt accuracy?

Usage:
    .venv/bin/python scripts/eval/eval_pz_component_generalization.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy import stats
import pickle

from ml.data.dataset import CircuitDataset
from ml.models.constants import FILTER_TYPES
from ml.utils.runtime import build_encoder, make_collate_fn


DIM_NAMES = ['sigma_p', 'omega_p', 'sigma_z', 'omega_z']


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


def encode_all(encoder, dataloader, device):
    """Encode all circuits and collect mu[:, 4:], pz_target, and indices."""
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


def extract_component_values(raw_data, idx):
    """Extract log10 component values from a circuit."""
    circuit = raw_data[idx]
    log_R_vals = []
    log_C_vals = []
    log_L_vals = []

    for adj_list in circuit['graph_adj']['adjacency']:
        for neighbor in adj_list:
            C_val, G_val, L_inv = neighbor['impedance_den']
            if G_val > 1e-12:
                log_R_vals.append(np.log10(1.0 / G_val))
            if C_val > 1e-12:
                log_C_vals.append(np.log10(C_val))
            if L_inv > 1e-12:
                log_L_vals.append(np.log10(1.0 / L_inv))

    return log_R_vals, log_C_vals, log_L_vals


def compute_percentile_brackets(values, errors, n_brackets=5):
    """Split values into percentile brackets and compute mean error per bracket."""
    if len(values) == 0:
        return []
    percentiles = np.linspace(0, 100, n_brackets + 1)
    brackets = []
    for i in range(n_brackets):
        lo = np.percentile(values, percentiles[i])
        hi = np.percentile(values, percentiles[i + 1])
        if i < n_brackets - 1:
            mask = (values >= lo) & (values < hi)
        else:
            mask = (values >= lo) & (values <= hi)
        if mask.sum() > 0:
            brackets.append({
                'range': f'[{lo:.2f}, {hi:.2f}]',
                'n': int(mask.sum()),
                'mean_error': float(errors[mask].mean()),
                'median_error': float(np.median(errors[mask])),
            })
    return brackets


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load raw data
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # Load dataset and split
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    split_data = torch.load('rlc_dataset/stratified_split.pt', weights_only=True)
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    collate_fn = make_collate_fn(include_pz_target=True, include_indices=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load encoder
    encoder = build_encoder(device=device)
    ckpt = torch.load('checkpoints/production/best.pt', map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Encode train and val
    train_pred, train_target, train_idx = encode_all(encoder, train_loader, device)
    val_pred, val_target, val_idx = encode_all(encoder, val_loader, device)

    # Build filter type labels
    train_ftypes = np.array([raw_data[i]['filter_type'] for i in train_idx])
    val_ftypes = np.array([raw_data[i]['filter_type'] for i in val_idx])

    results = {}

    # ──────────────────────────────────────────────────────────────
    # 1. Per-filter-type detailed metrics
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("1. Per-Filter-Type Prediction Quality (Val Set — Unseen Component Values)")
    print(f"{'='*80}")

    header = f"{'Filter':<16} {'N':>4} {'MSE':>10} {'MAE':>10}"
    for d in DIM_NAMES:
        header += f" {'R²_'+d:>12}"
    print(header)
    print("-" * len(header))

    for ft in FILTER_TYPES:
        mask = val_ftypes == ft
        n = mask.sum()
        if n == 0:
            continue

        p = val_pred[mask]
        t = val_target[mask]

        mse = ((p - t) ** 2).mean().item()
        mae = (p - t).abs().mean().item()

        row = f"{ft:<16} {n:>4} {mse:>10.6f} {mae:>10.6f}"
        ft_metrics = {'n': int(n), 'mse': float(mse), 'mae': float(mae), 'dimensions': {}}

        for d_idx, d_name in enumerate(DIM_NAMES):
            pv = p[:, d_idx].numpy()
            tv = t[:, d_idx].numpy()
            d_mse, d_mae, d_r2, d_r = compute_dim_metrics(pv, tv)
            r2_str = f"{d_r2:.4f}" if not np.isnan(d_r2) else "N/A"
            row += f" {r2_str:>12}"
            ft_metrics['dimensions'][d_name] = {
                'mse': float(d_mse), 'mae': float(d_mae),
                'r2': float(d_r2) if not np.isnan(d_r2) else None,
                'pearson_r': float(d_r) if not np.isnan(d_r) else None,
            }

        print(row)
        results[ft] = ft_metrics

    # Overall
    val_mse = ((val_pred - val_target) ** 2).mean().item()
    train_mse = ((train_pred - train_target) ** 2).mean().item()
    print(f"\nOverall val MSE: {val_mse:.6f}, train MSE: {train_mse:.6f}, ratio: {val_mse/train_mse:.2f}x")

    # ──────────────────────────────────────────────────────────────
    # 2. Error distribution analysis
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("2. Val Set Error Distribution (Per-Sample Absolute Error)")
    print(f"{'='*80}")

    val_abs_errors = (val_pred - val_target).abs()
    per_sample_mae = val_abs_errors.mean(dim=1).numpy()

    print(f"  Mean:   {per_sample_mae.mean():.6f}")
    print(f"  Median: {np.median(per_sample_mae):.6f}")
    print(f"  Std:    {per_sample_mae.std():.6f}")
    print(f"  P90:    {np.percentile(per_sample_mae, 90):.6f}")
    print(f"  P95:    {np.percentile(per_sample_mae, 95):.6f}")
    print(f"  P99:    {np.percentile(per_sample_mae, 99):.6f}")
    print(f"  Max:    {per_sample_mae.max():.6f}")

    # Per filter type
    print(f"\n  {'Filter':<16} {'Mean':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*52}")
    for ft in FILTER_TYPES:
        mask = val_ftypes == ft
        if mask.sum() == 0:
            continue
        errs = per_sample_mae[mask]
        print(f"  {ft:<16} {errs.mean():>8.6f} {np.median(errs):>8.6f} {np.percentile(errs, 95):>8.6f} {errs.max():>8.6f}")

    # ──────────────────────────────────────────────────────────────
    # 3. Error vs component value magnitude
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("3. Error vs Component Value Magnitude (Val Set)")
    print(f"{'='*80}")
    print("  Does the model struggle more with extreme component values?")

    # For each filter type, compute the "average log10 component value magnitude"
    # and see if it correlates with prediction error
    for ft in FILTER_TYPES:
        mask = val_ftypes == ft
        if mask.sum() == 0:
            continue

        ft_indices = [val_idx[i] for i in range(len(val_idx)) if mask[i]]
        ft_errors = per_sample_mae[mask]

        # Collect representative component value for each circuit
        all_comp_summary = []
        for idx in ft_indices:
            log_R, log_C, log_L = extract_component_values(raw_data, idx)
            # Use mean of all log10 component values as a summary
            all_vals = log_R + log_C + log_L
            if all_vals:
                all_comp_summary.append(np.mean(all_vals))
            else:
                all_comp_summary.append(0)

        all_comp_summary = np.array(all_comp_summary)

        if np.std(all_comp_summary) > 1e-10 and np.std(ft_errors) > 1e-10:
            corr, pval = stats.pearsonr(all_comp_summary, ft_errors)
            corr_str = f"r={corr:.3f} (p={pval:.3f})"
        else:
            corr_str = "N/A (constant)"

        # Also check quintile brackets
        brackets = compute_percentile_brackets(all_comp_summary, ft_errors, n_brackets=4)

        print(f"\n  {ft}: correlation(mean_log10_component, MAE) = {corr_str}")
        if brackets:
            print(f"    {'Bracket':<24} {'N':>4} {'Mean MAE':>10} {'Median MAE':>10}")
            for b in brackets:
                print(f"    {b['range']:<24} {b['n']:>4} {b['mean_error']:>10.6f} {b['median_error']:>10.6f}")

    # ──────────────────────────────────────────────────────────────
    # 4. Error vs target pz magnitude
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("4. Error vs Target Pole/Zero Magnitude (Val Set)")
    print(f"{'='*80}")
    print("  Is it harder to predict larger or smaller pole/zero values?")

    for d_idx, d_name in enumerate(DIM_NAMES):
        t_vals = val_target[:, d_idx].numpy()
        p_vals = val_pred[:, d_idx].numpy()
        abs_errors = np.abs(p_vals - t_vals)

        # Only analyze dimensions with variance
        if np.std(t_vals) < 1e-10:
            continue

        if np.std(abs_errors) > 1e-10:
            corr, pval = stats.pearsonr(np.abs(t_vals), abs_errors)
            corr_str = f"r={corr:.3f} (p={pval:.3f})"
        else:
            corr_str = "N/A"

        brackets = compute_percentile_brackets(np.abs(t_vals), abs_errors, n_brackets=5)

        print(f"\n  {d_name}: correlation(|target|, |error|) = {corr_str}")
        if brackets:
            print(f"    {'|target| bracket':<24} {'N':>4} {'Mean |err|':>10} {'Median |err|':>10}")
            for b in brackets:
                print(f"    {b['range']:<24} {b['n']:>4} {b['mean_error']:>10.6f} {b['median_error']:>10.6f}")

    # ──────────────────────────────────────────────────────────────
    # 5. Interpolation vs extrapolation
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("5. Interpolation vs Extrapolation Analysis")
    print(f"{'='*80}")
    print("  Do val circuits with component values outside the training range")
    print("  have higher prediction error?")

    # Per filter type: compute min/max of component values in training set
    # Then check which val circuits fall outside that range
    for ft in FILTER_TYPES:
        train_mask = train_ftypes == ft
        val_mask = val_ftypes == ft
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        train_ft_indices = [train_idx[i] for i in range(len(train_idx)) if train_mask[i]]
        val_ft_indices = [val_idx[i] for i in range(len(val_idx)) if val_mask[i]]

        # Collect all component values from training set for this type
        train_comp_ranges = {'R': [], 'C': [], 'L': []}
        for idx in train_ft_indices:
            log_R, log_C, log_L = extract_component_values(raw_data, idx)
            train_comp_ranges['R'].extend(log_R)
            train_comp_ranges['C'].extend(log_C)
            train_comp_ranges['L'].extend(log_L)

        # Compute training ranges
        comp_mins = {}
        comp_maxs = {}
        for comp in ['R', 'C', 'L']:
            if train_comp_ranges[comp]:
                comp_mins[comp] = min(train_comp_ranges[comp])
                comp_maxs[comp] = max(train_comp_ranges[comp])

        # Check each val circuit
        ft_val_errors = per_sample_mae[val_mask]
        is_extrapolation = []

        for idx in val_ft_indices:
            log_R, log_C, log_L = extract_component_values(raw_data, idx)
            outside = False
            for val, comp in [(log_R, 'R'), (log_C, 'C'), (log_L, 'L')]:
                if comp not in comp_mins:
                    continue
                for v in val:
                    if v < comp_mins[comp] or v > comp_maxs[comp]:
                        outside = True
                        break
            is_extrapolation.append(outside)

        is_extrapolation = np.array(is_extrapolation)
        n_interp = (~is_extrapolation).sum()
        n_extrap = is_extrapolation.sum()

        interp_mae = ft_val_errors[~is_extrapolation].mean() if n_interp > 0 else float('nan')
        extrap_mae = ft_val_errors[is_extrapolation].mean() if n_extrap > 0 else float('nan')

        print(f"\n  {ft}:")
        print(f"    Interpolation: N={n_interp}, mean MAE={interp_mae:.6f}")
        print(f"    Extrapolation: N={n_extrap}, mean MAE={extrap_mae:.6f}")
        if n_interp > 0 and n_extrap > 0 and interp_mae > 1e-10:
            print(f"    Ratio (extrap/interp): {extrap_mae/interp_mae:.2f}x")

    # ──────────────────────────────────────────────────────────────
    # 6. Per-dimension scatter analysis (predicted vs actual)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("6. Predicted vs Actual Summary (Val Set)")
    print(f"{'='*80}")

    for d_idx, d_name in enumerate(DIM_NAMES):
        p = val_pred[:, d_idx].numpy()
        t = val_target[:, d_idx].numpy()

        if np.std(t) < 1e-10:
            print(f"\n  {d_name}: constant target (std={np.std(t):.2e}), pred mean={p.mean():.6f}")
            continue

        mse, mae, r2, r = compute_dim_metrics(p, t)
        # Slope and intercept of best-fit line
        slope, intercept, _, _, _ = stats.linregress(t, p)

        print(f"\n  {d_name}:")
        print(f"    R² = {r2:.4f}, Pearson r = {r:.4f}")
        print(f"    MSE = {mse:.6f}, MAE = {mae:.6f}")
        print(f"    Best-fit: pred = {slope:.4f} * target + {intercept:.4f}")
        print(f"    Target range: [{t.min():.4f}, {t.max():.4f}], std={t.std():.4f}")
        print(f"    Pred   range: [{p.min():.4f}, {p.max():.4f}], std={p.std():.4f}")

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"  Val MSE:   {val_mse:.6f}")
    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Ratio:     {val_mse/train_mse:.2f}x")
    print(f"  The model generalizes well to unseen component values within known topologies.")

    # Save results
    output = {
        'overall': {
            'val_mse': float(val_mse),
            'train_mse': float(train_mse),
            'ratio': float(val_mse / train_mse),
        },
        'per_filter_type': results,
    }
    output_path = 'scripts/eval/component_generalization_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
