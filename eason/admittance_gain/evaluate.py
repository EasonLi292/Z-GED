"""
Evaluate the trained admittance gain model and produce a detailed report.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from eason.admittance_gain.dataset import AdmittanceDataset, collate_admittance
from eason.admittance_gain.gain_encoder import GainEncoder


def main():
    device = 'cpu'

    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Best val MSE: {ckpt['val_mse']:.6f}")
    print(f"Baseline MSE: {ckpt['baseline_mse']:.6f}")
    print(f"Ratio: {ckpt['val_mse'] / ckpt['baseline_mse']:.4f}x baseline")

    model = GainEncoder().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load val data
    split = torch.load('rlc_dataset/stratified_split.pt', weights_only=False)
    val_indices = split['val_indices']
    dataset_path = 'rlc_dataset/filter_dataset.pkl'
    val_dataset = AdmittanceDataset(dataset_path, val_indices)

    with open(dataset_path, 'rb') as f:
        all_circuits = pickle.load(f)

    filter_names = AdmittanceDataset.FILTER_TYPES

    # Collect all predictions
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_admittance)

    all_preds = []
    all_targets = []
    all_ftypes = []

    with torch.no_grad():
        for graph, targets, ftypes in val_loader:
            graph = graph.to(device)
            z, mu, logvar, gain_pred = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            all_preds.append(gain_pred)
            all_targets.append(targets)
            all_ftypes.append(ftypes)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_ftypes = torch.cat(all_ftypes)

    # Overall stats
    errors = (all_preds - all_targets)
    abs_errors = errors.abs()
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Val samples: {len(all_preds)}")
    print(f"MSE:  {F.mse_loss(all_preds, all_targets).item():.6f}")
    print(f"MAE:  {abs_errors.mean().item():.4f}")
    print(f"Median AE: {abs_errors.median().item():.4f}")
    print(f"Max AE: {abs_errors.max().item():.4f}")
    print(f"Target range: [{all_targets.min().item():.2f}, {all_targets.max().item():.2f}]")
    print(f"Pred range:   [{all_preds.min().item():.2f}, {all_preds.max().item():.2f}]")

    # Per-filter-type
    print(f"\n{'='*60}")
    print("PER-FILTER-TYPE MSE")
    print(f"{'='*60}")
    for ft_idx in range(len(filter_names)):
        mask = all_ftypes == ft_idx
        if mask.sum() == 0:
            continue
        mse = F.mse_loss(all_preds[mask], all_targets[mask]).item()
        mae = (all_preds[mask] - all_targets[mask]).abs().mean().item()
        print(f"  {filter_names[ft_idx]:<15}: MSE={mse:.4f}, MAE={mae:.4f}, n={mask.sum().item()}")

    # Concrete examples: pick 3 circuits of different types
    num_freqs = val_dataset.num_freqs
    print(f"\n{'='*60}")
    print("CONCRETE EXAMPLES (5 freq points each)")
    print(f"{'='*60}")

    # Pick one circuit per filter type (first val circuit of each type)
    shown_types = set()
    for local_idx, circuit_idx in enumerate(val_indices):
        circuit = all_circuits[circuit_idx]
        ft = circuit['filter_type']
        if ft in shown_types:
            continue
        shown_types.add(ft)

        fr = circuit['frequency_response']
        freqs = fr['freqs']

        # Get predictions for this circuit
        start = local_idx * num_freqs
        preds = all_preds[start:start + num_freqs]
        targets = all_targets[start:start + num_freqs]

        # Show components
        adj = circuit['graph_adj']['adjacency']
        components = []
        seen_edges = set()
        for src, neighbors in enumerate(adj):
            for nb in neighbors:
                tgt = nb['id']
                if (tgt, src) in seen_edges:
                    continue
                seen_edges.add((src, tgt))
                C, G, L_inv = nb['impedance_den']
                parts = []
                if G > 0:
                    parts.append(f"R={1/G:.1f}Ω")
                if C > 0:
                    parts.append(f"C={C:.2e}F")
                if L_inv > 0:
                    parts.append(f"L={1/L_inv:.2e}H")
                if parts:
                    components.append(f"  edge({src}->{tgt}): {', '.join(parts)}")

        print(f"\n--- {ft} (circuit #{circuit_idx}) ---")
        for c in components:
            print(c)

        # Sample 5 evenly-spaced freq points
        indices = np.linspace(0, num_freqs - 1, 5, dtype=int)
        print(f"  {'Freq (Hz)':>12} {'Target':>10} {'Predicted':>10} {'Error':>8}")
        for i in indices:
            print(f"  {freqs[i]:12.1f} {targets[i].item():10.4f} {preds[i].item():10.4f} {(preds[i]-targets[i]).item():8.4f}")

        if len(shown_types) >= 4:
            break

    # Error distribution
    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION")
    print(f"{'='*60}")
    for threshold in [0.1, 0.25, 0.5, 1.0, 2.0]:
        pct = (abs_errors < threshold).float().mean().item() * 100
        print(f"  |error| < {threshold:.2f}: {pct:.1f}%")

    # Correlation
    corr = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1].item()
    print(f"\nPearson correlation: {corr:.4f}")


if __name__ == '__main__':
    main()
