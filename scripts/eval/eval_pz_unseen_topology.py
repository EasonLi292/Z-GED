"""
Leave-One-Topology-Out (LOTO) cross-validation for pole/zero prediction.

For each of 8 filter types, trains the full encoder+decoder pipeline on the
other 7 types, then evaluates pole/zero prediction (mu[:, 4:8] vs pz_target)
on the held-out type. This tests whether the encoder can predict poles/zeros
for topologies it has never seen during training.

Usage:
    .venv/bin/python scripts/eval/eval_pz_unseen_topology.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy import stats
import pickle

from ml.data.dataset import CircuitDataset
from ml.losses.circuit_loss import CircuitLoss
from ml.models.constants import FILTER_TYPES
from ml.utils.runtime import build_encoder, build_decoder, make_collate_fn
from scripts.training.train import graph_to_dense_format


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
    """Encode all circuits and collect mu[:, 4:], pz_target."""
    encoder.eval()
    all_mu_pz = []
    all_target = []

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

    return torch.cat(all_mu_pz, dim=0), torch.cat(all_target, dim=0)


def train_one_fold(train_indices, val_in_dist_indices, device, dataset,
                   num_epochs=80, batch_size=32, lr=1e-3, patience=20):
    """Train encoder+decoder on given indices, return best models."""
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_in_dist_indices)

    collate_fn = make_collate_fn(include_pz_target=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    encoder = build_encoder(device=device)
    decoder = build_decoder(device=device, max_nodes=10)

    loss_fn = CircuitLoss(
        node_type_weight=1.0,
        node_count_weight=5.0,
        edge_component_weight=2.0,
        connectivity_weight=5.0,
        kl_weight=0.01,
        pz_weight=5.0,
        use_connectivity_loss=True,
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )

    best_val_loss = float('inf')
    best_encoder_state = None
    best_decoder_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # KL warmup
        target_kl_weight = 0.01
        loss_fn.kl_weight = min(target_kl_weight, target_kl_weight * epoch / 20)

        # Train
        encoder.train()
        decoder.train()
        for batch in train_loader:
            graph = batch['graph'].to(device)
            pz_target = batch['pz_target'].to(device)

            z, mu, logvar = encoder(
                graph.x, graph.edge_index, graph.edge_attr, graph.batch
            )

            std = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(std) * std

            targets = graph_to_dense_format(graph)

            target_edge_components = torch.where(
                targets['edge_existence'] > 0.5,
                targets['component_types'],
                torch.zeros_like(targets['component_types'])
            ).long()

            predictions = decoder(
                latent_code=latent,
                target_node_types=targets['node_types'],
                target_edges=target_edge_components,
            )

            loss, _ = loss_fn(
                predictions, targets,
                mu=mu, logvar=logvar,
                pz_target=pz_target,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        # Validate
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                graph = batch['graph'].to(device)
                pz_target = batch['pz_target'].to(device)

                z, mu, logvar = encoder(
                    graph.x, graph.edge_index, graph.edge_attr, graph.batch
                )

                targets = graph_to_dense_format(graph)

                target_edge_components = torch.where(
                    targets['edge_existence'] > 0.5,
                    targets['component_types'],
                    torch.zeros_like(targets['component_types'])
                ).long()

                predictions = decoder(
                    latent_code=mu,
                    target_node_types=targets['node_types'],
                    target_edges=target_edge_components,
                )

                loss, _ = loss_fn(
                    predictions, targets,
                    mu=mu, logvar=logvar,
                    pz_target=pz_target,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_decoder_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"    Early stopping at epoch {epoch} (patience {patience})")
            break

    # Restore best
    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder, best_val_loss


def evaluate_fold(encoder, dataset, indices, device, batch_size=32):
    """Evaluate pz prediction on given indices, return per-dim metrics."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collate_fn(include_pz_target=True),
    )
    pred, target = encode_all(encoder, loader, device)

    overall_mse = ((pred - target) ** 2).mean().item()

    dim_metrics = {}
    for d in range(4):
        p = pred[:, d].numpy()
        t = target[:, d].numpy()
        mse, mae, r2, r = compute_dim_metrics(p, t)
        dim_metrics[DIM_NAMES[d]] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2) if not np.isnan(r2) else None,
            'pearson_r': float(r) if not np.isnan(r) else None,
            'target_std': float(t.std()),
        }

    return {
        'n': len(indices),
        'overall_mse': float(overall_mse),
        'dimensions': dim_metrics,
    }


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load raw data for filter type lookup
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # Build index: filter_type -> list of dataset indices
    type_to_indices = {ft: [] for ft in FILTER_TYPES}
    for idx, circuit in enumerate(raw_data):
        type_to_indices[circuit['filter_type']].append(idx)

    print(f"Total circuits: {len(raw_data)}")
    for ft in FILTER_TYPES:
        print(f"  {ft}: {len(type_to_indices[ft])}")

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

    results = {}
    all_unseen_mse = []
    all_in_dist_mse = []

    print(f"\n{'='*70}")
    print("Leave-One-Topology-Out (LOTO) Cross-Validation")
    print(f"{'='*70}")

    for fold_idx, held_out_type in enumerate(FILTER_TYPES):
        print(f"\n{'─'*70}")
        print(f"Fold {fold_idx+1}/8: Holding out '{held_out_type}'")
        print(f"{'─'*70}")

        # Test split: all circuits of the held-out type
        test_indices = type_to_indices[held_out_type]

        # Train types: the other 7
        train_types = [ft for ft in FILTER_TYPES if ft != held_out_type]

        # Collect all indices for training types
        all_train_type_indices = []
        for ft in train_types:
            all_train_type_indices.extend(type_to_indices[ft])

        # Split training types into train (80%) and in-distribution val (20%)
        rng = np.random.RandomState(seed)
        all_train_type_indices = np.array(all_train_type_indices)
        rng.shuffle(all_train_type_indices)

        n_total = len(all_train_type_indices)
        n_val = int(n_total * 0.2)
        val_in_dist_indices = all_train_type_indices[:n_val].tolist()
        train_indices = all_train_type_indices[n_val:].tolist()

        print(f"  Train: {len(train_indices)}, Val (in-dist): {len(val_in_dist_indices)}, Test (unseen): {len(test_indices)}")

        t0 = time.time()
        encoder, decoder, best_val_loss = train_one_fold(
            train_indices, val_in_dist_indices, device, dataset,
            num_epochs=80, batch_size=32, lr=1e-3, patience=20,
        )
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.0f}s, best val loss: {best_val_loss:.4f}")

        # Evaluate on unseen topology
        unseen_metrics = evaluate_fold(encoder, dataset, test_indices, device)
        print(f"  Unseen topology ({held_out_type}): MSE={unseen_metrics['overall_mse']:.6f}")
        for dim_name in DIM_NAMES:
            dm = unseen_metrics['dimensions'][dim_name]
            r2_str = f"{dm['r2']:.3f}" if dm['r2'] is not None else "N/A"
            print(f"    {dim_name}: MSE={dm['mse']:.6f}, R²={r2_str}, tgt_std={dm['target_std']:.4f}")

        # Evaluate on in-distribution val set
        in_dist_metrics = evaluate_fold(encoder, dataset, val_in_dist_indices, device)
        print(f"  In-distribution val: MSE={in_dist_metrics['overall_mse']:.6f}")

        results[held_out_type] = {
            'unseen': unseen_metrics,
            'in_dist': in_dist_metrics,
            'train_size': len(train_indices),
            'val_size': len(val_in_dist_indices),
            'test_size': len(test_indices),
            'best_val_loss': float(best_val_loss),
            'training_time_s': float(elapsed),
        }

        all_unseen_mse.append(unseen_metrics['overall_mse'])
        all_in_dist_mse.append(in_dist_metrics['overall_mse'])

    # Summary table
    print(f"\n{'='*70}")
    print("LOTO Results Summary")
    print(f"{'='*70}")

    header = f"{'Held-out type':<16} {'Unseen MSE':>10} {'InDist MSE':>10}"
    for dim_name in DIM_NAMES:
        header += f" {'R²_'+dim_name:>10}"
    print(header)
    print("-" * len(header))

    for ft in FILTER_TYPES:
        r = results[ft]
        row = f"{ft:<16} {r['unseen']['overall_mse']:>10.6f} {r['in_dist']['overall_mse']:>10.6f}"
        for dim_name in DIM_NAMES:
            r2 = r['unseen']['dimensions'][dim_name]['r2']
            row += f" {r2:>10.3f}" if r2 is not None else f" {'N/A':>10}"
        print(row)

    avg_unseen = np.mean(all_unseen_mse)
    avg_in_dist = np.mean(all_in_dist_mse)
    print("-" * len(header))
    print(f"{'Average':<16} {avg_unseen:>10.6f} {avg_in_dist:>10.6f}")
    print(f"\nUnseen/InDist MSE ratio: {avg_unseen / avg_in_dist:.2f}x")

    # Save results
    output_path = 'scripts/eval/loto_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
