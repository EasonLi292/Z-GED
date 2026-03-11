"""
Training script for circuit generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

from ml.data.dataset import CircuitDataset
from ml.losses.circuit_loss import CircuitLoss
from ml.models.component_utils import masks_to_component_type
from ml.utils.runtime import (
    build_classification_mlp, build_decoder, build_encoder,
    build_regression_mlp, make_collate_fn,
)
from yubo.auxiliary_heads import ClassificationMLP, RegressionMLP


def graph_to_dense_format(graph, max_nodes=6):
    """Convert PyG graph to dense format for decoder."""
    batch_size = graph.batch.max().item() + 1
    device = graph.x.device

    # Initialize with MASK (4) for padding positions
    node_types = torch.full((batch_size, max_nodes), 4, dtype=torch.long, device=device)
    edge_existence = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    component_types = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)

    node_offset = 0
    for b in range(batch_size):
        mask = (graph.batch == b)
        num_nodes = mask.sum().item()

        # Node types
        graph_node_types = graph.x[mask].argmax(dim=-1)
        node_types[b, :num_nodes] = graph_node_types

        # Edges
        edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        graph_edges = graph.edge_index[:, edge_mask] - node_offset
        graph_edge_attr = graph.edge_attr[edge_mask]

        for idx, (src, dst) in enumerate(graph_edges.t()):
            edge_existence[b, src, dst] = 1.0

            # Component type from log10 values
            # Edge attr format: [log10(R), log10(C), log10(L)] where 0 = absent
            # masks_to_component_type expects: [mask_C, mask_G, mask_L]
            is_R = (graph_edge_attr[idx, 0].abs() > 0.01).float()
            is_C = (graph_edge_attr[idx, 1].abs() > 0.01).float()
            is_L = (graph_edge_attr[idx, 2].abs() > 0.01).float()
            masks = torch.stack([is_C, is_R, is_L])  # Reorder to [C, G, L]
            comp_type = masks_to_component_type(masks.unsqueeze(0))[0]
            component_types[b, src, dst] = comp_type

        node_offset += num_nodes

    return {
        'node_types': node_types,
        'edge_existence': edge_existence,
        'component_types': component_types,
    }


def train_epoch(encoder, decoder, reg_mlp, cls_mlp, dataloader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    reg_mlp.train()
    cls_mlp.train()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        graph = batch['graph'].to(device)
        pz_target = batch['pz_target'].to(device)           # [B, 4]
        filter_labels = batch['filter_type_label'].to(device)  # [B]

        # Encode
        z, mu, logvar = encoder(
            graph.x, graph.edge_index, graph.edge_attr,
            graph.batch
        )

        # Sample latent
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std

        # Auxiliary heads — use sampled z during training
        pole_pred = reg_mlp(latent)                          # [B, 2]
        reg_loss = F.mse_loss(pole_pred, pz_target[:, :2])

        cls_logits = cls_mlp(latent)                         # [B, 8]
        cls_loss = F.cross_entropy(cls_logits, filter_labels)

        targets = graph_to_dense_format(graph)

        # Unified edge-component target for teacher forcing (0=no edge, 1-7=type)
        target_edge_components = torch.where(
            targets['edge_existence'] > 0.5,
            targets['component_types'],
            torch.zeros_like(targets['component_types'])
        ).long()

        # Forward (latent only, no conditions)
        predictions = decoder(
            latent_code=latent,
            target_node_types=targets['node_types'],
            target_edges=target_edge_components
        )

        # Loss
        loss, metrics = loss_fn(
            predictions, targets,
            mu=mu, logvar=logvar,
        )
        loss = loss + 0.1 * reg_loss + 0.1 * cls_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()) +
            list(reg_mlp.parameters()) + list(cls_mlp.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        for key, value in metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'edge': f'{metrics["edge_exist_acc"]:.0f}%',
            'comp': f'{metrics["component_type_acc"]:.0f}%'
        })

    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    avg_metrics['total_loss'] = total_loss / num_batches
    return avg_metrics


def validate(encoder, decoder, reg_mlp, cls_mlp, dataloader, loss_fn, device):
    """Validate the model."""
    encoder.eval()
    decoder.eval()
    reg_mlp.eval()
    cls_mlp.eval()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            graph = batch['graph'].to(device)
            pz_target = batch['pz_target'].to(device)           # [B, 4]
            filter_labels = batch['filter_type_label'].to(device)  # [B]

            z, mu, logvar = encoder(
                graph.x, graph.edge_index, graph.edge_attr,
                graph.batch
            )

            latent = mu  # Use mean for validation

            # Auxiliary heads — use mu for stable evaluation
            pole_pred = reg_mlp(mu)                              # [B, 2]
            reg_loss = F.mse_loss(pole_pred, pz_target[:, :2])

            cls_logits = cls_mlp(mu)                             # [B, 8]
            cls_loss = F.cross_entropy(cls_logits, filter_labels)

            targets = graph_to_dense_format(graph)

            # Unified edge-component target for teacher forcing (0=no edge, 1-7=type)
            target_edge_components = torch.where(
                targets['edge_existence'] > 0.5,
                targets['component_types'],
                torch.zeros_like(targets['component_types'])
            ).long()

            predictions = decoder(
                latent_code=latent,
                target_node_types=targets['node_types'],
                target_edges=target_edge_components
            )

            loss, metrics = loss_fn(
                predictions, targets,
                mu=mu, logvar=logvar,
            )
            loss = loss + 0.1 * reg_loss + 0.1 * cls_loss

            total_loss += loss.item()
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0) + value
            num_batches += 1

    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    avg_metrics['total_loss'] = total_loss / num_batches
    return avg_metrics


def create_balanced_sampler(dataset_path, indices):
    """Create weighted sampler for balanced node count distribution."""
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)

    node_counts = []
    for idx in indices:
        circuit = raw_data[idx]
        nodes = set()
        for comp in circuit['components']:
            nodes.add(comp['node1'])
            nodes.add(comp['node2'])
        node_counts.append(len(nodes))

    count_freq = Counter(node_counts)
    total = len(node_counts)

    weights = [total / (len(count_freq) * count_freq[count]) for count in node_counts]

    print(f"Node count distribution: {dict(count_freq)}")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main():
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100

    print("="*60)
    print("Training Topology-Only Model")
    print("="*60)
    print(f"Device: {device}")
    print(f"Random seed: {seed}")

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    print(f"Dataset size: {len(dataset)}")

    # Load split
    split_data = torch.load('rlc_dataset/stratified_split.pt')
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_sampler = create_balanced_sampler('rlc_dataset/filter_dataset.pkl', train_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=make_collate_fn(include_pz_target=True, include_filter_type_label=True),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(include_pz_target=True, include_filter_type_label=True),
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create models
    encoder = build_encoder(device=device)
    decoder = build_decoder(device=device, max_nodes=10)
    reg_mlp = build_regression_mlp(device=device)
    cls_mlp = build_classification_mlp(device=device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Regression MLP params: {sum(p.numel() for p in reg_mlp.parameters()):,}")
    print(f"Classification MLP params: {sum(p.numel() for p in cls_mlp.parameters()):,}")

    # Loss function
    loss_fn = CircuitLoss(
        node_type_weight=1.0,
        node_count_weight=5.0,
        edge_component_weight=2.0,
        connectivity_weight=5.0,
        kl_weight=0.01,
        use_connectivity_loss=True,
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(reg_mlp.parameters()) + list(cls_mlp.parameters()),
        lr=learning_rate
    )

    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    best_val_loss = float('inf')
    checkpoint_dir = 'checkpoints/production'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # KL warmup - gradually increase to target weight over first 20 epochs
        target_kl_weight = 0.01
        loss_fn.kl_weight = min(target_kl_weight, target_kl_weight * epoch / 20)

        train_metrics = train_epoch(encoder, decoder, reg_mlp, cls_mlp, train_loader, loss_fn, optimizer, device, epoch)
        val_metrics = validate(encoder, decoder, reg_mlp, cls_mlp, val_loader, loss_fn, device)

        print(f"\nEpoch {epoch}")
        print(f"  Train - loss: {train_metrics['total_loss']:.4f}, "
              f"node_count: {train_metrics['node_count_acc']:.1f}%, "
              f"edge: {train_metrics['edge_exist_acc']:.1f}%, "
              f"comp: {train_metrics['component_type_acc']:.1f}%")
        print(f"  Val   - loss: {val_metrics['total_loss']:.4f}, "
              f"node_count: {val_metrics['node_count_acc']:.1f}%, "
              f"edge: {val_metrics['edge_exist_acc']:.1f}%, "
              f"comp: {val_metrics['component_type_acc']:.1f}%")

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'regression_mlp_state_dict': reg_mlp.state_dict(),
                'classification_mlp_state_dict': cls_mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best.pt')
            print(f"  ✓ Saved best model (loss: {best_val_loss:.4f})")

        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
            }
            torch.save(checkpoint, f'{checkpoint_dir}/epoch_{epoch}.pt')

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
