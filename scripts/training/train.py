"""
Training script for circuit generation with auxiliary spec prediction.

Includes spec prediction loss to prevent z[4:8] posterior collapse.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import SimplifiedCircuitDecoder
from ml.losses.gumbel_softmax_loss import GumbelSoftmaxCircuitLoss
from ml.models.gumbel_softmax_utils import masks_to_component_type
from torch_geometric.data import Batch


def collate_circuit_batch(batch_list):
    """Custom collate function for circuit graphs."""
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    specifications = torch.stack([item['specifications'] for item in batch_list])
    batched_graph = Batch.from_data_list(graphs)

    return {
        'graph': batched_graph,
        'poles': poles,
        'zeros': zeros,
        'specifications': specifications,  # [batch, 2] = [cutoff_hz, Q]
    }


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

            # Component type from masks
            # Edge attr format: [C_norm, G_norm, L_inv_norm, is_R, is_C, is_L, is_parallel]
            # masks_to_component_type expects: [mask_C, mask_G, mask_L] = [is_C, is_R, is_L]
            is_R = graph_edge_attr[idx, 3]
            is_C = graph_edge_attr[idx, 4]
            is_L = graph_edge_attr[idx, 5]
            masks = torch.stack([is_C, is_R, is_L])  # Reorder to [C, G, L]
            comp_type = masks_to_component_type(masks.unsqueeze(0))[0]
            component_types[b, src, dst] = comp_type

        node_offset += num_nodes

    return {
        'node_types': node_types,
        'edge_existence': edge_existence,
        'component_types': component_types,
    }


def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    # Set spec_predictor to train mode if it exists
    if hasattr(loss_fn, 'spec_predictor'):
        loss_fn.spec_predictor.train()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        graph = batch['graph'].to(device)
        poles_list = batch['poles']
        zeros_list = batch['zeros']
        specifications = batch['specifications'].to(device)  # [batch, 2]

        # Encode
        z, mu, logvar = encoder(
            graph.x, graph.edge_index, graph.edge_attr,
            graph.batch, poles_list, zeros_list
        )

        # Sample latent
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std

        targets = graph_to_dense_format(graph)

        # Forward (latent only, no conditions)
        predictions = decoder(
            latent_code=latent,
            target_node_types=targets['node_types']
        )

        # Loss (with spec prediction)
        loss, metrics = loss_fn(
            predictions, targets,
            mu=mu, logvar=logvar,
            latent=latent,
            target_specs=specifications
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        for key, value in metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value
        num_batches += 1

        # Show spec prediction metrics if available
        postfix = {
            'loss': f'{loss.item():.3f}',
            'edge': f'{metrics["edge_exist_acc"]:.0f}%',
            'comp': f'{metrics["component_type_acc"]:.0f}%'
        }
        if 'cutoff_error_decades' in metrics and metrics['cutoff_error_decades'] > 0:
            postfix['fc_err'] = f'{metrics["cutoff_error_decades"]:.2f}'
        pbar.set_postfix(postfix)

    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    avg_metrics['total_loss'] = total_loss / num_batches
    return avg_metrics


def validate(encoder, decoder, dataloader, loss_fn, device):
    """Validate the model."""
    encoder.eval()
    decoder.eval()

    # Set spec_predictor to eval mode if it exists
    if hasattr(loss_fn, 'spec_predictor'):
        loss_fn.spec_predictor.eval()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            graph = batch['graph'].to(device)
            poles_list = batch['poles']
            zeros_list = batch['zeros']
            specifications = batch['specifications'].to(device)

            z, mu, logvar = encoder(
                graph.x, graph.edge_index, graph.edge_attr,
                graph.batch, poles_list, zeros_list
            )

            latent = mu  # Use mean for validation
            targets = graph_to_dense_format(graph)

            predictions = decoder(
                latent_code=latent,
                target_node_types=targets['node_types']
            )

            loss, metrics = loss_fn(
                predictions, targets,
                mu=mu, logvar=logvar,
                latent=latent,
                target_specs=specifications
            )

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
    print("Training Simplified Topology-Only Model")
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
        sampler=train_sampler, collate_fn=collate_circuit_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_circuit_batch
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=7,
        gnn_hidden_dim=64,
        gnn_num_layers=3,
        latent_dim=8,
        topo_latent_dim=2,
        values_latent_dim=2,
        pz_latent_dim=4,
        dropout=0.1
    ).to(device)

    decoder = SimplifiedCircuitDecoder(
        latent_dim=8,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=10,
        dropout=0.1
    ).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    # Loss function with spec prediction
    # spec_weight forces z[4:8] to encode transfer function info
    loss_fn = GumbelSoftmaxCircuitLoss(
        node_type_weight=1.0,
        node_count_weight=5.0,
        edge_component_weight=2.0,
        connectivity_weight=5.0,
        kl_weight=0.01,  # Low KL for topology separation
        spec_weight=1.0,  # Spec prediction to prevent z[4:8] collapse
        use_connectivity_loss=True,
        use_spec_loss=True,
        pz_latent_start=4,
        pz_latent_dim=4
    )

    # Move spec_predictor to device
    if hasattr(loss_fn, 'spec_predictor'):
        loss_fn.spec_predictor = loss_fn.spec_predictor.to(device)
        spec_predictor_params = list(loss_fn.spec_predictor.parameters())
        print(f"SpecPredictor params: {sum(p.numel() for p in spec_predictor_params):,}")
    else:
        spec_predictor_params = []

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + spec_predictor_params,
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

        train_metrics = train_epoch(encoder, decoder, train_loader, loss_fn, optimizer, device, epoch)
        val_metrics = validate(encoder, decoder, val_loader, loss_fn, device)

        print(f"\nEpoch {epoch}")
        print(f"  Train - loss: {train_metrics['total_loss']:.4f}, "
              f"node_count: {train_metrics['node_count_acc']:.1f}%, "
              f"edge: {train_metrics['edge_exist_acc']:.1f}%, "
              f"comp: {train_metrics['component_type_acc']:.1f}%")
        if 'cutoff_error_decades' in train_metrics:
            print(f"          spec_loss: {train_metrics['loss_spec']:.4f}, "
                  f"fc_err: {train_metrics['cutoff_error_decades']:.3f} decades, "
                  f"Q_err: {train_metrics['q_error']:.3f}")
        print(f"  Val   - loss: {val_metrics['total_loss']:.4f}, "
              f"node_count: {val_metrics['node_count_acc']:.1f}%, "
              f"edge: {val_metrics['edge_exist_acc']:.1f}%, "
              f"comp: {val_metrics['component_type_acc']:.1f}%")
        if 'cutoff_error_decades' in val_metrics:
            print(f"          spec_loss: {val_metrics['loss_spec']:.4f}, "
                  f"fc_err: {val_metrics['cutoff_error_decades']:.3f} decades, "
                  f"Q_err: {val_metrics['q_error']:.3f}")

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }
            if hasattr(loss_fn, 'spec_predictor'):
                checkpoint['spec_predictor_state_dict'] = loss_fn.spec_predictor.state_dict()
            torch.save(checkpoint, f'{checkpoint_dir}/best.pt')
            print(f"  ✓ Saved best model (loss: {best_val_loss:.4f})")

        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
            }
            if hasattr(loss_fn, 'spec_predictor'):
                checkpoint['spec_predictor_state_dict'] = loss_fn.spec_predictor.state_dict()
            torch.save(checkpoint, f'{checkpoint_dir}/epoch_{epoch}.pt')

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
