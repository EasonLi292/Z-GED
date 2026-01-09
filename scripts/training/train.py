"""
Training script for latent-guided circuit generation model.

This script trains the circuit generation architecture with joint edge-component prediction.

Key features:
1. Unified edge-component head (class 0=no edge, 1-7=component type)
2. Latent-guided decoder with hierarchical latent space (8D = 2D topology + 2D values + 4D TF)
3. Gumbel-Softmax for discrete component selection (R, C, L, RC, RL, CL, RCL)
4. Cross-attention to latent components for context-aware generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pickle
from tqdm import tqdm

from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import LatentGuidedGraphGPTDecoder
from ml.losses.gumbel_softmax_loss import GumbelSoftmaxCircuitLoss
from ml.models.gumbel_softmax_utils import masks_to_component_type
from torch_geometric.data import Batch


def collate_circuit_batch(batch_list):
    """Custom collate function for circuit graphs."""
    import numpy as np

    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)

    # Extract and normalize specifications
    batch_size = len(batch_list)
    specifications = torch.zeros(batch_size, 2, dtype=torch.float32)

    for b, item in enumerate(batch_list):
        cutoff_freq = item['specifications'][0].item()
        q_factor = item['specifications'][1].item()

        # Normalize: log10(freq)/4.0, log10(Q)/2.0
        specifications[b, 0] = np.log10(max(cutoff_freq, 1.0)) / 4.0
        specifications[b, 1] = np.log10(max(q_factor, 0.01)) / 2.0

    return {
        'graph': batched_graph,
        'poles': poles,
        'zeros': zeros,
        'specifications': specifications
    }


def graph_to_dense_format(graph, max_nodes=None):
    """
    Convert PyG graph to dense format for decoder.

    Args:
        graph: Batched PyG graph
        max_nodes: Maximum nodes to pad to. If None, uses FIXED padding for proper stop learning.

    Returns:
        Dictionary with dense tensors padded to max_nodes
    """
    batch_size = graph.batch.max().item() + 1
    device = graph.x.device

    # FIXED PADDING: Always pad to a consistent max to enable stop criterion learning
    # This ensures every sample sees positions BEYOND its actual node count,
    # so the model can learn where to stop generating.
    # Without this, a batch of 3-node circuits would only see [0,1,2] positions
    # with stop_targets=[0,0,0] (never stop!) - causing mode collapse.
    if max_nodes is None:
        # Use fixed max = 6 (one more than largest circuits in dataset: 5)
        # This creates clear stopping positions:
        #   3-node: stop_targets = [0,0,0,1,1,1] -> learn to stop at position 3
        #   4-node: stop_targets = [0,0,0,0,1,1] -> learn to stop at position 4
        #   5-node: stop_targets = [0,0,0,0,0,1] -> learn to stop at position 5
        max_nodes = 6

    # Initialize dense tensors
    # CRITICAL: Initialize node_types to MASK (4) not GND (0)
    # Unfilled positions should be padding, not valid nodes!
    node_types = torch.full((batch_size, max_nodes), 4, dtype=torch.long, device=device)  # MASK = 4
    edge_existence = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    component_types = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)
    component_values = torch.zeros(batch_size, max_nodes, max_nodes, 3, device=device)
    is_parallel = torch.zeros(batch_size, max_nodes, max_nodes, device=device)

    # Process each graph in batch
    node_offset = 0
    for b in range(batch_size):
        # Get nodes for this graph
        mask = (graph.batch == b)
        num_nodes = mask.sum().item()

        # Node types
        graph_node_types = graph.x[mask].argmax(dim=-1)
        node_types[b, :num_nodes] = graph_node_types

        # Edges for this graph
        edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        graph_edges = graph.edge_index[:, edge_mask] - node_offset
        graph_edge_attr = graph.edge_attr[edge_mask]

        # Build edge existence matrix and extract attributes
        for idx, (src, dst) in enumerate(graph_edges.t()):
            edge_existence[b, src, dst] = 1.0

            # Extract edge attributes
            edge_vals = graph_edge_attr[idx]  # [7]

            # Component type from masks
            masks = edge_vals[3:6]  # [mask_C, mask_G, mask_L]
            comp_type = masks_to_component_type(masks.unsqueeze(0))[0]
            component_types[b, src, dst] = comp_type

            # Continuous values
            component_values[b, src, dst] = edge_vals[:3]  # [log(C), G, log(L_inv)]

            # Is parallel
            is_parallel[b, src, dst] = edge_vals[6]

        node_offset += num_nodes

    return {
        'node_types': node_types,
        'edge_existence': edge_existence,
        'component_types': component_types,
        'component_values': component_values,
        'is_parallel': is_parallel
    }


def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        graph = batch['graph'].to(device)
        poles_list = batch['poles']
        zeros_list = batch['zeros']

        # Forward pass through encoder
        z, mu, logvar = encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            poles_list,
            zeros_list
        )

        # Sample latent (reparameterization trick)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        # Use real circuit specifications from batch
        conditions = batch['specifications'].to(device)

        # Convert graph to dense format for decoder targets (VARIABLE LENGTH!)
        # Pass max_nodes=None to use actual max in batch, not fixed limit
        targets = graph_to_dense_format(graph, max_nodes=None)

        # Forward pass through decoder (now returns edge_component_logits)
        predictions = decoder(
            latent_code=latent,
            conditions=conditions,
            target_node_types=targets['node_types'],
            target_edges=targets['edge_existence']
        )

        # Compute loss (joint edge-component prediction with VAE regularization)
        loss, metrics = loss_fn(predictions, targets, mu=mu, logvar=logvar)

        # Monitor logit magnitudes (detect numerical issues early)
        if 'edge_component_logits' in predictions:
            logits = predictions['edge_component_logits']
            logit_max = logits.abs().max().item()
            metrics['logit_max'] = logit_max

            # Warn if logits are exploding
            if logit_max > 20 and num_batches == 0:
                print(f"\n⚠️  WARNING: Logits are large ({logit_max:.1f}). May indicate numerical instability.")

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for key, value in metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'comp_acc': f'{metrics.get("component_type_acc", 0):.1f}%'
        })

    # Compute averages
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    avg_metrics['total_loss'] = total_loss / num_batches

    return avg_metrics


def validate(encoder, decoder, dataloader, loss_fn, device):
    """Validate the model."""
    encoder.eval()
    decoder.eval()

    total_loss = 0
    metrics_sum = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            graph = batch['graph'].to(device)
            poles_list = batch['poles']
            zeros_list = batch['zeros']

            # Forward pass through encoder
            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles_list,
                zeros_list
            )

            # Use mean for validation (no sampling)
            latent = mu

            # Use real circuit specifications from batch
            conditions = batch['specifications'].to(device)

            # Convert graph to dense format (VARIABLE LENGTH!)
            targets = graph_to_dense_format(graph, max_nodes=None)

            # Forward pass through decoder
            predictions = decoder(
                latent_code=latent,
                conditions=conditions,
                target_node_types=targets['node_types'],
                target_edges=targets['edge_existence']
            )

            # Compute loss (with VAE regularization)
            loss, metrics = loss_fn(predictions, targets, mu=mu, logvar=logvar)

            # Monitor logit magnitudes (detect numerical issues early)
            if 'edge_component_logits' in predictions:
                logits = predictions['edge_component_logits']
                logit_max = logits.abs().max().item()
                metrics['logit_max'] = logit_max

            # Accumulate metrics
            total_loss += loss.item()
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0) + value
            num_batches += 1

    # Compute averages
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    avg_metrics['total_loss'] = total_loss / num_batches

    return avg_metrics


def get_kl_weight(epoch, warmup_epochs=20, target_weight=0.005):
    """KL warm-up schedule (Section 3.1 of Generation Plan).

    Gradually increase KL weight to keep latent active during early training.
    This prevents posterior collapse where the decoder ignores the latent.
    """
    if epoch < warmup_epochs:
        # Linear warm-up
        return target_weight * (epoch / warmup_epochs)
    return target_weight


def create_balanced_sampler(dataset_path, indices):
    """Create a weighted sampler for balanced node count distribution (Section 3.3).

    Ensures batches have balanced representation of 3, 4, and 5 node circuits
    to prevent the model from collapsing to the most common node count.
    """
    # Load raw data to count nodes per circuit
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)

    # Count nodes for each sample
    node_counts = []
    for idx in indices:
        circuit = raw_data[idx]
        nodes = set()
        for comp in circuit['components']:
            nodes.add(comp['node1'])
            nodes.add(comp['node2'])
        node_counts.append(len(nodes))

    # Calculate weights: inverse of class frequency
    from collections import Counter
    count_freq = Counter(node_counts)
    total = len(node_counts)

    # Weight = 1 / frequency (inverse frequency weighting)
    weights = []
    for count in node_counts:
        weights.append(total / (len(count_freq) * count_freq[count]))

    print(f"\nData balancing (Section 3.3):")
    print(f"  Node count distribution: {dict(count_freq)}")
    print(f"  Sample weights: 3-node={weights[node_counts.index(3)] if 3 in node_counts else 'N/A':.2f}, "
          f"4-node={weights[node_counts.index(4)] if 4 in node_counts else 'N/A':.2f}, "
          f"5-node={weights[node_counts.index(5)] if 5 in node_counts else 'N/A':.2f}")

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    kl_warmup_epochs = 20  # NEW: KL warm-up period

    print("="*70)
    print("Phase 3: Joint Edge-Component Prediction Training")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"KL warm-up epochs: {kl_warmup_epochs}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    print(f"Total samples: {len(dataset)}")

    # Load stratified split
    split_data = torch.load('rlc_dataset/stratified_split.pt')
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # NEW: Create balanced sampler for training (Section 3.3)
    train_sampler = create_balanced_sampler('rlc_dataset/filter_dataset.pkl', train_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # NEW: Use balanced sampler instead of shuffle
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create models
    print("\nCreating models...")
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

    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=50,  # CHANGED: Safety limit for generation (can generate up to 50 nodes!)
        max_training_nodes=10,  # Max nodes expected in training data
        enforce_vin_connectivity=True
    ).to(device)

    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # METHOD 1: LEARNED STOPPING CRITERION (from solution plan)
    # + FIX 2: Stop-Node Correlation Loss (ties stop and node type predictions)
    loss_fn = GumbelSoftmaxCircuitLoss(
        node_type_weight=1.0,
        stop_weight=2.0,              # Stopping criterion
        stop_node_correlation_weight=2.0,  # Middle ground: 1.0 too weak, 3.0 too aggressive
        edge_exist_weight=3.0,        # INCREASED from 1.0 (Phase 1 recommendation)
        component_type_weight=5.0,    # DECREASED from 10.0 (Phase 2 recommendation)
        component_value_weight=0.5,
        use_connectivity_loss=False,  # DISABLED - asymmetric penalties cause model collapse
        connectivity_weight=2.0,      # (not used when use_connectivity_loss=False)
        kl_weight=0.005              # VAE regularization
    )

    print("\nMethod 1: Learned Stopping Criterion + Fix 2 (Correlation Loss)")
    print(f"  stop_weight: 2.0 (learns when to stop generating nodes)")
    print(f"  stop_node_correlation_weight: 2.0 (middle ground: 1.0 too weak, 3.0 too aggressive)")
    print(f"  edge_exist_weight: 3.0 (increased from 1.0)")
    print(f"  component_type_weight: 5.0 (decreased from 10.0)")
    print(f"  use_connectivity_loss: False (DISABLED - was causing model collapse)")
    print(f"  → Model will learn: stop=1 → MASK, stop=0 → INTERNAL")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )

    # Optional: Load from previous checkpoint
    load_from_checkpoint = False
    if load_from_checkpoint and os.path.exists('checkpoints/gumbel_softmax/best.pt'):
        print("\nLoading weights from previous checkpoint...")
        checkpoint = torch.load('checkpoints/gumbel_softmax/best.pt', map_location=device)

        # Try to load encoder (should work)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print("✅ Loaded encoder weights")

        # Decoder has architecture changes, so only load compatible layers
        try:
            decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
            print("⚠️  Loaded decoder weights (with architecture changes)")
        except:
            print("⚠️  Decoder architecture changed, starting from scratch")

    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    best_val_loss = float('inf')
    checkpoint_dir = 'checkpoints/production'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # NEW: Update KL weight based on warm-up schedule (Section 3.1)
        current_kl_weight = get_kl_weight(epoch, warmup_epochs=kl_warmup_epochs)
        loss_fn.kl_weight = current_kl_weight

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs} (KL weight: {current_kl_weight:.5f})")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(encoder, decoder, train_loader, loss_fn, optimizer, device, epoch)

        # Validate
        val_metrics = validate(encoder, decoder, val_loader, loss_fn, device)

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Node Acc: {train_metrics['node_type_acc']:.1f}%")
        print(f"  Node Count Acc: {train_metrics.get('node_count_acc', 0):.1f}% ← KEY METRIC!")
        print(f"  Edge Acc: {train_metrics['edge_exist_acc']:.1f}%")
        print(f"  Component Type Acc: {train_metrics['component_type_acc']:.1f}%")
        if 'logit_max' in train_metrics:
            print(f"  Max Logit Magnitude: {train_metrics['logit_max']:.2f} (should be < 10)")

        print(f"\nVal Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Node Acc: {val_metrics['node_type_acc']:.1f}%")
        print(f"  Node Count Acc: {val_metrics.get('node_count_acc', 0):.1f}% ← KEY METRIC!")
        print(f"  Edge Acc: {val_metrics['edge_exist_acc']:.1f}%")
        print(f"  Component Type Acc: {val_metrics['component_type_acc']:.1f}%")
        if 'logit_max' in val_metrics:
            print(f"  Max Logit Magnitude: {val_metrics['logit_max']:.2f} (should be < 10)")

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, f'{checkpoint_dir}/best.pt')
            print(f"\n✅ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{checkpoint_dir}/epoch_{epoch}.pt')

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir}/best.pt")
    print(f"\nNext step: Run validation with scripts/validate.py")


if __name__ == '__main__':
    main()
