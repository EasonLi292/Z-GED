#!/usr/bin/env python3
"""
Evaluation script for GraphVAE.

Loads a trained model and evaluates it on the test set:
    - Computes reconstruction metrics
    - Analyzes latent space quality
    - Generates visualizations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import argparse
import numpy as np
from datetime import datetime
import json

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.data import CircuitDataset, collate_circuit_batch
from ml.utils.metrics import MetricsAggregator
from ml.utils.visualization import LatentSpaceVisualizer, TrainingVisualizer, ReconstructionVisualizer


def load_checkpoint(checkpoint_path: str, encoder, decoder, device: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Best val loss: {checkpoint['best_val_loss']:.4f}")

    return checkpoint


def extract_latent_representations(
    encoder,
    dataloader,
    device: str
) -> tuple:
    """
    Extract latent representations for all circuits in dataloader.

    Returns:
        latent_vectors: [N, latent_dim]
        filter_types: [N]
        indices: [N]
    """
    encoder.eval()

    all_latent = []
    all_filter_types = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch['graph'] = batch['graph'].to(device)
            batch['poles'] = [p.to(device) for p in batch['poles']]
            batch['zeros'] = [z.to(device) for z in batch['zeros']]

            # Encode
            z, mu, logvar = encoder(
                batch['graph'].x,
                batch['graph'].edge_index,
                batch['graph'].edge_attr,
                batch['graph'].batch,
                batch['poles'],
                batch['zeros']
            )

            all_latent.append(z.cpu().numpy())
            all_filter_types.append(batch['filter_type'].argmax(dim=-1).cpu().numpy())
            if 'idx' in batch:
                all_indices.append(batch['idx'].cpu().numpy())

    latent_vectors = np.vstack(all_latent)
    filter_types = np.concatenate(all_filter_types)
    indices = np.concatenate(all_indices) if all_indices else None

    return latent_vectors, filter_types, indices


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate GraphVAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/.../best.pt)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint dir)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--ged-matrix', type=str, default=None,
                       help='Path to GED matrix for correlation analysis')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n" + "="*70)
    print("GRAPHVAE EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print("="*70 + "\n")

    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config if args.config else checkpoint_dir / 'config.yaml'

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")

    # Create dataset
    dataset = CircuitDataset(
        dataset_path=config['data']['dataset_path'],
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    # Get splits
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        seed=config['data']['split_seed']
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_idx)} circuits")
    print(f"  Val:   {len(val_idx)} circuits")
    print(f"  Test:  {len(test_idx)} circuits")

    # Create dataloaders
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    # For visualizing all data
    full_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    decoder = HybridDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, encoder, decoder, device)

    # Load GED matrix if provided
    ged_matrix = None
    if args.ged_matrix and Path(args.ged_matrix).exists():
        ged_matrix = np.load(args.ged_matrix)
        print(f"\nLoaded GED matrix from {args.ged_matrix}")

    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)

    # Compute metrics
    aggregator = MetricsAggregator()
    metrics = aggregator.compute_all_metrics(
        encoder,
        decoder,
        test_loader,
        device=device,
        ged_matrix=ged_matrix
    )

    # Print metrics
    print("\n📊 Test Set Metrics:")
    print("-" * 70)
    print(f"Reconstruction:")
    print(f"  Topology Accuracy:     {metrics['topology_accuracy']:.2%}")
    print(f"  Pole Chamfer Distance: {metrics['pole_chamfer']:.4f}")
    print(f"  Zero Chamfer Distance: {metrics['zero_chamfer']:.4f}")
    print(f"\nLatent Space Quality:")
    print(f"  Silhouette Score:      {metrics['silhouette_score']:.4f}")
    print(f"  Cluster Purity:        {metrics['cluster_purity']:.2%}")
    print(f"  Mean Distance:         {metrics['mean_distance']:.4f}")
    print(f"  Std Distance:          {metrics['std_distance']:.4f}")

    if 'pearson_correlation' in metrics:
        print(f"\nGED Correlation:")
        print(f"  Pearson:               {metrics['pearson_correlation']:.4f}")
        print(f"  Spearman:              {metrics['spearman_correlation']:.4f}")

    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics to {metrics_path}")

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Extract latent representations for visualization
    print("\nExtracting latent representations...")
    latent_vectors, filter_types, indices = extract_latent_representations(
        encoder, full_loader, device
    )

    print(f"  Latent vectors: {latent_vectors.shape}")
    print(f"  Filter types: {len(np.unique(filter_types))} unique")

    # Create visualizers
    latent_viz = LatentSpaceVisualizer()
    training_viz = TrainingVisualizer()

    # 1. t-SNE and PCA
    print("\n1. Creating t-SNE and PCA visualizations...")
    latent_viz.plot_tsne_pca(
        latent_vectors,
        filter_types,
        save_path=output_dir / 'latent_tsne_pca.png'
    )

    # 2. Latent dimension distributions
    print("2. Analyzing latent dimension distributions...")
    latent_viz.plot_latent_dimensions(
        latent_vectors,
        filter_types,
        save_path=output_dir / 'latent_dimensions.png'
    )

    # 3. Hierarchical structure (if 24D latent space)
    if latent_vectors.shape[1] == 24:
        print("3. Visualizing hierarchical latent structure...")
        latent_viz.plot_hierarchical_structure(
            latent_vectors,
            filter_types,
            save_path=output_dir / 'hierarchical_structure.png'
        )

    # 4. Training history
    history_path = checkpoint_dir / 'training_history.json'
    if history_path.exists():
        print("4. Plotting training history...")
        training_viz.plot_training_history(
            str(history_path),
            save_path=output_dir / 'training_history.png'
        )

        print("5. Plotting loss components...")
        training_viz.plot_loss_components(
            str(history_path),
            save_path=output_dir / 'loss_components.png'
        )

    # 5. Pole-zero reconstruction (sample circuits)
    print("6. Visualizing pole-zero reconstruction...")

    # Get a few test samples
    encoder.eval()
    decoder.eval()

    sample_batch = next(iter(test_loader))
    sample_batch['graph'] = sample_batch['graph'].to(device)
    sample_batch['poles'] = [p.to(device) for p in sample_batch['poles']]
    sample_batch['zeros'] = [z.to(device) for z in sample_batch['zeros']]

    with torch.no_grad():
        z, mu, logvar = encoder(
            sample_batch['graph'].x,
            sample_batch['graph'].edge_index,
            sample_batch['graph'].edge_attr,
            sample_batch['graph'].batch,
            sample_batch['poles'],
            sample_batch['zeros']
        )

        decoder_output = decoder(z, hard=False)

    # Convert to numpy
    pred_poles_list = []
    pred_zeros_list = []
    target_poles_list = []
    target_zeros_list = []

    pred_poles = decoder_output['poles'].cpu().numpy()
    pred_zeros = decoder_output['zeros'].cpu().numpy()

    for i in range(min(4, len(sample_batch['poles']))):
        # Predicted (filter padding)
        pp = pred_poles[i]
        mag = np.sqrt((pp**2).sum(axis=-1))
        pp = pp[mag > 1e-6]
        pred_poles_list.append(pp)

        pz = pred_zeros[i]
        mag = np.sqrt((pz**2).sum(axis=-1))
        pz = pz[mag > 1e-6]
        pred_zeros_list.append(pz)

        # Target
        target_poles_list.append(sample_batch['poles'][i].cpu().numpy())
        target_zeros_list.append(sample_batch['zeros'][i].cpu().numpy())

    recon_viz = ReconstructionVisualizer()
    recon_viz.plot_pole_zero_comparison(
        pred_poles_list,
        pred_zeros_list,
        target_poles_list,
        target_zeros_list,
        save_path=output_dir / 'pole_zero_reconstruction.png'
    )

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n✅ All results saved to {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - test_metrics.json")
    print(f"  - latent_tsne_pca.png")
    print(f"  - latent_dimensions.png")
    if latent_vectors.shape[1] == 24:
        print(f"  - hierarchical_structure.png")
    if history_path.exists():
        print(f"  - training_history.png")
        print(f"  - loss_components.png")
    print(f"  - pole_zero_reconstruction.png")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
