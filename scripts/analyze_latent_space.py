#!/usr/bin/env python3
"""
Analyze latent space geometry using diffusion maps.

This script:
1. Loads a trained GraphVAE model
2. Encodes all circuits to latent space
3. Performs diffusion map analysis
4. Estimates intrinsic dimensionality
5. Visualizes diffusion coordinates

Usage:
    python scripts/analyze_latent_space.py --checkpoint checkpoints/best.pt
    python scripts/analyze_latent_space.py --checkpoint checkpoints/best.pt --n-components 24
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from datetime import datetime

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.data import CircuitDataset, collate_circuit_batch
from ml.analysis import estimate_intrinsic_dimension, visualize_diffusion_coordinates
from torch.utils.data import DataLoader


def load_model_and_config(checkpoint_path: str, device: str):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        encoder: Loaded encoder model
        decoder: Loaded decoder model
        config: Training configuration
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to load config from checkpoint directory
    config_path = checkpoint_path.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default config
        print("Warning: config.yaml not found, using base_config.yaml")
        with open('configs/base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    # Extract branch dimensions (optional, defaults to equal split)
    topo_dim = config['model'].get('topo_latent_dim', None)
    values_dim = config['model'].get('values_latent_dim', None)
    pz_dim = config['model'].get('pz_latent_dim', None)

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    decoder = HybridDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    print(f"\n✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Best val loss: {checkpoint['best_val_loss']:.4f}")

    return encoder, decoder, config


def encode_dataset(encoder, dataset, device: str, batch_size: int = 32):
    """
    Encode entire dataset to latent space.

    Args:
        encoder: Trained encoder model
        dataset: CircuitDataset
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        latent_vectors: Encoded latent vectors [N, latent_dim]
        filter_types: Filter type labels [N]
        poles_list: List of poles for each circuit
        zeros_list: List of zeros for each circuit
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_circuit_batch,
        num_workers=0
    )

    latent_vectors = []
    filter_types = []
    poles_list = []
    zeros_list = []

    print(f"\nEncoding {len(dataset)} circuits to latent space...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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

            # Use mu (mean) instead of sampled z for deterministic encoding
            latent_vectors.append(mu.cpu().numpy())

            # Get filter types (convert one-hot to class indices)
            filter_type_idx = batch['filter_type'].argmax(dim=1).cpu().numpy()
            filter_types.append(filter_type_idx)

            # Store poles/zeros
            for p, z_pz in zip(batch['poles'], batch['zeros']):
                poles_list.append(p.cpu().numpy())
                zeros_list.append(z_pz.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} circuits")

    latent_vectors = np.vstack(latent_vectors)
    filter_types = np.concatenate(filter_types)

    print(f"✅ Encoded {len(latent_vectors)} circuits")
    print(f"   Latent space: {latent_vectors.shape}")

    return latent_vectors, filter_types, poles_list, zeros_list


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze GraphVAE latent space')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--n-components', type=int, default=24,
                       help='Number of diffusion components to analyze')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')

    args = parser.parse_args()

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
    print("LATENT SPACE ANALYSIS - DIFFUSION MAPS")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Components to analyze: {args.n_components}")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load model
    encoder, decoder, config = load_model_and_config(args.checkpoint, device)

    # Get branch dimensions from config
    topo_dim = config['model'].get('topo_latent_dim', config['model']['latent_dim'] // 3)
    values_dim = config['model'].get('values_latent_dim', config['model']['latent_dim'] // 3)
    pz_dim = config['model'].get('pz_latent_dim', config['model']['latent_dim'] // 3)
    branch_dims = [topo_dim, values_dim, pz_dim]

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=args.dataset,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    print(f"\n📊 Dataset: {len(dataset)} circuits")

    # Encode dataset to latent space
    latent_vectors, filter_types, poles_list, zeros_list = encode_dataset(
        encoder, dataset, device, batch_size=args.batch_size
    )

    # Filter type names
    filter_type_names = ['Low-pass', 'High-pass', 'Band-pass',
                        'Band-stop', 'RLC Series', 'RLC Parallel']

    # Perform diffusion map analysis
    print("\n" + "="*70)
    print("PERFORMING DIFFUSION MAP ANALYSIS")
    print("="*70)

    results = estimate_intrinsic_dimension(
        latent_vectors,
        n_components=args.n_components,
        epsilon=None,  # Auto-select
        plot=True,
        save_path=str(output_dir / 'eigenspectrum.png')
    )

    # Visualize diffusion coordinates
    print("\n" + "="*70)
    print("VISUALIZING DIFFUSION COORDINATES")
    print("="*70)

    # 2D visualization
    visualize_diffusion_coordinates(
        latent_vectors,
        labels=filter_types,
        label_names=filter_type_names,
        n_components=2,
        save_path=str(output_dir / 'diffusion_coords_2d.png')
    )

    # 3D visualization
    visualize_diffusion_coordinates(
        latent_vectors,
        labels=filter_types,
        label_names=filter_type_names,
        n_components=3,
        save_path=str(output_dir / 'diffusion_coords_3d.png')
    )

    # Analyze latent space branches
    print("\n" + "="*70)
    print("ANALYZING LATENT SPACE BRANCHES")
    print("="*70)

    latent_dim = latent_vectors.shape[1]

    # Analyze each branch separately with actual branch dimensions
    branch_names = ['z_topo (topology)', 'z_values (component values)', 'z_pz (poles/zeros)']
    branch_results = {}

    # Calculate branch start indices
    branch_starts = [0, topo_dim, topo_dim + values_dim]

    for i, (branch_name, branch_dim) in enumerate(zip(branch_names, branch_dims)):
        print(f"\n{branch_name} [{branch_dim}D]:")
        print("-" * 70)

        # Extract branch using actual dimensions
        start_idx = branch_starts[i]
        end_idx = start_idx + branch_dim
        branch_vectors = latent_vectors[:, start_idx:end_idx]

        # Analyze
        branch_result = estimate_intrinsic_dimension(
            branch_vectors,
            n_components=min(branch_dim, args.n_components),
            epsilon=None,
            plot=True,
            save_path=str(output_dir / f'eigenspectrum_branch_{i}.png')
        )

        branch_results[branch_name] = branch_result

    # Save summary
    summary = {
        'checkpoint': str(args.checkpoint),
        'dataset': args.dataset,
        'num_circuits': len(dataset),
        'latent_dim': latent_dim,
        'full_latent_space': {
            'intrinsic_dim_gap': int(results['intrinsic_dim_gap']),
            'intrinsic_dim_variance': int(results['intrinsic_dim_variance']),
            'intrinsic_dim_elbow': int(results['intrinsic_dim_elbow']),
            'recommended_dim': int(results['recommended_dim']),
            'epsilon': float(results['epsilon'])
        },
        'branches': {
            name: {
                'intrinsic_dim_gap': int(res['intrinsic_dim_gap']),
                'intrinsic_dim_variance': int(res['intrinsic_dim_variance']),
                'intrinsic_dim_elbow': int(res['intrinsic_dim_elbow']),
                'recommended_dim': int(res['recommended_dim'])
            }
            for name, res in branch_results.items()
        }
    }

    # Save summary to YAML
    summary_path = output_dir / 'analysis_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📊 Summary saved to: {summary_path}")
    print(f"📈 Plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - eigenspectrum.png (full latent space)")
    print(f"  - eigenspectrum_branch_0.png (z_topo)")
    print(f"  - eigenspectrum_branch_1.png (z_values)")
    print(f"  - eigenspectrum_branch_2.png (z_pz)")
    print(f"  - diffusion_coords_2d.png")
    print(f"  - diffusion_coords_3d.png")
    print(f"  - analysis_summary.yaml")

    # Print final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Current latent space: {latent_dim}D ({topo_dim}D + {values_dim}D + {pz_dim}D branches)")
    print(f"Estimated intrinsic dimension: {results['recommended_dim']}D")

    if results['recommended_dim'] < latent_dim:
        print(f"\n✅ Consider reducing latent dimension to {results['recommended_dim']}D")
        print(f"   This would eliminate ~{latent_dim - results['recommended_dim']} redundant dimensions.")
    else:
        print(f"\n✅ Current latent dimension ({latent_dim}D) is well-matched.")

    # Branch-specific recommendations
    print("\nBranch-specific analysis:")
    for i, (branch_name, branch_res) in enumerate(zip(branch_names, branch_results.values())):
        current_dim = branch_dims[i]
        recommended = branch_res['recommended_dim']
        if recommended < current_dim:
            print(f"  {branch_name}: {current_dim}D → {recommended}D (reduce by {current_dim - recommended})")
        else:
            print(f"  {branch_name}: {current_dim}D (well-matched)")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
