#!/usr/bin/env python3
"""
Test reconstruction quality of variable-length decoder.

This tests the encode-decode cycle:
1. Encode real circuit → latent code z
2. Decode z → reconstructed circuit
3. Compare reconstruction quality

Usage:
    python scripts/test_reconstruction.py \
        --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
        --num-samples 10
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse

from ml.models import HierarchicalEncoder
from ml.models.variable_decoder import VariableLengthDecoder
from ml.data import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def infer_filter_type_from_structure(num_poles, num_zeros):
    """Infer filter type from pole/zero structure."""
    if num_poles == 1 and num_zeros == 0:
        return 'low_pass'
    elif num_poles == 2 and num_zeros == 0:
        return 'low_pass'
    elif num_poles == 1 and num_zeros == 1:
        return 'high_pass'
    elif num_poles == 2 and num_zeros == 2:
        return 'high_pass_or_band_stop'
    elif num_poles == 2 and num_zeros == 1:
        return 'band_pass'
    else:
        return 'unknown'


def test_reconstruction(checkpoint_path, dataset_path, num_samples, device):
    """Test reconstruction on real circuits."""

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"\n{'='*70}")
    print("VARIABLE-LENGTH DECODER RECONSTRUCTION TEST")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"{'='*70}\n")

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    decoder = VariableLengthDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        max_poles=config['model'].get('max_poles', 4),
        max_zeros=config['model'].get('max_zeros', 4),
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=dataset_path,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    # Create test split
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(config['data']['split_seed'])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Sample a few circuits
    sample_size = min(num_samples, len(test_dataset))
    sample_indices = torch.randperm(len(test_dataset))[:sample_size]

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    print(f"Testing reconstruction on {sample_size} test circuits...\n")

    with torch.no_grad():
        for idx in sample_indices:
            circuit = test_dataset[int(idx)]

            # Get ground truth
            filter_type_idx = circuit['filter_type'].argmax().item()
            filter_type_str = filter_type_names[filter_type_idx]
            poles_gt = circuit['poles'].numpy()
            zeros_gt = circuit['zeros'].numpy()
            num_poles_gt = circuit['num_poles'].item()
            num_zeros_gt = circuit['num_zeros'].item()

            # Encode
            graph = circuit['graph'].to(device)
            poles_list = [circuit['poles'].to(device)]
            zeros_list = [circuit['zeros'].to(device)]

            # Add batch dimension
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles_list,
                zeros_list
            )

            # Decode (with and without teacher forcing)
            outputs_with_tf = decoder(mu, hard=False, gt_filter_type=circuit['filter_type'].unsqueeze(0).to(device))
            outputs_no_tf = decoder(mu, hard=False, gt_filter_type=None)

            # Get predictions (without teacher forcing)
            pred_topo = outputs_no_tf['topo_logits'].argmax(dim=-1).item()
            pred_filter_type = filter_type_names[pred_topo]
            topo_confidence = torch.softmax(outputs_no_tf['topo_logits'], dim=-1)[0, pred_topo].item()

            pred_pole_count = outputs_no_tf['pole_count_logits'].argmax(dim=-1).item()
            pred_zero_count = outputs_no_tf['zero_count_logits'].argmax(dim=-1).item()

            # Extract predicted poles/zeros
            poles_pred = outputs_no_tf['poles_all'][0, :pred_pole_count].cpu().numpy() if pred_pole_count > 0 else np.array([])
            zeros_pred = outputs_no_tf['zeros_all'][0, :pred_zero_count].cpu().numpy() if pred_zero_count > 0 else np.array([])

            # Infer TF type from structure
            tf_type_gt = infer_filter_type_from_structure(num_poles_gt, num_zeros_gt)
            tf_type_pred = infer_filter_type_from_structure(pred_pole_count, pred_zero_count)

            # Check matches
            topo_match = (pred_filter_type == filter_type_str)
            pole_count_match = (pred_pole_count == num_poles_gt)
            zero_count_match = (pred_zero_count == num_zeros_gt)
            tf_match = (tf_type_pred == tf_type_gt)

            print(f"{'─'*70}")
            print(f"Circuit {int(idx) + 1} (Test Set)")
            print(f"{'─'*70}")
            print(f"GROUND TRUTH:")
            print(f"  Filter Type: {filter_type_str.upper()}")
            print(f"  Pole/Zero Count: {num_poles_gt} poles, {num_zeros_gt} zeros")
            print(f"  TF Structure: {tf_type_gt}")

            print(f"\nRECONSTRUCTION:")
            print(f"  Filter Type: {pred_filter_type.upper()} ({topo_confidence:.1%} conf) {'✅' if topo_match else '❌'}")
            print(f"  Pole/Zero Count: {pred_pole_count} poles {'✅' if pole_count_match else '❌'}, {pred_zero_count} zeros {'✅' if zero_count_match else '❌'}")
            print(f"  TF Structure: {tf_type_pred} {'✅' if tf_match else '❌'}")

            # Compare pole/zero values
            if len(poles_gt) > 0 and len(poles_pred) > 0:
                print(f"\n  POLES:")
                print(f"    Ground Truth ({len(poles_gt)}):")
                for i, pole in enumerate(poles_gt):
                    print(f"      {i+1}. {pole[0]:+.4f} {pole[1]:+.4f}j")
                print(f"    Predicted ({len(poles_pred)}):")
                for i, pole in enumerate(poles_pred):
                    print(f"      {i+1}. {pole[0]:+.4f} {pole[1]:+.4f}j")

            if len(zeros_gt) > 0 and len(zeros_pred) > 0:
                print(f"\n  ZEROS:")
                print(f"    Ground Truth ({len(zeros_gt)}):")
                for i, zero in enumerate(zeros_gt):
                    print(f"      {i+1}. {zero[0]:+.4f} {zero[1]:+.4f}j")
                print(f"    Predicted ({len(zeros_pred)}):")
                for i, zero in enumerate(zeros_pred):
                    print(f"      {i+1}. {zero[0]:+.4f} {zero[1]:+.4f}j")

            print()

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test reconstruction quality')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of circuits to test')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')

    args = parser.parse_args()

    test_reconstruction(args.checkpoint, args.dataset, args.num_samples, args.device)


if __name__ == '__main__':
    main()
