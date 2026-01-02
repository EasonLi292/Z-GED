"""Analyze potential overfitting in the circuit generation model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import LatentGuidedGraphGPTDecoder
from ml.data.dataset import CircuitDataset


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_complexity():
    """Analyze model complexity vs dataset size."""
    device = 'cpu'

    # Load models
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
        max_nodes=5
    ).to(device)

    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    print("="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)

    print("\n📊 MODEL COMPLEXITY:")
    print(f"  Encoder parameters:    {encoder_params:,}")
    print(f"  Decoder parameters:    {decoder_params:,}")
    print(f"  Total parameters:      {total_params:,}")

    print("\n📊 DATASET SIZE:")
    print(f"  Total circuits:        {dataset_size}")
    print(f"  Training circuits:     {train_size} (80%)")
    print(f"  Validation circuits:   {val_size} (20%)")

    print("\n📊 PARAMETERS PER TRAINING SAMPLE:")
    params_per_sample = total_params / train_size
    print(f"  {params_per_sample:,.0f} parameters per training circuit")

    print("\n⚠️  OVERFITTING RISK ASSESSMENT:")
    if params_per_sample > 10000:
        print(f"  🔴 SEVERE: {params_per_sample:,.0f} params/sample (>>10k)")
        print(f"     Model has {int(params_per_sample/1000)}x more parameters than needed")
        print(f"     Extremely high risk of memorization")
    elif params_per_sample > 5000:
        print(f"  🟠 HIGH: {params_per_sample:,.0f} params/sample (>5k)")
        print(f"     Model likely overfitting to training data")
    elif params_per_sample > 1000:
        print(f"  🟡 MODERATE: {params_per_sample:,.0f} params/sample (>1k)")
        print(f"     Some overfitting risk, monitor validation performance")
    else:
        print(f"  🟢 LOW: {params_per_sample:,.0f} params/sample (<1k)")
        print(f"     Reasonable parameter count")

    return {
        'encoder_params': encoder_params,
        'decoder_params': decoder_params,
        'total_params': total_params,
        'dataset_size': dataset_size,
        'train_size': train_size,
        'val_size': val_size,
        'params_per_sample': params_per_sample
    }


def check_generation_diversity():
    """Check if model generates diverse circuits or just memorizes templates."""
    device = 'cpu'

    # Load models
    encoder = HierarchicalEncoder(
        node_feature_dim=4, edge_feature_dim=7, gnn_hidden_dim=64,
        gnn_num_layers=3, latent_dim=8, topo_latent_dim=2,
        values_latent_dim=2, pz_latent_dim=4, dropout=0.1
    ).to(device)

    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8, conditions_dim=2, hidden_dim=256,
        num_heads=8, num_node_layers=4, max_nodes=5
    ).to(device)

    checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    print("\n" + "="*80)
    print("GENERATION DIVERSITY TEST")
    print("="*80)
    print("\nGenerating 10 circuits with SAME specification (10kHz, Q=0.707)...")
    print("If overfitting, model will generate identical circuits every time.\n")

    target_cutoff = 10000
    target_q = 0.707

    # Generate same latent code
    latent = torch.randn(1, 8, device=device)
    conditions = torch.tensor([[
        np.log10(target_cutoff) / 4.0,
        np.log10(target_q) / 2.0
    ]], dtype=torch.float32, device=device)

    generated_circuits = []

    for i in range(10):
        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Extract topology signature
        edge_exist = circuit['edge_existence'][0]
        num_edges = (edge_exist > 0.5).sum().item() // 2

        # Extract component values for first edge
        edge_vals = circuit['edge_values'][0]
        first_edge_vals = None
        for ii in range(5):
            for jj in range(ii+1, 5):
                if edge_exist[ii, jj] > 0.5:
                    first_edge_vals = edge_vals[ii, jj, :3]  # log(C), G, log(L_inv)
                    break
            if first_edge_vals is not None:
                break

        generated_circuits.append({
            'num_edges': num_edges,
            'first_edge_vals': first_edge_vals.tolist() if first_edge_vals is not None else None
        })

        print(f"  Run {i+1}: {num_edges} edges, first edge values: {first_edge_vals.tolist() if first_edge_vals is not None else 'None'}")

    # Check diversity
    num_edges_set = set(c['num_edges'] for c in generated_circuits)
    print(f"\n📊 Topology Diversity:")
    print(f"  Unique edge counts: {len(num_edges_set)} (out of 10 generations)")

    if len(num_edges_set) == 1:
        print(f"  🔴 ZERO DIVERSITY: All circuits have same topology")
        print(f"     Strong evidence of memorization/overfitting")
    elif len(num_edges_set) <= 2:
        print(f"  🟡 LOW DIVERSITY: Only {len(num_edges_set)} different topologies")
        print(f"     Possible overfitting to training templates")
    else:
        print(f"  🟢 GOOD DIVERSITY: {len(num_edges_set)} different topologies")


def main():
    complexity_info = analyze_model_complexity()
    check_generation_diversity()

    print("\n" + "="*80)
    print("SUMMARY: OVERFITTING ASSESSMENT")
    print("="*80)

    params_per_sample = complexity_info['params_per_sample']

    if params_per_sample > 10000:
        print("\n🔴 CRITICAL OVERFITTING DETECTED:")
        print(f"   • {params_per_sample:,.0f} parameters per training sample")
        print(f"   • Model has capacity to memorize every training circuit")
        print(f"   • {complexity_info['total_params']:,} params / {complexity_info['train_size']} samples")
        print("\n   RECOMMENDATIONS:")
        print("   1. Reduce model size (smaller hidden_dim: 256 → 128)")
        print("   2. Increase dropout (0.1 → 0.3)")
        print("   3. Collect more training data (120 → 500+ circuits)")
        print("   4. Add stronger regularization (weight decay)")
        print("   5. Use data augmentation (noise injection)")
    elif params_per_sample > 5000:
        print("\n🟠 LIKELY OVERFITTING:")
        print(f"   • {params_per_sample:,.0f} parameters per training sample")
        print(f"   • High risk of memorization")
        print("\n   RECOMMENDATIONS:")
        print("   1. Monitor validation loss closely")
        print("   2. Consider reducing hidden_dim (256 → 192)")
        print("   3. Increase training data if possible")
    else:
        print("\n🟢 REASONABLE MODEL CAPACITY:")
        print(f"   • {params_per_sample:,.0f} parameters per training sample")
        print(f"   • Model size is appropriate for dataset")


if __name__ == '__main__':
    main()
