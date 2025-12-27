"""
Circuit generation script using trained GraphGPT model.

Usage:
    python3 scripts/generate_graphgpt.py \\
        --checkpoint checkpoints/graphgpt_decoder/best.pt \\
        --cutoff 1000.0 \\
        --q-factor 0.707 \\
        --num-samples 5 \\
        --device mps
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder


NODE_TYPE_NAMES = {
    0: 'GND',
    1: 'VIN',
    2: 'VOUT',
    3: 'INTERNAL',
    4: 'MASK'
}


def load_model(checkpoint_path, device):
    """Load trained GraphGPT model."""
    print(f"\n📂 Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create encoder
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['encoder']['node_feature_dim'],
        edge_feature_dim=config['model']['encoder']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['encoder']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['encoder']['gnn_num_layers'],
        latent_dim=config['model']['encoder']['latent_dim'],
        topo_latent_dim=config['model']['encoder']['topo_latent_dim'],
        values_latent_dim=config['model']['encoder']['values_latent_dim'],
        pz_latent_dim=config['model']['encoder']['pz_latent_dim'],
        dropout=config['model']['encoder']['dropout']
    ).to(device)

    # Create decoder
    decoder = GraphGPTDecoder(
        latent_dim=config['model']['decoder']['latent_dim'],
        conditions_dim=config['model']['decoder']['conditions_dim'],
        hidden_dim=config['model']['decoder']['hidden_dim'],
        num_heads=config['model']['decoder']['num_heads'],
        num_node_layers=config['model']['decoder']['num_node_layers'],
        max_nodes=config['model']['decoder']['max_nodes'],
        max_poles=config['model']['decoder']['max_poles'],
        max_zeros=config['model']['decoder']['max_zeros'],
        dropout=config['model']['decoder']['dropout']
    ).to(device)

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")

    return encoder, decoder, config


def generate_circuits(
    decoder,
    cutoff_freq,
    q_factor,
    num_samples,
    device,
    edge_threshold=0.5
):
    """Generate circuits from specifications."""
    print(f"\n🎨 Generating {num_samples} circuits...")
    print(f"   Cutoff frequency: {cutoff_freq} Hz")
    print(f"   Q-factor: {q_factor}")

    # Sample random latent codes
    latent = torch.randn(num_samples, 8, device=device)

    # Prepare specifications (normalize)
    log_cutoff = np.log10(cutoff_freq) / 4.0  # Normalize to ~[0, 1]
    log_q = np.log10(max(q_factor, 0.1)) / 2.0

    conditions = torch.tensor(
        [[log_cutoff, log_q]] * num_samples,
        dtype=torch.float32,
        device=device
    )

    # Generate
    with torch.no_grad():
        circuits = decoder.generate(
            latent_code=latent,
            conditions=conditions,
            enforce_constraints=True,
            edge_threshold=edge_threshold
        )

    return circuits


def print_circuit(circuit_idx, circuit):
    """Print circuit details."""
    print(f"\n{'='*70}")
    print(f"Circuit {circuit_idx}:")
    print(f"{'='*70}")

    # Node types
    node_types = circuit['node_types'][circuit_idx].cpu().numpy()
    print("\nNodes:")
    for i, node_type in enumerate(node_types):
        print(f"  N{i}: {NODE_TYPE_NAMES.get(int(node_type), 'UNKNOWN')}")

    # Edges
    edge_matrix = circuit['edge_existence'][circuit_idx].cpu().numpy()
    num_edges = int(edge_matrix.sum()) // 2  # Divide by 2 for undirected graph

    print(f"\nEdges: {num_edges} total")
    edge_count = 0
    for i in range(edge_matrix.shape[0]):
        for j in range(i+1, edge_matrix.shape[1]):  # Only upper triangle
            if edge_matrix[i, j] > 0.5:
                edge_values = circuit['edge_values'][circuit_idx, i, j].cpu().numpy()
                # Decode edge values (these are log-scaled in training)
                C_log, G_log, L_log = edge_values[:3]
                masks = edge_values[3:]

                edge_count += 1
                print(f"  Edge {edge_count}: N{i} -- N{j}")
                print(f"    C_log: {C_log:.4f}, G_log: {G_log:.4f}, L_log: {L_log:.4f}")

    # Poles and zeros
    pole_count = circuit['pole_count'][circuit_idx].item()
    zero_count = circuit['zero_count'][circuit_idx].item()

    print(f"\nPoles: {pole_count}")
    if pole_count > 0:
        pole_values = circuit['pole_values'][circuit_idx, :pole_count].cpu().numpy()
        for i, pole in enumerate(pole_values):
            real, imag = pole
            if abs(imag) < 1e-6:
                print(f"  Pole {i+1}: {real:.4f}")
            else:
                print(f"  Pole {i+1}: {real:.4f} {'+'if imag >=0 else ''} {imag:.4f}j")

    print(f"\nZeros: {zero_count}")
    if zero_count > 0:
        zero_values = circuit['zero_values'][circuit_idx, :zero_count].cpu().numpy()
        for i, zero in enumerate(zero_values):
            real, imag = zero
            if abs(imag) < 1e-6:
                print(f"  Zero {i+1}: {real:.4f}")
            else:
                print(f"  Zero {i+1}: {real:.4f} {'+' if imag >= 0 else ''} {imag:.4f}j")


def analyze_generation_quality(circuits, num_samples):
    """Analyze quality of generated circuits."""
    print(f"\n{'='*70}")
    print("Generation Quality Analysis")
    print(f"{'='*70}")

    # Edge statistics
    edge_counts = []
    edge_probs = []

    for i in range(num_samples):
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()
        num_edges = int(edge_matrix.sum()) // 2
        edge_counts.append(num_edges)

        # Average edge probability (before thresholding)
        # Note: This is after sigmoid in generation, so already probabilities
        edge_probs.append(edge_matrix.mean())

    print(f"\nEdge Statistics:")
    print(f"  Average edges per circuit: {np.mean(edge_counts):.1f}")
    print(f"  Min edges: {np.min(edge_counts)}")
    print(f"  Max edges: {np.max(edge_counts)}")
    print(f"  Std edges: {np.std(edge_counts):.2f}")

    # Pole/zero statistics
    pole_counts = circuits['pole_count'].cpu().numpy()
    zero_counts = circuits['zero_count'].cpu().numpy()

    print(f"\nPole/Zero Statistics:")
    print(f"  Average poles: {np.mean(pole_counts):.1f}")
    print(f"  Average zeros: {np.mean(zero_counts):.1f}")

    # Check for valid circuits
    valid_circuits = sum(1 for ec in edge_counts if ec > 0)
    print(f"\nCircuit Validity:")
    print(f"  Circuits with edges: {valid_circuits}/{num_samples} ({100*valid_circuits/num_samples:.1f}%)")

    if valid_circuits < num_samples:
        print(f"  ⚠️  WARNING: {num_samples - valid_circuits} circuits have no edges!")
    else:
        print(f"  ✅ All circuits have edges (SPICE-simulatable)")


def main(args):
    """Main generation function."""
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("GraphGPT Circuit Generation")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load model
    encoder, decoder, config = load_model(args.checkpoint, device)

    # Generate circuits
    circuits = generate_circuits(
        decoder,
        args.cutoff,
        args.q_factor,
        args.num_samples,
        device,
        edge_threshold=args.edge_threshold
    )

    # Print circuits
    for i in range(min(args.num_samples, args.max_print)):
        print_circuit(i, circuits)

    # Analyze quality
    analyze_generation_quality(circuits, args.num_samples)

    print(f"\n{'='*70}")
    print("✅ Generation completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate circuits with GraphGPT')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='Target cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Target Q-factor')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of circuits to generate')
    parser.add_argument('--max-print', type=int, default=5,
                       help='Maximum number of circuits to print')
    parser.add_argument('--edge-threshold', type=float, default=0.5,
                       help='Threshold for edge existence')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')

    args = parser.parse_args()
    main(args)
