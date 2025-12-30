"""
Generate circuits with targeted transfer functions.

This script allows you to:
1. Specify exact poles/zeros you want
2. Generate multiple novel circuit topologies that implement them
3. Explore design space while maintaining TF characteristics

Usage:
    python scripts/generate_targeted_tf.py \\
        --poles "(-1000+2000j)" "(-1000-2000j)" \\
        --zeros "(-500+0j)" \\
        --cutoff 1000 \\
        --q-factor 0.707 \\
        --num-samples 10
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.tf_encoder import TransferFunctionEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder


def parse_complex(s: str) -> complex:
    """Parse complex number from string like '(-1000+2000j)'."""
    return complex(s.replace(' ', ''))


def generate_with_target_tf(
    tf_encoder,
    decoder,
    target_poles,
    target_zeros,
    cutoff_freq,
    q_factor,
    num_samples,
    topology_variation=0.3,
    values_variation=0.3,
    device='mps'
):
    """
    Generate circuits with targeted transfer function.

    Args:
        tf_encoder: Trained TF encoder
        decoder: Trained GraphGPT decoder
        target_poles: List of complex poles
        target_zeros: List of complex zeros
        cutoff_freq: Target cutoff frequency (Hz)
        q_factor: Target Q-factor
        num_samples: Number of circuit variations to generate
        topology_variation: Std dev for topology latent noise (0=fixed, 1=very random)
        values_variation: Std dev for values latent noise
        device: Device to use

    Returns:
        circuits: Generated circuit dictionary
    """
    device = torch.device(device)

    # Prepare target poles/zeros
    num_poles = len(target_poles)
    num_zeros = len(target_zeros)

    # Create tensors
    pole_values = torch.zeros(1, 4, 2, device=device)
    for i, pole in enumerate(target_poles[:4]):  # Max 4 poles
        pole_values[0, i, 0] = pole.real
        pole_values[0, i, 1] = pole.imag

    zero_values = torch.zeros(1, 4, 2, device=device)
    for i, zero in enumerate(target_zeros[:4]):  # Max 4 zeros
        zero_values[0, i, 0] = zero.real
        zero_values[0, i, 1] = zero.imag

    pole_count = torch.tensor([num_poles], device=device)
    zero_count = torch.tensor([num_zeros], device=device)

    # Encode target TF to latent[4:8]
    with torch.no_grad():
        tf_latent = tf_encoder.encode(
            pole_values, pole_count, zero_values, zero_count,
            deterministic=True  # Use mean for consistent TF
        )

    print(f"\n✅ Target TF encoded to latent[4:8]: {tf_latent.cpu().numpy()[0]}")

    # Generate multiple circuit variations
    print(f"\n🎨 Generating {num_samples} circuit variations...")

    all_latents = []

    for i in range(num_samples):
        # Sample random topology and values latents
        topo_latent = torch.randn(1, 2, device=device) * topology_variation
        values_latent = torch.randn(1, 2, device=device) * values_variation

        # Combine: [topology (2D) | values (2D) | TF (4D)]
        full_latent = torch.cat([topo_latent, values_latent, tf_latent], dim=1)
        all_latents.append(full_latent)

    all_latents = torch.cat(all_latents, dim=0)  # [num_samples, 8]

    # Prepare specifications
    log_cutoff = np.log10(cutoff_freq) / 4.0
    log_q = np.log10(max(q_factor, 0.1)) / 2.0

    conditions = torch.tensor(
        [[log_cutoff, log_q]] * num_samples,
        dtype=torch.float32,
        device=device
    )

    # Generate circuits
    with torch.no_grad():
        circuits = decoder.generate(
            latent_code=all_latents,
            conditions=conditions,
            enforce_constraints=True,
            edge_threshold=0.5
        )

    return circuits


def print_circuit_summary(circuits, num_samples):
    """Print summary of generated circuits."""
    print(f"\n{'='*70}")
    print("Generation Summary")
    print(f"{'='*70}\n")

    # Topology diversity
    topologies = set()
    for i in range(num_samples):
        node_types = circuits['node_types'][i].cpu().numpy()
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()

        num_nodes = sum(1 for nt in node_types if nt != 4)
        num_edges = int((edge_matrix > 0.5).sum()) // 2

        topologies.add((num_nodes, num_edges))

    print(f"Topology diversity: {len(topologies)} unique structures")
    for topo in sorted(topologies):
        nodes, edges = topo
        count = sum(1 for i in range(num_samples)
                   if (sum(1 for nt in circuits['node_types'][i].cpu().numpy() if nt != 4) == nodes
                       and int((circuits['edge_existence'][i].cpu().numpy() > 0.5).sum()) // 2 == edges))
        print(f"  {nodes} nodes, {edges} edges: {count}/{num_samples} circuits")

    # Transfer function accuracy
    print(f"\nTransfer Function Accuracy:")
    pole_counts = circuits['pole_count'].cpu().numpy()
    zero_counts = circuits['zero_count'].cpu().numpy()

    print(f"  Pole counts: {pole_counts.tolist()}")
    print(f"  Zero counts: {zero_counts.tolist()}")

    # Check stability
    stable_count = 0
    for i in range(num_samples):
        n_poles = pole_counts[i]
        if n_poles > 0:
            poles = circuits['pole_values'][i, :n_poles].cpu().numpy()
            if all(pole[0] < 0 for pole in poles):  # All real parts < 0
                stable_count += 1

    print(f"  Stable circuits: {stable_count}/{num_samples}")

    # Show first 3 circuits in detail
    print(f"\n{'='*70}")
    print("Sample Circuits (first 3):")
    print(f"{'='*70}\n")

    for i in range(min(3, num_samples)):
        print(f"Circuit {i}:")

        # Nodes
        node_types = circuits['node_types'][i].cpu().numpy()
        node_names = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INTERNAL', 4: 'MASK'}
        print(f"  Nodes: {[node_names[int(nt)] for nt in node_types]}")

        # Edges
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()
        num_edges = int((edge_matrix > 0.5).sum()) // 2
        print(f"  Edges: {num_edges}")

        # Poles/zeros
        n_poles = pole_counts[i]
        n_zeros = zero_counts[i]

        print(f"  Transfer Function: {n_poles} poles, {n_zeros} zeros")

        if n_poles > 0:
            poles = circuits['pole_values'][i, :n_poles].cpu().numpy()
            for j, pole in enumerate(poles):
                real, imag = pole
                if abs(imag) < 1e-6:
                    print(f"    Pole {j+1}: {real:.4f}")
                else:
                    print(f"    Pole {j+1}: {real:.4f} {'+' if imag >= 0 else ''}{imag:.4f}j")

        if n_zeros > 0:
            zeros = circuits['zero_values'][i, :n_zeros].cpu().numpy()
            for j, zero in enumerate(zeros):
                real, imag = zero
                if abs(imag) < 1e-6:
                    print(f"    Zero {j+1}: {real:.4f}")
                else:
                    print(f"    Zero {j+1}: {real:.4f} {'+' if imag >= 0 else ''}{imag:.4f}j")

        print()


def main(args):
    """Main generation function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("Targeted Transfer Function Circuit Generation")
    print(f"{'='*70}\n")

    # Parse target poles/zeros
    target_poles = [parse_complex(p) for p in args.poles] if args.poles else []
    target_zeros = [parse_complex(z) for z in args.zeros] if args.zeros else []

    print(f"Target Transfer Function:")
    print(f"  Poles ({len(target_poles)}):")
    for i, pole in enumerate(target_poles):
        print(f"    Pole {i+1}: {pole}")
    print(f"  Zeros ({len(target_zeros)}):")
    if target_zeros:
        for i, zero in enumerate(target_zeros):
            print(f"    Zero {i+1}: {zero}")
    else:
        print(f"    (None - all-pole filter)")

    print(f"\nSpecifications:")
    print(f"  Cutoff frequency: {args.cutoff} Hz")
    print(f"  Q-factor: {args.q_factor}")

    # Load TF encoder
    print(f"\n📂 Loading TF encoder from: {args.tf_encoder_checkpoint}")
    tf_checkpoint = torch.load(args.tf_encoder_checkpoint, map_location=device)

    tf_encoder = TransferFunctionEncoder(
        max_poles=tf_checkpoint['config']['max_poles'],
        max_zeros=tf_checkpoint['config']['max_zeros'],
        hidden_dim=tf_checkpoint['config']['hidden_dim'],
        latent_dim=tf_checkpoint['config']['latent_dim']
    ).to(device)

    tf_encoder.load_state_dict(tf_checkpoint['model_state_dict'])
    tf_encoder.eval()
    print("✅ TF encoder loaded")

    # Load decoder
    print(f"\n📂 Loading decoder from: {args.decoder_checkpoint}")
    decoder_checkpoint = torch.load(args.decoder_checkpoint, map_location=device)
    decoder_config = decoder_checkpoint['config']

    decoder = GraphGPTDecoder(
        latent_dim=decoder_config['model']['decoder']['latent_dim'],
        conditions_dim=decoder_config['model']['decoder']['conditions_dim'],
        hidden_dim=decoder_config['model']['decoder']['hidden_dim'],
        num_heads=decoder_config['model']['decoder']['num_heads'],
        num_node_layers=decoder_config['model']['decoder']['num_node_layers'],
        max_nodes=decoder_config['model']['decoder']['max_nodes'],
        max_poles=decoder_config['model']['decoder']['max_poles'],
        max_zeros=decoder_config['model']['decoder']['max_zeros'],
        dropout=decoder_config['model']['decoder']['dropout']
    ).to(device)

    decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
    decoder.eval()
    print("✅ Decoder loaded")

    # Generate circuits
    circuits = generate_with_target_tf(
        tf_encoder=tf_encoder,
        decoder=decoder,
        target_poles=target_poles,
        target_zeros=target_zeros,
        cutoff_freq=args.cutoff,
        q_factor=args.q_factor,
        num_samples=args.num_samples,
        topology_variation=args.topology_variation,
        values_variation=args.values_variation,
        device=device
    )

    # Print summary
    print_circuit_summary(circuits, args.num_samples)

    print(f"\n{'='*70}")
    print("✅ Generation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate circuits with targeted transfer functions')

    # Target TF specification
    parser.add_argument('--poles', nargs='+', type=str,
                       help='Target poles as complex numbers, e.g., "(-1000+2000j)" "(-1000-2000j)"')
    parser.add_argument('--zeros', nargs='+', type=str,
                       help='Target zeros as complex numbers, e.g., "(-500+0j)"')

    # Circuit specifications
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='Target cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Target Q-factor')

    # Generation settings
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of circuit variations to generate')
    parser.add_argument('--topology-variation', type=float, default=0.3,
                       help='Topology diversity (0=same, 1=very different)')
    parser.add_argument('--values-variation', type=float, default=0.3,
                       help='Component value diversity')

    # Model checkpoints
    parser.add_argument('--tf-encoder-checkpoint', type=str,
                       default='checkpoints/tf_encoder/best.pt',
                       help='Path to trained TF encoder')
    parser.add_argument('--decoder-checkpoint', type=str,
                       default='checkpoints/graphgpt_decoder/best.pt',
                       help='Path to trained decoder')

    # Device
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()
    main(args)
