"""
Generate circuits from high-level filter specifications.

Simple interface: Just specify filter type and specs!

Usage:
    python scripts/generate_circuit.py \\
        --filter-type butterworth \\
        --order 2 \\
        --cutoff 1000 \\
        --num-samples 10

That's it! No manual pole/zero calculation needed.
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.tf_encoder import TransferFunctionEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.utils.filter_design import calculate_filter_poles_zeros, poles_zeros_to_arrays


def print_filter_info(filter_type, order, cutoff, q_factor, poles, zeros):
    """Print filter design information."""
    print(f"\n{'='*70}")
    print(f"Filter Design")
    print(f"{'='*70}\n")

    print(f"Type: {filter_type.capitalize()}")
    print(f"Order: {order}")
    print(f"Cutoff: {cutoff} Hz")
    if q_factor is not None:
        print(f"Q-factor: {q_factor}")

    print(f"\nCalculated Transfer Function:")
    print(f"  Poles ({len(poles)}):")
    for i, pole in enumerate(poles):
        if abs(pole.imag) < 1e-6:
            print(f"    Pole {i+1}: {pole.real:.2f}")
        else:
            sign = '+' if pole.imag >= 0 else ''
            print(f"    Pole {i+1}: {pole.real:.2f} {sign}{pole.imag:.2f}j")

    if zeros:
        print(f"  Zeros ({len(zeros)}):")
        for i, zero in enumerate(zeros):
            if abs(zero.imag) < 1e-6:
                print(f"    Zero {i+1}: {zero.real:.2f}")
            else:
                sign = '+' if zero.imag >= 0 else ''
                print(f"    Zero {i+1}: {zero.real:.2f} {sign}{zero.imag:.2f}j")
    else:
        print(f"  Zeros: None (all-pole filter)")


def print_generation_summary(circuits, num_samples):
    """Print summary of generated circuits."""
    print(f"\n{'='*70}")
    print(f"Generation Summary")
    print(f"{'='*70}\n")

    # Topology diversity
    topologies = set()
    for i in range(num_samples):
        node_types = circuits['node_types'][i].cpu().numpy()
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()

        num_nodes = sum(1 for nt in node_types if nt != 4)
        num_edges = int((edge_matrix > 0.5).sum()) // 2

        topologies.add((num_nodes, num_edges))

    print(f"Generated: {num_samples} circuits")
    print(f"Topology diversity: {len(topologies)} unique structures")

    for topo in sorted(topologies):
        nodes, edges = topo
        count = sum(1 for i in range(num_samples)
                   if (sum(1 for nt in circuits['node_types'][i].cpu().numpy() if nt != 4) == nodes
                       and int((circuits['edge_existence'][i].cpu().numpy() > 0.5).sum()) // 2 == edges))
        print(f"  • {nodes} nodes, {edges} edges: {count} circuits")

    # Transfer function accuracy
    pole_counts = circuits['pole_count'].cpu().numpy()
    zero_counts = circuits['zero_count'].cpu().numpy()

    print(f"\nTransfer Function Accuracy:")
    print(f"  Pole count match: {sum(pole_counts == pole_counts[0])}/{num_samples}")
    print(f"  Zero count match: {sum(zero_counts == zero_counts[0])}/{num_samples}")

    # Stability
    stable_count = 0
    for i in range(num_samples):
        n_poles = pole_counts[i]
        if n_poles > 0:
            poles = circuits['pole_values'][i, :n_poles].cpu().numpy()
            if all(pole[0] < 0 for pole in poles):
                stable_count += 1
        else:
            stable_count += 1

    print(f"  Stable circuits: {stable_count}/{num_samples} ✓")


def print_sample_circuits(circuits, num_samples, max_display=3):
    """Print detailed info for first few circuits."""
    print(f"\n{'='*70}")
    print(f"Sample Circuits (showing first {min(max_display, num_samples)}):")
    print(f"{'='*70}\n")

    node_names = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INTERNAL', 4: 'MASK'}

    for i in range(min(max_display, num_samples)):
        print(f"Circuit {i}:")

        # Nodes
        node_types = circuits['node_types'][i].cpu().numpy()
        active_nodes = [node_names[int(nt)] for nt in node_types if nt != 4]
        print(f"  Nodes: {active_nodes}")

        # Edges
        edge_matrix = circuits['edge_existence'][i].cpu().numpy()
        num_edges = int((edge_matrix > 0.5).sum()) // 2
        print(f"  Edges: {num_edges}")

        # Transfer function
        pole_count = circuits['pole_count'][i].item()
        zero_count = circuits['zero_count'][i].item()

        print(f"  Transfer Function: {pole_count} poles, {zero_count} zeros")

        if pole_count > 0:
            poles = circuits['pole_values'][i, :pole_count].cpu().numpy()
            for j, pole in enumerate(poles):
                real, imag = pole
                if abs(imag) < 1e-6:
                    print(f"    Pole {j+1}: {real:.2f}")
                else:
                    sign = '+' if imag >= 0 else ''
                    print(f"    Pole {j+1}: {real:.2f} {sign}{imag:.2f}j")

        print()


def main(args):
    """Main generation function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("Circuit Generation from Filter Specifications")
    print(f"{'='*70}\n")

    # === Step 1: Calculate poles/zeros from filter specs ===
    print("📐 Calculating transfer function...")

    try:
        poles, zeros = calculate_filter_poles_zeros(
            filter_type=args.filter_type,
            order=args.order,
            cutoff_freq=args.cutoff,
            q_factor=args.q_factor,
            gain_db=args.gain_db,
            ripple_db=args.ripple_db
        )
    except Exception as e:
        print(f"❌ Error calculating filter: {e}")
        return

    print_filter_info(args.filter_type, args.order, args.cutoff, args.q_factor, poles, zeros)

    # Convert to arrays
    pole_array, num_poles, zero_array, num_zeros = poles_zeros_to_arrays(poles, zeros)

    # === Step 2: Load models ===
    print(f"\n📂 Loading models...")

    # Load TF encoder
    tf_checkpoint = torch.load(args.tf_encoder_checkpoint, map_location=device)
    tf_encoder = TransferFunctionEncoder(
        max_poles=tf_checkpoint['config']['max_poles'],
        max_zeros=tf_checkpoint['config']['max_zeros'],
        hidden_dim=tf_checkpoint['config']['hidden_dim'],
        latent_dim=tf_checkpoint['config']['latent_dim']
    ).to(device)
    tf_encoder.load_state_dict(tf_checkpoint['model_state_dict'])
    tf_encoder.eval()

    # Load decoder
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

    print("✅ Models loaded")

    # === Step 3: Encode TF to latent ===
    print(f"\n🎯 Encoding transfer function to latent space...")

    with torch.no_grad():
        pole_values = torch.from_numpy(pole_array).unsqueeze(0).to(device)  # [1, 4, 2]
        zero_values = torch.from_numpy(zero_array).unsqueeze(0).to(device)  # [1, 4, 2]
        pole_count = torch.tensor([num_poles], device=device)
        zero_count = torch.tensor([num_zeros], device=device)

        tf_latent = tf_encoder.encode(
            pole_values, pole_count, zero_values, zero_count,
            deterministic=True
        )

    print(f"✅ TF encoded to latent[4:8]: {tf_latent.cpu().numpy()[0]}")

    # === Step 4: Generate circuits ===
    print(f"\n🎨 Generating {args.num_samples} circuit variations...")

    # Sample random topology/values latents
    all_latents = []
    for i in range(args.num_samples):
        topo_latent = torch.randn(1, 2, device=device) * args.topology_variation
        values_latent = torch.randn(1, 2, device=device) * args.values_variation

        # Combine: [topology | values | TF]
        full_latent = torch.cat([topo_latent, values_latent, tf_latent], dim=1)
        all_latents.append(full_latent)

    all_latents = torch.cat(all_latents, dim=0)  # [num_samples, 8]

    # Prepare conditions
    log_cutoff = np.log10(args.cutoff) / 4.0
    log_q = np.log10(max(args.q_factor if args.q_factor else 0.707, 0.1)) / 2.0

    conditions = torch.tensor(
        [[log_cutoff, log_q]] * args.num_samples,
        dtype=torch.float32,
        device=device
    )

    # Generate
    with torch.no_grad():
        circuits = decoder.generate(
            latent_code=all_latents,
            conditions=conditions,
            enforce_constraints=True,
            edge_threshold=0.5
        )

    print("✅ Generation complete!")

    # === Step 5: Display results ===
    print_generation_summary(circuits, args.num_samples)
    print_sample_circuits(circuits, args.num_samples, max_display=3)

    # === Step 6: Optionally export to SPICE ===
    if args.export_spice:
        print(f"\n💾 Exporting to SPICE netlists...")
        # TODO: Integrate with export_spice_netlists.py
        print("   (SPICE export coming soon - use scripts/export_spice_netlists.py manually)")

    print(f"\n{'='*70}")
    print("✅ All done!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate circuits from high-level filter specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2nd-order Butterworth lowpass at 1kHz
  python scripts/generate_circuit.py --filter-type butterworth --order 2 --cutoff 1000

  # 3rd-order Bessel lowpass at 5kHz
  python scripts/generate_circuit.py --filter-type bessel --order 3 --cutoff 5000

  # Notch filter rejecting 60Hz
  python scripts/generate_circuit.py --filter-type notch --order 2 --cutoff 60 --q-factor 10

  # High-Q bandpass (coming soon)
  python scripts/generate_circuit.py --filter-type chebyshev --order 2 --cutoff 1000 --q-factor 2
        """
    )

    # Filter specifications
    parser.add_argument('--filter-type', type=str, required=True,
                       choices=['butterworth', 'butter', 'bessel', 'chebyshev', 'cheby',
                               'notch', 'lowpass', 'highpass', 'bandpass'],
                       help='Filter type (butterworth, bessel, chebyshev, notch, etc.)')
    parser.add_argument('--order', type=int, required=True,
                       help='Filter order (1-4)')
    parser.add_argument('--cutoff', type=float, required=True,
                       help='Cutoff frequency in Hz')
    parser.add_argument('--q-factor', type=float, default=None,
                       help='Q-factor (default: 0.707 for Butterworth)')
    parser.add_argument('--gain-db', type=float, default=0.0,
                       help='DC gain in dB (default: 0)')
    parser.add_argument('--ripple-db', type=float, default=1.0,
                       help='Passband ripple for Chebyshev in dB (default: 1.0)')

    # Generation settings
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of circuit variations to generate (default: 10)')
    parser.add_argument('--topology-variation', type=float, default=0.5,
                       help='Topology diversity (0=same, 1=very different, default: 0.5)')
    parser.add_argument('--values-variation', type=float, default=0.3,
                       help='Component value diversity (default: 0.3)')

    # Model checkpoints
    parser.add_argument('--tf-encoder-checkpoint', type=str,
                       default='checkpoints/tf_encoder/20251227_203319/best.pt',
                       help='Path to trained TF encoder')
    parser.add_argument('--decoder-checkpoint', type=str,
                       default='checkpoints/graphgpt_decoder/best.pt',
                       help='Path to trained decoder')

    # Export options
    parser.add_argument('--export-spice', action='store_true',
                       help='Export circuits to SPICE netlists')

    # Device
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    # Set default Q-factor based on filter type
    if args.q_factor is None:
        if args.filter_type in ['butterworth', 'butter', 'lowpass']:
            args.q_factor = 0.707  # Butterworth
        elif args.filter_type == 'bessel':
            args.q_factor = 0.577  # Bessel
        elif args.filter_type in ['chebyshev', 'cheby']:
            args.q_factor = 0.8636  # Chebyshev 1dB
        elif args.filter_type == 'notch':
            args.q_factor = 10.0  # High Q for sharp notch
        else:
            args.q_factor = 0.707

    main(args)
