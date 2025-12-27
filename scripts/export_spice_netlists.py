"""
Export GraphGPT circuits to SPICE netlist files.

Creates standard SPICE .cir files that can be simulated with
any SPICE simulator (ngspice, LTspice, etc.)
"""

import argparse
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.data.dataset import CircuitDataset


NODE_TYPE_NAMES = {
    0: 'GND',
    1: 'VIN',
    2: 'VOUT',
    3: 'INTERNAL',
    4: 'MASK'
}


def denormalize_impedance(log_values, mean, std):
    """Denormalize log-scaled impedance values."""
    denorm_log = (log_values * std) + mean
    return np.exp(denorm_log)


def format_value(value, unit=''):
    """Format component value in engineering notation."""
    if value >= 1e6:
        return f"{value/1e6:.6g}Meg{unit}"
    elif value >= 1e3:
        return f"{value/1e3:.6g}k{unit}"
    elif value >= 1:
        return f"{value:.6g}{unit}"
    elif value >= 1e-3:
        return f"{value*1e3:.6g}m{unit}"
    elif value >= 1e-6:
        return f"{value*1e6:.6g}u{unit}"
    elif value >= 1e-9:
        return f"{value*1e9:.6g}n{unit}"
    elif value >= 1e-12:
        return f"{value*1e12:.6g}p{unit}"
    else:
        return f"{value:.6g}{unit}"


def export_spice_netlist(circuit_dict, circuit_id, normalization_stats, output_dir):
    """
    Export circuit to SPICE netlist file.

    Args:
        circuit_dict: Dictionary with circuit structure
        circuit_id: Circuit identifier
        normalization_stats: Impedance normalization parameters
        output_dir: Output directory for .cir files

    Returns:
        (filepath, num_components, is_valid)
    """
    # Get normalization stats
    imp_mean = normalization_stats['impedance_mean']
    imp_std = normalization_stats['impedance_std']

    # Node mapping
    node_types = circuit_dict['node_types']
    node_map = {}
    internal_counter = 3

    for i, ntype in enumerate(node_types):
        if ntype == 0:  # GND
            node_map[i] = 0
        elif ntype == 1:  # VIN
            node_map[i] = 1
        elif ntype == 2:  # VOUT
            node_map[i] = 2
        elif ntype == 3:  # INTERNAL
            node_map[i] = internal_counter
            internal_counter += 1

    # Build netlist
    netlist_lines = []
    netlist_lines.append(f"* GraphGPT Generated Circuit {circuit_id}")
    netlist_lines.append(f"* Auto-generated from trained GraphGPT model")
    netlist_lines.append("")

    # Circuit title
    netlist_lines.append(f".title GraphGPT_Circuit_{circuit_id}")
    netlist_lines.append("")

    # Voltage source
    netlist_lines.append("* Input voltage source")
    netlist_lines.append("Vin 1 0 AC 1.0")
    netlist_lines.append("")

    # Process edges
    edge_existence = circuit_dict['edge_existence']
    edge_values = circuit_dict['edge_values']

    comp_counter = {'R': 0, 'L': 0, 'C': 0}
    components = []

    netlist_lines.append("* Circuit components")

    for i in range(len(node_types)):
        if node_types[i] == 4:  # Skip MASK
            continue
        for j in range(i+1, len(node_types)):
            if node_types[j] == 4:  # Skip MASK
                continue

            if edge_existence[i, j] > 0.5:  # Edge exists
                n1 = node_map[i]
                n2 = node_map[j]

                # Get edge values
                C_log, G_log, L_inv_log = edge_values[i, j, :3]
                has_C, has_R, has_L = edge_values[i, j, 3:6]

                # Denormalize
                C_denorm = denormalize_impedance(C_log, imp_mean[0], imp_std[0])
                G_denorm = denormalize_impedance(G_log, imp_mean[1], imp_std[1])
                L_inv_denorm = denormalize_impedance(L_inv_log, imp_mean[2], imp_std[2])

                # Convert to component values
                C = C_denorm  # Farads
                R = 1.0 / (G_denorm + 1e-12)  # Ohms
                L = 1.0 / (L_inv_denorm + 1e-12)  # Henrys

                # Add components
                if has_C > 0.5:
                    comp_counter['C'] += 1
                    netlist_lines.append(f"C{comp_counter['C']} {n1} {n2} {format_value(C, 'F')}")
                    components.append(('C', C))

                if has_R > 0.5:
                    comp_counter['R'] += 1
                    netlist_lines.append(f"R{comp_counter['R']} {n1} {n2} {format_value(R, 'Ohm')}")
                    components.append(('R', R))

                if has_L > 0.5:
                    comp_counter['L'] += 1
                    netlist_lines.append(f"L{comp_counter['L']} {n1} {n2} {format_value(L, 'H')}")
                    components.append(('L', L))

    netlist_lines.append("")

    # AC analysis commands
    netlist_lines.append("* AC Analysis from 1Hz to 100kHz")
    netlist_lines.append(".ac dec 100 1 100k")
    netlist_lines.append("")

    # Print output voltage
    netlist_lines.append("* Output")
    netlist_lines.append(".print ac v(2)")
    netlist_lines.append("")
    netlist_lines.append(".end")

    # Write to file
    output_file = Path(output_dir) / f"circuit_{circuit_id}.cir"
    with open(output_file, 'w') as f:
        f.write('\n'.join(netlist_lines))

    num_components = sum(comp_counter.values())
    is_valid = num_components > 0

    return output_file, num_components, components, is_valid


def analyze_circuit_components(components):
    """Analyze component values for practical validity."""
    analysis = {
        'has_resistors': False,
        'has_capacitors': False,
        'has_inductors': False,
        'practical_resistors': 0,
        'practical_capacitors': 0,
        'practical_inductors': 0,
        'total_components': len(components)
    }

    # Practical ranges (from real filter designs)
    R_RANGE = (10, 100e3)      # 10Ω to 100kΩ
    C_RANGE = (1e-12, 1e-6)    # 1pF to 1μF
    L_RANGE = (1e-9, 10e-3)    # 1nH to 10mH

    for comp_type, value in components:
        if comp_type == 'R':
            analysis['has_resistors'] = True
            if R_RANGE[0] <= value <= R_RANGE[1]:
                analysis['practical_resistors'] += 1
        elif comp_type == 'C':
            analysis['has_capacitors'] = True
            if C_RANGE[0] <= value <= C_RANGE[1]:
                analysis['practical_capacitors'] += 1
        elif comp_type == 'L':
            analysis['has_inductors'] = True
            if L_RANGE[0] <= value <= L_RANGE[1]:
                analysis['practical_inductors'] += 1

    return analysis


def main(args):
    """Main export function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("GraphGPT SPICE Netlist Export")
    print(f"{'='*70}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Load model
    print("\n📂 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

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

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")

    # Get normalization stats
    dataset = CircuitDataset(config['data']['dataset_path'])
    normalization_stats = {
        'impedance_mean': dataset.impedance_mean.cpu().numpy(),
        'impedance_std': dataset.impedance_std.cpu().numpy()
    }

    # Generate circuits
    print(f"\n🎨 Generating {args.num_samples} circuits...")

    latent = torch.randn(args.num_samples, config['model']['decoder']['latent_dim'], device=device)

    log_cutoff = np.log10(args.cutoff) / 4.0
    log_q = np.log10(max(args.q_factor, 0.1)) / 2.0

    conditions = torch.tensor(
        [[log_cutoff, log_q]] * args.num_samples,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        circuits = decoder.generate(
            latent_code=latent,
            conditions=conditions,
            enforce_constraints=True,
            edge_threshold=args.edge_threshold
        )

    print(f"✅ Generated {args.num_samples} circuits\n")

    # Export each circuit
    valid_count = 0
    practical_count = 0

    for i in range(args.num_samples):
        circuit_dict = {
            'node_types': circuits['node_types'][i].cpu().numpy(),
            'edge_existence': circuits['edge_existence'][i].cpu().numpy(),
            'edge_values': circuits['edge_values'][i].cpu().numpy()
        }

        filepath, num_comp, components, is_valid = export_spice_netlist(
            circuit_dict, i, normalization_stats, output_dir
        )

        if is_valid:
            valid_count += 1
            analysis = analyze_circuit_components(components)

            print(f"Circuit {i}:")
            print(f"  ✅ Exported: {filepath.name}")
            print(f"  Components: {num_comp} total")
            print(f"    R: {analysis['practical_resistors']}/{sum(1 for c,_ in components if c=='R')} practical")
            print(f"    C: {analysis['practical_capacitors']}/{sum(1 for c,_ in components if c=='C')} practical")
            print(f"    L: {analysis['practical_inductors']}/{sum(1 for c,_ in components if c=='L')} practical")

            total_practical = (analysis['practical_resistors'] +
                             analysis['practical_capacitors'] +
                             analysis['practical_inductors'])

            if total_practical == num_comp:
                practical_count += 1
                print(f"  🎯 All components in practical range!")
            print()

    # Summary
    print(f"{'='*70}")
    print("Export Summary")
    print(f"{'='*70}")
    print(f"Total circuits: {args.num_samples}")
    print(f"Valid circuits: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.1f}%)")
    print(f"Practical circuits: {practical_count}/{args.num_samples} ({100*practical_count/args.num_samples:.1f}%)")
    print(f"\n💡 To simulate, run:")
    print(f"   ngspice {output_dir}/circuit_0.cir")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export GraphGPT circuits to SPICE netlists')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/graphgpt_decoder/best.pt')
    parser.add_argument('--cutoff', type=float, default=1000.0)
    parser.add_argument('--q-factor', type=float, default=0.707)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--edge-threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='generated_circuits')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()
    main(args)
