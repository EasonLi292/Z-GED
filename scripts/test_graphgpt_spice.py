"""
Test GraphGPT generated circuits with SPICE simulation.

Converts GraphGPT output to SPICE netlists and runs AC analysis
to verify circuits are electrically valid.
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

# PySpice imports
try:
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import u_Ohm, u_H, u_F
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    PYSPICE_AVAILABLE = True
except ImportError:
    print("⚠️  PySpice not available. Install with: pip install PySpice")
    PYSPICE_AVAILABLE = False


NODE_TYPE_NAMES = {
    0: 'GND',
    1: 'VIN',
    2: 'VOUT',
    3: 'INTERNAL',
    4: 'MASK'
}


def denormalize_impedance(log_values, mean, std):
    """Denormalize log-scaled impedance values."""
    # Reverse normalization: x = (log_val * std) + mean
    # Then reverse log: val = exp(x)
    denorm_log = (log_values * std) + mean
    return np.exp(denorm_log)


def graphgpt_to_spice(circuit_dict, circuit_id, normalization_stats):
    """
    Convert GraphGPT circuit to PySpice Circuit object.

    Args:
        circuit_dict: Dictionary with circuit structure from GraphGPT
        circuit_id: Unique identifier for this circuit
        normalization_stats: Dictionary with impedance mean/std

    Returns:
        PySpice Circuit object
    """
    if not PYSPICE_AVAILABLE:
        raise ImportError("PySpice required for SPICE simulation")

    circuit = Circuit(f'GraphGPT_Circuit_{circuit_id}')

    # Get normalization stats
    imp_mean = normalization_stats['impedance_mean']
    imp_std = normalization_stats['impedance_std']

    # Node mapping: GraphGPT node indices -> SPICE node numbers
    # GND=0, VIN=1, VOUT=2, INTERNAL nodes get sequential numbers
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
        # Skip MASK nodes (ntype == 4)

    # Add voltage source at VIN (node 1) referenced to GND (node 0)
    circuit.SinusoidalVoltageSource('in', 1, 0, amplitude=1.0, ac_magnitude=1.0)

    # Process edges
    edge_existence = circuit_dict['edge_existence']
    edge_values = circuit_dict['edge_values']

    comp_counter = {'R': 0, 'L': 0, 'C': 0}

    for i in range(len(node_types)):
        if node_types[i] == 4:  # Skip MASK nodes
            continue
        for j in range(i+1, len(node_types)):
            if node_types[j] == 4:  # Skip MASK nodes
                continue

            if edge_existence[i, j] > 0.5:  # Edge exists
                n1 = node_map[i]
                n2 = node_map[j]

                # Get edge values: [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
                C_log, G_log, L_inv_log = edge_values[i, j, :3]
                has_C, has_R, has_L = edge_values[i, j, 3:6]

                # Denormalize
                C_denorm = denormalize_impedance(C_log, imp_mean[0], imp_std[0])
                G_denorm = denormalize_impedance(G_log, imp_mean[1], imp_std[1])
                L_inv_denorm = denormalize_impedance(L_inv_log, imp_mean[2], imp_std[2])

                # Convert to component values
                C = C_denorm  # Capacitance in Farads
                R = 1.0 / (G_denorm + 1e-12)  # Resistance in Ohms (G = 1/R)
                L = 1.0 / (L_inv_denorm + 1e-12)  # Inductance in Henrys

                # Add components if they exist (based on masks)
                # In parallel configuration, all components are between same nodes
                if has_C > 0.5:
                    comp_counter['C'] += 1
                    circuit.Capacitor(f'C{comp_counter["C"]}', n1, n2, C@u_F)

                if has_R > 0.5:
                    comp_counter['R'] += 1
                    circuit.Resistor(f'R{comp_counter["R"]}', n1, n2, R@u_Ohm)

                if has_L > 0.5:
                    comp_counter['L'] += 1
                    circuit.Inductor(f'L{comp_counter["L"]}', n1, n2, L@u_H)

    return circuit


def run_ac_analysis(circuit, freq_start=1, freq_stop=100e3, num_points=100):
    """
    Run AC analysis on circuit.

    Args:
        circuit: PySpice Circuit object
        freq_start: Start frequency (Hz)
        freq_stop: Stop frequency (Hz)
        num_points: Number of frequency points

    Returns:
        (frequencies, magnitude, phase) or None if simulation fails
    """
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        # AC analysis
        analysis = simulator.ac(
            start_frequency=freq_start,
            stop_frequency=freq_stop,
            number_of_points=num_points,
            variation='dec'  # Decade variation
        )

        # Get output voltage (at node 2 = VOUT)
        v_out = np.array(analysis['2'])

        # Calculate magnitude and phase
        magnitude = np.abs(v_out)
        phase = np.angle(v_out, deg=True)
        frequencies = np.array(analysis.frequency.as_ndarray())

        return frequencies, magnitude, phase

    except Exception as e:
        print(f"   ❌ Simulation failed: {str(e)}")
        return None


def test_circuit(circuit_dict, circuit_id, normalization_stats):
    """Test a single generated circuit."""
    print(f"\n{'='*70}")
    print(f"Testing Circuit {circuit_id}")
    print(f"{'='*70}")

    # Print circuit structure
    node_types = circuit_dict['node_types']
    edge_existence = circuit_dict['edge_existence']

    print("\nCircuit Structure:")
    print(f"  Nodes: {[NODE_TYPE_NAMES[int(nt)] for nt in node_types if nt != 4]}")

    num_edges = int((edge_existence > 0.5).sum()) // 2
    print(f"  Edges: {num_edges}")

    # Convert to SPICE
    try:
        spice_circuit = graphgpt_to_spice(circuit_dict, circuit_id, normalization_stats)
        print("  ✅ SPICE netlist created")
    except Exception as e:
        print(f"  ❌ Failed to create SPICE netlist: {e}")
        return False

    # Run simulation
    print("\nRunning AC Analysis...")
    result = run_ac_analysis(spice_circuit)

    if result is None:
        return False

    frequencies, magnitude, phase = result

    # Analyze results
    print("  ✅ Simulation successful")
    print(f"\nFrequency Response:")
    print(f"  DC gain (1 Hz): {magnitude[0]:.4f}")
    print(f"  Gain at 1 kHz: {magnitude[len(magnitude)//2]:.4f}")
    print(f"  Gain at 100 kHz: {magnitude[-1]:.4f}")

    # Check if circuit has filtering behavior
    gain_ratio = magnitude[0] / (magnitude[-1] + 1e-10)
    if gain_ratio > 2:
        print(f"  📊 Low-pass behavior (DC/HF ratio: {gain_ratio:.2f})")
    elif gain_ratio < 0.5:
        print(f"  📊 High-pass behavior (DC/HF ratio: {gain_ratio:.2f})")
    else:
        print(f"  📊 Flat response (DC/HF ratio: {gain_ratio:.2f})")

    return True


def main(args):
    """Main testing function."""
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print("GraphGPT SPICE Validation")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print("\n📂 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

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

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")

    # Get normalization stats from dataset
    from ml.data.dataset import CircuitDataset
    dataset = CircuitDataset(config['data']['dataset_path'])
    normalization_stats = {
        'impedance_mean': dataset.impedance_mean.cpu().numpy(),
        'impedance_std': dataset.impedance_std.cpu().numpy()
    }

    print(f"\nNormalization stats loaded:")
    print(f"  Mean: {normalization_stats['impedance_mean']}")
    print(f"  Std:  {normalization_stats['impedance_std']}")

    # Generate circuits
    print(f"\n🎨 Generating {args.num_samples} circuits...")

    latent = torch.randn(args.num_samples, config['model']['decoder']['latent_dim'], device=device)

    # Normalize specifications
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

    print(f"✅ Generated {args.num_samples} circuits")

    # Test each circuit
    success_count = 0
    for i in range(args.num_samples):
        circuit_dict = {
            'node_types': circuits['node_types'][i].cpu().numpy(),
            'edge_existence': circuits['edge_existence'][i].cpu().numpy(),
            'edge_values': circuits['edge_values'][i].cpu().numpy()
        }

        if test_circuit(circuit_dict, i, normalization_stats):
            success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print("SPICE Validation Summary")
    print(f"{'='*70}")
    print(f"Total circuits generated: {args.num_samples}")
    print(f"Successfully simulated: {success_count}/{args.num_samples} ({100*success_count/args.num_samples:.1f}%)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GraphGPT circuits with SPICE')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/graphgpt_decoder/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--cutoff', type=float, default=1000.0,
                       help='Target cutoff frequency (Hz)')
    parser.add_argument('--q-factor', type=float, default=0.707,
                       help='Target Q-factor')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of circuits to generate and test')
    parser.add_argument('--edge-threshold', type=float, default=0.5,
                       help='Threshold for edge existence')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')

    args = parser.parse_args()
    main(args)
