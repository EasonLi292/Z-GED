"""
Evaluate ACTUAL Transfer Function of Generated Circuits.

This script:
1. Generates circuits from filter specifications
2. Analyzes each circuit using nodal analysis (not decoder predictions)
3. Compares actual circuit TF with target TF
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.models.tf_encoder import TransferFunctionEncoder
from ml.utils.filter_design import calculate_filter_poles_zeros, poles_zeros_to_arrays
from ml.utils.circuit_analysis import compute_transfer_function_simple, compare_transfer_functions
import argparse

def complex_to_str(c):
    """Format complex number for display."""
    if abs(c.imag) < 1e-6:
        return f"{c.real:.2f}"
    elif c.imag > 0:
        return f"{c.real:.2f} +{c.imag:.2f}j"
    else:
        return f"{c.real:.2f} {c.imag:.2f}j"

def denormalize_edge_values(edge_values_norm, impedance_mean, impedance_std):
    """
    Denormalize edge values from decoder output.

    Args:
        edge_values_norm: [7] array with [C_norm, G_norm, L_inv_norm, has_C, has_R, has_L, parallel]
        impedance_mean: [3] array of [C_mean, G_mean, L_inv_mean]
        impedance_std: [3] array of [C_std, G_std, L_inv_std]

    Returns:
        (C, R, L) in actual units (Farads, Ohms, Henries)
    """
    C_norm, G_norm, L_inv_norm = edge_values_norm[:3]
    has_C, has_R, has_L = edge_values_norm[3:6]

    # Denormalize from z-score
    log_C = C_norm * impedance_std[0] + impedance_mean[0]
    log_G = G_norm * impedance_std[1] + impedance_mean[1]
    log_L_inv = L_inv_norm * impedance_std[2] + impedance_mean[2]

    # Exponentiate
    C = np.exp(log_C) if has_C > 0.5 else 0.0
    G = np.exp(log_G)
    L_inv = np.exp(log_L_inv) if has_L > 0.5 else 0.0

    # Convert to R and L
    R = 1.0 / (G + 1e-15) if has_R > 0.5 else 1e15
    L = 1.0 / (L_inv + 1e-15) if has_L > 0.5 else 1e15

    return C, R, L

def build_circuit_graph_from_decoder(outputs, impedance_mean, impedance_std):
    """
    Build circuit graph from decoder output.

    Returns graph in format expected by compute_transfer_function_simple.
    """
    node_types = outputs['node_types'][0].cpu().numpy()  # [max_nodes]
    edge_existence = outputs['edge_existence'][0].cpu().numpy()  # [max_nodes, max_nodes]
    edge_values = outputs['edge_values'][0].cpu().numpy()  # [max_nodes, max_nodes, 7]

    num_nodes = len(node_types)

    # Node type mapping
    node_type_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']

    # Build nodes list
    nodes = []
    for i in range(num_nodes):
        if node_types[i] < 4:  # Not MASK
            nodes.append({
                'id': i,
                'type': node_type_names[node_types[i]],
                'features': [1 if j == node_types[i] else 0 for j in range(4)]
            })

    # Build adjacency list
    adjacency = [[] for _ in range(num_nodes)]

    for src in range(num_nodes):
        for dst in range(num_nodes):
            if edge_existence[src, dst] > 0.5 and node_types[src] < 4 and node_types[dst] < 4:
                # Denormalize edge values
                edge_norm = edge_values[src, dst]
                C, R, L = denormalize_edge_values(edge_norm, impedance_mean, impedance_std)

                # Check which components are present
                has_C = edge_norm[3] > 0.5
                has_R = edge_norm[4] > 0.5
                has_L = edge_norm[5] > 0.5

                # Create impedance_den [C, G, L_inv]
                impedance_den = [
                    C if has_C else 0.0,
                    1.0/R if has_R else 0.0,
                    1.0/L if has_L else 0.0
                ]

                adjacency[src].append({
                    'id': dst,
                    'impedance_den': impedance_den
                })

    return {
        'nodes': nodes,
        'adjacency': adjacency
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-type', type=str, default='butterworth')
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--cutoff', type=float, default=1000.0)
    parser.add_argument('--q-factor', type=float, default=0.707)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--tf-encoder-checkpoint', type=str,
                       default='checkpoints/tf_encoder/20251228_002821/best.pt')
    parser.add_argument('--decoder-checkpoint', type=str,
                       default='checkpoints/graphgpt_decoder_new_norm/best.pt')
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    device = args.device

    print("="*70)
    print("ACTUAL Transfer Function Evaluation")
    print("(Analyzing generated circuits, not decoder predictions)")
    print("="*70)

    # Calculate target transfer function
    print(f"\n📐 Target Filter Specification:")
    print(f"   Type: {args.filter_type.capitalize()}")
    print(f"   Order: {args.order}")
    print(f"   Cutoff: {args.cutoff} Hz")
    print(f"   Q-factor: {args.q_factor}")

    target_poles, target_zeros = calculate_filter_poles_zeros(
        filter_type=args.filter_type,
        order=args.order,
        cutoff_freq=args.cutoff,
        q_factor=args.q_factor
    )

    print(f"\n🎯 Target Transfer Function:")
    print(f"   Poles ({len(target_poles)}):")
    for i, p in enumerate(target_poles):
        print(f"      #{i+1}: {complex_to_str(p)}")
    if target_zeros:
        print(f"   Zeros ({len(target_zeros)}):")
        for i, z in enumerate(target_zeros):
            print(f"      #{i+1}: {complex_to_str(z)}")
    else:
        print(f"   Zeros: None (all-pole filter)")

    # Load models
    print(f"\n📂 Loading models...")

    checkpoint = torch.load(args.decoder_checkpoint, map_location=device)
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
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    decoder = GraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=5,
        max_poles=4,
        max_zeros=4,
        dropout=0.1
    ).to(device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    tf_checkpoint = torch.load(args.tf_encoder_checkpoint, map_location=device)
    tf_encoder = TransferFunctionEncoder(
        max_poles=tf_checkpoint['config']['max_poles'],
        max_zeros=tf_checkpoint['config']['max_zeros'],
        hidden_dim=tf_checkpoint['config']['hidden_dim'],
        latent_dim=tf_checkpoint['config']['latent_dim']
    ).to(device)
    tf_encoder.load_state_dict(tf_checkpoint['model_state_dict'])
    tf_encoder.eval()

    # Get normalization stats
    from ml.data.dataset import CircuitDataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl', normalize_features=True)
    impedance_mean = dataset.impedance_mean.cpu().numpy()
    impedance_std = dataset.impedance_std.cpu().numpy()

    print(f"✅ Models loaded")

    # Encode target TF to latent
    print(f"\n🎯 Encoding target TF to latent space...")

    # Convert poles/zeros to padded arrays
    pole_array, num_poles, zero_array, num_zeros = poles_zeros_to_arrays(
        target_poles, target_zeros,
        max_poles=tf_checkpoint['config']['max_poles'],
        max_zeros=tf_checkpoint['config']['max_zeros']
    )

    with torch.no_grad():
        pole_values = torch.from_numpy(pole_array).unsqueeze(0).to(device)  # [1, 4, 2]
        zero_values = torch.from_numpy(zero_array).unsqueeze(0).to(device)  # [1, 4, 2]
        pole_count = torch.tensor([num_poles], device=device)
        zero_count = torch.tensor([num_zeros], device=device)

        mu_pz = tf_encoder.encode(pole_values, pole_count, zero_values, zero_count,
                                  deterministic=True)

    # Generate circuits and analyze
    print(f"\n🎨 Generating {args.num_samples} circuits and analyzing ACTUAL transfer functions...")

    # Prepare conditions (cutoff, Q)
    conditions = torch.tensor([[
        np.log10(args.cutoff) / 4.0,  # Normalized cutoff
        np.log10(args.q_factor) / 2.0  # Normalized Q
    ]], dtype=torch.float32, device=device)

    results = []
    valid_circuits = 0
    analysis_failures = 0

    for i in range(args.num_samples):
        # Sample topology/values latent
        z_topo = torch.randn(1, 2, device=device) * 0.5
        z_values = torch.randn(1, 2, device=device) * 0.3
        z_latent = torch.cat([z_topo, z_values, mu_pz], dim=1)

        # Generate circuit
        with torch.no_grad():
            outputs = decoder.generate(z_latent, conditions)

        # Debug raw decoder output for first circuit
        if i == 0:
            node_types = outputs['node_types'][0].cpu().numpy()
            edge_existence = outputs['edge_existence'][0].cpu().numpy()
            print(f"\n" + "="*70)
            print(f"RAW DECODER OUTPUT (Circuit 0):")
            print(f"Node types: {node_types}")
            print(f"Edge existence matrix:")
            for src in range(len(node_types)):
                for dst in range(len(node_types)):
                    if edge_existence[src, dst] > 0.5:
                        print(f"  Edge {src}→{dst}: exists (prob={edge_existence[src, dst]:.3f})")
            print("="*70)

        # Build circuit graph
        try:
            circuit_graph = build_circuit_graph_from_decoder(outputs, impedance_mean, impedance_std)

            # Analyze ACTUAL circuit transfer function using nodal analysis
            actual_poles, actual_zeros, freqs, H_mag = compute_transfer_function_simple(
                circuit_graph,
                freq_range=np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz
            )

            # Debug: check circuit structure and frequency response
            if i < 3:  # Show debug for first 3 circuits
                print(f"\n" + "-"*70)
                print(f"Circuit {i} DEBUG:")
                print(f"  Nodes: {len(circuit_graph['nodes'])}")
                print(f"  Node types: {[n['type'] for n in circuit_graph['nodes']]}")

                # Count edges
                total_edges = sum(len(neighbors) for neighbors in circuit_graph['adjacency'])
                print(f"  Edges: {total_edges}")

                # Show edge details
                if total_edges > 0 and total_edges < 10:
                    print(f"  Edge details:")
                    for src_id, neighbors in enumerate(circuit_graph['adjacency']):
                        for edge in neighbors:
                            dst_id = edge['id']
                            C, G, L_inv = edge['impedance_den']
                            R = 1.0 / (G + 1e-15) if G > 1e-10 else float('inf')
                            L = 1.0 / (L_inv + 1e-15) if L_inv > 1e-10 else float('inf')
                            print(f"    {src_id}→{dst_id}: C={C:.2e}F, R={R:.2e}Ω, L={L:.2e}H")

                print(f"  Frequency response: min={np.min(H_mag):.6f}, max={np.max(H_mag):.6f}, mean={np.mean(H_mag):.6f}")
                print(f"  Extracted poles: {len(actual_poles)}")
                print(f"  Extracted zeros: {len(actual_zeros)}")

            if len(actual_poles) == 0 and len(actual_zeros) == 0:
                analysis_failures += 1
                results.append({
                    'circuit_id': i,
                    'valid': False,
                    'error': 'TF analysis failed (no poles/zeros extracted)',
                    'freq_response_valid': np.max(H_mag) > 1e-10
                })
                continue

            # Compare with target
            comparison = compare_transfer_functions(
                target_poles, target_zeros,
                actual_poles, actual_zeros
            )

            results.append({
                'circuit_id': i,
                'valid': True,
                'actual_poles': actual_poles,
                'actual_zeros': actual_zeros,
                'comparison': comparison,
                'freq_response': (freqs, H_mag)
            })

            valid_circuits += 1

            # Show first 3 circuits in detail
            if valid_circuits <= 3:
                print(f"\n" + "-"*70)
                print(f"Circuit {i}:")
                print(f"  Actual Poles ({len(actual_poles)}):")
                for j, p in enumerate(actual_poles):
                    print(f"     #{j+1}: {complex_to_str(p)} (|z|={abs(p):.1f})")

                if len(actual_zeros) > 0:
                    print(f"  Actual Zeros ({len(actual_zeros)}):")
                    for j, z in enumerate(actual_zeros):
                        print(f"     #{j+1}: {complex_to_str(z)}")

                # Show pole comparison
                if comparison['pole_errors']:
                    print(f"\n  Pole Comparison:")
                    for err in comparison['pole_errors']:
                        print(f"     Target:   {complex_to_str(err['target'])} (|z|={abs(err['target']):.1f})")
                        print(f"     Actual:   {complex_to_str(err['actual'])} (|z|={abs(err['actual']):.1f})")
                        print(f"     Mag error: {100*err['mag_error']:.1f}%")
                        print(f"     Phase error: {err['phase_error']*180/np.pi:.1f}°")
                        print()

        except Exception as e:
            analysis_failures += 1
            results.append({
                'circuit_id': i,
                'valid': False,
                'error': str(e)
            })

    # Summary statistics
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    print(f"\n📊 Generation Statistics:")
    print(f"   Valid circuits: {valid_circuits}/{args.num_samples} ({100*valid_circuits/args.num_samples:.1f}%)")
    print(f"   Analysis failures: {analysis_failures}/{args.num_samples} ({100*analysis_failures/args.num_samples:.1f}%)")

    if valid_circuits > 0:
        valid_results = [r for r in results if r.get('valid', False)]

        # Count accuracy
        pole_count_matches = sum(1 for r in valid_results if r['comparison']['pole_count_match'])
        zero_count_matches = sum(1 for r in valid_results if r['comparison']['zero_count_match'])

        print(f"\n📊 Pole/Zero Count Accuracy:")
        print(f"   Pole count match: {pole_count_matches}/{valid_circuits} = {100*pole_count_matches/valid_circuits:.1f}%")
        print(f"   Zero count match: {zero_count_matches}/{valid_circuits} = {100*zero_count_matches/valid_circuits:.1f}%")

        # Pole location errors
        all_pole_errors = []
        for r in valid_results:
            if r['comparison']['pole_errors']:
                for err in r['comparison']['pole_errors']:
                    all_pole_errors.append(err['mag_error'])

        if all_pole_errors:
            all_pole_errors = np.array(all_pole_errors)
            print(f"\n📊 Pole Location Error:")
            print(f"   Mean: {100*np.mean(all_pole_errors):.1f}%")
            print(f"   Median: {100*np.median(all_pole_errors):.1f}%")
            print(f"   Min: {100*np.min(all_pole_errors):.1f}%")
            print(f"   Max: {100*np.max(all_pole_errors):.1f}%")

        # Overall assessment
        print("\n" + "="*70)
        print("Overall Assessment")
        print("="*70)

        avg_pole_error_pct = 100 * np.mean(all_pole_errors) if all_pole_errors else float('inf')
        pole_count_pct = 100 * pole_count_matches / valid_circuits

        print(f"\nTransfer Function Accuracy:")
        print(f"  Pole count accuracy: {pole_count_pct:.1f}%")
        print(f"  Pole location error: {avg_pole_error_pct:.1f}%")

        if pole_count_pct > 90 and avg_pole_error_pct < 10:
            print(f"\n✅ EXCELLENT: Actual TF accuracy is very good!")
            grade = "A"
        elif pole_count_pct > 70 and avg_pole_error_pct < 30:
            print(f"\n⚠️  GOOD: Actual TF accuracy is acceptable")
            grade = "B"
        elif pole_count_pct > 50 and avg_pole_error_pct < 50:
            print(f"\n⚠️  FAIR: Actual TF accuracy needs improvement")
            grade = "C"
        else:
            print(f"\n❌ NEEDS WORK: Actual TF accuracy needs significant improvement")
            grade = "F"

        print(f"\nGrade: {grade}")
    else:
        print("\n❌ No valid circuits generated - cannot evaluate TF accuracy")

    print()

if __name__ == '__main__':
    main()
