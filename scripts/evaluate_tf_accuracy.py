"""
Evaluate Transfer Function Accuracy of Generated Circuits.

This script compares the target transfer function (poles/zeros) with the
actual transfer function computed from generated circuits.
"""

import sys
sys.path.append('.')

import numpy as np
import torch
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder import GraphGPTDecoder
from ml.models.tf_encoder import TransferFunctionEncoder
from ml.utils.filter_design import calculate_filter_poles_zeros
import argparse
from pathlib import Path

# Import circuit analysis tools
sys.path.insert(0, 'tools')
from rlc_graph_utils import graph_to_transfer_function

def complex_to_str(c):
    """Format complex number for display."""
    if abs(c.imag) < 1e-6:
        return f"{c.real:.2f}"
    elif c.imag > 0:
        return f"{c.real:.2f} +{c.imag:.2f}j"
    else:
        return f"{c.real:.2f} {c.imag:.2f}j"

def pole_zero_error(target, actual):
    """
    Calculate error between target and actual poles/zeros.

    Uses Hungarian algorithm to match poles, then computes relative error.
    """
    if len(target) == 0 and len(actual) == 0:
        return 0.0, []

    if len(target) == 0 or len(actual) == 0:
        return float('inf'), []

    from scipy.optimize import linear_sum_assignment

    # Build cost matrix (distance between each pair)
    n_target = len(target)
    n_actual = len(actual)

    # Pad to same length
    max_len = max(n_target, n_actual)
    target_padded = list(target) + [0] * (max_len - n_target)
    actual_padded = list(actual) + [0] * (max_len - n_actual)

    cost_matrix = np.zeros((max_len, max_len))
    for i, t in enumerate(target_padded):
        for j, a in enumerate(actual_padded):
            cost_matrix[i, j] = abs(t - a)

    # Find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate errors for matched pairs
    errors = []
    total_error = 0.0
    for i, j in zip(row_ind[:min(n_target, n_actual)], col_ind[:min(n_target, n_actual)]):
        if i < n_target and j < n_actual:
            t = target[i]
            a = actual[j]
            mag_error = abs(abs(t) - abs(a)) / (abs(t) + 1e-10)
            phase_error = abs(np.angle(t) - np.angle(a))
            error = mag_error + phase_error / np.pi  # Combined error metric
            errors.append({
                'target': t,
                'actual': a,
                'mag_error': mag_error,
                'phase_error': phase_error,
                'combined_error': error
            })
            total_error += error

    avg_error = total_error / len(errors) if errors else float('inf')
    return avg_error, errors

def denormalize_edge_values(edge_values_normalized, impedance_mean, impedance_std):
    """Denormalize edge values to actual component values."""
    C_norm, G_norm, L_inv_norm = edge_values_normalized[:3]

    log_C = C_norm * impedance_std[0] + impedance_mean[0]
    log_G = G_norm * impedance_std[1] + impedance_mean[1]
    log_L_inv = L_inv_norm * impedance_std[2] + impedance_mean[2]

    C = np.exp(log_C)
    G = np.exp(log_G)
    L_inv = np.exp(log_L_inv)

    R = 1.0 / (G + 1e-15)
    L = 1.0 / (L_inv + 1e-15)

    return C, R, L

def build_circuit_graph(node_types, edge_existence, edge_values, impedance_mean, impedance_std):
    """
    Build circuit graph from decoder output.

    Returns graph in format expected by graph_to_transfer_function.
    """
    num_nodes = len(node_types)

    # Map node types (0=GND, 1=VIN, 2=VOUT, 3=INTERNAL, 4=MASK)
    node_type_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']

    # Build adjacency list
    nodes = []
    for i in range(num_nodes):
        if node_types[i] < 4:  # Not MASK
            nodes.append({
                'id': i,
                'type': node_type_names[node_types[i]],
                'features': [1 if j == node_types[i] else 0 for j in range(4)]
            })

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
    parser.add_argument('--filter-type', type=str, default='butterworth',
                       choices=['butterworth', 'bessel', 'chebyshev', 'notch'])
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
    print("Transfer Function Accuracy Evaluation")
    print("="*70)

    # Calculate target transfer function
    print(f"\n📐 Calculating target transfer function...")
    print(f"   Filter: {args.filter_type.capitalize()}")
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
        print(f"      Pole {i+1}: {complex_to_str(p)}")
    if target_zeros:
        print(f"   Zeros ({len(target_zeros)}):")
        for i, z in enumerate(target_zeros):
            print(f"      Zero {i+1}: {complex_to_str(z)}")
    else:
        print(f"   Zeros: None (all-pole filter)")

    # Load models
    print(f"\n📂 Loading models...")

    # Load encoder+decoder
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

    # Load TF encoder
    tf_checkpoint = torch.load(args.tf_encoder_checkpoint, map_location=device)
    tf_encoder = TransferFunctionEncoder(latent_dim=4).to(device)
    tf_encoder.load_state_dict(tf_checkpoint['model_state_dict'])
    tf_encoder.eval()

    # Get normalization stats
    from ml.data.dataset import CircuitDataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl', normalize_features=True)
    impedance_mean = dataset.impedance_mean.numpy()
    impedance_std = dataset.impedance_std.numpy()

    print(f"✅ Models loaded")

    # Encode target TF to latent
    print(f"\n🎯 Encoding target TF to latent space...")

    pole_values = torch.tensor([[p.real, p.imag] for p in target_poles],
                               dtype=torch.float32, device=device).unsqueeze(0)
    pole_count = torch.tensor([len(target_poles)], dtype=torch.long, device=device)

    if target_zeros:
        zero_values = torch.tensor([[z.real, z.imag] for z in target_zeros],
                                   dtype=torch.float32, device=device).unsqueeze(0)
        zero_count = torch.tensor([len(target_zeros)], dtype=torch.long, device=device)
    else:
        zero_values = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
        zero_count = torch.tensor([0], dtype=torch.long, device=device)

    with torch.no_grad():
        mu_pz, _ = tf_encoder.forward(pole_values, pole_count, zero_values, zero_count)

    print(f"✅ TF encoded to latent[4:8]: {mu_pz[0].cpu().numpy()}")

    # Generate circuits
    print(f"\n🎨 Generating {args.num_samples} circuits...")

    results = []

    for i in range(args.num_samples):
        # Sample topology/values latent
        z_topo = torch.randn(1, 2, device=device) * 0.5
        z_values = torch.randn(1, 2, device=device) * 0.3
        z_latent = torch.cat([z_topo, z_values, mu_pz.unsqueeze(0)], dim=1)

        # Generate circuit
        with torch.no_grad():
            outputs = decoder.generate(z_latent, num_samples=1)

        # Extract circuit structure
        node_types = outputs['node_types'][0].cpu().numpy()
        edge_existence = outputs['edge_existence'][0].cpu().numpy()
        edge_values = outputs['edge_values'][0].cpu().numpy()

        # Build circuit graph
        try:
            graph = build_circuit_graph(node_types, edge_existence, edge_values,
                                       impedance_mean, impedance_std)

            # Compute transfer function
            tf_result = graph_to_transfer_function(graph, freq_points=np.logspace(0, 5, 100))

            if tf_result is None:
                results.append({
                    'circuit_id': i,
                    'valid': False,
                    'error': 'TF computation failed'
                })
                continue

            # Extract poles and zeros
            actual_poles = tf_result.get('poles', [])
            actual_zeros = tf_result.get('zeros', [])

            # Calculate errors
            pole_error, pole_matches = pole_zero_error(target_poles, actual_poles)
            zero_error, zero_matches = pole_zero_error(target_zeros, actual_zeros)

            results.append({
                'circuit_id': i,
                'valid': True,
                'num_nodes': sum(1 for nt in node_types if nt < 4),
                'num_edges': int(edge_existence.sum()),
                'target_poles': target_poles,
                'actual_poles': actual_poles,
                'target_zeros': target_zeros,
                'actual_zeros': actual_zeros,
                'pole_error': pole_error,
                'zero_error': zero_error,
                'pole_matches': pole_matches,
                'zero_matches': zero_matches,
                'pole_count_match': len(actual_poles) == len(target_poles),
                'zero_count_match': len(actual_zeros) == len(target_zeros)
            })

        except Exception as e:
            results.append({
                'circuit_id': i,
                'valid': False,
                'error': str(e)
            })

    # Analyze results
    valid_results = [r for r in results if r.get('valid', False)]

    print(f"✅ Generated {len(results)} circuits ({len(valid_results)} valid)")

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    # Count accuracy
    pole_count_correct = sum(1 for r in valid_results if r['pole_count_match'])
    zero_count_correct = sum(1 for r in valid_results if r['zero_count_match'])

    print(f"\n📊 Pole/Zero Count Accuracy:")
    print(f"   Pole count match: {pole_count_correct}/{len(valid_results)} = {100*pole_count_correct/len(valid_results):.1f}%")
    print(f"   Zero count match: {zero_count_correct}/{len(valid_results)} = {100*zero_count_correct/len(valid_results):.1f}%")

    # Error statistics
    pole_errors = [r['pole_error'] for r in valid_results if r['pole_error'] < float('inf')]
    zero_errors = [r['zero_error'] for r in valid_results if r['zero_error'] < float('inf')]

    if pole_errors:
        print(f"\n📊 Pole Location Error:")
        print(f"   Mean: {np.mean(pole_errors):.3f}")
        print(f"   Median: {np.median(pole_errors):.3f}")
        print(f"   Min: {np.min(pole_errors):.3f}")
        print(f"   Max: {np.max(pole_errors):.3f}")

    if zero_errors:
        print(f"\n📊 Zero Location Error:")
        print(f"   Mean: {np.mean(zero_errors):.3f}")
        print(f"   Median: {np.median(zero_errors):.3f}")

    # Show detailed comparison for first 3 circuits
    print("\n" + "="*70)
    print("Detailed Comparison (First 3 Circuits)")
    print("="*70)

    for i, result in enumerate(valid_results[:3]):
        print(f"\nCircuit {result['circuit_id']}:")
        print(f"  Structure: {result['num_nodes']} nodes, {result['num_edges']} edges")

        print(f"\n  Target Poles ({len(result['target_poles'])}):")
        for j, p in enumerate(result['target_poles']):
            print(f"     {complex_to_str(p)}")

        print(f"\n  Actual Poles ({len(result['actual_poles'])}):")
        for j, p in enumerate(result['actual_poles']):
            print(f"     {complex_to_str(p)}")

        if result['pole_matches']:
            print(f"\n  Pole Matching:")
            for match in result['pole_matches']:
                print(f"     Target: {complex_to_str(match['target'])}")
                print(f"     Actual: {complex_to_str(match['actual'])}")
                print(f"     Mag error: {100*match['mag_error']:.1f}%")
                print(f"     Phase error: {match['phase_error']*180/np.pi:.1f}°")
                print()

        if len(result['target_zeros']) > 0:
            print(f"\n  Target Zeros ({len(result['target_zeros'])}):")
            for j, z in enumerate(result['target_zeros']):
                print(f"     {complex_to_str(z)}")

            print(f"\n  Actual Zeros ({len(result['actual_zeros'])}):")
            for j, z in enumerate(result['actual_zeros']):
                print(f"     {complex_to_str(z)}")

    # Overall assessment
    print("\n" + "="*70)
    print("Overall Assessment")
    print("="*70)

    avg_pole_error = np.mean(pole_errors) if pole_errors else float('inf')
    pole_count_pct = 100 * pole_count_correct / len(valid_results) if valid_results else 0

    print(f"\nTransfer Function Accuracy:")
    print(f"  Pole count accuracy: {pole_count_pct:.1f}%")
    print(f"  Pole location error: {avg_pole_error:.3f}")

    if avg_pole_error < 0.1 and pole_count_pct > 90:
        print(f"\n✅ EXCELLENT: TF accuracy is very good!")
    elif avg_pole_error < 0.3 and pole_count_pct > 70:
        print(f"\n⚠️  GOOD: TF accuracy is acceptable but could improve")
    else:
        print(f"\n❌ NEEDS WORK: TF accuracy needs improvement")

    print()

if __name__ == '__main__':
    main()
