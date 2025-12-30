"""
Test hybrid/cross-type specifications that force interpolation across filter types.

This tests the key advantage of the current [cutoff, Q] approach: the ability
to generate novel topologies by blending different filter types from training data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pickle
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from ml.utils.spice_simulator import CircuitSimulator, extract_cutoff_and_q


def collate_circuit_batch(batch_list):
    """Custom collate function."""
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    specifications = torch.stack([item['specifications'] for item in batch_list])

    batched_graph = Batch.from_data_list(graphs)
    return {
        'graph': batched_graph,
        'poles': poles,
        'zeros': zeros,
        'specifications': specifications
    }


def build_specification_database(encoder, dataset, device='cpu'):
    """Build database of specifications → latent codes."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_circuit_batch)

    all_specs = []
    all_latents = []
    all_filter_types = []

    # Load raw circuits to get filter types
    with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
        circuits_raw = pickle.load(f)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            poles = batch['poles']
            zeros = batch['zeros']

            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles,
                zeros
            )

            specs = batch['specifications'][0]
            all_specs.append(specs)
            all_latents.append(mu[0])
            all_filter_types.append(circuits_raw[idx]['filter_type'])

    specs_tensor = torch.stack(all_specs)
    latents_tensor = torch.stack(all_latents)

    return specs_tensor, latents_tensor, all_filter_types


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, filter_types_db, k=5):
    """Interpolate latent codes from k-nearest neighbors."""
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    spec_distances = ((db_normalized - target)**2).sum(dim=1).sqrt()
    spec_nearest_indices = spec_distances.argsort()[:k]
    spec_dists_k = spec_distances[spec_nearest_indices]
    weights = 1.0 / (spec_dists_k + 1e-6)
    weights = weights / weights.sum()

    interpolated = (latents_db[spec_nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

    # Gather neighbor filter types
    neighbor_types = [filter_types_db[i] for i in spec_nearest_indices.numpy()]

    info = {
        'neighbor_indices': spec_nearest_indices.numpy(),
        'neighbor_specs': specs_db[spec_nearest_indices].numpy(),
        'neighbor_types': neighbor_types,
        'weights': weights.numpy(),
    }

    return interpolated, info


def analyze_topology(circuit):
    """Analyze generated circuit topology."""
    edge_exist = circuit['edge_existence'][0]
    edge_values = circuit['edge_values'][0]

    num_edges = (edge_exist > 0.5).sum().item() // 2

    has_R = False
    has_L = False
    has_C = False

    for i in range(5):
        for j in range(i+1, 5):
            if edge_exist[i, j] > 0.5:
                log_C = abs(edge_values[i, j, 0].item())
                log_G = abs(edge_values[i, j, 1].item())
                log_L_inv = abs(edge_values[i, j, 2].item())

                if log_C > 0.1:
                    has_C = True
                if log_G > 0.1:
                    has_R = True
                if log_L_inv > 0.1:
                    has_L = True

    components = []
    if has_R:
        components.append('R')
    if has_L:
        components.append('L')
    if has_C:
        components.append('C')

    return {
        'num_edges': num_edges,
        'components': '+'.join(components) if components else 'None'
    }


def main():
    device = 'cpu'

    print("="*80)
    print("Hybrid/Cross-Type Specification Test")
    print("="*80)
    print("\nTesting specifications that force blending across filter types")
    print()

    # Load models
    print("Loading models...")
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

    checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db, filter_types_db = build_specification_database(encoder, dataset, device)

    impedance_mean = dataset.impedance_mean.numpy()
    impedance_std = dataset.impedance_std.numpy()

    simulator = CircuitSimulator(
        simulator='ngspice',
        freq_points=200,
        freq_start=1.0,
        freq_stop=1e6,
        impedance_mean=impedance_mean,
        impedance_std=impedance_std
    )

    # Hybrid test specifications
    test_cases = [
        # Between low-pass and band-pass (Q between 0.707 and 1.5)
        (10000, 1.0, "Between low-pass and band-pass (Q=1.0)"),
        (5000, 1.2, "Between low-pass and band-pass (Q=1.2)"),

        # Between band-pass and high-Q (Q between 3 and 5)
        (10000, 4.0, "Between band-pass and high-Q (Q=4.0)"),
        (15000, 4.5, "Between band-pass and high-Q (Q=4.5)"),

        # Between overdamped and Butterworth (Q between 0.3 and 0.707)
        (10000, 0.5, "Between overdamped and Butterworth (Q=0.5)"),
        (20000, 0.4, "Between overdamped and Butterworth (Q=0.4)"),

        # Frequency edge cases (between typical ranges)
        (200, 0.707, "Very low freq edge (200 Hz)"),
        (300000, 0.707, "High freq edge (300 kHz)"),

        # Unusual but valid combinations
        (8000, 1.8, "Mid-frequency, mid-Q (8 kHz, Q=1.8)"),
        (25000, 2.3, "Mid-high frequency, moderate-Q"),
    ]

    results = []

    for idx, (target_cutoff, target_q, description) in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Test {idx+1}/{len(test_cases)}: {description}")
        print(f"Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}")
        print(f"{'='*80}")

        # K-NN interpolation
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, filter_types_db, k=5)

        # Analyze neighbor filter types
        print(f"\n🔍 K-NN Neighbors (k=5):")
        neighbor_type_counts = {}
        for i in range(5):
            neighbor_specs = info['neighbor_specs'][i]
            neighbor_type = info['neighbor_types'][i]
            weight = info['weights'][i]

            neighbor_type_counts[neighbor_type] = neighbor_type_counts.get(neighbor_type, 0) + 1

            print(f"  {i+1}. {neighbor_type:15s} | {neighbor_specs[0]:>8.1f} Hz, Q={neighbor_specs[1]:>6.3f} | weight={weight:.3f}")

        # Check if neighbors span multiple types
        unique_types = list(neighbor_type_counts.keys())
        is_hybrid = len(unique_types) > 1

        print(f"\n📊 Filter Type Distribution:")
        for ftype, count in neighbor_type_counts.items():
            print(f"  {ftype}: {count}/5 neighbors")

        if is_hybrid:
            print(f"  ✅ HYBRID: Blending {len(unique_types)} filter types!")
        else:
            print(f"  ⚠️  PURE: All neighbors are {unique_types[0]}")

        # Generate circuit
        latent = latent.unsqueeze(0).to(device).float()
        conditions = torch.tensor([[
            np.log10(max(target_cutoff, 1.0)) / 4.0,
            np.log10(max(target_q, 0.01)) / 2.0
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Analyze topology
        topo = analyze_topology(circuit)

        # Check validity
        edge_exist = circuit['edge_existence'][0]
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
        valid = vin_connected and vout_connected

        print(f"\n🔧 Generated Topology:")
        print(f"  Edges: {topo['num_edges']}")
        print(f"  Components: {topo['components']}")
        print(f"  Valid: {'✅' if valid else '❌'}")

        # SPICE simulation
        actual_cutoff = None
        actual_q = None
        sim_success = False

        if valid:
            try:
                node_types_indices = circuit['node_types'][0]
                node_types_onehot = torch.zeros(5, 5)
                for i, node_type_idx in enumerate(node_types_indices):
                    node_types_onehot[i, int(node_type_idx.item())] = 1.0

                netlist = simulator.circuit_to_netlist(
                    node_types=node_types_onehot,
                    edge_existence=circuit['edge_existence'][0],
                    edge_values=circuit['edge_values'][0]
                )

                frequencies, response = simulator.run_ac_analysis(netlist)
                specs = extract_cutoff_and_q(frequencies, response)

                actual_cutoff = specs['cutoff_freq']
                actual_q = specs['q_factor']
                sim_success = True

                cutoff_error = abs(target_cutoff - actual_cutoff) / target_cutoff * 100
                q_error = abs(target_q - actual_q) / target_q * 100

                print(f"\n📈 SPICE Results:")
                print(f"  Actual: {actual_cutoff:.1f} Hz, Q={actual_q:.3f}")
                print(f"  Error: {cutoff_error:.1f}% (cutoff), {q_error:.1f}% (Q)")

            except Exception as e:
                print(f"\n❌ SPICE simulation failed: {str(e)}")

        results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'is_hybrid': is_hybrid,
            'neighbor_types': unique_types,
            'topology': topo,
            'valid': valid,
            'actual_cutoff': actual_cutoff,
            'actual_q': actual_q,
            'sim_success': sim_success,
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Hybrid Generation Capability")
    print("="*80)

    hybrid_count = sum(1 for r in results if r['is_hybrid'])
    print(f"\n🎯 Hybrid Specifications: {hybrid_count}/{len(results)} ({100*hybrid_count/len(results):.1f}%)")
    print(f"   (Specs that blend multiple filter types)")

    valid_count = sum(1 for r in results if r['valid'])
    sim_count = sum(1 for r in results if r['sim_success'])
    print(f"\n✅ Valid Circuits: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"✅ Successful Simulations: {sim_count}/{len(results)} ({100*sim_count/len(results):.1f}%)")

    # Analyze hybrid results
    print("\n" + "-"*80)
    print("Hybrid Results Analysis:")
    print("-"*80)

    hybrid_results = [r for r in results if r['is_hybrid']]
    if hybrid_results:
        for r in hybrid_results:
            print(f"\n{r['description']}")
            print(f"  Blended types: {', '.join(r['neighbor_types'])}")
            print(f"  Topology: {r['topology']['num_edges']} edges, {r['topology']['components']}")
            if r['sim_success']:
                cutoff_err = abs(r['target_cutoff'] - r['actual_cutoff']) / r['target_cutoff'] * 100
                q_err = abs(r['target_q'] - r['actual_q']) / r['target_q'] * 100
                print(f"  Error: {cutoff_err:.1f}% (cutoff), {q_err:.1f}% (Q)")

    # Save results
    with open('docs/HYBRID_GENERATION_RESULTS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Hybrid/Cross-Type Generation Test Results\n")
        f.write("="*80 + "\n\n")

        for r in results:
            f.write(f"\n{r['description']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target: {r['target_cutoff']:.1f} Hz, Q={r['target_q']:.3f}\n")
            f.write(f"Hybrid: {'YES - ' + ', '.join(r['neighbor_types']) if r['is_hybrid'] else 'NO'}\n")
            f.write(f"Topology: {r['topology']['num_edges']} edges, {r['topology']['components']}\n")

            if r['sim_success']:
                cutoff_err = abs(r['target_cutoff'] - r['actual_cutoff']) / r['target_cutoff'] * 100
                q_err = abs(r['target_q'] - r['actual_q']) / r['target_q'] * 100
                f.write(f"Actual: {r['actual_cutoff']:.1f} Hz, Q={r['actual_q']:.3f}\n")
                f.write(f"Error: {cutoff_err:.1f}% (cutoff), {q_err:.1f}% (Q)\n")
            else:
                f.write("Simulation: Failed\n")

    print("\n💾 Results saved to docs/HYBRID_GENERATION_RESULTS.txt")


if __name__ == '__main__':
    main()
