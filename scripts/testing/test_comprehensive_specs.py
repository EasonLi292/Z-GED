"""
Comprehensive test of various input specifications.

Tests different combinations of cutoff frequency and Q-factor to document
what topologies are generated for different specifications.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import LatentGuidedGraphGPTDecoder
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

    specs_tensor = torch.stack(all_specs)
    latents_tensor = torch.stack(all_latents)

    return specs_tensor, latents_tensor


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5):
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

    info = {
        'neighbor_indices': spec_nearest_indices.numpy(),
        'neighbor_specs': specs_db[spec_nearest_indices].numpy(),
        'weights': weights.numpy(),
    }

    return interpolated, info


def analyze_topology(circuit):
    """Analyze generated circuit topology."""
    edge_exist = circuit['edge_existence'][0]
    edge_values = circuit['edge_values'][0]
    node_types = circuit['node_types'][0]

    num_edges = (edge_exist > 0.5).sum().item() // 2

    # Analyze components
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

    # Load models
    print("Loading models...", flush=True)
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
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

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

    # Test specifications
    test_cases = [
        # Low-pass filters (Q ≈ 0.707)
        (100, 0.707, "100 Hz, Q=0.707"),
        (10000, 0.707, "10 kHz, Q=0.707"),
        (100000, 0.707, "100 kHz, Q=0.707"),

        # High-pass filters (Q ≈ 0.707, but should differ by frequency)
        (500, 0.707, "500 Hz, Q=0.707"),
        (50000, 0.707, "50 kHz, Q=0.707"),

        # Band-pass filters (moderate Q)
        (1000, 1.5, "1 kHz, Q=1.5"),
        (5000, 2.0, "5 kHz, Q=2.0"),
        (15000, 3.0, "15 kHz, Q=3.0"),
        (50000, 2.5, "50 kHz, Q=2.5"),

        # High-Q resonators
        (1000, 5.0, "1 kHz, Q=5.0"),
        (10000, 10.0, "10 kHz, Q=10.0"),
        (5000, 20.0, "5 kHz, Q=20.0"),

        # Overdamped filters (low Q)
        (1000, 0.3, "1 kHz, Q=0.3"),
        (50000, 0.1, "50 kHz, Q=0.1"),

        # Edge cases
        (50, 0.707, "50 Hz, Q=0.707"),
        (500000, 0.707, "500 kHz, Q=0.707"),
        (10000, 0.05, "10 kHz, Q=0.05"),
        (5000, 30.0, "5 kHz, Q=30.0"),
    ]

    results = []

    print("\n" + "="*80)
    print("Comprehensive Specification Test")
    print("="*80)
    print(f"Testing {len(test_cases)} different specifications\n")

    for idx, (target_cutoff, target_q, description) in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] {description}", flush=True)
        print(f"  Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}", flush=True)

        # K-NN interpolation
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

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

        print(f"  Generated: {topo['num_edges']} edges, {topo['components']}, valid={valid}", flush=True)

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

                print(f"  Actual: {actual_cutoff:.1f} Hz, Q={actual_q:.3f}", flush=True)
                print(f"  Error: {cutoff_error:.1f}% (cutoff), {q_error:.1f}% (Q)\n", flush=True)

            except Exception as e:
                print(f"  Simulation failed: {str(e)}\n", flush=True)

        results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'topology': topo,
            'valid': valid,
            'actual_cutoff': actual_cutoff,
            'actual_q': actual_q,
            'sim_success': sim_success,
            'nearest_neighbor': info['neighbor_specs'][0]
        })

    # Generate summary report
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print("\n📊 Overall Statistics:")
    valid_count = sum(1 for r in results if r['valid'])
    sim_count = sum(1 for r in results if r['sim_success'])
    print(f"  Valid circuits: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"  Successful simulations: {sim_count}/{len(results)} ({100*sim_count/len(results):.1f}%)")

    # Topology distribution
    print("\n🔧 Topology Distribution:")
    topo_counts = {}
    for r in results:
        key = f"{r['topology']['num_edges']} edges ({r['topology']['components']})"
        topo_counts[key] = topo_counts.get(key, 0) + 1

    for topo, count in sorted(topo_counts.items()):
        print(f"  {topo}: {count} circuits")

    # Accuracy by category
    categories = {
        'Low-pass (Q≈0.707)': [r for r in results if 0.5 < r['target_q'] < 1.0 and r['target_cutoff'] < 150000],
        'Band-pass (1<Q<5)': [r for r in results if 1.0 < r['target_q'] < 5.0],
        'High-Q (Q≥5)': [r for r in results if r['target_q'] >= 5.0],
        'Overdamped (Q<0.5)': [r for r in results if r['target_q'] < 0.5],
    }

    print("\n📈 Accuracy by Category:")
    for cat_name, cat_results in categories.items():
        sim_results = [r for r in cat_results if r['sim_success']]
        if sim_results:
            avg_cutoff_err = np.mean([
                abs(r['target_cutoff'] - r['actual_cutoff']) / r['target_cutoff'] * 100
                for r in sim_results
            ])
            avg_q_err = np.mean([
                abs(r['target_q'] - r['actual_q']) / r['target_q'] * 100
                for r in sim_results
            ])
            print(f"  {cat_name}:")
            print(f"    Cutoff error: {avg_cutoff_err:.1f}%")
            print(f"    Q error: {avg_q_err:.1f}%")

    # Save detailed results
    print("\n💾 Saving detailed results...")
    with open('docs/GENERATION_TEST_RESULTS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Comprehensive Generation Test Results\n")
        f.write("="*80 + "\n\n")

        for r in results:
            f.write(f"\n{r['description']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target: {r['target_cutoff']:.1f} Hz, Q={r['target_q']:.3f}\n")
            f.write(f"Topology: {r['topology']['num_edges']} edges, {r['topology']['components']}\n")
            f.write(f"Valid: {'✅' if r['valid'] else '❌'}\n")

            if r['sim_success']:
                cutoff_err = abs(r['target_cutoff'] - r['actual_cutoff']) / r['target_cutoff'] * 100
                q_err = abs(r['target_q'] - r['actual_q']) / r['target_q'] * 100
                f.write(f"Actual: {r['actual_cutoff']:.1f} Hz, Q={r['actual_q']:.3f}\n")
                f.write(f"Error: {cutoff_err:.1f}% (cutoff), {q_err:.1f}% (Q)\n")
            else:
                f.write("Simulation: Failed\n")

            f.write(f"Nearest neighbor: {r['nearest_neighbor'][0]:.1f} Hz, Q={r['nearest_neighbor'][1]:.3f}\n")

    print("✅ Results saved to docs/GENERATION_TEST_RESULTS.txt")


if __name__ == '__main__':
    main()
