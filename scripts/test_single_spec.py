"""
Test circuit generation on a single unseen specification.

This script generates a circuit for one target specification and provides
detailed analysis of the result.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
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
        'spec_distances': spec_distances[spec_nearest_indices].numpy()
    }

    return interpolated, info


def analyze_circuit_structure(circuit):
    """Analyze the generated circuit structure."""
    edge_exist = circuit['edge_existence'][0]
    edge_values = circuit['edge_values'][0]
    node_types = circuit['node_types'][0]

    # Count edges
    num_edges = (edge_exist > 0.5).sum().item() // 2

    # Analyze node types
    node_type_names = ['GND', 'VIN', 'VOUT', 'Internal', 'Internal']
    node_list = []
    for i, nt_idx in enumerate(node_types):
        node_list.append(f"Node {i}: {node_type_names[int(nt_idx.item())]}")

    # Analyze edges and components
    component_names = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']
    edges_list = []

    for i in range(5):
        for j in range(i+1, 5):
            if edge_exist[i, j] > 0.5:
                # Get component values (normalized)
                log_C_norm = edge_values[i, j, 0].item()
                log_G_norm = edge_values[i, j, 1].item()
                log_L_inv_norm = edge_values[i, j, 2].item()

                edges_list.append({
                    'nodes': (i, j),
                    'log_C_norm': log_C_norm,
                    'log_G_norm': log_G_norm,
                    'log_L_inv_norm': log_L_inv_norm
                })

    return {
        'num_edges': num_edges,
        'node_list': node_list,
        'edges_list': edges_list
    }


def main():
    device = 'cpu'

    # Get target specification from command line or use default
    if len(sys.argv) >= 3:
        target_cutoff = float(sys.argv[1])
        target_q = float(sys.argv[2])
    else:
        # Default: test a mid-range specification
        target_cutoff = 15000.0  # 15 kHz
        target_q = 1.5

    print("="*80)
    print("Single Specification Test")
    print("="*80)
    print(f"\nTarget: {target_cutoff:.1f} Hz, Q={target_q:.3f}")
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
    print("Loading dataset and building specification database...")
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

    print(f"Dataset: {len(specs_db)} circuits")
    print(f"  Cutoff range: {specs_db[:, 0].min():.1f} - {specs_db[:, 0].max():.1f} Hz")
    print(f"  Q range: {specs_db[:, 1].min():.3f} - {specs_db[:, 1].max():.3f}")

    # Check if target is in training data
    target_tensor = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([torch.log10(specs_db[:, 0]), specs_db[:, 1]], dim=1)
    distances = ((db_normalized - target_tensor)**2).sum(dim=1).sqrt()
    min_dist = distances.min().item()

    print(f"\nTarget distance from training data: {min_dist:.4f}")
    if min_dist < 0.05:
        print("  ⚠️  WARNING: Very close to training data")
    else:
        print("  ✓ Truly unseen specification")

    # Find k-NN neighbors and interpolate
    print("\n" + "="*80)
    print("K-NN Interpolation (k=5)")
    print("="*80)

    latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

    print("\nNearest 5 training circuits:")
    for i in range(5):
        neighbor_specs = info['neighbor_specs'][i]
        weight = info['weights'][i]
        dist = info['spec_distances'][i]
        print(f"  {i+1}. {neighbor_specs[0]:>8.1f} Hz, Q={neighbor_specs[1]:>6.3f}  "
              f"(weight={weight:.3f}, dist={dist:.3f})")

    print(f"\nInterpolated latent code: {latent.numpy()}")

    # Generate circuit
    print("\n" + "="*80)
    print("Circuit Generation")
    print("="*80)

    latent = latent.unsqueeze(0).to(device).float()
    conditions = torch.tensor([[
        np.log10(max(target_cutoff, 1.0)) / 4.0,
        np.log10(max(target_q, 0.01)) / 2.0
    ]], dtype=torch.float32, device=device)

    print(f"\nConditions (normalized): {conditions[0].numpy()}")

    with torch.no_grad():
        circuit = decoder.generate(latent, conditions, verbose=True)

    # Analyze structure
    print("\n" + "="*80)
    print("Circuit Structure Analysis")
    print("="*80)

    structure = analyze_circuit_structure(circuit)

    print(f"\nNumber of edges: {structure['num_edges']}")
    print("\nNodes:")
    for node in structure['node_list']:
        print(f"  {node}")

    print(f"\nEdges ({len(structure['edges_list'])} total):")
    for edge in structure['edges_list']:
        i, j = edge['nodes']
        print(f"  Edge ({i},{j}): log_C_norm={edge['log_C_norm']:>6.3f}, "
              f"log_G_norm={edge['log_G_norm']:>6.3f}, log_L_inv_norm={edge['log_L_inv_norm']:>6.3f}")

    # Check validity
    edge_exist = circuit['edge_existence'][0]
    vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
    vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
    valid = vin_connected and vout_connected

    print(f"\nValidity check:")
    print(f"  VIN connected: {'✓' if vin_connected else '✗'}")
    print(f"  VOUT connected: {'✓' if vout_connected else '✗'}")
    print(f"  Overall: {'✓ VALID' if valid else '✗ INVALID'}")

    # SPICE simulation
    print("\n" + "="*80)
    print("SPICE Simulation")
    print("="*80)

    if not valid:
        print("\n✗ Cannot simulate: circuit is invalid")
        return

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

    try:
        # Convert node types to one-hot
        node_types_indices = circuit['node_types'][0]
        node_types_onehot = torch.zeros(5, 5)
        for i, node_type_idx in enumerate(node_types_indices):
            node_types_onehot[i, int(node_type_idx.item())] = 1.0

        print("\nGenerating SPICE netlist...")
        netlist = simulator.circuit_to_netlist(
            node_types=node_types_onehot,
            edge_existence=circuit['edge_existence'][0],
            edge_values=circuit['edge_values'][0]
        )

        print("\nSPICE Netlist:")
        print("-" * 80)
        print(netlist)
        print("-" * 80)

        print("\nRunning AC analysis...")
        frequencies, response = simulator.run_ac_analysis(netlist)

        print("Extracting specifications...")
        specs = extract_cutoff_and_q(frequencies, response)

        print("\n" + "="*80)
        print("Results")
        print("="*80)

        actual_cutoff = specs['cutoff_freq']
        actual_q = specs['q_factor']

        cutoff_error = abs(target_cutoff - actual_cutoff) / target_cutoff * 100
        q_error = abs(target_q - actual_q) / target_q * 100

        print(f"\nTarget specifications:")
        print(f"  Cutoff: {target_cutoff:.1f} Hz")
        print(f"  Q-factor: {target_q:.3f}")

        print(f"\nGenerated specifications:")
        print(f"  Cutoff: {actual_cutoff:.1f} Hz")
        print(f"  Q-factor: {actual_q:.3f}")

        print(f"\nAccuracy:")
        print(f"  Cutoff error: {cutoff_error:.1f}%")
        print(f"  Q-factor error: {q_error:.1f}%")

        if cutoff_error < 20:
            print(f"\n✓ GOOD: Cutoff error < 20%")
        elif cutoff_error < 50:
            print(f"\n⚠️  MODERATE: Cutoff error 20-50%")
        else:
            print(f"\n✗ POOR: Cutoff error > 50%")

        print(f"\nFilter type: {specs['filter_type']}")
        print(f"Peak gain: {specs['peak_gain_db']:.1f} dB")
        print(f"DC gain: {specs['dc_gain_db']:.1f} dB")

    except Exception as e:
        print(f"\n✗ SPICE simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
