"""
Extract detailed circuit topology for drawing diagrams.

This script generates circuits and outputs detailed topology information
that can be used to create circuit diagrams in the documentation.
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

    return interpolated


def extract_circuit_topology(circuit, simulator):
    """Extract detailed circuit topology for diagram generation."""
    edge_exist = circuit['edge_existence'][0]
    edge_values = circuit['edge_values'][0]
    node_types = circuit['node_types'][0]

    node_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']

    # Get node types
    nodes = []
    for i in range(5):
        if node_types[i].item() < 4:  # Not MASK
            nodes.append({
                'id': i,
                'type': node_names[node_types[i].item()],
                'label': f'n{i}'
            })

    # Get edges and components
    edges = []
    for i in range(5):
        for j in range(i+1, 5):
            if edge_exist[i, j] > 0.5:
                # Extract component values
                log_C = edge_values[i, j, 0].item()
                G = edge_values[i, j, 1].item()
                log_L_inv = edge_values[i, j, 2].item()
                mask_C = edge_values[i, j, 3].item()
                mask_G = edge_values[i, j, 4].item()
                mask_L = edge_values[i, j, 5].item()
                is_parallel = edge_values[i, j, 6].item()

                # Denormalize values
                impedance_mean = simulator.impedance_mean
                impedance_std = simulator.impedance_std

                components = []
                if mask_C > 0.5:
                    C_norm = abs(log_C)
                    C_denorm = C_norm * impedance_std[0] + impedance_mean[0]
                    C = 10 ** C_denorm
                    if C < 1e-9:
                        components.append(f'{C*1e12:.1f}pF')
                    elif C < 1e-6:
                        components.append(f'{C*1e9:.1f}nF')
                    else:
                        components.append(f'{C*1e6:.1f}uF')

                if mask_G > 0.5:
                    G_norm = abs(G)
                    G_denorm = G_norm * impedance_std[1] + impedance_mean[1]
                    R = 1.0 / (10 ** G_denorm)
                    if R < 1000:
                        components.append(f'{R:.1f}Ω')
                    elif R < 1e6:
                        components.append(f'{R/1000:.1f}kΩ')
                    else:
                        components.append(f'{R/1e6:.1f}MΩ')

                if mask_L > 0.5:
                    L_inv_norm = abs(log_L_inv)
                    L_inv_denorm = L_inv_norm * impedance_std[2] + impedance_mean[2]
                    L = 1.0 / (10 ** L_inv_denorm)
                    if L < 1e-6:
                        components.append(f'{L*1e9:.1f}nH')
                    elif L < 1e-3:
                        components.append(f'{L*1e6:.1f}uH')
                    else:
                        components.append(f'{L*1e3:.1f}mH')

                topology = 'parallel' if is_parallel > 0.5 else 'series'
                component_str = f"{'+' if is_parallel > 0.5 else '/'}.join(components)"

                edges.append({
                    'from': i,
                    'to': j,
                    'components': components,
                    'topology': topology,
                    'label': component_str
                })

    return {
        'nodes': nodes,
        'edges': edges
    }


def draw_ascii_circuit(topology):
    """Generate ASCII art circuit diagram."""
    nodes = topology['nodes']
    edges = topology['edges']

    # Create simple ASCII representation
    lines = []
    lines.append("Circuit Topology:")
    lines.append("")

    # Draw nodes
    for node in nodes:
        lines.append(f"  {node['label']}: {node['type']}")
    lines.append("")

    # Draw edges
    lines.append("Connections:")
    for edge in edges:
        from_node = f"n{edge['from']}"
        to_node = f"n{edge['to']}"
        components = ', '.join(edge['components'])
        topo = edge['topology']
        lines.append(f"  {from_node} --[{components} ({topo})]-- {to_node}")

    return '\n'.join(lines)


def main():
    device = 'cpu'

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

    # Test cases to diagram
    test_cases = [
        (10000, 0.707, "Low-pass 10kHz Butterworth"),
        (15000, 3.0, "Band-pass 15kHz Q=3.0"),
        (10000, 10.0, "High-Q Resonator"),
        (10000, 1.0, "Hybrid: Low-pass + Band-pass"),
        (15000, 4.5, "Hybrid: Band-pass + High-Q (BEST RESULT)"),
    ]

    for target_cutoff, target_q, description in test_cases:
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"Target: {target_cutoff} Hz, Q={target_q}")
        print(f"{'='*70}")

        # Generate circuit
        latent = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)
        latent = latent.unsqueeze(0).to(device).float()
        conditions = torch.tensor([[
            np.log10(max(target_cutoff, 1.0)) / 4.0,
            np.log10(max(target_q, 0.01)) / 2.0
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Extract topology
        topology = extract_circuit_topology(circuit, simulator)

        # Draw ASCII circuit
        ascii_diagram = draw_ascii_circuit(topology)
        print(f"\n{ascii_diagram}")

        # Run SPICE simulation
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

            cutoff_error = abs(target_cutoff - actual_cutoff) / target_cutoff * 100
            q_error = abs(target_q - actual_q) / target_q * 100

            print(f"\nSPICE Results:")
            print(f"  Actual: {actual_cutoff:.1f} Hz, Q={actual_q:.3f}")
            print(f"  Error: {cutoff_error:.1f}% (cutoff), {q_error:.1f}% (Q)")

        except Exception as e:
            print(f"\nSPICE simulation failed: {str(e)}")


if __name__ == '__main__':
    main()
