"""
Generate detailed circuit diagrams and SPICE netlists for all test cases.
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


def format_component_value(value, unit):
    """Format component value with engineering notation."""
    if value >= 1e6:
        return f"{value/1e6:.1f}M{unit}"
    elif value >= 1e3:
        return f"{value/1e3:.1f}k{unit}"
    elif value >= 1:
        return f"{value:.1f}{unit}"
    elif value >= 1e-3:
        return f"{value*1e3:.1f}m{unit}"
    elif value >= 1e-6:
        return f"{value*1e6:.1f}u{unit}"
    elif value >= 1e-9:
        return f"{value*1e9:.1f}n{unit}"
    elif value >= 1e-12:
        return f"{value*1e12:.1f}p{unit}"
    else:
        return f"{value:.2e}{unit}"


def parse_spice_netlist(netlist):
    """Parse SPICE netlist and extract components."""
    import re

    components = []
    lines = netlist.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.') or line.startswith('VIN'):
            continue

        # Parse component lines: "C1 n1 n2 value" or "R1 n1 n2 value" or "L1 n1 n2 value"
        parts = line.split()
        if len(parts) >= 4:
            comp_name = parts[0]
            node1 = parts[1]
            node2 = parts[2]
            value = float(parts[3])

            comp_type = comp_name[0]  # C, R, or L

            # Convert node names
            node_map = {'0': 'GND', 'n1': 'VIN', 'n2': 'VOUT', 'n3': 'n3', 'n4': 'n4'}
            node1_name = node_map.get(node1, node1)
            node2_name = node_map.get(node2, node2)

            components.append({
                'type': comp_type,
                'name': comp_name,
                'node1': node1_name,
                'node2': node2_name,
                'value': value
            })

    return components


def generate_circuit_diagram(netlist):
    """Generate ASCII circuit diagram from SPICE netlist."""
    components = parse_spice_netlist(netlist)

    if not components:
        return "No components found"

    # Group components by edge (node pair)
    edges = {}
    for comp in components:
        # Normalize edge key (always smaller node first for consistency)
        nodes = tuple(sorted([comp['node1'], comp['node2']]))
        if nodes not in edges:
            edges[nodes] = []
        edges[nodes].append(comp)

    # Build diagram
    diagram_lines = []
    for (node1, node2), comps in sorted(edges.items()):
        comp_strs = []
        for comp in comps:
            value = comp['value']
            comp_type = comp['type']

            if comp_type == 'C':
                formatted = format_component_value(value, 'F')
            elif comp_type == 'L':
                formatted = format_component_value(value, 'H')
            elif comp_type == 'R':
                formatted = format_component_value(value, 'Ω')
            else:
                formatted = f"{value:.2e}"

            comp_strs.append(f"{comp['name']}={formatted}")

        components_str = ', '.join(comp_strs)
        diagram_lines.append(f"{node1} ────── {components_str} ────── {node2}")

    return '\n'.join(diagram_lines)


def main():
    device = 'cpu'

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
        (100, 0.707, "Low-pass (100 Hz, Butterworth)"),
        (10000, 0.707, "Low-pass (10 kHz, Butterworth)"),
        (100000, 0.707, "Low-pass (100 kHz, Butterworth)"),

        # High-pass filters (Q ≈ 0.707, but should differ by frequency)
        (500, 0.707, "High-pass-like (500 Hz, Butterworth)"),
        (50000, 0.707, "High-pass-like (50 kHz, Butterworth)"),

        # Band-pass filters (moderate Q)
        (1000, 1.5, "Band-pass (1 kHz, Q=1.5)"),
        (5000, 2.0, "Band-pass (5 kHz, Q=2.0)"),
        (15000, 3.0, "Band-pass (15 kHz, Q=3.0)"),
        (50000, 2.5, "Band-pass (50 kHz, Q=2.5)"),

        # High-Q resonators
        (1000, 5.0, "Resonator (1 kHz, Q=5.0)"),
        (10000, 10.0, "Resonator (10 kHz, Q=10.0)"),
        (5000, 20.0, "Sharp resonator (5 kHz, Q=20.0)"),

        # Overdamped filters (low Q)
        (1000, 0.3, "Overdamped (1 kHz, Q=0.3)"),
        (50000, 0.1, "Very overdamped (50 kHz, Q=0.1)"),

        # Edge cases
        (50, 0.707, "Very low frequency (50 Hz)"),
        (500000, 0.707, "Very high frequency (500 kHz)"),
        (10000, 0.05, "Very low Q (10 kHz, Q=0.05)"),
        (5000, 30.0, "Very high Q (5 kHz, Q=30.0)"),
    ]

    detailed_results = []

    print("\n" + "="*80)
    print("Generating Detailed Circuit Information")
    print("="*80)
    print(f"Processing {len(test_cases)} test cases...\n")

    for idx, (target_cutoff, target_q, description) in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] {description}")

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

        # Count edges
        edge_exist = circuit['edge_existence'][0]
        num_edges = (edge_exist > 0.5).sum().item() // 2

        # Check validity
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
        valid = vin_connected and vout_connected

        # Generate netlist and diagram
        netlist = ""
        diagram = ""
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

                diagram = generate_circuit_diagram(netlist)

                frequencies, response = simulator.run_ac_analysis(netlist)
                specs = extract_cutoff_and_q(frequencies, response)

                actual_cutoff = specs['cutoff_freq']
                actual_q = specs['q_factor']
                sim_success = True

            except Exception as e:
                print(f"  Error: {str(e)}")

        detailed_results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'num_edges': num_edges,
            'valid': valid,
            'netlist': netlist,
            'diagram': diagram,
            'actual_cutoff': actual_cutoff,
            'actual_q': actual_q,
            'sim_success': sim_success
        })

    # Save to file
    output_file = 'docs/DETAILED_CIRCUITS.txt'
    print(f"\n💾 Saving detailed results to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED CIRCUIT INFORMATION FOR ALL TEST CASES\n")
        f.write("="*80 + "\n\n")

        for idx, r in enumerate(detailed_results):
            f.write(f"\n{'='*80}\n")
            f.write(f"Example {idx+1}: {r['description']}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"**Target Specification:**\n")
            f.write(f"- Cutoff frequency: {r['target_cutoff']:,.0f} Hz\n")
            f.write(f"- Q-factor: {r['target_q']:.3f}\n\n")

            f.write(f"**Generated Circuit:**\n")
            f.write(f"- Topology: {r['num_edges']} edges\n")
            f.write(f"- Valid: {'✅ Yes' if r['valid'] else '❌ No'}\n\n")

            if r['sim_success']:
                cutoff_error = abs(r['target_cutoff'] - r['actual_cutoff']) / r['target_cutoff'] * 100
                q_error = abs(r['target_q'] - r['actual_q']) / r['target_q'] * 100

                f.write(f"**Measured Performance:**\n")
                f.write(f"- Actual Cutoff: {r['actual_cutoff']:,.1f} Hz\n")
                f.write(f"- Actual Q: {r['actual_q']:.3f}\n")
                f.write(f"- **Cutoff Error: {cutoff_error:.1f}%**")
                if cutoff_error < 20:
                    f.write(" ✅")
                elif cutoff_error < 50:
                    f.write(" ⚠️")
                else:
                    f.write(" ❌")
                f.write("\n")
                f.write(f"- **Q Error: {q_error:.1f}%**")
                if q_error < 10:
                    f.write(" ✅")
                elif q_error < 50:
                    f.write(" ⚠️")
                else:
                    f.write(" ❌")
                f.write("\n\n")

                f.write(f"**SPICE Netlist:**\n")
                f.write("```spice\n")
                f.write(r['netlist'])
                if not r['netlist'].endswith('\n'):
                    f.write('\n')
                f.write("```\n\n")

                f.write(f"**Circuit Diagram:**\n")
                f.write("```\n")
                f.write(r['diagram'])
                f.write("\n```\n\n")

                # Analysis
                f.write(f"**Analysis:**\n")
                if cutoff_error < 20 and q_error < 20:
                    f.write("- Excellent accuracy on both metrics\n")
                elif cutoff_error < 50 and q_error < 50:
                    f.write("- Moderate accuracy, within acceptable range\n")
                else:
                    f.write("- Poor accuracy, likely due to training data bias\n")

                if r['target_q'] >= 5.0:
                    f.write("- High-Q specification outside typical training data\n")
                elif abs(r['target_q'] - 0.707) < 0.1:
                    f.write("- Butterworth filter (Q≈0.707) matches training data well\n")

            else:
                f.write("**Simulation failed or circuit invalid**\n\n")

            f.write("\n" + "-"*80 + "\n")

    print(f"✅ Detailed results saved to {output_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
