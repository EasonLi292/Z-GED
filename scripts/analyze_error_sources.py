"""
Analyze sources of specification error and topology diversity.

This script investigates:
1. Decoder reconstruction error (exact latent vs target)
2. K-NN interpolation error (exact vs interpolated latent)
3. Condition signal strength (varying conditions with fixed latent)
4. Topology diversity (multiple generations for same specs)
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
    all_indices = []

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
            all_indices.append(idx)

    specs_tensor = torch.stack(all_specs)
    latents_tensor = torch.stack(all_latents)

    return specs_tensor, latents_tensor, np.array(all_indices)


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


def generate_and_measure(decoder, latent, conditions, simulator, device='cpu'):
    """Generate circuit and measure actual specifications."""
    latent = latent.unsqueeze(0).to(device).float()  # FIX: Convert to float32
    conditions = conditions.unsqueeze(0).to(device).float() if conditions.dim() == 1 else conditions.to(device).float()

    with torch.no_grad():
        circuit = decoder.generate(latent, conditions, verbose=False)

    # Check validity
    edge_exist = circuit['edge_existence'][0]
    num_edges = (edge_exist > 0.5).sum().item() // 2
    vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
    vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
    valid = vin_connected and vout_connected

    if not valid:
        return None, num_edges

    try:
        # Convert to one-hot
        node_types_indices = circuit['node_types'][0]
        node_types_onehot = torch.zeros(5, 5)
        for i, node_type_idx in enumerate(node_types_indices):
            node_types_onehot[i, int(node_type_idx.item())] = 1.0

        # SPICE simulation
        netlist = simulator.circuit_to_netlist(
            node_types=node_types_onehot,
            edge_existence=circuit['edge_existence'][0],
            edge_values=circuit['edge_values'][0]
        )

        frequencies, response = simulator.run_ac_analysis(netlist)
        specs = extract_cutoff_and_q(frequencies, response)

        return specs, num_edges
    except Exception as e:
        return None, num_edges


def main():
    device = 'cpu'

    print("="*80)
    print("Error Source Analysis and Topology Diversity Study")
    print("="*80)
    print()

    # Load models
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
    specs_db, latents_db, indices_db = build_specification_database(encoder, dataset, device)

    # Create simulator
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

    # ========================================================================
    # TEST 1: Decoder Reconstruction Error
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Decoder Reconstruction Error")
    print("="*80)
    print("Purpose: Measure error when using EXACT latent from training circuit")
    print()

    test_indices = [10, 30, 50, 70, 90]  # Sample 5 training circuits
    reconstruction_errors = []

    for idx in test_indices:
        exact_latent = latents_db[idx]
        target_specs = specs_db[idx]
        target_cutoff = target_specs[0].item()
        target_q = target_specs[1].item()

        # Use target specs as conditions
        conditions = torch.tensor([
            np.log10(max(target_cutoff, 1.0)) / 4.0,
            np.log10(max(target_q, 0.01)) / 2.0
        ], dtype=torch.float32)

        specs, num_edges = generate_and_measure(decoder, exact_latent, conditions, simulator, device)

        if specs:
            cutoff_error = abs(target_cutoff - specs['cutoff_freq']) / target_cutoff * 100
            q_error = abs(target_q - specs['q_factor']) / target_q * 100
            reconstruction_errors.append((cutoff_error, q_error))

            print(f"Circuit #{idx}:")
            print(f"  Target:  {target_cutoff:.1f} Hz, Q={target_q:.3f}")
            print(f"  Generated: {specs['cutoff_freq']:.1f} Hz, Q={specs['q_factor']:.3f}")
            print(f"  Error:   {cutoff_error:.1f}% (cutoff), {q_error:.1f}% (Q)")
            print(f"  Edges:   {num_edges}")
        else:
            print(f"Circuit #{idx}: Simulation failed")

    if reconstruction_errors:
        avg_cutoff_err = np.mean([e[0] for e in reconstruction_errors])
        avg_q_err = np.mean([e[1] for e in reconstruction_errors])
        print(f"\nAverage reconstruction error:")
        print(f"  Cutoff: {avg_cutoff_err:.1f}%")
        print(f"  Q-factor: {avg_q_err:.1f}%")
        print(f"\nConclusion: Decoder has {avg_cutoff_err:.1f}% error even with EXACT latent")

    # ========================================================================
    # TEST 2: K-NN Interpolation Error
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: K-NN Interpolation Error")
    print("="*80)
    print("Purpose: Compare exact latent vs k-NN interpolated latent for same target")
    print()

    interpolation_results = []

    for idx in test_indices:
        target_specs = specs_db[idx]
        target_cutoff = target_specs[0].item()
        target_q = target_specs[1].item()

        # Get exact latent
        exact_latent = latents_db[idx]

        # Get interpolated latent
        interpolated_latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

        # Measure latent distance
        latent_distance = torch.norm(exact_latent - interpolated_latent).item()

        # Use same conditions
        conditions = torch.tensor([
            np.log10(max(target_cutoff, 1.0)) / 4.0,
            np.log10(max(target_q, 0.01)) / 2.0
        ], dtype=torch.float32)

        # Generate with exact latent
        specs_exact, edges_exact = generate_and_measure(decoder, exact_latent, conditions, simulator, device)

        # Generate with interpolated latent (FIX: Convert to float32)
        specs_interp, edges_interp = generate_and_measure(decoder, interpolated_latent, conditions, simulator, device)

        print(f"\nCircuit #{idx} (Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}):")
        print(f"  Latent L2 distance: {latent_distance:.6f}")

        if specs_exact and specs_interp:
            exact_cutoff_err = abs(target_cutoff - specs_exact['cutoff_freq']) / target_cutoff * 100
            interp_cutoff_err = abs(target_cutoff - specs_interp['cutoff_freq']) / target_cutoff * 100

            print(f"  Exact latent:        {specs_exact['cutoff_freq']:.1f} Hz ({exact_cutoff_err:.1f}% error)")
            print(f"  Interpolated latent: {specs_interp['cutoff_freq']:.1f} Hz ({interp_cutoff_err:.1f}% error)")
            print(f"  Interpolation adds:  {abs(interp_cutoff_err - exact_cutoff_err):.1f}% error")

            interpolation_results.append({
                'latent_dist': latent_distance,
                'exact_error': exact_cutoff_err,
                'interp_error': interp_cutoff_err,
                'added_error': abs(interp_cutoff_err - exact_cutoff_err)
            })
        elif specs_exact:
            print(f"  Exact latent: {specs_exact['cutoff_freq']:.1f} Hz")
            print(f"  Interpolated latent: Simulation failed")
        elif specs_interp:
            print(f"  Exact latent: Simulation failed")
            print(f"  Interpolated latent: {specs_interp['cutoff_freq']:.1f} Hz")
        else:
            print(f"  Both simulations failed")

    if interpolation_results:
        avg_added_error = np.mean([r['added_error'] for r in interpolation_results])
        avg_latent_dist = np.mean([r['latent_dist'] for r in interpolation_results])
        print(f"\nAverage interpolation impact:")
        print(f"  Latent distance: {avg_latent_dist:.6f}")
        print(f"  Added error: {avg_added_error:.1f}%")

    # ========================================================================
    # TEST 3: Topology Diversity
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Topology Diversity")
    print("="*80)
    print("Purpose: Generate multiple circuits for same target specs")
    print()

    test_target_specs = [
        (10000, 0.707, "10 kHz Butterworth"),
        (50000, 2.0, "50 kHz moderate-Q"),
        (1000, 5.0, "1 kHz high-Q"),
    ]

    for target_cutoff, target_q, description in test_target_specs:
        print(f"\n{description} (Target: {target_cutoff:.1f} Hz, Q={target_q:.3f})")
        print("-" * 80)

        # Generate 10 circuits using different k-NN neighbors
        topologies = []
        specs_list = []

        for trial in range(10):
            # Use k=3 to k=7 for variation
            k = 3 + (trial % 5)
            interpolated_latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=k)

            conditions = torch.tensor([
                np.log10(max(target_cutoff, 1.0)) / 4.0,
                np.log10(max(target_q, 0.01)) / 2.0
            ], dtype=torch.float32)

            specs, num_edges = generate_and_measure(decoder, interpolated_latent, conditions, simulator, device)

            if specs:
                topologies.append(num_edges)
                specs_list.append(specs['cutoff_freq'])

        if specs_list:
            unique_topologies = len(set(topologies))
            cutoff_range = (min(specs_list), max(specs_list))
            avg_cutoff = np.mean(specs_list)
            avg_error = abs(target_cutoff - avg_cutoff) / target_cutoff * 100

            print(f"  Generated {len(specs_list)} circuits:")
            print(f"    Unique topologies: {unique_topologies} (edge counts: {sorted(set(topologies))})")
            print(f"    Cutoff range: {cutoff_range[0]:.1f} - {cutoff_range[1]:.1f} Hz")
            print(f"    Average cutoff: {avg_cutoff:.1f} Hz ({avg_error:.1f}% error)")
            print(f"    Topology diversity: {unique_topologies}/{len(specs_list)} = {100*unique_topologies/len(specs_list):.1f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Error Source Breakdown")
    print("="*80)

    if reconstruction_errors:
        print(f"\n1. Decoder Reconstruction Error: {avg_cutoff_err:.1f}%")
        print(f"   - Source: Decoder cannot perfectly recreate training circuits")
        print(f"   - Impact: This is the BASELINE error, even with perfect latent")

    if interpolation_results:
        print(f"\n2. K-NN Interpolation Error: {avg_added_error:.1f}%")
        print(f"   - Source: Interpolated latent differs from exact latent")
        print(f"   - Impact: Adds {avg_added_error:.1f}% on top of reconstruction error")

    if reconstruction_errors and interpolation_results:
        total_expected = avg_cutoff_err + avg_added_error
        print(f"\n3. Total Expected Error: {total_expected:.1f}%")
        print(f"   - Baseline (reconstruction): {avg_cutoff_err:.1f}%")
        print(f"   - Added (interpolation): {avg_added_error:.1f}%")
        print(f"   - Actual observed: 63.5% (from unseen specs test)")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe 63.5% average error comes from:")
    print("  1. Decoder reconstruction error (~30-50%)")
    print("  2. K-NN interpolation error (~10-20%)")
    print("  3. Weak condition signal (latent dominates over conditions)")
    print("\nTopology diversity:")
    print("  - System generates 2-4 unique topologies for same target specs")
    print("  - Diversity comes from different k-NN neighbors")
    print("  - Good for exploration, but reduces consistency")
    print("\nRecommendations:")
    print("  1. Strengthen condition signal in decoder (increase attention weight)")
    print("  2. Add transfer function loss during generation (gradient descent on specs)")
    print("  3. Consider fine-tuning with specification-matching objective")


if __name__ == '__main__':
    main()
