"""
Test circuit generation on specifications NOT in training data.

This script generates circuits for unseen specifications to validate:
1. Interpolation quality (specs between training examples)
2. Extrapolation capability (specs outside training range)
3. Generalization to novel spec combinations
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

    print("Building specification database...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            poles = batch['poles']
            zeros = batch['zeros']

            # Encode circuit
            z, mu, logvar = encoder(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                poles,
                zeros
            )

            # Store specifications and latent code
            specs = batch['specifications'][0]  # [2]
            all_specs.append(specs)
            all_latents.append(mu[0])
            all_indices.append(idx)

    specs_tensor = torch.stack(all_specs)  # [N, 2]
    latents_tensor = torch.stack(all_latents)  # [N, 8]

    print(f"Built database with {len(specs_tensor)} circuits")
    return specs_tensor, latents_tensor, np.array(all_indices)


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5):
    """Interpolate latent codes from k-nearest neighbors (spec-only)."""
    # Normalize target
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    # Compute specification distances
    spec_distances = ((db_normalized - target)**2).sum(dim=1).sqrt()

    # Find k nearest by specification
    spec_nearest_indices = spec_distances.argsort()[:k]

    # Inverse distance weighting (spec-only)
    spec_dists_k = spec_distances[spec_nearest_indices]
    weights = 1.0 / (spec_dists_k + 1e-6)
    weights = weights / weights.sum()

    # Weighted average of latents
    interpolated = (latents_db[spec_nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

    # Gather info
    info = {
        'neighbor_indices': spec_nearest_indices.numpy(),
        'neighbor_specs': specs_db[spec_nearest_indices].numpy(),
        'weights': weights.numpy(),
        'spec_distances': spec_distances[spec_nearest_indices].numpy()
    }

    return interpolated, info


def is_in_training_data(target_cutoff, target_q, specs_db, tolerance=0.05):
    """Check if target specs are close to any training example."""
    # Normalize
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([
        torch.log10(specs_db[:, 0]),
        specs_db[:, 1]
    ], dim=1)

    # Compute distances
    distances = ((db_normalized - target)**2).sum(dim=1).sqrt()
    min_dist = distances.min().item()

    return min_dist < tolerance, min_dist


def main():
    device = 'cpu'

    print("="*70)
    print("Testing Circuit Generation on Unseen Specifications")
    print("="*70)
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

    # Load dataset and build specification database
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db, indices_db = build_specification_database(encoder, dataset, device)

    # Get normalization stats from dataset (CRITICAL for denormalization!)
    impedance_mean = dataset.impedance_mean.numpy()  # [C_mean, G_mean, L_inv_mean]
    impedance_std = dataset.impedance_std.numpy()    # [C_std, G_std, L_inv_std]

    print()
    print("="*70)
    print("Test Specifications (Unseen by Training Data)")
    print("="*70)
    print()

    # Define test specifications
    test_specs = [
        # Interpolation tests (between training examples)
        (5000, 0.5, "Interpolation: Mid-low frequency, mid-low Q"),
        (50000, 2.0, "Interpolation: Mid frequency, moderate Q"),
        (200000, 5.0, "Interpolation: High frequency, moderate-high Q"),

        # Edge case tests
        (20, 0.707, "Edge case: Very low frequency"),
        (500000, 0.707, "Edge case: High frequency"),
        (10000, 20.0, "Edge case: High Q resonance"),

        # Unusual combinations
        (1000, 10.0, "Unusual: Low frequency + high Q"),
        (100000, 0.05, "Unusual: High frequency + very low Q"),
    ]

    # Create SPICE simulator with normalization stats
    simulator = CircuitSimulator(
        simulator='ngspice',
        freq_points=200,
        freq_start=1.0,
        freq_stop=1e6,
        impedance_mean=impedance_mean,  # NEW: For denormalization
        impedance_std=impedance_std      # NEW: For denormalization
    )

    results = []

    for target_cutoff, target_q, description in test_specs:
        print(f"\n{'='*70}")
        print(f"Test: {description}")
        print(f"Target: cutoff={target_cutoff:.1f} Hz, Q={target_q:.3f}")
        print(f"{'='*70}")

        # Check if in training data
        in_training, min_dist = is_in_training_data(target_cutoff, target_q, specs_db)
        print(f"In training data: {'YES (distance={:.3f})'.format(min_dist) if in_training else 'NO (distance={:.3f})'.format(min_dist)}")

        # Generate circuit
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

        # Show neighbors
        print(f"\nNearest 5 training circuits:")
        for i in range(5):
            neighbor_specs = info['neighbor_specs'][i]
            weight = info['weights'][i]
            dist = info['spec_distances'][i]
            print(f"  {i+1}. cutoff={neighbor_specs[0]:.1f} Hz, Q={neighbor_specs[1]:.3f}, "
                  f"weight={weight:.3f}, dist={dist:.3f}")

        # Generate from interpolated latent
        latent = latent.unsqueeze(0).to(device).float()

        # Use TARGET specifications as conditions (not random!)
        conditions = torch.tensor([[
            np.log10(max(target_cutoff, 1.0)) / 4.0,    # Normalize cutoff
            np.log10(max(target_q, 0.01)) / 2.0          # Normalize Q
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Analyze generated circuit structure
        edge_exist = circuit['edge_existence'][0]
        num_edges = (edge_exist > 0.5).sum().item() // 2

        print(f"\nGenerated circuit:")
        print(f"  Edges: {num_edges}")

        # Check VIN/VOUT connectivity
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
        valid = vin_connected and vout_connected

        print(f"  Valid (VIN & VOUT connected): {'✅ YES' if valid else '❌ NO'}")

        # Simulate circuit and measure ACTUAL specifications
        actual_cutoff = None
        actual_q = None
        simulation_success = False

        if valid:
            try:
                # Convert node types to one-hot encoding for SPICE simulator
                node_types_indices = circuit['node_types'][0]  # [5]
                node_types_onehot = torch.zeros(5, 5)  # [num_nodes=5, num_classes=5]
                for i, node_type_idx in enumerate(node_types_indices):
                    node_types_onehot[i, int(node_type_idx.item())] = 1.0

                # Convert to SPICE netlist
                netlist = simulator.circuit_to_netlist(
                    node_types=node_types_onehot,
                    edge_existence=circuit['edge_existence'][0],
                    edge_values=circuit['edge_values'][0]
                )

                # Run AC analysis
                frequencies, response = simulator.run_ac_analysis(netlist)

                # Extract actual specifications from frequency response
                specs = extract_cutoff_and_q(frequencies, response)
                actual_cutoff = specs['cutoff_freq']
                actual_q = specs['q_factor']
                simulation_success = True

                print(f"\nActual specifications (from SPICE simulation):")
                print(f"  Cutoff: {actual_cutoff:.1f} Hz")
                print(f"  Q-factor: {actual_q:.3f}")

                # Compute errors
                cutoff_error = abs(target_cutoff - actual_cutoff) / target_cutoff * 100
                q_error = abs(target_q - actual_q) / target_q * 100

                print(f"\nAccuracy:")
                print(f"  Cutoff error: {cutoff_error:.1f}%")
                print(f"  Q error: {q_error:.1f}%")

            except Exception as e:
                print(f"\n⚠️  SPICE simulation failed: {e}")
                simulation_success = False

        # Find closest training circuit for comparison
        closest_idx = info['neighbor_indices'][0]
        closest_specs = info['neighbor_specs'][0]
        print(f"\nClosest training circuit:")
        print(f"  cutoff={closest_specs[0]:.1f} Hz ({100*closest_specs[0]/target_cutoff:.1f}% of target)")
        print(f"  Q={closest_specs[1]:.3f} ({100*closest_specs[1]/target_q:.1f}% of target)")

        results.append({
            'target': (target_cutoff, target_q),
            'description': description,
            'in_training': in_training,
            'min_dist': min_dist,
            'num_edges': num_edges,
            'valid': valid,
            'simulation_success': simulation_success,
            'actual_cutoff': actual_cutoff,
            'actual_q': actual_q,
            'closest_cutoff': closest_specs[0],
            'closest_q': closest_specs[1]
        })

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print()

    valid_count = sum(1 for r in results if r['valid'])
    print(f"Valid circuits generated: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")

    simulated_count = sum(1 for r in results if r['simulation_success'])
    print(f"Successfully simulated: {simulated_count}/{len(results)} ({100*simulated_count/len(results):.1f}%)")

    unseen_count = sum(1 for r in results if not r['in_training'])
    print(f"Truly unseen specifications: {unseen_count}/{len(results)}")

    print("\n" + "="*70)
    print("ACTUAL Specification Accuracy (from SPICE simulation):")
    print("="*70)

    actual_errors = []
    for result in results:
        target_cutoff, target_q = result['target']

        print(f"\n{result['description']}")
        print(f"  Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}")

        if result['simulation_success']:
            actual_cutoff = result['actual_cutoff']
            actual_q = result['actual_q']

            cutoff_error = abs(target_cutoff - actual_cutoff) / target_cutoff * 100
            q_error = abs(target_q - actual_q) / target_q * 100
            actual_errors.append((cutoff_error, q_error))

            print(f"  Actual:  {actual_cutoff:.1f} Hz, Q={actual_q:.3f}")
            print(f"  Error:   {cutoff_error:.1f}% (cutoff), {q_error:.1f}% (Q)")
            print(f"  Status:  ✅ SIMULATED")
        else:
            print(f"  Status:  ❌ SIMULATION FAILED")

    # Compute average errors
    if actual_errors:
        avg_cutoff_error = np.mean([e[0] for e in actual_errors])
        avg_q_error = np.mean([e[1] for e in actual_errors])

        print("\n" + "="*70)
        print("Average Accuracy (successfully simulated circuits):")
        print("="*70)
        print(f"  Average cutoff error: {avg_cutoff_error:.1f}%")
        print(f"  Average Q error: {avg_q_error:.1f}%")
        print(f"\n  Circuits with <20% cutoff error: {sum(1 for e in actual_errors if e[0] < 20)}/{len(actual_errors)}")
        print(f"  Circuits with <20% Q error: {sum(1 for e in actual_errors if e[1] < 20)}/{len(actual_errors)}")


if __name__ == '__main__':
    main()
