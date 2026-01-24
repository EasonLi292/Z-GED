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
    component_types = circuit['component_types'][0]
    node_types = circuit['node_types'][0]

    num_edges = (edge_exist > 0.5).sum().item() // 2

    # Component type names: 0=None, 1=R, 2=C, 3=L, 4=RC, 5=RL, 6=CL, 7=RCL
    component_names = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']

    # Analyze components
    has_R = False
    has_L = False
    has_C = False

    num_nodes = edge_exist.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if edge_exist[i, j] > 0.5:
                comp_type = component_types[i, j].item()
                if comp_type in [1, 4, 5, 7]:  # R, RC, RL, RCL
                    has_R = True
                if comp_type in [2, 4, 6, 7]:  # C, RC, CL, RCL
                    has_C = True
                if comp_type in [3, 5, 6, 7]:  # L, RL, CL, RCL
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
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=50
    ).to(device)

    checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

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

        # K-NN interpolation to find latent code
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

        # Generate circuit from latent (no conditions)
        latent = latent.unsqueeze(0).to(device).float()

        with torch.no_grad():
            circuit = decoder.generate(latent, verbose=False)

        # Analyze topology
        topo = analyze_topology(circuit)

        # Check validity
        edge_exist = circuit['edge_existence'][0]
        vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
        vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()
        valid = vin_connected and vout_connected

        print(f"  Generated: {topo['num_edges']} edges, {topo['components']}, valid={valid}", flush=True)
        print(f"  Nearest neighbor: {info['neighbor_specs'][0][0]:.1f} Hz, Q={info['neighbor_specs'][0][1]:.3f}\n", flush=True)

        results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'topology': topo,
            'valid': valid,
            'nearest_neighbor': info['neighbor_specs'][0]
        })

    # Generate summary report
    print("\n" + "="*80)
    print("RESULTS SUMMARY (Topology-Only)")
    print("="*80)

    print("\n📊 Overall Statistics:")
    valid_count = sum(1 for r in results if r['valid'])
    print(f"  Valid circuits: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")

    # Topology distribution
    print("\n🔧 Topology Distribution:")
    topo_counts = {}
    for r in results:
        key = f"{r['topology']['num_edges']} edges ({r['topology']['components']})"
        topo_counts[key] = topo_counts.get(key, 0) + 1

    for topo, count in sorted(topo_counts.items()):
        print(f"  {topo}: {count} circuits")

    # Validity by category
    categories = {
        'Low-pass (Q≈0.707)': [r for r in results if 0.5 < r['target_q'] < 1.0 and r['target_cutoff'] < 150000],
        'Band-pass (1<Q<5)': [r for r in results if 1.0 < r['target_q'] < 5.0],
        'High-Q (Q≥5)': [r for r in results if r['target_q'] >= 5.0],
        'Overdamped (Q<0.5)': [r for r in results if r['target_q'] < 0.5],
    }

    print("\n📈 Validity by Category:")
    for cat_name, cat_results in categories.items():
        valid_in_cat = sum(1 for r in cat_results if r['valid'])
        if cat_results:
            print(f"  {cat_name}: {valid_in_cat}/{len(cat_results)} valid ({100*valid_in_cat/len(cat_results):.1f}%)")

    # Save detailed results
    print("\n💾 Saving detailed results...")
    os.makedirs('docs', exist_ok=True)
    with open('docs/GENERATION_TEST_RESULTS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("Comprehensive Topology Generation Test Results\n")
        f.write("="*80 + "\n\n")
        f.write("Note: This is topology-only generation (no component values).\n\n")

        for r in results:
            f.write(f"\n{r['description']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target: {r['target_cutoff']:.1f} Hz, Q={r['target_q']:.3f}\n")
            f.write(f"Topology: {r['topology']['num_edges']} edges, {r['topology']['components']}\n")
            f.write(f"Valid: {'Yes' if r['valid'] else 'No'}\n")
            f.write(f"Nearest neighbor: {r['nearest_neighbor'][0]:.1f} Hz, Q={r['nearest_neighbor'][1]:.3f}\n")

    print("Results saved to docs/GENERATION_TEST_RESULTS.txt")


if __name__ == '__main__':
    main()
