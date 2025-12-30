"""
Test if generated topologies are theoretically viable for target specifications.

This script focuses on topology generation capability, not exact component values.
It verifies that generated topologies have the necessary components to achieve
target transfer functions if values were properly tuned.
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


def analyze_topology_viability(circuit, target_cutoff, target_q):
    """
    Analyze if topology is theoretically capable of achieving target specs.

    Returns:
        viable: bool - Can this topology achieve target specs?
        reason: str - Why or why not?
        components: dict - Component analysis
    """
    edge_exist = circuit['edge_existence'][0]
    edge_values = circuit['edge_values'][0]

    # Count components
    has_capacitor = False
    has_inductor = False
    has_resistor = False
    num_edges = 0

    for i in range(5):
        for j in range(i+1, 5):
            if edge_exist[i, j] > 0.5:
                num_edges += 1
                log_C = edge_values[i, j, 0].item()
                log_G = edge_values[i, j, 1].item()
                log_L_inv = edge_values[i, j, 2].item()

                # Simple heuristic: check if component value is "active"
                # (normalized values should be non-zero for active components)
                if abs(log_C) > 0.1:
                    has_capacitor = True
                if abs(log_G) > 0.1:
                    has_resistor = True
                if abs(log_L_inv) > 0.1:
                    has_inductor = True

    components = {
        'num_edges': num_edges,
        'has_R': has_resistor,
        'has_L': has_inductor,
        'has_C': has_capacitor
    }

    # Viability rules based on filter theory

    # Rule 1: Need at least 1 reactive element for any frequency response
    if not has_capacitor and not has_inductor:
        return False, "No reactive elements (L or C) - cannot create frequency-dependent response", components

    # Rule 2: High Q (>1) requires resonance → need BOTH L and C
    if target_q > 1.0:
        if not (has_capacitor and has_inductor):
            missing = []
            if not has_capacitor:
                missing.append("C")
            if not has_inductor:
                missing.append("L")
            return False, f"High-Q resonance requires LC tank, missing: {', '.join(missing)}", components

    # Rule 3: Very high Q (>5) needs low damping → need at least 3 edges for isolation
    if target_q > 5.0 and num_edges < 3:
        return False, f"Very high Q={target_q:.1f} needs complex topology, only {num_edges} edges", components

    # Rule 4: Butterworth (Q ≈ 0.707) can use simple RC or RL
    if 0.5 < target_q < 1.0:
        if (has_capacitor or has_inductor) and has_resistor:
            return True, f"Simple RC/RL filter viable for Butterworth Q={target_q:.3f}", components

    # Rule 5: General case - has reactive elements, should be viable
    if has_capacitor or has_inductor:
        return True, f"Has reactive elements, topology is viable", components

    return False, "Unknown viability issue", components


def main():
    device = 'cpu'

    print("="*80)
    print("Topology Viability Test")
    print("="*80)
    print("\nTesting if decoder generates topologies CAPABLE of achieving target specs")
    print("(Component values will be tuned later)")
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
    print("Loading dataset...")
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
    specs_db, latents_db = build_specification_database(encoder, dataset, device)

    # Test specifications covering different requirements
    test_specs = [
        # Butterworth filters (Q ≈ 0.707)
        (100, 0.707, "Low-pass Butterworth (100 Hz)"),
        (10000, 0.707, "Low-pass Butterworth (10 kHz)"),
        (100000, 0.707, "Low-pass Butterworth (100 kHz)"),

        # Moderate Q resonance (1 < Q < 5)
        (1000, 1.5, "Band-pass moderate-Q (1 kHz, Q=1.5)"),
        (5000, 2.0, "Band-pass moderate-Q (5 kHz, Q=2.0)"),
        (50000, 3.0, "Band-pass moderate-Q (50 kHz, Q=3.0)"),

        # High Q resonance (Q > 5)
        (1000, 5.0, "Band-pass high-Q (1 kHz, Q=5.0)"),
        (10000, 10.0, "Band-pass high-Q (10 kHz, Q=10.0)"),
        (5000, 20.0, "Band-pass very-high-Q (5 kHz, Q=20.0)"),

        # Low Q (overdamped)
        (1000, 0.1, "Overdamped (1 kHz, Q=0.1)"),
        (50000, 0.05, "Very overdamped (50 kHz, Q=0.05)"),
    ]

    results = []

    for target_cutoff, target_q, description in test_specs:
        print(f"\n{'='*80}")
        print(f"Test: {description}")
        print(f"Target: {target_cutoff:.1f} Hz, Q={target_q:.3f}")
        print(f"{'='*80}")

        # K-NN interpolation
        latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

        print(f"\nNearest neighbors:")
        for i in range(3):  # Show top 3
            neighbor_specs = info['neighbor_specs'][i]
            weight = info['weights'][i]
            print(f"  {i+1}. {neighbor_specs[0]:>8.1f} Hz, Q={neighbor_specs[1]:>6.3f} (weight={weight:.3f})")

        # Generate circuit
        latent = latent.unsqueeze(0).to(device).float()
        conditions = torch.tensor([[
            np.log10(max(target_cutoff, 1.0)) / 4.0,
            np.log10(max(target_q, 0.01)) / 2.0
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            circuit = decoder.generate(latent, conditions, verbose=False)

        # Analyze topology viability
        viable, reason, components = analyze_topology_viability(circuit, target_cutoff, target_q)

        print(f"\nGenerated topology:")
        print(f"  Edges: {components['num_edges']}")
        print(f"  Components: ", end="")
        comp_list = []
        if components['has_R']:
            comp_list.append("R")
        if components['has_L']:
            comp_list.append("L")
        if components['has_C']:
            comp_list.append("C")
        print(", ".join(comp_list) if comp_list else "None")

        print(f"\nViability assessment:")
        if viable:
            print(f"  ✅ VIABLE: {reason}")
        else:
            print(f"  ❌ NOT VIABLE: {reason}")

        results.append({
            'description': description,
            'target_cutoff': target_cutoff,
            'target_q': target_q,
            'viable': viable,
            'reason': reason,
            'components': components
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    viable_count = sum(1 for r in results if r['viable'])
    total_count = len(results)

    print(f"\nTopology Viability: {viable_count}/{total_count} ({100*viable_count/total_count:.1f}%)")

    # Break down by category
    print("\n" + "-"*80)
    print("Butterworth filters (Q ≈ 0.707):")
    butterworth = [r for r in results if 0.5 < r['target_q'] < 1.0]
    butterworth_viable = sum(1 for r in butterworth if r['viable'])
    print(f"  {butterworth_viable}/{len(butterworth)} viable")
    for r in butterworth:
        status = "✅" if r['viable'] else "❌"
        print(f"    {status} {r['description']}: {r['components']['num_edges']} edges")

    print("\n" + "-"*80)
    print("Moderate-Q resonance (1 < Q < 5):")
    moderate = [r for r in results if 1.0 < r['target_q'] < 5.0]
    moderate_viable = sum(1 for r in moderate if r['viable'])
    print(f"  {moderate_viable}/{len(moderate)} viable")
    for r in moderate:
        status = "✅" if r['viable'] else "❌"
        comps = []
        if r['components']['has_R']:
            comps.append("R")
        if r['components']['has_L']:
            comps.append("L")
        if r['components']['has_C']:
            comps.append("C")
        comp_str = "+".join(comps)
        print(f"    {status} {r['description']}: {r['components']['num_edges']} edges ({comp_str})")

    print("\n" + "-"*80)
    print("High-Q resonance (Q ≥ 5):")
    high_q = [r for r in results if r['target_q'] >= 5.0]
    high_q_viable = sum(1 for r in high_q if r['viable'])
    print(f"  {high_q_viable}/{len(high_q)} viable")
    for r in high_q:
        status = "✅" if r['viable'] else "❌"
        comps = []
        if r['components']['has_R']:
            comps.append("R")
        if r['components']['has_L']:
            comps.append("L")
        if r['components']['has_C']:
            comps.append("C")
        comp_str = "+".join(comps)
        print(f"    {status} {r['description']}: {r['components']['num_edges']} edges ({comp_str})")

    print("\n" + "-"*80)
    print("Low-Q overdamped (Q < 0.5):")
    low_q = [r for r in results if r['target_q'] < 0.5]
    low_q_viable = sum(1 for r in low_q if r['viable'])
    print(f"  {low_q_viable}/{len(low_q)} viable")
    for r in low_q:
        status = "✅" if r['viable'] else "❌"
        print(f"    {status} {r['description']}: {r['components']['num_edges']} edges")

    # Failure analysis
    failures = [r for r in results if not r['viable']]
    if failures:
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)
        for r in failures:
            print(f"\n❌ {r['description']}")
            print(f"   Reason: {r['reason']}")
            print(f"   Components: {r['components']}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if viable_count == total_count:
        print("\n✅ EXCELLENT: All generated topologies are viable!")
        print("   Component value tuning should enable matching all target specs.")
    elif viable_count >= 0.8 * total_count:
        print(f"\n✅ GOOD: {viable_count}/{total_count} topologies are viable")
        print("   Most specs can be achieved with component tuning.")
        print("   Some edge cases need architecture improvements.")
    elif viable_count >= 0.5 * total_count:
        print(f"\n⚠️  MODERATE: {viable_count}/{total_count} topologies are viable")
        print("   Common specs work, but many fail.")
        print("   Decoder needs better topology selection for unusual specs.")
    else:
        print(f"\n❌ POOR: Only {viable_count}/{total_count} topologies are viable")
        print("   Decoder is not generating appropriate topologies.")
        print("   Needs fundamental architecture changes.")


if __name__ == '__main__':
    main()
