"""Debug script to see actual SPICE netlist generated."""

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
    encoder.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_circuit_batch)
    all_specs = []
    all_latents = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            graph = batch['graph'].to(device)
            poles = batch['poles']
            zeros = batch['zeros']
            z, mu, logvar = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch, poles, zeros)
            specs = batch['specifications'][0]
            all_specs.append(specs)
            all_latents.append(mu[0])

    return torch.stack(all_specs), torch.stack(all_latents)


def interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5):
    target = torch.tensor([np.log10(target_cutoff), target_q])
    db_normalized = torch.stack([torch.log10(specs_db[:, 0]), specs_db[:, 1]], dim=1)
    spec_distances = ((db_normalized - target)**2).sum(dim=1).sqrt()
    spec_nearest_indices = spec_distances.argsort()[:k]
    spec_dists_k = spec_distances[spec_nearest_indices]
    weights = 1.0 / (spec_dists_k + 1e-6)
    weights = weights / weights.sum()
    interpolated = (latents_db[spec_nearest_indices] * weights.unsqueeze(1)).sum(dim=0)
    return interpolated


def main():
    device = 'cpu'

    # Load models
    encoder = HierarchicalEncoder(
        node_feature_dim=4, edge_feature_dim=7, gnn_hidden_dim=64,
        gnn_num_layers=3, latent_dim=8, topo_latent_dim=2,
        values_latent_dim=2, pz_latent_dim=4, dropout=0.1
    ).to(device)

    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8, conditions_dim=2, hidden_dim=256,
        num_heads=8, num_node_layers=4, max_nodes=5
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

    # Generate 10kHz low-pass
    print("="*80)
    print("Generating 10 kHz, Q=0.707 Low-Pass Filter")
    print("="*80)

    target_cutoff = 10000
    target_q = 0.707

    latent = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)
    latent = latent.unsqueeze(0).to(device).float()
    conditions = torch.tensor([[
        np.log10(max(target_cutoff, 1.0)) / 4.0,
        np.log10(max(target_q, 0.01)) / 2.0
    ]], dtype=torch.float32, device=device)

    with torch.no_grad():
        circuit = decoder.generate(latent, conditions, verbose=False)

    # Print circuit structure
    print("\n📊 Circuit Structure:")
    print(f"Node types: {circuit['node_types'][0]}")
    print(f"Edge existence shape: {circuit['edge_existence'][0].shape}")
    print(f"Edge values shape: {circuit['edge_values'][0].shape}")

    # Print edge existence matrix
    print("\n📊 Edge Existence Matrix:")
    edge_exist = circuit['edge_existence'][0]
    for i in range(5):
        for j in range(5):
            if edge_exist[i, j] > 0.5:
                print(f"  Edge {i} → {j}: EXISTS (value={edge_exist[i, j]:.3f})")

    # Print edge values for existing edges
    print("\n📊 Edge Values (for existing edges):")
    edge_values = circuit['edge_values'][0]
    node_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']

    for i in range(5):
        for j in range(5):
            if edge_exist[i, j] > 0.5:
                vals = edge_values[i, j]
                print(f"\n  {node_names[i]} → {node_names[j]}:")
                print(f"    log(C):     {vals[0].item():.4f}")
                print(f"    G:          {vals[1].item():.4f}")
                print(f"    log(L_inv): {vals[2].item():.4f}")
                print(f"    mask_C:     {vals[3].item():.4f}")
                print(f"    mask_G:     {vals[4].item():.4f}")
                print(f"    mask_L:     {vals[5].item():.4f}")
                print(f"    is_parallel:{vals[6].item():.4f}")

    # Convert to SPICE netlist
    print("\n" + "="*80)
    print("SPICE NETLIST:")
    print("="*80)

    node_types_indices = circuit['node_types'][0]
    node_types_onehot = torch.zeros(5, 5)
    for i, node_type_idx in enumerate(node_types_indices):
        node_types_onehot[i, int(node_type_idx.item())] = 1.0

    netlist = simulator.circuit_to_netlist(
        node_types=node_types_onehot,
        edge_existence=circuit['edge_existence'][0],
        edge_values=circuit['edge_values'][0]
    )

    print(netlist)

    # Run simulation
    print("\n" + "="*80)
    print("SPICE SIMULATION:")
    print("="*80)

    try:
        frequencies, response = simulator.run_ac_analysis(netlist)
        specs = extract_cutoff_and_q(frequencies, response)
        print(f"✅ Simulation successful!")
        print(f"   Actual cutoff: {specs['cutoff_freq']:.1f} Hz")
        print(f"   Actual Q:      {specs['q_factor']:.3f}")
        print(f"   Error: {abs(target_cutoff - specs['cutoff_freq'])/target_cutoff*100:.1f}% (cutoff)")
    except Exception as e:
        print(f"❌ Simulation failed: {e}")


if __name__ == '__main__':
    main()
