#!/usr/bin/env python3
"""
Test script for GraphVAE model components.

Tests encoder, decoder, and end-to-end VAE functionality.
"""

import sys
import torch
sys.path.insert(0, '.')

from ml.models import ImpedanceConv, ImpedanceGNN, GlobalPooling, DeepSets
from ml.models import HierarchicalEncoder, SimplifiedCircuitDecoder, CIRCUIT_TEMPLATES
from ml.data import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def test_impedance_conv():
    """Test ImpedanceConv layer."""
    print("="*70)
    print("TEST: ImpedanceConv Layer")
    print("="*70)

    # Create simple graph with edge_dim=3
    # Edge attr: [log10(R), log10(C), log10(L)] where 0 = absent
    x = torch.randn(5, 4)  # 5 nodes, 4 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    # Create edge_attr with log10 component values
    edge_attr = torch.zeros(4, 3)
    edge_attr[0, 0] = 3.0   # R edge: log10(1000 Ohm)
    edge_attr[1, 1] = -7.0  # C edge: log10(100nF)
    edge_attr[2, 2] = -3.0  # L edge: log10(1mH)
    edge_attr[3, 0] = 2.5   # RCL edge: R = log10(316 Ohm)
    edge_attr[3, 1] = -8.0  # RCL edge: C = log10(10nF)
    edge_attr[3, 2] = -2.5  # RCL edge: L = log10(3.16mH)

    # Create layer
    conv = ImpedanceConv(in_channels=4, out_channels=16, edge_dim=3)

    # Forward pass
    out = conv(x, edge_index, edge_attr)

    print(f"\nInput shape:  {x.shape}")
    print(f"Edge shape:   {edge_attr.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == (5, 16), f"Expected (5, 16), got {out.shape}"
    print("\n  ImpedanceConv test passed")


def test_impedance_gnn():
    """Test ImpedanceGNN multi-layer."""
    print("\n" + "="*70)
    print("TEST: ImpedanceGNN Multi-Layer")
    print("="*70)

    x = torch.randn(10, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)

    # Create edge_attr with log10 component values (edge_dim=3)
    edge_attr = torch.zeros(6, 3)
    edge_attr[:, 0] = 3.0  # All R edges: log10(1000 Ohm)

    batch = torch.zeros(10, dtype=torch.long)

    gnn = ImpedanceGNN(
        in_channels=4,
        hidden_channels=32,
        out_channels=64,
        num_layers=3,
        edge_dim=3
    )

    out = gnn(x, edge_index, edge_attr, batch)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in gnn.parameters()):,}")

    assert out.shape == (10, 64)
    print("\n  ImpedanceGNN test passed")


def test_deep_sets():
    """Test DeepSets for poles/zeros encoding."""
    print("\n" + "="*70)
    print("TEST: DeepSets for Poles/Zeros")
    print("="*70)

    # Test with variable-length sets
    poles1 = torch.randn(1, 2)  # 1 pole
    poles2 = torch.randn(2, 2)  # 2 poles

    deepsets = DeepSets(input_dim=2, hidden_dim=32, output_dim=8)

    out1 = deepsets(poles1)
    out2 = deepsets(poles2)

    print(f"\n1 pole:  Input {poles1.shape} -> Output {out1.shape}")
    print(f"2 poles: Input {poles2.shape} -> Output {out2.shape}")

    assert out1.shape == (1, 8)
    assert out2.shape == (1, 8)
    print("\n  DeepSets test passed")


def test_encoder():
    """Test HierarchicalEncoder."""
    print("\n" + "="*70)
    print("TEST: HierarchicalEncoder")
    print("="*70)

    batch_size = 4
    latent_dim = 8

    # Create synthetic batch with edge_dim=3
    x = torch.randn(15, 4)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9]
    ], dtype=torch.long)
    edge_attr = torch.zeros(12, 3)
    edge_attr[:, 0] = 3.0  # All R edges: log10(1000 Ohm)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    poles_list = [
        torch.randn(1, 2),
        torch.randn(2, 2),
        torch.randn(1, 2),
        torch.randn(2, 2)
    ]
    zeros_list = [
        torch.zeros(0, 2),
        torch.randn(1, 2),
        torch.randn(1, 2),
        torch.randn(2, 2)
    ]

    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        gnn_hidden_dim=64,
        latent_dim=latent_dim
    )

    z, mu, logvar = encoder(x, edge_index, edge_attr, batch, poles_list, zeros_list)

    print(f"\nInput nodes: {x.shape}")
    print(f"Batch size:  {batch_size}")
    print(f"\nOutput:")
    print(f"  z shape:      {z.shape}")
    print(f"  mu shape:     {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")

    z_topo, z_values, z_pz = encoder.get_latent_split(z)
    print(f"\nLatent split:")
    print(f"  z_topo:   {z_topo.shape}")
    print(f"  z_values: {z_values.shape}")
    print(f"  z_pz:     {z_pz.shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    assert z.shape == (batch_size, latent_dim)
    assert z_topo.shape == (batch_size, 2)
    assert z_values.shape == (batch_size, 2)
    assert z_pz.shape == (batch_size, 4)
    print("\n  HierarchicalEncoder test passed")


def test_decoder():
    """Test SimplifiedCircuitDecoder."""
    print("\n" + "="*70)
    print("TEST: SimplifiedCircuitDecoder")
    print("="*70)

    # Generate uses batch_size=1
    batch_size = 1
    latent_dim = 8

    decoder = SimplifiedCircuitDecoder(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_heads=4,
        num_node_layers=2,
        max_nodes=5
    )

    z = torch.randn(batch_size, latent_dim)

    print(f"\nInput z shape: {z.shape}")

    decoder.eval()
    with torch.no_grad():
        output = decoder.generate(z, verbose=False)

    num_nodes = output['node_types'].shape[1]
    print(f"\nOutput:")
    print(f"  node_types:      {output['node_types'].shape}")
    print(f"  edge_existence:  {output['edge_existence'].shape}")
    print(f"  component_types: {output['component_types'].shape}")
    print(f"  num_nodes:       {num_nodes}")

    assert output['node_types'].shape[0] == batch_size
    assert output['node_types'].shape[1] <= 5
    assert output['edge_existence'].shape == (batch_size, num_nodes, num_nodes)

    # Test forward with teacher forcing
    print("\n  Testing forward with teacher forcing...")
    decoder.train()
    target_nodes = torch.randint(0, 4, (batch_size, 4))
    target_edges = torch.randint(0, 8, (batch_size, 4, 4))
    out = decoder(z, target_node_types=target_nodes, target_edges=target_edges)
    assert out['node_types'].shape == (batch_size, 4, 5)
    assert out['edge_component_logits'].shape == (batch_size, 4, 4, 8)
    print(f"  Teacher forcing output: node_types={out['node_types'].shape}, "
          f"edges={out['edge_component_logits'].shape}")

    # Test forward without teacher forcing (autoregressive with own predictions)
    out2 = decoder(z, target_node_types=target_nodes)
    assert out2['edge_component_logits'].shape == (batch_size, 4, 4, 8)
    print(f"  No teacher forcing output: edges={out2['edge_component_logits'].shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print("\n  SimplifiedCircuitDecoder test passed")


def test_with_real_data():
    """Test with real dataset."""
    print("\n" + "="*70)
    print("TEST: With Real Dataset")
    print("="*70)

    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_circuit_batch
    )

    batch = next(iter(loader))

    print(f"\nBatch loaded:")
    print(f"  Graphs:       {batch['graph'].num_graphs}")
    print(f"  Total nodes:  {batch['graph'].num_nodes}")
    print(f"  Total edges:  {batch['graph'].num_edges}")

    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=3,
        latent_dim=8
    )

    z, mu, logvar = encoder(
        batch['graph'].x,
        batch['graph'].edge_index,
        batch['graph'].edge_attr,
        batch['graph'].batch,
        batch['poles'],
        batch['zeros']
    )

    print(f"\nEncoded:")
    print(f"  z:      {z.shape}")
    print(f"  mu:     {mu.shape}")
    print(f"  logvar: {logvar.shape}")

    decoder = SimplifiedCircuitDecoder(
        latent_dim=8,
        hidden_dim=128,
        num_heads=4,
        num_node_layers=2,
        max_nodes=5
    )

    decoder.eval()
    # Generate uses batch_size=1, so take first sample
    z_single = z[:1]
    with torch.no_grad():
        output = decoder.generate(z_single, verbose=False)

    print(f"\nGenerated:")
    print(f"  node_types:     {output['node_types'].shape}")
    print(f"  edge_existence: {output['edge_existence'].shape}")

    print("\n  Real data test passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GRAPHVAE MODEL COMPONENT TEST SUITE")
    print("="*70)

    test_impedance_conv()
    test_impedance_gnn()
    test_deep_sets()
    test_encoder()
    test_decoder()
    test_with_real_data()

    print("\n" + "="*70)
    print("ALL MODEL TESTS PASSED!")
    print("="*70)


if __name__ == '__main__':
    main()
