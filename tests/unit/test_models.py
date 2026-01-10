#!/usr/bin/env python3
"""
Test script for GraphVAE model components.

Tests encoder, decoder, and end-to-end VAE functionality.
"""

import sys
import torch
import torch.nn as nn
sys.path.insert(0, 'ml')
sys.path.insert(0, '.')

from ml.models import ImpedanceConv, ImpedanceGNN, GlobalPooling, DeepSets
from ml.models import HierarchicalEncoder, LatentGuidedGraphGPTDecoder, CIRCUIT_TEMPLATES
from ml.data import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def test_impedance_conv():
    """Test ImpedanceConv layer."""
    print("="*70)
    print("TEST: ImpedanceConv Layer")
    print("="*70)

    # Create simple graph
    x = torch.randn(5, 4)  # 5 nodes, 4 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 3)  # 4 edges, 3 features

    # Create layer
    conv = ImpedanceConv(in_channels=4, out_channels=16, edge_dim=3)

    # Forward pass
    out = conv(x, edge_index, edge_attr)

    print(f"\nInput shape:  {x.shape}")
    print(f"Edge shape:   {edge_attr.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == (5, 16), f"Expected (5, 16), got {out.shape}"
    print("\n✅ ImpedanceConv test passed")


def test_impedance_gnn():
    """Test ImpedanceGNN multi-layer."""
    print("\n" + "="*70)
    print("TEST: ImpedanceGNN Multi-Layer")
    print("="*70)

    x = torch.randn(10, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
    edge_attr = torch.randn(6, 3)

    gnn = ImpedanceGNN(
        in_channels=4,
        hidden_channels=32,
        out_channels=64,
        num_layers=3,
        edge_dim=3
    )

    out = gnn(x, edge_index, edge_attr)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in gnn.parameters()):,}")

    assert out.shape == (10, 64)
    print("\n✅ ImpedanceGNN test passed")


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

    print(f"\n1 pole:  Input {poles1.shape} → Output {out1.shape}")
    print(f"2 poles: Input {poles2.shape} → Output {out2.shape}")

    assert out1.shape == (1, 8)
    assert out2.shape == (1, 8)
    print("\n✅ DeepSets test passed")


def test_encoder():
    """Test HierarchicalEncoder."""
    print("\n" + "="*70)
    print("TEST: HierarchicalEncoder")
    print("="*70)

    batch_size = 4
    latent_dim = 8  # Production config: 2D topo + 2D values + 4D pz

    # Create synthetic batch
    x = torch.randn(15, 4)  # Mixed nodes from 4 graphs
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9]
    ], dtype=torch.long)
    edge_attr = torch.randn(12, 7)  # 7 edge features
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    # Poles and zeros for each graph
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
        edge_feature_dim=7,
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

    # Test latent split
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
    print("\n✅ HierarchicalEncoder test passed")


def test_decoder():
    """Test LatentGuidedGraphGPTDecoder."""
    print("\n" + "="*70)
    print("TEST: LatentGuidedGraphGPTDecoder")
    print("="*70)

    batch_size = 2
    latent_dim = 8
    conditions_dim = 2

    # Create decoder
    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=latent_dim,
        conditions_dim=conditions_dim,
        hidden_dim=128,
        num_heads=4,
        num_node_layers=2,
        max_nodes=5
    )

    # Create inputs
    z = torch.randn(batch_size, latent_dim)
    conditions = torch.randn(batch_size, conditions_dim)

    print(f"\nInput z shape: {z.shape}")
    print(f"Conditions shape: {conditions.shape}")

    # Test generation
    decoder.eval()
    with torch.no_grad():
        output = decoder.generate(z, conditions, verbose=False)

    num_nodes = output['node_types'].shape[1]
    print(f"\nOutput:")
    print(f"  node_types:      {output['node_types'].shape}")
    print(f"  edge_existence:  {output['edge_existence'].shape}")
    print(f"  edge_values:     {output['edge_values'].shape}")
    print(f"  num_nodes:       {num_nodes}")

    # Verify shapes (num_nodes is variable, typically 3-5)
    assert output['node_types'].shape[0] == batch_size
    assert output['node_types'].shape[1] <= 5
    assert output['edge_existence'].shape == (batch_size, num_nodes, num_nodes)
    assert output['edge_values'].shape == (batch_size, num_nodes, num_nodes, 7)

    print(f"\nModel parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print("\n✅ LatentGuidedGraphGPTDecoder test passed")


def test_end_to_end():
    """Test end-to-end VAE (encode → decode)."""
    print("\n" + "="*70)
    print("TEST: End-to-End VAE")
    print("="*70)

    batch_size = 2
    latent_dim = 8

    # Create synthetic batch (simpler)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, 7)  # 7 features for edge
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    poles_list = [torch.randn(1, 2), torch.randn(2, 2)]
    zeros_list = [torch.zeros(0, 2), torch.randn(1, 2)]

    # Create encoder and decoder
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=7,
        latent_dim=latent_dim
    )
    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=latent_dim,
        conditions_dim=2,
        hidden_dim=128,
        num_heads=4,
        num_node_layers=2,
        max_nodes=5
    )

    # Encode
    z, mu, logvar = encoder(x, edge_index, edge_attr, batch, poles_list, zeros_list)
    print(f"\nEncoded to latent: {z.shape}")

    # Decode (generation mode)
    decoder.eval()
    conditions = torch.randn(batch_size, 2)
    with torch.no_grad():
        output = decoder.generate(z, conditions, verbose=False)

    print(f"Generated {batch_size} circuits")
    print(f"  node_types shape: {output['node_types'].shape}")
    print(f"  edge_existence shape: {output['edge_existence'].shape}")

    print("\n✅ End-to-end test passed")


def test_with_real_data():
    """Test with real dataset."""
    print("\n" + "="*70)
    print("TEST: With Real Dataset")
    print("="*70)

    # Load dataset
    dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_circuit_batch
    )

    # Get one batch
    batch = next(iter(loader))

    print(f"\nBatch loaded:")
    print(f"  Graphs:       {batch['graph'].num_graphs}")
    print(f"  Total nodes:  {batch['graph'].num_nodes}")
    print(f"  Total edges:  {batch['graph'].num_edges}")
    print(f"  Poles:        {len(batch['poles'])} lists")
    print(f"  Zeros:        {len(batch['zeros'])} lists")

    # Create encoder with matching dimensions
    encoder = HierarchicalEncoder(
        node_feature_dim=4,
        edge_feature_dim=7,
        latent_dim=8
    )

    # Encode
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

    # Create decoder and generate
    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=128,
        num_heads=4,
        num_node_layers=2,
        max_nodes=5
    )

    decoder.eval()
    # Create dummy conditions (normalized frequency, Q)
    conditions = torch.randn(z.shape[0], 2)
    with torch.no_grad():
        output = decoder.generate(z, conditions, verbose=False)

    print(f"\nGenerated:")
    print(f"  node_types:     {output['node_types'].shape}")
    print(f"  edge_existence: {output['edge_existence'].shape}")

    print("\n✅ Real data test passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GRAPHVAE MODEL COMPONENT TEST SUITE")
    print("="*70)

    # Test individual components
    test_impedance_conv()
    test_impedance_gnn()
    test_deep_sets()
    test_encoder()
    test_decoder()
    test_end_to_end()
    test_with_real_data()

    print("\n" + "="*70)
    print("ALL MODEL TESTS PASSED!")
    print("="*70)
    print("\n✅ Phase 2 Complete: Model Architecture Ready")
    print()


if __name__ == '__main__':
    main()
