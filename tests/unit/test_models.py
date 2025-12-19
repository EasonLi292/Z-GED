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
from ml.models import HierarchicalEncoder, HybridDecoder, CIRCUIT_TEMPLATES
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

    # Create synthetic batch
    x = torch.randn(15, 4)  # Mixed nodes from 4 graphs
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9]
    ], dtype=torch.long)
    edge_attr = torch.randn(12, 3)
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
        edge_feature_dim=3,
        gnn_hidden_dim=64,
        latent_dim=24
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

    assert z.shape == (batch_size, 24)
    assert z_topo.shape == (batch_size, 8)
    print("\n✅ HierarchicalEncoder test passed")


def test_decoder():
    """Test HybridDecoder."""
    print("\n" + "="*70)
    print("TEST: HybridDecoder")
    print("="*70)

    batch_size = 4
    z = torch.randn(batch_size, 24)

    decoder = HybridDecoder(
        latent_dim=24,
        edge_feature_dim=3,
        hidden_dim=128
    )

    # Test soft decoding (training mode)
    decoder.train()
    output = decoder(z, temperature=1.0, hard=False)

    print(f"\nInput z shape: {z.shape}")
    print(f"\nOutput (soft):")
    print(f"  topo_logits:    {output['topo_logits'].shape}")
    print(f"  topo_probs:     {output['topo_probs'].shape}")
    print(f"  edge_features:  {output['edge_features'].shape}")
    print(f"  poles:          {output['poles'].shape}")
    print(f"  zeros:          {output['zeros'].shape}")

    # Check topology probabilities sum to 1
    topo_sum = output['topo_probs'].sum(dim=-1)
    print(f"\nTopology prob sums: {topo_sum}")
    assert torch.allclose(topo_sum, torch.ones_like(topo_sum), atol=1e-5)

    # Test hard decoding (inference mode)
    decoder.eval()
    output_hard = decoder(z, hard=True)

    print(f"\nOutput (hard):")
    print(f"  graphs: {len(output_hard['graphs'])} PyG Data objects")

    for i, graph in enumerate(output_hard['graphs']):
        print(f"    Graph {i}: {graph.filter_type}, {graph.num_nodes} nodes, {graph.num_edges} edges")

    print(f"\nModel parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    assert len(output_hard['graphs']) == batch_size
    print("\n✅ HybridDecoder test passed")


def test_end_to_end():
    """Test end-to-end VAE (encode → decode)."""
    print("\n" + "="*70)
    print("TEST: End-to-End VAE")
    print("="*70)

    batch_size = 2

    # Create synthetic batch (simpler)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    poles_list = [torch.randn(1, 2), torch.randn(2, 2)]
    zeros_list = [torch.zeros(0, 2), torch.randn(1, 2)]

    # Create encoder and decoder
    encoder = HierarchicalEncoder(latent_dim=24)
    decoder = HybridDecoder(latent_dim=24)

    # Encode
    z, mu, logvar = encoder(x, edge_index, edge_attr, batch, poles_list, zeros_list)
    print(f"\nEncoded to latent: {z.shape}")

    # Decode
    decoder.eval()
    output = decoder(z, hard=True)
    print(f"Decoded to {len(output['graphs'])} graphs")

    for i, graph in enumerate(output['graphs']):
        print(f"  Graph {i}: {graph.filter_type}")

    # Check that we can backprop through the whole pipeline
    encoder.train()
    decoder.train()

    z, mu, logvar = encoder(x, edge_index, edge_attr, batch, poles_list, zeros_list)
    output = decoder(z, hard=False)

    # Dummy loss
    loss = output['topo_logits'].sum() + output['edge_features'].sum()
    loss.backward()

    print(f"\n✅ Gradient flow works (loss = {loss.item():.4f})")
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

    # Create models
    encoder = HierarchicalEncoder()
    decoder = HybridDecoder()

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

    # Decode
    decoder.eval()
    output = decoder(z, hard=True)

    print(f"\nDecoded:")
    print(f"  Graphs: {len(output['graphs'])}")

    for i, (graph, true_type) in enumerate(zip(output['graphs'], batch['circuit_id'])):
        true_type_idx = batch['filter_type'][i].argmax().item()
        true_type_name = dataset.FILTER_TYPES[true_type_idx]
        print(f"  Graph {i}: Predicted={graph.filter_type}, True={true_type_name}")

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
