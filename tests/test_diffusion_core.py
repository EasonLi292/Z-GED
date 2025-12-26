#!/usr/bin/env python3
"""
Unit tests for diffusion model core components.

Tests noise schedules, time embeddings, graph transformers, and denoising network.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest


def test_noise_schedules():
    """Test cosine noise schedule and discrete transition matrices."""
    from ml.models.diffusion import (
        get_cosine_schedule,
        get_discrete_transition_matrix,
        get_cumulative_transition_matrix
    )

    print("\n" + "=" * 70)
    print("Testing Noise Schedules")
    print("=" * 70)

    # Test cosine schedule
    timesteps = 1000
    betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_cosine_schedule(timesteps)

    print(f"\n✓ Cosine schedule created for {timesteps} timesteps")
    print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
    print(f"  Alpha_bar range: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")

    assert betas.shape[0] == timesteps
    assert alphas.shape[0] == timesteps
    assert alphas_cumprod.shape[0] == timesteps
    assert torch.all(betas >= 0) and torch.all(betas <= 1)
    assert torch.all(alphas_cumprod >= 0) and torch.all(alphas_cumprod <= 1)

    # Test discrete transition matrix
    num_classes = 5
    Q_uniform = get_discrete_transition_matrix(num_classes, timesteps, 'uniform')

    print(f"\n✓ Uniform transition matrices created")
    print(f"  Shape: {Q_uniform.shape}")
    print(f"  Q[0] sample:\n{Q_uniform[0][:3, :3]}")

    assert Q_uniform.shape == (timesteps, num_classes, num_classes)

    # Test cumulative transition matrix
    Q_bar = get_cumulative_transition_matrix(Q_uniform)

    print(f"\n✓ Cumulative transition matrices computed")
    print(f"  Shape: {Q_bar.shape}")

    assert Q_bar.shape == (timesteps, num_classes, num_classes)

    print("\n✅ Noise schedule tests passed!")


def test_time_embedding():
    """Test sinusoidal time embedding."""
    from ml.models.diffusion import SinusoidalTimeEmbedding, TimeEmbeddingMLP

    print("\n" + "=" * 70)
    print("Testing Time Embeddings")
    print("=" * 70)

    # Test sinusoidal embedding
    embedding_dim = 128
    time_embed = SinusoidalTimeEmbedding(embedding_dim)

    t = torch.tensor([0, 10, 100, 999])
    emb = time_embed(t)

    print(f"\n✓ Sinusoidal embedding created (dim={embedding_dim})")
    print(f"  Input timesteps: {t.tolist()}")
    print(f"  Output shape: {emb.shape}")
    print(f"  Embedding range: [{emb.min():.3f}, {emb.max():.3f}]")

    assert emb.shape == (4, embedding_dim)

    # Test time embedding MLP
    time_mlp = TimeEmbeddingMLP(embedding_dim=128, hidden_dim=256)

    time_features = time_mlp(t)

    print(f"\n✓ Time embedding MLP created")
    print(f"  Output shape: {time_features.shape}")
    print(f"  Feature range: [{time_features.min():.3f}, {time_features.max():.3f}]")

    assert time_features.shape == (4, 256)

    print("\n✅ Time embedding tests passed!")


def test_graph_transformer():
    """Test graph transformer components."""
    from ml.models.diffusion import (
        MultiHeadGraphAttention,
        GraphTransformerLayer,
        GraphTransformerStack,
        GraphPooling
    )

    print("\n" + "=" * 70)
    print("Testing Graph Transformer")
    print("=" * 70)

    batch_size = 4
    num_nodes = 5
    hidden_dim = 256
    num_heads = 8

    # Test multi-head attention
    attn = MultiHeadGraphAttention(hidden_dim=hidden_dim, num_heads=num_heads)

    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    attn_out = attn(node_features)

    print(f"\n✓ Multi-head attention created")
    print(f"  Input shape: {node_features.shape}")
    print(f"  Output shape: {attn_out.shape}")

    assert attn_out.shape == (batch_size, num_nodes, hidden_dim)

    # Test transformer layer
    layer = GraphTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads)

    layer_out = layer(node_features)

    print(f"\n✓ Transformer layer created")
    print(f"  Output shape: {layer_out.shape}")

    assert layer_out.shape == (batch_size, num_nodes, hidden_dim)

    # Test transformer stack
    num_layers = 6
    stack = GraphTransformerStack(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )

    stack_out = stack(node_features)

    print(f"\n✓ Transformer stack created ({num_layers} layers)")
    print(f"  Output shape: {stack_out.shape}")

    assert stack_out.shape == (batch_size, num_nodes, hidden_dim)

    # Test graph pooling
    pooling = GraphPooling(pooling_type='attention', hidden_dim=hidden_dim)

    graph_features = pooling(node_features)

    print(f"\n✓ Graph pooling created")
    print(f"  Input shape: {node_features.shape}")
    print(f"  Output shape: {graph_features.shape}")

    assert graph_features.shape == (batch_size, hidden_dim)

    print("\n✅ Graph transformer tests passed!")


def test_denoising_network():
    """Test diffusion graph transformer (denoising network)."""
    from ml.models.diffusion import DiffusionGraphTransformer

    print("\n" + "=" * 70)
    print("Testing Denoising Network")
    print("=" * 70)

    batch_size = 4
    max_nodes = 5
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    latent_dim = 8
    conditions_dim = 2
    max_poles = 4
    max_zeros = 4

    # Create model
    model = DiffusionGraphTransformer(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        latent_dim=latent_dim,
        conditions_dim=conditions_dim,
        max_nodes=max_nodes,
        max_poles=max_poles,
        max_zeros=max_zeros
    )

    print(f"\n✓ DiffusionGraphTransformer created")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Num heads: {num_heads}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Create dummy inputs
    num_node_types = 5
    edge_feature_dim = 7

    noisy_nodes = torch.randn(batch_size, max_nodes, num_node_types)
    noisy_edges = torch.randn(batch_size, max_nodes, max_nodes, edge_feature_dim)
    t = torch.randint(0, 1000, (batch_size,))
    latent_code = torch.randn(batch_size, latent_dim)
    conditions = torch.randn(batch_size, conditions_dim)

    print(f"\n✓ Created dummy inputs")
    print(f"  Noisy nodes: {noisy_nodes.shape}")
    print(f"  Noisy edges: {noisy_edges.shape}")
    print(f"  Timesteps: {t.shape}")
    print(f"  Latent code: {latent_code.shape}")
    print(f"  Conditions: {conditions.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(noisy_nodes, noisy_edges, t, latent_code, conditions)

    print(f"\n✓ Forward pass completed")
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key:25s}: {tuple(value.shape)}")

    # Verify output shapes
    assert outputs['node_types'].shape == (batch_size, max_nodes, num_node_types)
    assert outputs['edge_existence'].shape == (batch_size, max_nodes, max_nodes)
    assert outputs['edge_values'].shape == (batch_size, max_nodes, max_nodes, edge_feature_dim)
    assert outputs['pole_count_logits'].shape == (batch_size, max_poles + 1)
    assert outputs['zero_count_logits'].shape == (batch_size, max_zeros + 1)
    assert outputs['pole_values'].shape == (batch_size, max_poles, 2)
    assert outputs['zero_values'].shape == (batch_size, max_zeros, 2)

    print("\n✅ Denoising network tests passed!")


def test_noise_addition():
    """Test noise addition for continuous and discrete diffusion."""
    from ml.models.diffusion import (
        get_cosine_schedule,
        get_discrete_transition_matrix,
        get_cumulative_transition_matrix,
        add_noise_continuous,
        add_noise_discrete
    )

    print("\n" + "=" * 70)
    print("Testing Noise Addition")
    print("=" * 70)

    timesteps = 1000
    batch_size = 4

    # Get noise schedules
    betas, alphas, alphas_cumprod, alphas_cumprod_prev = get_cosine_schedule(timesteps)

    # Test continuous noise addition
    x_0 = torch.randn(batch_size, 8)  # Clean latent code
    t = torch.randint(0, timesteps, (batch_size,))

    x_t, noise = add_noise_continuous(x_0, t, alphas_cumprod)

    print(f"\n✓ Continuous noise addition")
    print(f"  Clean data shape: {x_0.shape}")
    print(f"  Noisy data shape: {x_t.shape}")
    print(f"  Noise shape: {noise.shape}")
    print(f"  SNR at t={t[0].item()}: {(x_0[0]**2).mean() / (noise[0]**2).mean():.4f}")

    assert x_t.shape == x_0.shape
    assert noise.shape == x_0.shape

    # Test discrete noise addition
    num_classes = 5
    Q_matrices = get_discrete_transition_matrix(num_classes, timesteps, 'uniform')
    Q_bar = get_cumulative_transition_matrix(Q_matrices)

    # Create one-hot encoded data
    d_0 = torch.zeros(batch_size, num_classes)
    d_0.scatter_(1, torch.randint(0, num_classes, (batch_size, 1)), 1.0)

    d_t = add_noise_discrete(d_0, t, Q_bar)

    print(f"\n✓ Discrete noise addition")
    print(f"  Clean data (one-hot): {d_0[0].tolist()}")
    print(f"  Noisy data (one-hot): {d_t[0].tolist()}")

    assert d_t.shape == d_0.shape
    assert torch.allclose(d_t.sum(dim=1), torch.ones(batch_size))  # Still one-hot

    print("\n✅ Noise addition tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DIFFUSION MODEL CORE COMPONENT TESTS")
    print("=" * 70)

    try:
        test_noise_schedules()
        test_time_embedding()
        test_graph_transformer()
        test_denoising_network()
        test_noise_addition()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
