"""
Tests for production model components.

Tests v1 HierarchicalEncoder, v2 AdmittanceEncoder, and SequenceDecoder.
"""

import torch
import pytest

from ml.models import (
    ImpedanceConv, ImpedanceGNN, GlobalPooling,
    HierarchicalEncoder, AdmittanceEncoder, SequenceDecoder,
    CircuitVocabulary, CIRCUIT_TEMPLATES,
)


# ── ImpedanceConv (v1 GNN layer) ────────────────────────────────

class TestImpedanceConv:
    def test_output_shape(self):
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.zeros(4, 3)
        edge_attr[0, 0] = 3.0   # R
        edge_attr[1, 1] = -7.0  # C
        edge_attr[2, 2] = -3.0  # L

        conv = ImpedanceConv(in_channels=4, out_channels=16, edge_dim=3)
        out = conv(x, edge_index, edge_attr)
        assert out.shape == (5, 16)

    def test_zero_edge_attr_produces_output(self):
        x = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.zeros(2, 3)

        conv = ImpedanceConv(in_channels=4, out_channels=8, edge_dim=3)
        out = conv(x, edge_index, edge_attr)
        assert out.shape == (3, 8)


# ── ImpedanceGNN (v1 multi-layer) ───────────────────────────────

class TestImpedanceGNN:
    def test_output_shape(self):
        x = torch.randn(10, 4)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
        edge_attr = torch.zeros(6, 3)
        edge_attr[:, 0] = 3.0
        batch = torch.zeros(10, dtype=torch.long)

        gnn = ImpedanceGNN(
            in_channels=4, hidden_channels=32, out_channels=64,
            num_layers=3, edge_dim=3)
        out = gnn(x, edge_index, edge_attr, batch)
        assert out.shape == (10, 64)


# ── HierarchicalEncoder (v1) ────────────────────────────────────

class TestHierarchicalEncoder:
    @pytest.fixture
    def encoder(self):
        torch.manual_seed(0)
        return HierarchicalEncoder(
            node_feature_dim=4, edge_feature_dim=3,
            gnn_hidden_dim=64, latent_dim=8)

    def test_output_shapes(self, encoder):
        x = torch.randn(6, 4)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]], dtype=torch.long)
        edge_attr = torch.zeros(6, 3)
        edge_attr[:, 0] = 3.0
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        z, mu, logvar = encoder(x, edge_index, edge_attr, batch)
        assert z.shape == (2, 8)
        assert mu.shape == (2, 8)
        assert logvar.shape == (2, 8)

    def test_latent_split(self, encoder):
        z = torch.randn(2, 8)
        z_topo, z_values, z_pz = encoder.get_latent_split(z)
        assert z_topo.shape == (2, 2)
        assert z_values.shape == (2, 2)
        assert z_pz.shape == (2, 4)


# ── SequenceDecoder ──────────────────────────────────────────────

class TestSequenceDecoder:
    @pytest.fixture
    def decoder(self):
        torch.manual_seed(0)
        vocab = CircuitVocabulary(max_internal=10, max_components=10)
        dec = SequenceDecoder(
            vocab_size=vocab.vocab_size, latent_dim=8, d_model=64,
            n_heads=4, n_layers=2, max_seq_len=33, dropout=0.0,
            pad_id=vocab.pad_id)
        dec.eval()
        return dec, vocab

    def test_forward_shape(self, decoder):
        dec, vocab = decoder
        dec.train()
        z = torch.randn(2, 8)
        seq_len = 10
        tgt = torch.randint(0, vocab.vocab_size, (2, seq_len))
        lengths = torch.tensor([seq_len, seq_len])
        logits = dec(z, tgt, lengths)
        assert logits.shape == (2, seq_len, vocab.vocab_size)

    def test_generate(self, decoder):
        dec, vocab = decoder
        z = torch.randn(1, 8)
        with torch.no_grad():
            gen = dec.generate(z, max_length=32, temperature=0.1,
                               greedy=True, eos_id=vocab.eos_id)
        assert len(gen) == 1
        assert all(isinstance(t, int) for t in gen[0])
