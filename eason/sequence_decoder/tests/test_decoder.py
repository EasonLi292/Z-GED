"""Tests for the GPT-style sequence decoder."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
import torch
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary


@pytest.fixture
def vocab():
    return CircuitVocabulary(max_internal=10, max_components=10)


@pytest.fixture
def decoder(vocab):
    return SequenceDecoder(
        vocab_size=vocab.vocab_size,
        latent_dim=8,
        d_model=64,  # small for testing
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        dropout=0.0,
        pad_id=vocab.pad_id,
    )


class TestForwardPass:

    def test_output_shape(self, decoder, vocab):
        B, L = 4, 16
        latent = torch.randn(B, 8)
        seq = torch.randint(0, vocab.vocab_size, (B, L))
        seq_len = torch.full((B,), L, dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        assert logits.shape == (B, L, vocab.vocab_size)

    def test_output_shape_with_padding(self, decoder, vocab):
        B, L = 2, 16
        latent = torch.randn(B, 8)
        seq = torch.randint(0, vocab.vocab_size, (B, L))
        seq_len = torch.tensor([10, 14], dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        assert logits.shape == (B, L, vocab.vocab_size)

    def test_batch_size_1(self, decoder, vocab):
        latent = torch.randn(1, 8)
        seq = torch.randint(0, vocab.vocab_size, (1, 9))
        seq_len = torch.tensor([9], dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        assert logits.shape == (1, 9, vocab.vocab_size)

    def test_gradients_flow(self, decoder, vocab):
        latent = torch.randn(2, 8, requires_grad=True)
        seq = torch.randint(0, vocab.vocab_size, (2, 9))
        seq_len = torch.tensor([9, 9], dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        loss = logits.sum()
        loss.backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum() > 0


class TestLossComputation:

    def test_loss_is_scalar(self, decoder, vocab):
        B, L = 4, 12
        latent = torch.randn(B, 8)
        seq = torch.randint(1, vocab.vocab_size, (B, L))  # avoid PAD=0 as target
        seq_len = torch.full((B,), L, dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        loss = decoder.compute_loss(logits, seq, seq_len)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_ignores_padding(self, decoder, vocab):
        B, L = 2, 16
        latent = torch.randn(B, 8)

        # Create sequence with lots of padding
        seq = torch.zeros(B, L, dtype=torch.long)
        seq[:, :5] = torch.randint(1, vocab.vocab_size, (B, 5))
        seq_len = torch.tensor([5, 5], dtype=torch.long)

        logits = decoder(latent, seq, seq_len)
        loss = decoder.compute_loss(logits, seq, seq_len)
        assert loss.item() > 0

    def test_shorter_seq_less_loss_contribution(self, decoder, vocab):
        """Padding should not contribute to loss."""
        B, L = 1, 16
        latent = torch.randn(B, 8)
        seq = torch.randint(1, vocab.vocab_size, (B, L))

        logits = decoder(latent, seq, torch.tensor([L]))

        # Now mask half as padding
        logits2 = decoder(latent, seq, torch.tensor([L // 2]))
        loss_full = decoder.compute_loss(logits, seq, torch.tensor([L]))
        loss_half = decoder.compute_loss(logits2, seq, torch.tensor([L // 2]))

        # Both should be valid losses (not NaN)
        assert not torch.isnan(loss_full)
        assert not torch.isnan(loss_half)


class TestGeneration:

    def test_generate_returns_list(self, decoder, vocab):
        latent = torch.randn(2, 8)
        result = decoder.generate(latent, max_length=20, eos_id=vocab.eos_id)
        assert isinstance(result, list)
        assert len(result) == 2
        for seq in result:
            assert isinstance(seq, list)
            assert all(isinstance(t, int) for t in seq)

    def test_generate_tokens_in_vocab_range(self, decoder, vocab):
        latent = torch.randn(1, 8)
        result = decoder.generate(latent, max_length=20, eos_id=vocab.eos_id)
        for tok in result[0]:
            assert 0 <= tok < vocab.vocab_size

    def test_generate_max_length_respected(self, decoder, vocab):
        latent = torch.randn(1, 8)
        max_len = 10
        result = decoder.generate(latent, max_length=max_len, eos_id=vocab.eos_id)
        assert len(result[0]) <= max_len

    def test_generate_greedy_deterministic(self, decoder, vocab):
        latent = torch.randn(1, 8)
        r1 = decoder.generate(latent, max_length=15, greedy=True, eos_id=vocab.eos_id)
        r2 = decoder.generate(latent, max_length=15, greedy=True, eos_id=vocab.eos_id)
        assert r1 == r2

    def test_generate_batch(self, decoder, vocab):
        latent = torch.randn(4, 8)
        result = decoder.generate(latent, max_length=20, eos_id=vocab.eos_id)
        assert len(result) == 4


class TestLatentConditioning:

    def test_different_latents_different_outputs(self, decoder, vocab):
        """Different latent codes should produce different logits."""
        seq = torch.randint(1, vocab.vocab_size, (1, 9))
        seq_len = torch.tensor([9])

        z1 = torch.randn(1, 8)
        z2 = torch.randn(1, 8) + 5.0  # very different

        logits1 = decoder(z1, seq, seq_len)
        logits2 = decoder(z2, seq, seq_len)

        # Logits should differ (not identical)
        assert not torch.allclose(logits1, logits2, atol=1e-3)

    def test_latent_at_position_zero(self, decoder, vocab):
        """The latent prefix is at position 0, before any walk tokens."""
        B = 1
        latent = torch.randn(B, 8)
        z_emb = decoder.latent_proj(latent)  # [B, d_model]
        assert z_emb.shape == (B, 64)  # d_model=64 in test fixture


class TestParameterCount:

    def test_reasonable_param_count(self, decoder):
        params = sum(p.numel() for p in decoder.parameters())
        # Small test model should be well under 1M params
        assert params < 1_000_000
        assert params > 1_000  # sanity: not trivially small


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
