"""
GPT-style sequence decoder for circuit generation.

The latent code from the encoder is projected to an initial context
embedding at position 0 (prefix token). The decoder then autoregressively
predicts circuit walk tokens via causal self-attention.

Architecture:
    Position 0:    latent_proj(z)          — no vocabulary token
    Position 1..L: token_embed + pos_embed — circuit walk tokens

    Target at position i: walk[i]  (the next token in the sequence)

    Loss: cross-entropy on positions 0..L-1 predicting walk[0..L-1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SequenceDecoder(nn.Module):
    """
    Decoder-only transformer for circuit sequence generation.

    The latent code is the ONLY conditioning signal, injected as the
    first position in the causal attention window.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        latent_dim: Dimension of the encoder's latent code (default 8).
        d_model: Transformer hidden dimension (default 256).
        n_heads: Number of attention heads (default 8).
        n_layers: Number of transformer blocks (default 4).
        max_seq_len: Maximum sequence length including latent prefix (default 65).
        dropout: Dropout probability (default 0.1).
        pad_id: Padding token ID (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 65,  # 1 (latent prefix) + 64 (max walk tokens)
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # Latent projection: 8D → d_model (prefix token at position 0)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Token embeddings (for positions 1+)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Positional embeddings (position 0 = latent, 1..L = walk tokens)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Layer norm after embedding
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer decoder blocks (using encoder layers with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm (like GPT-2)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: d_model → vocab_size
        self.output_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = blocked)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        latent: torch.Tensor,
        seq: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        The input sequence is shifted: position 0 is the latent prefix,
        positions 1..L hold walk tokens [0..L-1]. We predict walk[0..L-1]
        from positions 0..L-1.

        Args:
            latent:  [B, latent_dim] — encoder latent code.
            seq:     [B, max_walk_len] — padded walk token IDs.
            seq_len: [B] — un-padded walk lengths.

        Returns:
            logits: [B, max_walk_len, vocab_size] — next-token logits.
                    logits[:, t, :] predicts seq[:, t].
        """
        B, L = seq.shape
        device = seq.device

        # Position 0: projected latent — [B, 1, d_model]
        z_emb = self.latent_proj(latent).unsqueeze(1)

        # Positions 1..L-1: token embeddings for seq[:, :-1] (shifted input)
        # We feed tokens [0..L-2] to predict tokens [0..L-1]
        # But position 0 (latent) predicts token 0, so:
        #   input = [z_emb, embed(seq[0]), embed(seq[1]), ..., embed(seq[L-2])]
        #   target = [seq[0], seq[1], ..., seq[L-1]]
        tok_emb = self.token_embedding(seq[:, :-1])  # [B, L-1, d_model]

        # Concatenate: [z_emb, tok_emb] → [B, L, d_model]
        x = torch.cat([z_emb, tok_emb], dim=1)  # [B, L, d_model]

        # Add positional embeddings
        positions = torch.arange(L, device=device)  # [L]
        x = x + self.pos_embedding(positions).unsqueeze(0)  # broadcast [1, L, d_model]

        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # Causal mask + padding mask
        causal = self._causal_mask(L, device)

        # Padding mask: positions beyond seq_len should not attend or be attended to.
        # For each sample, valid positions are 0..seq_len-1 (shifted, so up to seq_len in the L-length input).
        # We mask the key positions that are padding.
        # Input length is L = max_walk_len, but valid input positions are:
        #   position 0 (latent): always valid
        #   positions 1..seq_len-1: valid (tokens 0..seq_len-2)
        #   positions seq_len..L-1: padding
        pad_mask = torch.arange(L, device=device).unsqueeze(0) >= seq_len.unsqueeze(1)  # [B, L]

        # Transformer with causal + padding mask
        x = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)
        x = self.final_norm(x)

        # Project to vocab
        logits = self.output_head(x)  # [B, L, vocab_size]

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        seq: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss, ignoring padding positions.

        Args:
            logits:  [B, L, vocab_size] — output of forward().
            seq:     [B, L] — full target sequence (including padding).
            seq_len: [B] — un-padded lengths.

        Returns:
            Scalar loss.
        """
        B, L, V = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, V)   # [B*L, V]
        targets_flat = seq.reshape(-1)        # [B*L]

        # Mask: only compute loss for non-padding positions
        mask = torch.arange(L, device=seq.device).unsqueeze(0) < seq_len.unsqueeze(1)  # [B, L]
        mask_flat = mask.reshape(-1)  # [B*L]

        # Cross-entropy with masking
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*L]
        loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum()

        return loss

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 64,
        temperature: float = 1.0,
        greedy: bool = True,
        eos_id: int = 1,
    ) -> List[List[int]]:
        """
        Autoregressive generation from latent code.

        Starts with the latent prefix, generates tokens one at a time.
        Stops when EOS is generated or max_length reached.

        Args:
            latent: [B, latent_dim] — latent codes.
            max_length: Maximum walk length.
            temperature: Sampling temperature (ignored if greedy=True).
            greedy: If True, use argmax; otherwise sample.
            eos_id: Token ID for EOS (stop signal).

        Returns:
            List of B token ID sequences (variable length, no padding,
            EOS excluded from output).
        """
        B = latent.shape[0]
        device = latent.device

        # Start with latent prefix
        z_emb = self.latent_proj(latent).unsqueeze(1)  # [B, 1, d_model]

        # Track generated tokens per sample
        generated: List[List[int]] = [[] for _ in range(B)]
        active = torch.ones(B, dtype=torch.bool, device=device)

        # Current input sequence (starts with just z_emb)
        # We'll maintain the full embedding sequence for simplicity
        current_emb = z_emb  # [B, 1, d_model]

        for step in range(max_length):
            seq_len = current_emb.shape[1]

            # Add positional embeddings
            positions = torch.arange(seq_len, device=device)
            x = current_emb + self.pos_embedding(positions).unsqueeze(0)
            x = self.embed_norm(x)

            # Causal attention
            causal = self._causal_mask(seq_len, device)
            x = self.transformer(x, mask=causal)
            x = self.final_norm(x)

            # Get logits for last position
            logits = self.output_head(x[:, -1, :])  # [B, vocab_size]

            if greedy:
                next_token = logits.argmax(dim=-1)  # [B]
            else:
                scaled = logits / max(temperature, 1e-8)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Record tokens for active samples
            for b in range(B):
                if active[b]:
                    tok = next_token[b].item()
                    if tok == eos_id:
                        active[b] = False  # stop, don't include EOS in output
                    else:
                        generated[b].append(tok)

            if not active.any():
                break

            # Append next token embedding to current sequence
            next_emb = self.token_embedding(next_token).unsqueeze(1)  # [B, 1, d_model]
            current_emb = torch.cat([current_emb, next_emb], dim=1)

        return generated
