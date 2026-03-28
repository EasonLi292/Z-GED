"""
Training script for the sequence-based circuit decoder.

Uses the existing HierarchicalEncoder (frozen or joint training) with
the new SequenceDecoder (GPT-style, latent-conditioned).

Loss = CE (next-token prediction) + kl_weight * KL (latent regularization)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ml.data.sequence_dataset import SequenceDataset, collate_sequence_batch
from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from ml.utils.runtime import build_encoder


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || p(z)) where p(z) = N(0, I)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(
    encoder, decoder, dataloader, optimizer, device, epoch,
    kl_weight=0.01, freeze_encoder=False,
):
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()
    decoder.train()

    total_ce = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        graph = batch['graph'].to(device)
        seq = batch['seq'].to(device)        # [B, max_seq_len]
        seq_len = batch['seq_len'].to(device) # [B]

        # Encode
        if freeze_encoder:
            with torch.no_grad():
                z, mu, logvar = encoder(
                    graph.x, graph.edge_index, graph.edge_attr, graph.batch
                )
            latent = mu  # deterministic when frozen
        else:
            z, mu, logvar = encoder(
                graph.x, graph.edge_index, graph.edge_attr, graph.batch
            )
            std = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(std) * std

        # Decode (teacher forcing)
        logits = decoder(latent, seq, seq_len)  # [B, L, vocab_size]

        # Losses
        ce_loss = decoder.compute_loss(logits, seq, seq_len)
        kl_loss = kl_divergence(mu, logvar)
        loss = ce_loss + kl_weight * kl_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters()) + (list(encoder.parameters()) if not freeze_encoder else []),
            max_norm=1.0,
        )
        optimizer.step()

        # Accuracy: fraction of correctly predicted non-padding tokens
        B, L, V = logits.shape
        preds = logits.argmax(dim=-1)  # [B, L]
        mask = torch.arange(L, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        correct = ((preds == seq) & mask).sum().item()
        total_tok = mask.sum().item()

        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        total_loss += loss.item()
        total_correct += correct
        total_tokens += total_tok
        num_batches += 1

        acc = correct / max(total_tok, 1) * 100
        pbar.set_postfix({
            'CE': f'{ce_loss.item():.3f}',
            'KL': f'{kl_loss.item():.3f}',
            'acc': f'{acc:.1f}%',
        })

    return {
        'ce_loss': total_ce / num_batches,
        'kl_loss': total_kl / num_batches,
        'total_loss': total_loss / num_batches,
        'accuracy': total_correct / max(total_tokens, 1) * 100,
    }


@torch.no_grad()
def validate(encoder, decoder, dataloader, device, kl_weight=0.01):
    encoder.eval()
    decoder.eval()

    total_ce = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in dataloader:
        graph = batch['graph'].to(device)
        seq = batch['seq'].to(device)
        seq_len = batch['seq_len'].to(device)

        z, mu, logvar = encoder(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )
        latent = mu  # deterministic for validation

        logits = decoder(latent, seq, seq_len)
        ce_loss = decoder.compute_loss(logits, seq, seq_len)
        kl_loss = kl_divergence(mu, logvar)
        loss = ce_loss + kl_weight * kl_loss

        B, L, V = logits.shape
        preds = logits.argmax(dim=-1)
        mask = torch.arange(L, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        correct = ((preds == seq) & mask).sum().item()
        total_tok = mask.sum().item()

        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        total_loss += loss.item()
        total_correct += correct
        total_tokens += total_tok
        num_batches += 1

    return {
        'ce_loss': total_ce / num_batches,
        'kl_loss': total_kl / num_batches,
        'total_loss': total_loss / num_batches,
        'accuracy': total_correct / max(total_tokens, 1) * 100,
    }


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 100
    kl_weight = 0.01
    freeze_encoder = False
    max_seq_len = 32

    print("=" * 60)
    print("Training Sequence Decoder (Bipartite Euler Walk)")
    print("=" * 60)
    print(f"Device: {device}")

    # Vocabulary
    vocab = CircuitVocabulary(max_internal=10, max_components=10)
    print(f"Vocabulary: {vocab}")

    # Load split
    split_data = torch.load('rlc_dataset/stratified_split.pt', weights_only=False)
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']

    # Datasets
    train_dataset = SequenceDataset(
        'rlc_dataset/filter_dataset.pkl', train_indices, vocab,
        augment=True, max_seq_len=max_seq_len,
    )
    val_dataset = SequenceDataset(
        'rlc_dataset/filter_dataset.pkl', val_indices, vocab,
        augment=False, max_seq_len=max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sequence_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Encoder (existing)
    encoder = build_encoder(device=device)

    # Optionally load pretrained encoder weights
    encoder_ckpt_path = 'checkpoints/production/best.pt'
    if os.path.exists(encoder_ckpt_path):
        ckpt = torch.load(encoder_ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"Loaded pretrained encoder from {encoder_ckpt_path}")

    # Sequence decoder (new)
    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size,
        latent_dim=8,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=max_seq_len + 1,  # +1 for latent prefix
        dropout=0.1,
        pad_id=vocab.pad_id,
    ).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {enc_params:,}")
    print(f"Decoder params: {dec_params:,}")

    # Optimizer
    if freeze_encoder:
        params = decoder.parameters()
    else:
        params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    checkpoint_dir = 'checkpoints/production'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # KL warmup
        current_kl = min(kl_weight, kl_weight * epoch / 20)

        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, device, epoch,
            kl_weight=current_kl, freeze_encoder=freeze_encoder,
        )
        val_metrics = validate(
            encoder, decoder, val_loader, device, kl_weight=current_kl,
        )

        scheduler.step(val_metrics['total_loss'])

        print(f"\nEpoch {epoch}")
        print(f"  Train — CE: {train_metrics['ce_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f}, "
              f"acc: {train_metrics['accuracy']:.1f}%")
        print(f"  Val   — CE: {val_metrics['ce_loss']:.4f}, "
              f"KL: {val_metrics['kl_loss']:.4f}, "
              f"acc: {val_metrics['accuracy']:.1f}%")

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'vocab_config': {
                    'max_internal': vocab.max_internal,
                    'max_components': vocab.max_components,
                },
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best.pt')
            print(f"  -> Saved best (loss: {best_val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
