"""Train the AdmittanceEncoder v2 + SequenceDecoder as an inverse-design model.

Pipeline:
    circuit graph  --AdmittanceEncoder(vae=True)-->  z in R^5
                                                      |
                                                      v
                   SequenceDecoder --> autoregressive Eulerian walk

Architecture:
  - phi_G/C/L are 2-layer MLPs (nonlinear, more expressive)
  - Learned coefficient scaling (linear + log1p blend)
  - 5D structured VAE latent: [topo(2D) | VIN(1D) | VOUT(1D) | GND(1D)]
  - Attribute heads on mu: FreqHead, GainHead, TypeHead
  - Loss: CE + beta_topo*KL_topo + beta_term*KL_term + alpha*(freq_MSE + gain_MSE + type_CE)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.models.vocabulary import CircuitVocabulary
from ml.models.decoder import SequenceDecoder
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.attribute_heads import FreqHead, GainHead, TypeHead, kl_divergence
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from ml.data.cross_topo_dataset import CrossTopoSequenceDataset, collate_fn


ALL_TYPES = set(FILTER_TYPES_V2)


# -- Training / validation -------------------------------------------

def train_epoch(encoder, decoder, heads, loader, optimizer, device, epoch,
                beta_topo=0.1, beta_term=0.02, beta_warmup=20,
                alpha_freq=0.5, alpha_gain=0.5, alpha_type=0.5):
    encoder.train(); decoder.train()
    for h in heads.values():
        h.train()

    ramp = min(1.0, epoch / beta_warmup)
    cur_bt = beta_topo * ramp
    cur_bm = beta_term * ramp

    totals = {'ce': 0, 'kl_topo': 0, 'kl_term': 0,
              'freq': 0, 'gain': 0, 'type': 0,
              'correct': 0, 'tokens': 0, 'type_correct': 0}
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        g = batch['graph'].to(device)
        seq = batch['seq'].to(device)
        seq_len = batch['seq_len'].to(device)
        beh = batch['behavior'].to(device)
        gain_1k = batch['gain_1k'].to(device)
        type_ids = torch.tensor(
            [TYPE_TO_IDX[ft] for ft in batch['filter_types']],
            dtype=torch.long, device=device)

        z, mu, logvar = encoder(g.x, g.edge_index, g.edge_attr, g.batch)

        logits = decoder(z, seq, seq_len)
        ce = decoder.compute_loss(logits, seq, seq_len)

        kl_topo = kl_divergence(mu[:, :2], logvar[:, :2])
        kl_term = kl_divergence(mu[:, 2:], logvar[:, 2:])

        freq_pred = heads['freq'](mu)
        gain_pred = heads['gain'](mu)
        type_logits = heads['type'](mu)

        freq_loss = F.mse_loss(freq_pred, beh)
        gain_loss = F.mse_loss(gain_pred, gain_1k)
        type_loss = F.cross_entropy(type_logits, type_ids)

        loss = (ce
                + cur_bt * kl_topo + cur_bm * kl_term
                + alpha_freq * freq_loss
                + alpha_gain * gain_loss
                + alpha_type * type_loss)

        optimizer.zero_grad()
        loss.backward()
        all_params = (list(encoder.parameters()) + list(decoder.parameters())
                      + [p for h in heads.values() for p in h.parameters()])
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        B, L, _ = logits.shape
        mask = torch.arange(L, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        totals['correct'] += ((preds == seq) & mask).sum().item()
        totals['tokens'] += mask.sum().item()
        totals['ce'] += ce.item()
        totals['kl_topo'] += kl_topo.item()
        totals['kl_term'] += kl_term.item()
        totals['freq'] += freq_loss.item()
        totals['gain'] += gain_loss.item()
        totals['type'] += type_loss.item()
        totals['type_correct'] += (type_logits.argmax(-1) == type_ids).sum().item()
        n += 1

        pbar.set_postfix(
            CE=f'{ce.item():.3f}',
            KL=f'{(kl_topo + kl_term).item():.2f}',
            acc=f"{100 * totals['correct'] / max(totals['tokens'], 1):.1f}%",
        )

    N = max(n, 1)
    return {
        'ce': totals['ce'] / N,
        'kl_topo': totals['kl_topo'] / N,
        'kl_term': totals['kl_term'] / N,
        'freq_mse': totals['freq'] / N,
        'gain_mse': totals['gain'] / N,
        'type_ce': totals['type'] / N,
        'acc': 100 * totals['correct'] / max(totals['tokens'], 1),
        'type_acc': 100 * totals['type_correct'] / max(n * loader.batch_size, 1),
    }


@torch.no_grad()
def validate(encoder, decoder, heads, loader, device,
             beta_topo, beta_term, alpha_freq, alpha_gain, alpha_type):
    encoder.eval(); decoder.eval()
    for h in heads.values():
        h.eval()

    totals = {'ce': 0, 'kl_topo': 0, 'kl_term': 0,
              'freq': 0, 'gain': 0, 'type': 0,
              'correct': 0, 'tokens': 0, 'type_correct': 0, 'n_samples': 0}
    n = 0

    for batch in loader:
        g = batch['graph'].to(device)
        seq = batch['seq'].to(device)
        seq_len = batch['seq_len'].to(device)
        beh = batch['behavior'].to(device)
        gain_1k = batch['gain_1k'].to(device)
        type_ids = torch.tensor(
            [TYPE_TO_IDX[ft] for ft in batch['filter_types']],
            dtype=torch.long, device=device)

        z, mu, logvar = encoder(g.x, g.edge_index, g.edge_attr, g.batch)
        logits = decoder(mu, seq, seq_len)
        ce = decoder.compute_loss(logits, seq, seq_len)

        kl_topo = kl_divergence(mu[:, :2], logvar[:, :2])
        kl_term = kl_divergence(mu[:, 2:], logvar[:, 2:])

        freq_pred = heads['freq'](mu)
        gain_pred = heads['gain'](mu)
        type_logits = heads['type'](mu)

        totals['ce'] += ce.item()
        totals['kl_topo'] += kl_topo.item()
        totals['kl_term'] += kl_term.item()
        totals['freq'] += F.mse_loss(freq_pred, beh).item()
        totals['gain'] += F.mse_loss(gain_pred, gain_1k).item()
        totals['type'] += F.cross_entropy(type_logits, type_ids).item()

        preds = logits.argmax(dim=-1)
        B, L, _ = logits.shape
        mask = torch.arange(L, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        totals['correct'] += ((preds == seq) & mask).sum().item()
        totals['tokens'] += mask.sum().item()
        totals['type_correct'] += (type_logits.argmax(-1) == type_ids).sum().item()
        totals['n_samples'] += len(batch['filter_types'])
        n += 1

    N = max(n, 1)
    return {
        'ce': totals['ce'] / N,
        'kl_topo': totals['kl_topo'] / N,
        'kl_term': totals['kl_term'] / N,
        'freq_mse': totals['freq'] / N,
        'gain_mse': totals['gain'] / N,
        'type_ce': totals['type'] / N,
        'acc': 100 * totals['correct'] / max(totals['tokens'], 1),
        'type_acc': 100 * totals['type_correct'] / max(totals['n_samples'], 1),
    }


@torch.no_grad()
def decode_per_type(encoder, decoder, vocab, dataset, device, label):
    encoder.eval(); decoder.eval()
    by_type = {}
    for i in range(len(dataset)):
        circ = dataset.circuits[i]
        graph = dataset.pyg_graphs[i].to(device)
        batch_idx = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
        result = encoder(graph.x, graph.edge_index, graph.edge_attr, batch_idx)
        mu = result[1] if isinstance(result, tuple) else result
        gen = decoder.generate(mu, max_length=32, greedy=True, eos_id=vocab.eos_id)
        tokens = tuple(t for t in vocab.decode(gen[0])
                       if t not in ('BOS', 'EOS', 'PAD'))
        by_type.setdefault(circ['filter_type'], []).append(' '.join(tokens))

    print(f"\n{label} decoded-walk diversity:")
    for ft in sorted(by_type.keys()):
        seqs = by_type[ft]
        uniq = len(set(seqs))
        print(f"  {ft:>15s}: {uniq} unique / {len(seqs)} circuits")


def main():
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    lr = 3e-4
    epochs = 80
    max_seq_len = 32
    latent_dim = 5
    n_augment_walks = 32

    beta_topo = 0.1
    beta_term = 0.02
    beta_warmup = 20
    alpha_freq = 0.5
    alpha_gain = 0.5
    alpha_type = 0.5

    pkl_main = 'rlc_dataset/filter_dataset.pkl'
    pkl_rl = 'rlc_dataset/rl_dataset.pkl'

    print("=" * 72)
    print("INVERSE-DESIGN MODEL v2 (AdmittanceEncoder VAE + SequenceDecoder)")
    print(f"  latent_dim = {latent_dim} (2D topo + 1Dx3 terminal)")
    print(f"  beta_topo={beta_topo}, beta_term={beta_term}, warmup={beta_warmup}")
    print(f"  alpha_freq={alpha_freq}, alpha_gain={alpha_gain}, alpha_type={alpha_type}")
    print("=" * 72)

    vocab = CircuitVocabulary(max_internal=10, max_components=10)

    full_ds = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        augment=False, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial')

    type_indices = {}
    for i, circ in enumerate(full_ds.circuits):
        type_indices.setdefault(circ['filter_type'], []).append(i)

    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for ft, idxs in sorted(type_indices.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_idx.extend(idxs[:split].tolist())
        val_idx.extend(idxs[split:].tolist())
    del full_ds

    train_dataset = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        indices=train_idx, augment=True, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial', n_augment_walks=n_augment_walks)
    val_dataset = CrossTopoSequenceDataset(
        [pkl_main, pkl_rl], ALL_TYPES, vocab,
        indices=val_idx, augment=False, max_seq_len=max_seq_len,
        edge_feature_mode='polynomial')

    print(f"Train: {len(train_dataset)} circuits")
    print(f"Val:   {len(val_dataset)} circuits")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    encoder = AdmittanceEncoder(
        node_feature_dim=4, hidden_dim=64, latent_dim=latent_dim,
        num_layers=3, dropout=0.1, vae=True,
    ).to(device)
    decoder = SequenceDecoder(
        vocab_size=vocab.vocab_size, latent_dim=latent_dim,
        d_model=128, n_heads=4, n_layers=2,
        max_seq_len=max_seq_len + 1, dropout=0.15, pad_id=vocab.pad_id,
    ).to(device)

    heads = {
        'freq': FreqHead(latent_dim).to(device),
        'gain': GainHead(latent_dim).to(device),
        'type': TypeHead(latent_dim).to(device),
    }

    enc_p = sum(p.numel() for p in encoder.parameters())
    dec_p = sum(p.numel() for p in decoder.parameters())
    head_p = sum(p.numel() for h in heads.values() for p in h.parameters())
    print(f"Encoder: {enc_p:,}  |  Decoder: {dec_p:,}  |  Heads: {head_p:,}")

    all_params = (list(encoder.parameters()) + list(decoder.parameters())
                  + [p for h in heads.values() for p in h.parameters()])
    optimizer = torch.optim.AdamW(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8)

    ckpt_path = 'checkpoints/production/best_v2.pt'
    best_val, best_epoch = float('inf'), 0

    for epoch in range(1, epochs + 1):
        t = train_epoch(encoder, decoder, heads, train_loader,
                        optimizer, device, epoch,
                        beta_topo=beta_topo, beta_term=beta_term,
                        beta_warmup=beta_warmup,
                        alpha_freq=alpha_freq, alpha_gain=alpha_gain,
                        alpha_type=alpha_type)
        v = validate(encoder, decoder, heads, val_loader, device,
                     beta_topo=beta_topo, beta_term=beta_term,
                     alpha_freq=alpha_freq, alpha_gain=alpha_gain,
                     alpha_type=alpha_type)
        scheduler.step(v['ce'])

        star = ' *' if v['ce'] < best_val else ''
        print(f"Epoch {epoch:>3d}  "
              f"t_CE={t['ce']:.3f} t_acc={t['acc']:.1f}%  |  "
              f"v_CE={v['ce']:.3f} v_acc={v['acc']:.1f}%  "
              f"KL={v['kl_topo']:.2f}+{v['kl_term']:.2f}  "
              f"freq={v['freq_mse']:.3f} gain={v['gain_mse']:.4f} "
              f"type={v['type_acc']:.0f}%{star}")

        if v['ce'] < best_val:
            best_val = v['ce']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'freq_head_state_dict': heads['freq'].state_dict(),
                'gain_head_state_dict': heads['gain'].state_dict(),
                'type_head_state_dict': heads['type'].state_dict(),
                'val_ce': best_val,
                'latent_dim': latent_dim,
                'type_to_idx': TYPE_TO_IDX,
                'vocab_config': {'max_internal': vocab.max_internal,
                                 'max_components': vocab.max_components},
                'hyperparams': {
                    'beta_topo': beta_topo, 'beta_term': beta_term,
                    'alpha_freq': alpha_freq, 'alpha_gain': alpha_gain,
                    'alpha_type': alpha_type,
                },
            }, ckpt_path)

    print(f"\nBest epoch {best_epoch}, val CE={best_val:.4f}")

    ckpt = torch.load(ckpt_path, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decode_per_type(encoder, decoder, vocab, val_dataset, device, 'Val')


if __name__ == '__main__':
    main()
