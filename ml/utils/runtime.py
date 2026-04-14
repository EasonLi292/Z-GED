"""Runtime helpers for model setup and data collation."""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from ml.models.decoder import SequenceDecoder
from ml.models.vocabulary import CircuitVocabulary
from ml.models.encoder import HierarchicalEncoder
from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.attribute_heads import FreqHead, GainHead, TypeHead
from ml.models.constants import FILTER_TYPES_V2, TYPE_TO_IDX
from yubo.auxiliary_heads import ClassificationMLP, RegressionMLP

DEFAULT_ENCODER_CONFIG: Dict[str, Any] = {
    'node_feature_dim': 4,
    'edge_feature_dim': 3,
    'gnn_hidden_dim': 64,
    'gnn_num_layers': 3,
    'latent_dim': 8,
    'topo_latent_dim': 2,
    'values_latent_dim': 2,
    'pz_latent_dim': 4,
    'dropout': 0.1,
}

DEFAULT_DECODER_CONFIG: Dict[str, Any] = {
    'vocab_size': 86,  # CircuitVocabulary defaults
    'latent_dim': 8,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 4,
    'max_seq_len': 33,  # 32 + 1 for latent prefix
    'dropout': 0.1,
    'pad_id': 0,
}

def build_encoder(device: str = 'cpu', **overrides: Any) -> HierarchicalEncoder:
    """Build an encoder with repository defaults."""
    config = dict(DEFAULT_ENCODER_CONFIG)
    config.update(overrides)
    return HierarchicalEncoder(**config).to(device)


def build_vocab(**overrides: Any) -> CircuitVocabulary:
    """Build vocabulary with defaults matching DEFAULT_DECODER_CONFIG."""
    max_internal = overrides.get('max_internal', 10)
    max_components = overrides.get('max_components', 10)
    return CircuitVocabulary(max_internal=max_internal, max_components=max_components)


def build_decoder(device: str = 'cpu', **overrides: Any) -> SequenceDecoder:
    """Build a sequence decoder with repository defaults."""
    config = dict(DEFAULT_DECODER_CONFIG)
    config.update(overrides)
    return SequenceDecoder(**config).to(device)


def load_encoder_decoder(
    checkpoint_path: str,
    device: str = 'cpu',
    encoder_overrides: Optional[Dict[str, Any]] = None,
    decoder_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[HierarchicalEncoder, SequenceDecoder, CircuitVocabulary, Dict[str, Any]]:
    """Load encoder and decoder weights from a checkpoint.

    Returns (encoder, decoder, vocab, checkpoint).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build vocab from checkpoint config if available
    vocab_cfg = checkpoint.get('vocab_config', {})
    vocab = build_vocab(**vocab_cfg)

    encoder = build_encoder(device=device, **(encoder_overrides or {}))

    # Merge decoder overrides with vocab-derived settings
    dec_cfg = dict(decoder_overrides or {})
    dec_cfg.setdefault('vocab_size', vocab.vocab_size)
    dec_cfg.setdefault('pad_id', vocab.pad_id)
    decoder = build_decoder(device=device, **dec_cfg)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab, checkpoint


def load_decoder(
    checkpoint_path: str,
    device: str = 'cpu',
    decoder_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[SequenceDecoder, CircuitVocabulary, Dict[str, Any]]:
    """Load decoder weights from a checkpoint.

    Returns (decoder, vocab, checkpoint).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab_cfg = checkpoint.get('vocab_config', {})
    vocab = build_vocab(**vocab_cfg)

    dec_cfg = dict(decoder_overrides or {})
    dec_cfg.setdefault('vocab_size', vocab.vocab_size)
    dec_cfg.setdefault('pad_id', vocab.pad_id)
    decoder = build_decoder(device=device, **dec_cfg)

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    return decoder, vocab, checkpoint


def collate_circuit_batch(
    batch_list: List[Dict[str, Any]],
    include_specifications: bool = False,
    include_pz_target: bool = False,
    include_indices: bool = False,
    include_filter_type_label: bool = False,
) -> Dict[str, Any]:
    """Collate circuit dataset samples into a PyG batch."""
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]

    output: Dict[str, Any] = {
        'graph': Batch.from_data_list(graphs),
        'poles': poles,
        'zeros': zeros,
    }

    if include_specifications:
        output['specifications'] = torch.stack([item['specifications'] for item in batch_list])

    if include_pz_target:
        output['pz_target'] = torch.stack([item['pz_target'] for item in batch_list])

    if include_indices:
        output['indices'] = [item['idx'] for item in batch_list]

    if include_filter_type_label:
        output['filter_type_label'] = torch.stack(
            [item['filter_type'].argmax().long() for item in batch_list]
        )

    return output


def make_collate_fn(
    include_specifications: bool = False,
    include_pz_target: bool = False,
    include_indices: bool = False,
    include_filter_type_label: bool = False,
):
    """Create a partial collate function for DataLoader."""
    return partial(
        collate_circuit_batch,
        include_specifications=include_specifications,
        include_pz_target=include_pz_target,
        include_indices=include_indices,
        include_filter_type_label=include_filter_type_label,
    )


# ── v2 Admittance Encoder builders ──────────────────────────────

DEFAULT_V2_ENCODER_CONFIG: Dict[str, Any] = {
    'node_feature_dim': 4,
    'hidden_dim': 64,
    'latent_dim': 5,
    'num_layers': 3,
    'dropout': 0.0,
    'vae': True,
}

DEFAULT_V2_DECODER_CONFIG: Dict[str, Any] = {
    'vocab_size': 86,
    'latent_dim': 5,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'max_seq_len': 33,
    'dropout': 0.0,
    'pad_id': 0,
}


def build_v2_encoder(device: str = 'cpu', **overrides: Any) -> AdmittanceEncoder:
    """Build a v2 AdmittanceEncoder with repository defaults."""
    config = dict(DEFAULT_V2_ENCODER_CONFIG)
    config.update(overrides)
    return AdmittanceEncoder(**config).to(device)


def build_v2_decoder(device: str = 'cpu', **overrides: Any) -> SequenceDecoder:
    """Build a v2 SequenceDecoder with repository defaults."""
    config = dict(DEFAULT_V2_DECODER_CONFIG)
    config.update(overrides)
    return SequenceDecoder(**config).to(device)


def build_attribute_heads(
    latent_dim: int = 5,
    n_types: int = 10,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Build attribute prediction heads for the v2 model."""
    return {
        'freq': FreqHead(latent_dim).to(device),
        'gain': GainHead(latent_dim).to(device),
        'type': TypeHead(latent_dim, n_types).to(device),
    }


def load_v2_model(
    checkpoint_path: str,
    device: str = 'cpu',
) -> Tuple[AdmittanceEncoder, SequenceDecoder, CircuitVocabulary,
           Dict[str, Any], Dict[str, Any]]:
    """Load v2 model from checkpoint.

    Returns (encoder, decoder, vocab, heads_dict, checkpoint).
    heads_dict has keys 'freq', 'gain', 'type'.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = build_vocab()
    latent_dim = ckpt.get('latent_dim', 5)

    encoder = build_v2_encoder(device=device, latent_dim=latent_dim)
    decoder = build_v2_decoder(device=device, latent_dim=latent_dim,
                               vocab_size=vocab.vocab_size, pad_id=vocab.pad_id)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    heads = build_attribute_heads(latent_dim=latent_dim, device=device)
    heads['freq'].load_state_dict(ckpt['freq_head_state_dict'])
    heads['gain'].load_state_dict(ckpt['gain_head_state_dict'])
    heads['type'].load_state_dict(ckpt['type_head_state_dict'])
    for h in heads.values():
        h.eval()

    return encoder, decoder, vocab, heads, ckpt
