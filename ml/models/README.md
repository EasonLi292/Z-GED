# Model Architecture

This directory contains the core neural network architecture for Z-GED.

## Primary Modules

- `encoder.py` — `HierarchicalEncoder`
  - 8D latent: `[z_topology(2) | z_values(2) | z_pz(4)]`
  - Uses impedance-aware GNN layers over edge features `[log10(R), log10(C), log10(L)]`
- `decoder.py` — `SequenceDecoder`
  - GPT-style autoregressive transformer
  - Generates circuits as Eulerian walk token sequences
  - Latent prefix conditioning (z projected to a prefix token)
- `vocabulary.py` — `CircuitVocabulary`
  - 86-token vocabulary: PAD, EOS, net tokens, component tokens
  - Encode/decode between token strings and integer IDs
- `gnn_layers.py`
  - `ImpedanceConv`, `ImpedanceGNN`, pooling/deepsets components
- `constants.py`
  - Shared model constants (`FILTER_TYPES`, `CIRCUIT_TEMPLATES`, `PZ_LOG_SCALE`)

## Archived Modules

The old adjacency-matrix decoder is preserved in `archive/`:
- `archive/decoder_adjacency.py` — `SimplifiedCircuitDecoder`
- `archive/node_decoder.py` — `AutoregressiveNodeDecoder`
- `archive/decoder_components.py` — `LatentGuidedEdgeDecoder`
- `archive/circuit_loss.py` — adjacency-based loss function

## Typical Flow

```text
graph -> HierarchicalEncoder -> latent z -> SequenceDecoder -> Eulerian walk tokens
```

## Config Source of Truth

Model defaults used by scripts are centralized in:
- `ml/utils/runtime.py` (`DEFAULT_ENCODER_CONFIG`, `DEFAULT_DECODER_CONFIG`)
