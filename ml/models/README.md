# Model Architecture

This directory contains the core neural network architecture for Z-GED.

## Primary Modules

- `encoder.py` - `HierarchicalEncoder`
  - 8D latent: `[z_topology(2) | z_values(2) | z_pz(4)]`
  - Uses impedance-aware GNN layers over edge features `[log10(R), log10(C), log10(L)]`
- `decoder.py` - `SimplifiedCircuitDecoder`
  - Autoregressive node decoding
  - Autoregressive edge-component decoding (8 classes)
- `decoder_components.py` - `LatentGuidedEdgeDecoder`
  - Causal transformer over edge decisions
- `node_decoder.py` - `AutoregressiveNodeDecoder`
  - Transformer node decoder used by `decoder.py`
- `gnn_layers.py`
  - `ImpedanceConv`, `ImpedanceGNN`, pooling/deepsets components
- `constants.py`
  - Shared model constants (`FILTER_TYPES`, `CIRCUIT_TEMPLATES`, `PZ_LOG_SCALE`)

## Typical Flow

```text
graph -> HierarchicalEncoder -> latent z -> SimplifiedCircuitDecoder -> topology
```

## Config Source of Truth

Model defaults used by scripts are centralized in:
- `ml/utils/runtime.py` (`DEFAULT_ENCODER_CONFIG`, `DEFAULT_DECODER_CONFIG`)
