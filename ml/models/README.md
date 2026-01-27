# Model Architecture

This directory contains the neural network architectures for circuit generation.

## Production Models

### Core Components

- **encoder.py** - `HierarchicalEncoder`
  - 8D latent space VAE encoder (~97,600 params)
  - 3-layer ImpedanceGNN + hierarchical branches:
    - Branch 1: Mean+max pooling → z[0:2] (topology)
    - Branch 2: GND/VIN/VOUT node embeddings → z[2:4] (values)
    - Branch 3: DeepSets on poles/zeros → z[4:8] (transfer function)

- **decoder.py** - `SimplifiedCircuitDecoder`
  - Autoregressive decoder for circuit generation (~5.6M params)
  - Autoregressive node generation (transformer-based)
  - Autoregressive edge generation (GRU-based, GraphRNN "Edge-level RNN" concept)
  - Teacher forcing during training; own predictions fed back at inference

- **decoder_components.py** - `LatentGuidedEdgeDecoder`
  - Edge-level autoregressive generation with latent guidance
  - GRU cell maintains hidden state across all edge decisions
  - Cross-attention to latent code + fusion with GRU hidden state
  - 8-way classification (0=no edge, 1-7=component type)

- **node_decoder.py** - `AutoregressiveNodeDecoder`
  - Transformer-based node-level autoregressive generation
  - Used as a component by the main decoder

### Supporting Modules

- **gnn_layers.py** - Graph neural network layers
  - `ImpedanceConv` - Component-aware graph convolution (separate R/C/L transformations)
  - `ImpedanceGNN` - Multi-layer GNN architecture
  - `GlobalPooling`, `DeepSets` - Aggregation layers

- **component_utils.py** - Component type utilities
  - `masks_to_component_type()` - Convert R/C/L masks to 8-way classification

- **constants.py** - Filter types and circuit templates
  - `FILTER_TYPES` - 6 supported filter types
  - `CIRCUIT_TEMPLATES` - Standard circuit topology templates

## Model Flow

```
Input Circuit → HierarchicalEncoder → 8D Latent Code → SimplifiedCircuitDecoder → Output Circuit
                     (encoder.py)                         (decoder.py)

Decoding pipeline:
  z (8D) → Context Encoder → Node Count Predictor
                            → Autoregressive Node Decoder (transformer)
                            → Autoregressive Edge Decoder (GRU + cross-attention)
                            → node_types, edge_existence, component_types

Latent code alone determines the generated circuit (no external conditions).
```

## Configuration

Production models are configured in `scripts/training/train.py` with hardcoded defaults.
Config files in `configs/` are available for experimental training variants.
