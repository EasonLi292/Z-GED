# Model Architecture

This directory contains the neural network architectures for circuit generation.

## Production Models

### Core Components

- **encoder.py** - `HierarchicalEncoder`
  - 8D latent space VAE encoder (~97,600 params)
  - 3-layer ImpedanceGNN → hierarchical branches:
    - Branch 1: Mean+max pooling → z[0:2] (topology)
    - Branch 2: GND/VIN/VOUT node embeddings → z[2:4] (values)
    - Branch 3: DeepSets on poles/zeros → z[4:8] (transfer function)

- **decoder.py** - `SimplifiedCircuitDecoder`
  - Autoregressive decoder for circuit generation (~4.9M params)
  - Latent-guided generation with cross-attention
  - Produces node types, edge existence, and component types

- **decoder_components.py** - Helper classes for the decoder
  - `LatentDecomposer` - Splits latent code into semantic parts
  - `LatentGuidedEdgeDecoder` - Edge-level generation with latent guidance

- **node_decoder.py** - `AutoregressiveNodeDecoder`
  - Node-level autoregressive generation
  - Used as a component by the main decoder

### Specialized Models

- **tf_encoder.py** - Transfer function encoder
  - Encodes poles/zeros to latent space
  - Used for targeted circuit generation

- **gnn_layers.py** - Graph neural network layers
  - `ImpedanceConv` - Impedance-aware graph convolution
  - `ImpedanceGNN` - Full GNN architecture
  - `GlobalPooling`, `DeepSets` - Aggregation layers

### Utilities

- **component_utils.py** - Component type utilities
  - Component type encoding and conversion

- **constants.py** - Filter types and circuit templates
  - Standard filter definitions
  - Circuit topology templates

- **guided_generation.py** - Guided generation utilities
  - Connectivity guarantees
  - Post-processing for valid circuits

## Archive

Older/experimental models preserved for reference:
- `conditional_decoder.py` - Early conditional decoder (unused)
- `conditional_encoder.py` - Early conditional encoder (unused)

## Model Flow

```
Input Circuit → HierarchicalEncoder → 8D Latent Code → SimplifiedCircuitDecoder → Output Circuit
                     (encoder.py)                         (decoder.py)

Latent code alone determines the generated circuit (no external conditions).
```

## Configuration

Production models are configured in `scripts/training/train.py` with hardcoded defaults.
Config files in `configs/` are available for experimental training variants.
