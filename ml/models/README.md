# Model Architecture

This directory contains the neural network architectures for circuit generation.

## Production Models

### Core Components

- **encoder.py** - `HierarchicalEncoder`
  - 8D latent space VAE encoder
  - Hierarchical decomposition: [2D topology | 2D values | 4D transfer function]
  - Graph neural network-based encoding

- **decoder.py** - `LatentGuidedGraphGPTDecoder`
  - Main autoregressive decoder for circuit generation
  - Latent-guided generation with cross-attention
  - Produces graph structure and component values

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

- **gumbel_softmax_utils.py** - Gumbel-Softmax sampling utilities
  - Discrete component type sampling
  - Component masking and conversion

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
Input Circuit → HierarchicalEncoder → 8D Latent Code → LatentGuidedGraphGPTDecoder → Output Circuit
                     (encoder.py)                         (decoder.py)

Specifications [cutoff, Q] → Decoder Conditions → Circuit Generation
```

## Configuration

Production models use configuration from `configs/production.yaml` (formerly `latent_guided_decoder.yaml`).
