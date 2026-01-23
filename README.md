# Z-GED: Specification-Driven Circuit Generation

Automated RLC filter circuit synthesis from user specifications using a conditional VAE with component-aware GNN encoder.

## What It Does

Z-GED generates RLC filter circuits that match target frequency response characteristics. You provide a cutoff frequency and Q-factor, and it outputs a valid circuit topology.

**Supported filter types:** Low-pass, High-pass, Band-pass, Band-stop, RLC parallel, RLC series

**Model Version:** v4.7 (Component-Aware Message Passing)

### Current Performance

| Metric | Validation |
|--------|------------|
| Node Count | 100% |
| Edge Existence | 100% |
| Component Type | 90% |

**Example outputs:**
- High-Q resonant (Q=10): `GND--RCL--VOUT, VIN--R--VOUT`
- Overdamped (Q=0.2): `GND--R--VOUT, VIN--L--N3, VOUT--C--N3`
- Standard filter (Q=0.707): `GND--RCL--VOUT, VIN--R--VOUT`

## Quick Start

### Generate a Circuit

```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

This generates a filter with ~10 kHz cutoff and Butterworth response (Q=0.707).

### Train the Model

```bash
python scripts/training/train.py
```

Training takes ~10 minutes on CPU for 100 epochs.

### Validate

```bash
python scripts/training/validate.py
```

## Installation

```bash
pip install torch torch-geometric numpy scipy

# ngspice required for SPICE simulation
# macOS: brew install ngspice
# Ubuntu: apt install ngspice
```

## Usage

### Command Line

```bash
# Generate with specific specs
python scripts/generation/generate_from_specs.py --cutoff 5000 --q-factor 1.0

# Test on multiple specifications
python scripts/testing/test_comprehensive_specs.py
```

### Python API

```python
import torch
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import SimplifiedCircuitDecoder

# Load trained model
device = 'cpu'
encoder = HierarchicalEncoder(
    node_feature_dim=4, edge_feature_dim=7, gnn_hidden_dim=64,
    gnn_num_layers=3, latent_dim=8, topo_latent_dim=2,
    values_latent_dim=2, pz_latent_dim=4, dropout=0.1
).to(device)

decoder = SimplifiedCircuitDecoder(
    latent_dim=8, conditions_dim=2, hidden_dim=256,
    num_heads=8, num_node_layers=4, max_nodes=10, dropout=0.1
).to(device)

checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()

# Build specification database from training data
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
# ... encode all circuits to build specs_db and latents_db ...

# Generate circuit using K-NN interpolation
target_cutoff = 10000  # Hz
target_q = 0.707

circuit = decoder.generate_from_specification(
    frequency=target_cutoff,
    q_factor=target_q,
    specs_db=specs_db,
    latents_db=latents_db,
    k=5,  # Use 5 nearest neighbors
    edge_threshold=0.5
)
```

## Project Structure

```
Z-GED/
├── ml/
│   ├── models/          # Encoder and decoder architectures
│   ├── losses/          # Training loss functions
│   ├── data/            # Dataset loading
│   └── utils/           # SPICE simulator interface
├── scripts/
│   ├── generation/      # Circuit generation scripts
│   ├── training/        # Training and validation
│   └── testing/         # Test scripts
├── tools/               # Dataset generation utilities
├── checkpoints/         # Trained models
└── rlc_dataset/         # Training data
```

## How It Works

1. **Encode specifications** - Target cutoff and Q-factor are normalized and used as conditioning
2. **Find similar circuits** - K-nearest neighbor search in latent space finds circuits with similar specs
3. **Interpolate latents** - Weighted average of nearest latent codes
4. **Decode circuit** - Transformer decoder generates nodes and edges with component types
5. **Validate with SPICE** - ngspice simulation verifies the circuit meets specifications

### One-to-Many Mapping

The specification-to-circuit mapping is **one-to-many**: multiple topologies can achieve the same transfer function. For overlapping frequency ranges, both simple RC filters and RLC resonant circuits can meet the same specifications. The model generates a valid topology from the possible set.

## Specifications Range

Based on training data (120 circuits):

| Filter Type | Frequency Range | Q-Factor Range |
|-------------|-----------------|----------------|
| low_pass | 1.7 Hz - 65 kHz | 0.707 (fixed) |
| high_pass | 5 Hz - 480 kHz | 0.707 (fixed) |
| band_pass | 2.6 - 284 kHz | 0.01 - 5.5 |
| band_stop | 2.3 - 278 kHz | ~0.01 |
| rlc_parallel | 3.3 - 239 kHz | 0.12 - 10.8 |
| rlc_series | 2.1 - 265 kHz | 0.01 - 1.4 |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture details
- [USAGE.md](USAGE.md) - Detailed API reference
- [GENERATION_RESULTS.md](GENERATION_RESULTS.md) - Test results

## License

MIT License - see [LICENSE](LICENSE)
