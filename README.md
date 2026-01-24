# Z-GED: Specification-Driven Circuit Generation

Automated RLC filter circuit synthesis using a VAE with component-aware GNN encoder. Generate circuit topologies by specifying **cutoff frequency** and **Q-factor**, or by sampling/interpolating in an 8-dimensional latent space.

## What It Does

Z-GED generates RLC filter circuits that match target specifications:

```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
# Output: GND--RCL--VOUT, VIN--R--VOUT
```

**Supported filter types:** Low-pass, High-pass, Band-pass, Band-stop, RLC series, RLC parallel

## Performance

| Metric | Validation |
|--------|------------|
| Node Count | 100% |
| Edge Existence | 100% |
| Component Type | 100% |

**Dataset:** 360 circuits (60 per filter type)

## Quick Start

### Generate from Specifications

```bash
# Standard Butterworth filter at 10kHz
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707

# High-Q resonant at 5kHz
python scripts/generation/generate_from_specs.py --cutoff 5000 --q-factor 5.0

# Band-pass at 20kHz
python scripts/generation/generate_from_specs.py --cutoff 20000 --q-factor 2.0
```

### Example Results

| Cutoff | Q | Generated Circuit |
|--------|---|-------------------|
| 1 kHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` (high-pass) |
| 10 kHz | 0.707 | `GND--RCL--VOUT, VIN--R--VOUT` (RLC parallel) |
| 10 kHz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` (high-Q resonant) |
| 100 kHz | 0.707 | `GND--C--VOUT, VIN--R--VOUT` (low-pass) |

### Python API

```python
import torch
from ml.models.decoder import SimplifiedCircuitDecoder

decoder = SimplifiedCircuitDecoder(latent_dim=8, hidden_dim=256)
checkpoint = torch.load('checkpoints/production/best.pt')
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder.eval()

# Generate from latent code
z = torch.randn(1, 8)
circuit = decoder.generate(z)

print(f"Nodes: {circuit['node_types'].shape[1]}")
print(f"Edges: {int(circuit['edge_existence'].sum().item() // 2)}")
```

### Train the Model

```bash
python scripts/training/train.py
```

## Installation

```bash
pip install torch torch-geometric numpy scipy networkx pyyaml tqdm
```

## Project Structure

```
Z-GED/
├── ml/
│   ├── models/          # Encoder and decoder architectures
│   ├── losses/          # Training loss functions
│   ├── data/            # Dataset loading
│   └── utils/           # Utilities
├── scripts/
│   ├── generation/      # Circuit generation scripts
│   ├── training/        # Training and validation
│   └── testing/         # Test scripts
├── tools/               # Dataset generation utilities
├── checkpoints/         # Trained models
└── rlc_dataset/         # Training data (360 circuits)
```

## How It Works

1. **Encoder** - Component-aware GNN encodes circuits into 8D latent space
2. **Latent Space** - Hierarchical structure: `[topology(2D) | values(2D) | transfer_function(4D)]`
3. **Decoder** - Autoregressive transformer generates nodes and edges with component types

### Latent Space Structure

The 8D latent space is hierarchically organized:
- `z[0:2]` - Topology encoding (graph structure, filter type)
- `z[2:4]` - Component values encoding
- `z[4:8]` - Transfer function encoding (poles/zeros)

## Example Outputs

### By Filter Type

| Filter Type | Example Circuit |
|-------------|-----------------|
| low_pass | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT1, INT1--L--INT2` |
| rlc_series | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1` |
| rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` |

### Interpolation Example

Low-pass to High-pass transition:
```
alpha=0.00: GND--C--VOUT, VIN--R--VOUT  (low-pass)
alpha=0.25: GND--C--VOUT, VIN--R--VOUT
alpha=0.50: GND--R--VOUT, VIN--C--VOUT  (transition)
alpha=0.75: GND--R--VOUT, VIN--C--VOUT
alpha=1.00: GND--R--VOUT, VIN--C--VOUT  (high-pass)
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture details
- [USAGE.md](USAGE.md) - Detailed API reference
- [GENERATION_RESULTS.md](GENERATION_RESULTS.md) - Examples and results

## License

MIT License - see [LICENSE](LICENSE)
