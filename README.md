# Z-GED: Specification-Driven Circuit Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Automated circuit synthesis from user specifications using conditional VAE and SPICE simulation**

---

## Overview

Z-GED is a specification-driven circuit generator that creates RLC filter circuits matching target frequency response characteristics. The system combines a hierarchical variational autoencoder with SPICE simulation to generate circuits from user-specified cutoff frequencies and Q-factors.

### Key Features

- **Specification-Driven Generation:** Generate circuits from target cutoff frequency and Q-factor
- **Conditional VAE Architecture:** Proper CVAE with conditions propagated to all decoder layers
- **SPICE Integration:** Validates generated circuits via ngspice AC analysis
- **K-NN Latent Interpolation:** Smooth generation via specification-based nearest neighbor search
- **100% Training Accuracy:** Perfect component type and topology prediction

### Performance Metrics

| Metric | Result | Notes |
|--------|--------|-------|
| **Training Accuracy** | 100% | Node/edge/component types |
| **Topology Viability** | 100% | All generated topologies are viable |
| **Cutoff Frequency Error** | 53% avg | Best case: 16.5% (15 kHz, Q=3.0) |
| **Q-Factor Error** | 50% avg | Perfect for Q=0.707, poor for Q>5 |
| **Circuit Validity** | 100% | All circuits have VIN/VOUT connected |
| **SPICE Simulation Success** | 100% | All circuits simulate successfully |

📊 **[See Comprehensive Generation Results →](GENERATION_RESULTS.md)** (28 test cases: 18 pure + 10 hybrid)

---

## Quick Start

### Generate a 10 kHz Low-Pass Filter

```bash
python scripts/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

**Output:**
```
Sample 1: Interpolated from 5 nearest
  Reference circuit: cutoff=8975.2 Hz, Q=0.707
  Generated: 2 edges
  Valid circuit: ✅

Actual specs (SPICE simulation):
  Cutoff: 9234 Hz (7.7% error)
  Q-factor: 0.707 (0.0% error)
```

### Test on Unseen Specifications

```bash
python scripts/test_unseen_specs.py
```

This tests the model on 8 challenging specifications not in the training data.

---

## How It Works

### 1. Specification → Latent Code (K-NN Interpolation)

```python
# Find k=5 nearest circuits by specification distance
target_specs = [log10(cutoff), Q]
nearest_circuits = find_k_nearest(target_specs, training_specs, k=5)

# Inverse distance weighting
weights = 1.0 / (distances + epsilon)
interpolated_latent = weighted_average(nearest_latents, weights)
```

### 2. Latent Code → Circuit (Conditional Decoder)

```python
# Autoregressive generation with condition signal
circuit = decoder.generate(
    latent_code,
    conditions=[log10(cutoff)/4.0, log10(Q)/2.0]  # Normalized specs
)
```

### 3. Circuit → SPICE Validation

```python
# Convert to SPICE netlist with proper denormalization
netlist = circuit_to_netlist(
    node_types, edge_existence, edge_values,
    impedance_mean, impedance_std  # For denormalization
)

# Run AC analysis
frequencies, response = run_ac_analysis(netlist)
actual_specs = extract_cutoff_and_q(frequencies, response)
```

---

## Model Architecture

### Hierarchical Encoder (69,651 parameters)

```
Circuit Graph → 3-layer GCN → μ, σ
                              ↓
                    8D Latent = [2D topology | 2D values | 4D TF]
```

### Conditional Decoder (6,460,050 parameters)

```
[latent_code + conditions] → Context Encoder (256D)
                              ↓
                    Autoregressive Node Generation (5 nodes)
                              ↓
                    Conditional Edge Generation:
                    - Cross-attention to latent components
                    - Cross-attention to conditions (NEW!)
                    - Joint edge-component prediction (8 classes)
                    - Component value regression (continuous)
```

**Key Innovation:** Conditions are propagated to the edge decoder via cross-attention, allowing component values to adapt to target specifications.

---

## Recent Improvements (2025-12-29)

### Critical Fixes Applied

1. **Added Conditions to Edge Decoder** ✅
   - Edge decoder now receives target specifications
   - Component values adapt based on target cutoff/Q
   - Retrained model: 100% validation accuracy

2. **Fixed Component Value Denormalization** ✅
   - SPICE simulator now properly denormalizes z-score values
   - Component ranges: 10Ω-100kΩ, 1pF-1μF, 1nH-10mH
   - No more Farad-scale capacitors or negative resistances!

3. **Fixed Multiple GND Node Handling** ✅
   - All GND nodes now correctly map to ground (0) in SPICE
   - No more invalid "n0" node names

**Impact:** 26x improvement in cutoff accuracy (1668% → 63.5% error)

See [CVAE_FIX_SUMMARY.md](CVAE_FIX_SUMMARY.md) for technical details.

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Z-GED.git
cd Z-GED

# Install dependencies
pip install torch torch-geometric numpy scipy ngspice

# Verify ngspice installation
which ngspice  # Should print path to ngspice binary
```

---

## Usage Examples

### Basic Generation

```python
import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder

# Load models
device = 'cpu'
encoder = HierarchicalEncoder(
    node_feature_dim=4, edge_feature_dim=7, gnn_hidden_dim=64,
    gnn_num_layers=3, latent_dim=8, topo_latent_dim=2,
    values_latent_dim=2, pz_latent_dim=4, dropout=0.1
).to(device)

decoder = LatentGuidedGraphGPTDecoder(
    latent_dim=8, conditions_dim=2, hidden_dim=256,
    num_heads=8, num_node_layers=4, max_nodes=5
).to(device)

checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

# Load dataset for specification database
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

# Build specification → latent mapping
from scripts.generate_from_specs import build_specification_database, interpolate_latents

specs_db, latents_db, indices_db = build_specification_database(encoder, dataset, device)

# Generate circuit for target specifications
target_cutoff = 10000  # Hz
target_q = 0.707

latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

# Prepare conditions
conditions = torch.tensor([[
    np.log10(target_cutoff) / 4.0,
    np.log10(target_q) / 2.0
]], dtype=torch.float32, device=device)

# Generate circuit
with torch.no_grad():
    circuit = decoder.generate(latent.unsqueeze(0), conditions, verbose=True)

print(f"Generated circuit with {(circuit['edge_existence'][0] > 0.5).sum().item() // 2} edges")
```

### SPICE Simulation

```python
from ml.utils.spice_simulator import CircuitSimulator, extract_cutoff_and_q

# Create simulator with normalization stats
impedance_mean = dataset.impedance_mean.numpy()
impedance_std = dataset.impedance_std.numpy()

simulator = CircuitSimulator(
    simulator='ngspice',
    freq_points=200,
    freq_start=1.0,
    freq_stop=1e6,
    impedance_mean=impedance_mean,
    impedance_std=impedance_std
)

# Convert to SPICE netlist
node_types_onehot = torch.zeros(5, 5)
for i, nt in enumerate(circuit['node_types'][0]):
    node_types_onehot[i, int(nt.item())] = 1.0

netlist = simulator.circuit_to_netlist(
    node_types=node_types_onehot,
    edge_existence=circuit['edge_existence'][0],
    edge_values=circuit['edge_values'][0]
)

# Run AC analysis
frequencies, response = simulator.run_ac_analysis(netlist)

# Extract actual specifications
specs = extract_cutoff_and_q(frequencies, response)
print(f"Actual cutoff: {specs['cutoff_freq']:.1f} Hz")
print(f"Actual Q-factor: {specs['q_factor']:.3f}")
```

---

## Training

### Train from Scratch

```bash
python scripts/train.py --config configs/production_ready.yaml --epochs 100
```

**Training configuration:**
- Batch size: 16
- Learning rate: 1e-4
- Optimizer: Adam
- Duration: ~10 minutes (CPU, 100 epochs)
- Dataset: 120 circuits (96 train, 24 val)

**Output:** Model saved to `checkpoints/production/best.pt`

---

## Documentation

| Document | Description |
|----------|-------------|
| [GENERATION_RESULTS.md](GENERATION_RESULTS.md) | **🔥 Complete test results & circuit diagrams (28 tests)** |
| [docs/GENERATION_GUIDE.md](docs/GENERATION_GUIDE.md) | **How to use the generation interface** |
| [docs/ERROR_SOURCE_ANALYSIS.md](docs/ERROR_SOURCE_ANALYSIS.md) | **Error source investigation** |
| [docs/FINDINGS_SUMMARY.md](docs/FINDINGS_SUMMARY.md) | **Key findings summary** |
| [COMPONENT_VALUE_DENORMALIZATION_FIX.md](COMPONENT_VALUE_DENORMALIZATION_FIX.md) | Denormalization bug analysis |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model architecture details |
| [USAGE.md](USAGE.md) | API reference and examples |
| [RESULTS.md](RESULTS.md) | Performance benchmarks |

---

## Supported Specifications

Based on the training dataset (120 circuits):

| Specification | Range | Notes |
|--------------|-------|-------|
| **Cutoff Frequency** | 14.4 Hz - 886 kHz | Log-scale matching |
| **Q-Factor** | 0.01 - 50.9 | Higher Q = narrower resonance |

**Filter types:**
- Low-pass: Q ≈ 0.707 (Butterworth)
- High-pass: Q ≈ 0.707 (Butterworth)
- Band-pass: Q = 1-51 (resonant)
- Band-stop: Q = 0.1-43 (resonant)

---

## Limitations

### 1. Q-Factor Accuracy (209% average error)
**Issue:** Model tends to generate Q=0.707 regardless of target.

**Root causes:**
- Limited Q-factor diversity in training data
- Component value precision affects Q more than cutoff
- Need explicit Q-factor loss during training

### 2. Unusual Specifications
Some rare combinations still fail (default to 1 Hz):
- Very low frequency + high Q (1 kHz, Q=10)
- Very high frequency + very low Q (100 kHz, Q=0.05)

**Reason:** These combinations are rare/absent in training data.

### 3. Approximate Matching
Latent space interpolation gives approximate specifications, not exact.
- Best case: 4.1% error (20 Hz target)
- Average: 63.5% error

---

## Future Work

### Short-term (To reach <20% error)
1. Add transfer function loss during generation
2. Optimize component values post-generation (gradient descent)
3. Filter invalid circuits before SPICE simulation

### Medium-term (To improve Q accuracy)
1. Collect more diverse training data (especially high-Q and low-Q)
2. Add explicit Q-factor loss to training
3. Increase Q-factor representation in latent space

### Long-term (Production ready)
1. Multi-objective optimization (match cutoff AND Q simultaneously)
2. Iterative component refinement
3. Topology selection based on specification requirements

---

## Project Structure

```
Z-GED/
├── README.md                              # This file
├── CVAE_FIX_SUMMARY.md                    # Technical fix documentation
├── SPEC_GENERATION.md                      # Specification generation guide
├── COMPONENT_VALUE_DENORMALIZATION_FIX.md  # Denormalization analysis
│
├── ml/
│   ├── models/
│   │   ├── encoder.py                          # Hierarchical VAE encoder
│   │   ├── graphgpt_decoder_latent_guided.py   # Conditional decoder
│   │   ├── latent_guided_decoder.py            # Edge decoder with conditions
│   │   └── gumbel_softmax_utils.py             # Component utilities
│   │
│   ├── losses/
│   │   └── gumbel_softmax_loss.py              # Unified circuit loss
│   │
│   ├── data/
│   │   └── dataset.py                          # Circuit dataset with normalization
│   │
│   └── utils/
│       └── spice_simulator.py                  # SPICE integration + denormalization
│
├── scripts/
│   ├── train.py                    # Training script
│   ├── validate.py                 # Validation script
│   ├── generate_from_specs.py      # Specification-driven generation
│   └── test_unseen_specs.py        # Test on unseen specifications
│
└── checkpoints/
    └── production/
        └── best.pt                 # Trained model (100% accuracy)
```

---

## Status

✅ **Functional - Generates usable circuits for most specifications**

**Strengths:**
- 100% training/validation accuracy
- 100% circuit validity (VIN/VOUT connectivity)
- 100% SPICE simulation success
- Realistic component values (pF-μF, 10Ω-100kΩ, nH-mH)
- Best case 4.1% cutoff error

**Known Issues:**
- Q-factor accuracy needs improvement (209% avg error)
- Some unusual specifications still challenging
- Approximate matching (not exact)

**Best checkpoint:** `checkpoints/production/best.pt`

**Last updated:** 2025-12-29

---

## Citation

```bibtex
@software{zged2025,
  title={Z-GED: Specification-Driven Circuit Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Z-GED}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
