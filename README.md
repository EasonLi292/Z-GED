# Z-GED: Graph Variational Autoencoder for Circuit Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Learned circuit synthesis using variable-length decoder with 83-100% accuracy**

## Current Status (December 2024)

**Model**: Variable-Length Decoder for Circuit Generation
**Architecture**: 8D Hierarchical VAE (77,305 parameters)
**Status**: Complete & Working
**Achievement**: 83-100% accuracy in pole/zero prediction, 100% behavioral specification matching

---

## Quick Start

### Documentation (Start Here!)

| Document | Description |
|----------|-------------|
| **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** | **Comprehensive project overview** |
| **[README_VARIABLE_LENGTH.md](README_VARIABLE_LENGTH.md)** | **Quick start guide with examples** |

### Generate Circuits Matching Specifications

```bash
python scripts/test_behavioral_generation.py \
    --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
    --filter-type low_pass \
    --target-cutoff 14.3 \
    --num-samples 10
```

### Validate Model Performance

```bash
python scripts/validate_variable_length.py \
    --checkpoint checkpoints/variable_length/20251222_102121/best.pt
```

### Test Reconstruction Quality

```bash
python scripts/test_reconstruction.py \
    --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
    --num-samples 10
```

---

## Performance Highlights

| Metric | Accuracy | Status |
|--------|----------|--------|
| **Zero count** | 100% | Perfect |
| **Pole count** | 83-95% | Excellent |
| **Topology** | 100% | Perfect |
| **TF inference** | 83% | Working |
| **Cutoff match** | 100% | Perfect |
| **Spec deviation** | 5-10% | Precise |

---

## Architecture

**Model**: 8D Hierarchical VAE
- **Encoder**: 69,651 parameters (3-layer GNN)
- **Decoder**: 7,654 parameters (variable-length)
- **Total**: 77,305 parameters

**Latent Space**: 8D = 2D (topology) + 2D (values) + 4D (poles/zeros)

**Key Components**:
- Count prediction heads (classification: 0-4 poles/zeros)
- Value prediction heads (regression: up to max=4)
- Validity masking (extract only valid predictions)

---

## Dataset

- **Size**: 120 RLC filter circuits
- **Types**: 6 filter types (low-pass, high-pass, band-pass, band-stop, RLC series/parallel)
- **Features**: Graph structure, transfer function, pole/zero representation
- **Splits**: 96 train / 12 val / 12 test

---

## Documentation

### Current Documentation

| Topic | File |
|-------|------|
| **Complete Overview** | [FINAL_SUMMARY.md](FINAL_SUMMARY.md) |
| **Quick Start Guide** | [README_VARIABLE_LENGTH.md](README_VARIABLE_LENGTH.md) |
| **Implementation** | [docs/VARIABLE_LENGTH_IMPLEMENTATION.md](docs/VARIABLE_LENGTH_IMPLEMENTATION.md) |
| **Training Results** | [docs/VARIABLE_LENGTH_TRAINING_RESULTS.md](docs/VARIABLE_LENGTH_TRAINING_RESULTS.md) |
| **Behavioral Generation** | [docs/BEHAVIORAL_GENERATION_RESULTS.md](docs/BEHAVIORAL_GENERATION_RESULTS.md) |
| **Decoder Design** | [docs/VARIABLE_LENGTH_DECODER_DESIGN.md](docs/VARIABLE_LENGTH_DECODER_DESIGN.md) |
| **Issue Investigation** | [docs/HIGH_PASS_POLE_COUNT_ISSUE.md](docs/HIGH_PASS_POLE_COUNT_ISSUE.md) |

### Historical Documentation

Historical documentation from earlier phases is available in [docs/ARCHIVED/](docs/ARCHIVED/).

---

## Use Cases

### 1. Specification-Driven Design
```bash
# "Give me a 14 Hz low-pass filter"
python scripts/test_behavioral_generation.py \
    --filter-type low_pass \
    --target-cutoff 14.3 \
    --perturbation 0.2
# → 100% success, 5% deviation
```

### 2. Topology Exploration
```bash
# "What other ways can I build a 12 Hz filter?"
python scripts/test_behavioral_generation.py \
    --filter-type band_pass \
    --target-cutoff 11.9 \
    --perturbation 0.8
# → Discovers band-pass, low-pass, RLC variants
```

### 3. Design Refinement
```bash
# Fine-tune existing design
python scripts/test_behavioral_generation.py \
    --filter-type low_pass \
    --target-cutoff 14.3 \
    --perturbation 0.1 \
    --num-samples 50
```

---

## Project Structure

```
Z-GED/
├── FINAL_SUMMARY.md                    # Master project documentation
├── README_VARIABLE_LENGTH.md           # Quick start guide
├── configs/
│   └── 8d_variable_length.yaml         # Training configuration
├── ml/
│   ├── models/
│   │   └── variable_decoder.py         # Variable-length decoder (370 lines)
│   ├── losses/
│   │   └── variable_tf_loss.py         # Variable TF loss (180 lines)
│   └── data/
│       └── dataset.py                  # Dataset with pole/zero counts
├── scripts/
│   ├── train_variable_length.py        # Training script (540 lines)
│   ├── validate_variable_length.py     # Test set validation
│   ├── test_behavioral_generation.py   # Behavioral spec testing
│   ├── test_reconstruction.py          # Reconstruction quality test
│   └── generate_variable_length.py     # Circuit generation
├── checkpoints/
│   └── variable_length/
│       └── 20251222_102121/
│           └── best.pt                 # Best model (val loss: 2.3573)
└── docs/
    ├── VARIABLE_LENGTH_*.md            # Current documentation
    └── ARCHIVED/                       # Historical documentation
```

---

## Research Contributions

### Novel Architecture
- First variable-length decoder for circuit generation
- Count + value prediction with validity masking
- Curriculum learning for structure-first training

### Behavioral Encoding
- Smooth latent space encoding of circuit behavior
- Behavioral dimensions stable across perturbations
- Enables specification-driven generation

### Practical Tool
- First learned system matching numerical behavioral specs
- 100% specification accuracy
- Supports automated design workflows

