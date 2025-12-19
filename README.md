# Graph Variational Autoencoder for Circuit Latent Space Discovery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Discover intrinsic circuit properties through learned latent representations**

This project implements a Graph Variational Autoencoder (GraphVAE) that learns to encode analog circuits into a structured 24-dimensional latent space, enabling circuit generation, latent space interpolation, and property discovery.

🎯 **Research Goal**: Discover intrinsic circuit properties through latent space learning—not about accuracy, about new representations.

## Quick Start

```bash
# Quick 2-epoch test
python3 scripts/train.py --config configs/test.yaml --epochs 2

# Full training (200 epochs)
python3 scripts/train.py --config configs/optimized.yaml

# Evaluate
python3 scripts/evaluate.py \
  --checkpoint experiments/exp002_optimized_2epochs/checkpoints/best.pt \
  --output-dir results/
```

## Project Structure

```
Z-GED/
├── configs/          # Training configurations (default, optimized, test)
├── ml/               # Core ML code (data, models, losses, training, utils)
├── scripts/          # Executable scripts (train.py, evaluate.py)
├── experiments/      # Training runs & results
├── docs/             # Documentation (status, phases, analysis, guides)
├── tests/            # Unit & integration tests
└── tools/            # Utility tools (GED, circuit generator)
```

## Current Results (2 Epochs)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Loss** | 2.78 | 🟢 53% improvement from baseline |
| **Cluster Purity** | 100% | 🟢 Perfect filter type separation |
| **Silhouette Score** | 0.62 | 🟢 Good latent space structure |

## Documentation

- **[Project Status](docs/PROJECT_STATUS.md)**: Current progress (5/7 phases complete)
- **[Latent Space](docs/analysis/LATENT_SPACE_ORGANIZATION.md)**: 24D hierarchical structure explained
- **[Optimization](docs/analysis/OPTIMIZATION_ANALYSIS.md)**: Loss analysis, 53% improvement
- **[Config Guide](configs/README.md)**: Configuration options
- **[Experiment Guide](experiments/README.md)**: Running experiments

## Architecture

**24D Hierarchical Latent Space**:
- 8D topology (filter type classification)
- 8D component values (R, L, C magnitudes)
- 8D poles/zeros (transfer function)

**Model**: 101,919 parameters (Encoder: 68,915, Decoder: 33,004)

## Dataset

- **Size**: 120 RLC circuits
- **Types**: 6 filter types
- **Features**: Graph, transfer function, frequency response
- **Splits**: 96 train / 12 val / 12 test

## Status

**Phase 5 Complete** ✅ | **Next**: Phase 6 (Circuit Generation)

Last Updated: December 19, 2025
