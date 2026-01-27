# Configuration Files

This directory contains YAML configuration files for training the GraphVAE model.

## Available Configurations

### production.yaml
**Purpose**: Production training configuration for the latent-guided decoder
**Use case**: Full training runs with encoder + decoder
**Characteristics**:
- 8D latent space (2 topology + 2 values + 4 transfer function)
- Two-phase training (100 epochs decoder-only + 100 epochs joint)
- Connectivity loss enabled
- KL weight: 0.01

**When to use**:
- Full production training
- Retraining after architecture changes

---

### optimized_8d.yaml
**Purpose**: Optimized 8D latent space configuration
**Use case**: Best performing configuration after latent space analysis
**Characteristics**:
- 8D latent: 2D topology + 2D values + 4D transfer function
- Cosine annealing scheduler
- Curriculum learning for topology weight
- Early stopping patience: 30 epochs
- Optimized loss weights (recon=1.0, tf=0.01, kl=0.1)

**When to use**:
- Standard training (most cases)
- For best results with the 8D architecture

---

### test.yaml
**Purpose**: Quick 2-epoch test for pipeline verification
**Use case**: Testing code changes, debugging
**Characteristics**:
- Only 2 epochs
- No scheduler (faster iteration)
- 24D latent space (legacy)
- More frequent logging

**When to use**:
- After code changes (verify no errors)
- Quick sanity checks
- Debugging training issues

---

## Notes

The production training script (`scripts/training/train.py`) uses **hardcoded defaults** rather than these config files. These configs are available for experimental training variants and reference.

Production defaults (from `train.py`):
- Encoder: 8D latent, 64-dim GNN, 3 layers
- Decoder: 256 hidden, 8 heads, 4 layers, max 10 nodes
- Loss: node=1.0, count=5.0, edge-component=2.0, connectivity=5.0, KL=0.01
- Training: 100 epochs, batch=16, lr=1e-4, Adam
