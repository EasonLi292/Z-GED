# Training Experiments History

This document summarizes key training experiments that led to the current production model.

---

## Early Loss Weight Optimization (December 2024)

### Experiment 1: Baseline Configuration
**Date:** December 17, 2024
**Duration:** 2 epochs
**Configuration:**
- Reconstruction loss weight: 1.0
- Transfer Function loss weight: 0.5
- KL divergence weight: 0.05

**Results:**
- Validation loss: 5.97
- Topology accuracy: 33.33%
- **Issue:** TF loss dominated (55% of total), causing imbalanced training

---

### Experiment 2: Optimized Loss Weights
**Date:** December 19, 2024
**Duration:** 2 epochs
**Configuration:**
- Reconstruction loss weight: 1.0
- Transfer Function loss weight: 0.01 (reduced 50×)
- KL divergence weight: 0.1 (increased 2×)

**Results:**
- Validation loss: 2.78 (**53% improvement**)
- TF loss contribution: 2% of total (down from 55%)
- Better regularization from increased KL weight

**Evaluation Metrics:**
- Silhouette Score: 0.62
- Cluster Purity: 100%
- Pole Chamfer: 4.85
- Zero Chamfer: 4.05

**Key Finding:** Dramatically reduced TF loss weight enabled balanced training

---

## Production Model (Current)

**Configuration:** `configs/production.yaml`
**Architecture:** 8D latent space (2D topology + 2D values + 4D TF)
**Training:** 100 epochs, batch_size=4, lr=1e-4

**Results:**
- Validation accuracy: **100%** on all metrics
  - Component type accuracy: 100%
  - Edge count match: 100%
  - VIN/VOUT connectivity: 100%
- Best validation loss: 0.1643 (epoch 95)
- Model size: 77,305 parameters

**Key Innovations:**
1. Joint edge-component prediction (8-way classification)
2. Latent-guided decoder with cross-attention
3. Gumbel-Softmax for differentiable sampling
4. Strong connectivity loss (weight=5.0)

---

## Lessons Learned

1. **Loss Weight Balancing is Critical**
   - TF loss was too dominant initially (0.5 → 0.01)
   - Increased KL weight improved regularization (0.05 → 0.1)

2. **Joint Prediction Superior to Separate Heads**
   - Unified edge-component classification eliminated coordination issues
   - Achieved 100% accuracy vs. 33% with separate predictions

3. **Hierarchical Latent Space Works Well**
   - 8D total: 2D topology + 2D values + 4D TF
   - Enables semantic control and interpolation

4. **Small Model, High Accuracy**
   - 77K parameters sufficient for 120-circuit dataset
   - Params/sample ratio: ~804 (well-balanced)

---

**Current Status:** Production model achieves 100% topology accuracy. See [GENERATION_RESULTS.md](../GENERATION_RESULTS.md) for comprehensive generation test results.
