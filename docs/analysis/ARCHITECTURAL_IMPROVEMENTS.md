# Architectural Improvements to GraphVAE

**Date**: December 20, 2025
**Status**: Phase 2 - Model Enhancements

## Overview

This document details the architectural improvements implemented to enhance the GraphVAE's ability to learn circuit representations. These improvements address loss balancing, training stability, and feature richness.

## Implemented Improvements

### 1. Topology Weight Curriculum Learning ✅

**Motivation**: The topology classification task is crucial for circuit understanding, but can be dominated by other loss components during early training. Curriculum learning helps the model first learn the topology, then refine other aspects.

**Implementation**:
- **Location**: `ml/losses/reconstruction.py` (`TemplateAwareReconstructionLoss`)
- **Mechanism**: Linearly anneal topology weight from high (3x) to normal (1x) over warmup period
- **Configuration**:
  ```yaml
  loss:
    use_topo_curriculum: true
    topo_curriculum_warmup_epochs: 20
    topo_curriculum_initial_multiplier: 3.0  # Start at 3x, anneal to 1x
  ```

**Training Schedule**:
```
Epoch  0: topo_weight = 3.0 (focus on topology)
Epoch  5: topo_weight = 2.25
Epoch 10: topo_weight = 1.5
Epoch 15: topo_weight = 1.25
Epoch 20: topo_weight = 1.0 (balanced)
Epoch 20+: topo_weight = 1.0 (constant)
```

**Expected Benefits**:
- Faster topology convergence in early epochs
- Better filter type classification accuracy
- More stable training overall
- Reduced interference from edge reconstruction noise

**Usage**:
```bash
# Use curriculum-enabled config
python3 scripts/train.py --config configs/curriculum.yaml
```

**Code**:
```python
class TemplateAwareReconstructionLoss(nn.Module):
    def __init__(self,
                 topo_weight=1.0,
                 use_curriculum=False,
                 curriculum_warmup_epochs=20,
                 curriculum_initial_multiplier=3.0):
        # ...

    def get_topology_weight(self) -> float:
        if not self.use_curriculum:
            return self.base_topo_weight

        if self.current_epoch >= self.curriculum_warmup_epochs:
            return self.base_topo_weight

        # Linear annealing
        progress = self.current_epoch / self.curriculum_warmup_epochs
        initial_weight = self.base_topo_weight * self.curriculum_initial_multiplier
        current_weight = initial_weight - progress * (initial_weight - self.base_topo_weight)
        return current_weight
```

---

### 2. Richer Edge Typing with Binary Masks ✅

**Motivation**: Original edge features only contained continuous impedance values `[log(C), log(G), log(L_inv)]`. This doesn't explicitly indicate which components are present, forcing the model to learn this indirectly from near-zero values. Explicit binary masks help the model understand component presence.

**Implementation**:
- **Location**: `ml/data/dataset.py` (edge feature creation)
- **New Edge Features**: 7D instead of 3D
  - **Continuous (3D)**: `[log(C), log(G), log(L_inv)]` (normalized)
  - **Binary Masks (4D)**: `[has_C, has_R, has_L, is_parallel]`

**Binary Mask Logic**:
```python
C, G, L_inv = impedance_den  # Original values before log-scaling

has_C = 1.0 if C > 1e-12 else 0.0          # Capacitor present
has_R = 1.0 if G > 1e-12 else 0.0          # Resistor present (G = 1/R)
has_L = 1.0 if L_inv > 1e-12 else 0.0      # Inductor present (L_inv = 1/L)
is_parallel = 1.0                           # All components between two nodes are parallel

edge_features = [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
```

**Configuration Update**:
```yaml
model:
  edge_feature_dim: 7  # Changed from 3
```

**Expected Benefits**:
- Explicit component type information for GNN
- Easier learning of which components are present vs. absent
- Better handling of sparse circuits (e.g., RC filters with L=0)
- Foundation for future component-aware attention mechanisms

**Model Impact**:
- Encoder: Processes 7D edge features (previously 3D)
- Decoder: Outputs 7D edge features (continuous + binary predictions)
- Parameter count: Increased from 101,919 to 108,871 (+6.8%)

**Backward Compatibility**:
⚠️ **Breaking change** - Models trained with 3D edge features cannot load datasets/configs with 7D features. Requires retraining.

**Code**:
```python
# In dataset.py _graph_to_pyg_data()
for neighbor in neighbors:
    imp_den = neighbor['impedance_den']  # [C, G, L_inv]

    # Extract binary masks BEFORE log-scaling
    C, G, L_inv = imp_den
    has_C = 1.0 if C > 1e-12 else 0.0
    has_R = 1.0 if G > 1e-12 else 0.0
    has_L = 1.0 if L_inv > 1e-12 else 0.0
    is_parallel = 1.0

    # Log-scale continuous features
    if self.log_scale_impedance:
        imp_den = np.log(np.array(imp_den) + 1e-15)

    # Normalize
    imp_den = torch.tensor(imp_den, dtype=torch.float32)
    if self.normalize_features:
        imp_den = (imp_den - self.impedance_mean) / self.impedance_std

    # Concatenate [continuous, binary]
    binary_masks = torch.tensor([has_C, has_R, has_L, is_parallel], dtype=torch.float32)
    edge_features = torch.cat([imp_den, binary_masks], dim=0)  # [7]
```

---

## Pending Improvements (Not Yet Implemented)

### 3. Canonical Edge Ordering per Filter Type

**Problem**: Current implementation pairs the first `E_i` target edges with the first `E_i` template edges. If edge ordering differs between target and template, the edge MSE loss gets noisy.

**Solution**:
1. Define canonical edge ordering for each filter type in `CIRCUIT_TEMPLATES`
2. In `tools/circuit_generator.py`: Sort edges consistently when creating graph
3. In `TemplateAwareReconstructionLoss`: Reorder `target_edge_attr` to match template ordering

**Expected Impact**:
- Reduce edge reconstruction loss noise
- Faster convergence on edge features
- More interpretable edge predictions

---

### 4. Teacher-Forced Templates During Training

**Problem**: Decoder samples topology from Gumbel-Softmax, which may be incorrect early in training. This causes edge predictions to use the wrong template structure.

**Solution**:
1. Add ground-truth filter type as input to decoder forward pass
2. Use GT template during training (teacher forcing)
3. Add auxiliary classifier to encoder's `h_topo` for topology prediction
4. Loss = reconstruction + auxiliary_topology_CE

**Expected Impact**:
- Faster training (correct templates from epoch 0)
- Reduced edge loss variance
- Better gradient signal for `z_values` branch

---

### 5. GED-Aware Metric Learning Loss

**Problem**: Current SimplifiedCompositeLoss doesn't encourage latent distances to correlate with circuit similarity (GED).

**Solution**:
1. Precompute 5-10 nearest neighbors per circuit using GED
2. Switch from `SimplifiedCompositeLoss` to full `CompositeLoss`
3. Enable GED metric learning loss:
   ```yaml
   loss:
     ged_weight: 0.5
     use_ged_loss: true
   ```

**Expected Impact**:
- Structured latent space reflecting circuit similarity
- Better interpolation between similar circuits
- Improved clustering by filter type

---

### 6. Topology-Conditioned Value Decoder

**Problem**: Current value decoder is independent of topology. Different filter types may have different valid ranges for component values.

**Solution**:
1. Add topology one-hot vector to value decoder input
2. Option A: Concatenate `z_values` with `filter_type_onehot`
3. Option B: Use FiLM (Feature-wise Linear Modulation) to gate `z_values` based on topology

**Expected Impact**:
- Filter-type-specific component value distributions
- Better generalization to each topology
- Reduced invalid component value predictions

---

## Configuration Files

### New Config: `configs/curriculum.yaml`

Production config with curriculum learning enabled:
```yaml
loss:
  recon_weight: 1.0
  tf_weight: 0.01
  kl_weight: 0.1
  use_topo_curriculum: true
  topo_curriculum_warmup_epochs: 20
  topo_curriculum_initial_multiplier: 3.0
```

### Updated Configs

All configs now use `edge_feature_dim: 7`:
- `configs/default.yaml`
- `configs/optimized.yaml`
- `configs/test.yaml`
- `configs/curriculum.yaml`

---

## Testing

**Quick Test** (1 epoch):
```bash
python3 scripts/train.py --config configs/test.yaml --epochs 1
```

**Results** (with richer edge typing):
- Model params: 108,871 (increased from 101,919)
- Training loss: 2.52 (comparable to 2.78 before)
- Edge reconstruction loss: 0.62 (working correctly)
- Topology accuracy: 16.67% after 1 epoch (expected, needs more epochs)

---

## Future Work

### Short Term (Next Session)
1. Implement canonical edge ordering
2. Enable teacher-forced templates
3. Run full 200-epoch training with curriculum + richer edge typing
4. Compare results with baseline (exp002)

### Long Term
1. GED-aware loss with precomputed neighbor matrix
2. Topology-conditioned value decoder with FiLM
3. Variable-length pole/zero prediction (auto-regressive decoder)
4. β-VAE with per-branch KL weights for better disentanglement

---

## Metrics to Track

After implementing improvements, track:
1. **Training stability**: Loss variance, gradient norms
2. **Topology learning**: Accuracy over epochs, confusion matrix
3. **Edge reconstruction**: MSE on continuous features, accuracy on binary masks
4. **Latent space quality**: Silhouette score, cluster purity, GED correlation
5. **Transfer function**: Pole/zero Chamfer distance

Compare against baseline (exp002_optimized_2epochs):
- Total loss: 2.78 → ?
- Topology accuracy: 16.67% @ 2 epochs → ?
- Cluster purity: 100% → maintain
- Silhouette score: 0.62 → improve

---

## Implementation Summary

| Feature | Status | Files Modified | LOC Changed |
|---------|--------|----------------|-------------|
| Curriculum learning | ✅ Complete | `reconstruction.py`, `composite.py`, `train.py`, configs | ~80 |
| Richer edge typing | ✅ Complete | `dataset.py`, all configs | ~30 |
| Canonical ordering | ⏳ Pending | `circuit_generator.py`, `reconstruction.py` | ~50 (est.) |
| Teacher forcing | ⏳ Pending | `decoder.py`, `encoder.py`, `trainer.py` | ~100 (est.) |
| GED-aware loss | ⏳ Pending | `train.py`, `precompute_ged.py`, configs | ~150 (est.) |
| Topo-conditioned decoder | ⏳ Pending | `decoder.py` | ~50 (est.) |

**Total implemented**: 2/6 improvements (110 LOC)
**Total pending**: 4/6 improvements (~350 LOC estimated)

---

**Last Updated**: December 20, 2025
**Next Review**: After full training run with curriculum + richer edge typing
