# Architectural Improvements to GraphVAE

**Date**: December 20, 2025
**Status**: Phase 2 - Model Enhancements

## Overview

This document details the architectural improvements implemented to enhance the GraphVAE's ability to learn circuit representations. These improvements address loss balancing, training stability, and feature richness.

## Implemented Improvements (4/6)

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

### 3. Canonical Edge Ordering per Filter Type ✅

**Motivation**: Original implementation paired the first `E_i` target edges with the first `E_i` template edges. If edge ordering differed between target and template, the edge MSE loss would get noisy due to permutation mismatches.

**Implementation**:
- **Location**: `ml/models/decoder.py` (CIRCUIT_TEMPLATES), `ml/losses/reconstruction.py`
- **Mechanism**: Sort edges canonically by (source, target) and reorder target edges to match template

**Template Updates**:
```python
# Before: arbitrary edge ordering
'band_pass': {
    'edges': [(0, 2), (1, 2), (2, 3), (3, 0), (2, 0), (2, 1), (0, 3)]
}

# After: canonical ordering (sorted by source, then target)
'band_pass': {
    'edges': [(0, 2), (0, 3), (1, 2), (2, 0), (2, 1), (2, 3), (3, 0)]
}
```

**Edge Reordering Function**:
```python
def _reorder_edges_to_template(self, edge_index, edge_attr, filter_type_idx):
    """
    Reorder edges to match canonical template ordering.

    Creates mapping: (source, target) → edge_features
    Then reorders according to template edge list.
    Pads with zeros if target has fewer edges than template.
    """
    filter_type = FILTER_TYPES[filter_type_idx]
    template_edges = CIRCUIT_TEMPLATES[filter_type]['edges']

    # Map edges
    edge_map = {(src, tgt): features for src, tgt, features in zip(...)}

    # Reorder to match template
    reordered = [edge_map.get(e, zero_padding) for e in template_edges]
    return torch.stack(reordered)
```

**Loss Function Update**:
- Added `edge_index` parameter to `TemplateAwareReconstructionLoss.forward()`
- For each graph, extract edge_index and call `_reorder_edges_to_template()`
- Compare reordered target edges to predicted edges

**Configuration**: No config changes needed (always enabled)

**Expected Benefits**:
- Reduce edge reconstruction loss noise (no permutation mismatch)
- Faster convergence on edge features
- More interpretable edge predictions (consistent ordering)

**Code Changes**:
- `ml/models/decoder.py`: Updated all 6 templates with canonical orderings
- `ml/losses/reconstruction.py`: Added `_reorder_edges_to_template()` method
- `ml/losses/composite.py`: Pass `edge_index` parameter through
- `ml/training/trainer.py`: Provide `batch['graph'].edge_index` to loss

---

### 4. Teacher-Forced Templates During Training ✅

**Motivation**: Decoder samples topology from Gumbel-Softmax, which may be incorrect early in training. This causes edge predictions to use the wrong template structure, making the value decoder learn on incorrect topologies.

**Implementation**:
- **Location**: `ml/models/decoder.py`, `ml/training/trainer.py`, `scripts/train.py`
- **Mechanism**: Pass ground-truth filter type to decoder; use it instead of sampling

**Decoder Update**:
```python
def forward(self, z, temperature=1.0, hard=False, gt_filter_type=None):
    topo_logits = self.topo_classifier(z_topo)

    # Teacher forcing: use ground-truth topology if provided
    if gt_filter_type is not None:
        topo_probs = gt_filter_type.float()  # Use GT
    elif self.training:
        topo_probs = F.gumbel_softmax(topo_logits, ...)  # Sample
    else:
        topo_probs = F.softmax(topo_logits, ...)  # Inference
```

**Trainer Update**:
```python
# In VAETrainer.__init__:
self.use_teacher_forcing = use_teacher_forcing

# In _forward_pass:
gt_filter_type = batch['filter_type'] if self.use_teacher_forcing else None
decoder_output = self.decoder(z, hard=False, gt_filter_type=gt_filter_type)
```

**Configuration**:
```yaml
training:
  use_teacher_forcing: true  # Enable teacher forcing
```

**Training Behavior**:
- **Training**: Decoder uses ground-truth filter type → correct template always used
- **Validation/Inference**: Decoder samples from Gumbel-Softmax → tests learned topology

**Expected Benefits**:
- Faster training (correct templates from epoch 0)
- Reduced edge loss variance (no wrong-template noise)
- Better gradient signal for `z_values` branch
- Value decoder learns on correct topologies

**Trade-offs**:
- Model becomes reliant on correct topology during training
- Must still learn to predict topology via `topo_logits` (topology loss still active)
- May need to disable teacher forcing later in training for end-to-end learning

**Code Changes**:
- `ml/models/decoder.py`: Added `gt_filter_type` parameter to `forward()`
- `ml/training/trainer.py`: Added `use_teacher_forcing` parameter
- `scripts/train.py`: Read `use_teacher_forcing` from config
- `configs/curriculum.yaml`: Enabled teacher forcing by default

---

## Pending Improvements (2/6 remaining)

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

Production config with all improvements enabled:
```yaml
loss:
  recon_weight: 1.0
  tf_weight: 0.01
  kl_weight: 0.1
  use_topo_curriculum: true
  topo_curriculum_warmup_epochs: 20
  topo_curriculum_initial_multiplier: 3.0

training:
  use_teacher_forcing: true
```

### Updated Configs

All configs now use `edge_feature_dim: 7`:
- `configs/default.yaml`
- `configs/optimized.yaml`
- `configs/test.yaml`
- `configs/curriculum.yaml`

---

## Testing

**Test 1: Richer Edge Typing** (1 epoch, test.yaml):
```bash
python3 scripts/train.py --config configs/test.yaml --epochs 1
```

Results:
- Model params: 108,871 (increased from 101,919, +6.8%)
- Training loss: 2.52 (comparable to 2.78 baseline)
- Edge reconstruction loss: 0.62 (working correctly)
- Topology accuracy: 16.67% after 1 epoch

**Test 2: Curriculum + Teacher Forcing** (1 epoch, curriculum.yaml):
```bash
python3 scripts/train.py --config configs/curriculum.yaml --epochs 1
```

Results:
- Training loss: 5.62 (higher due to 3x topology weight)
- Topology loss: 1.80 (heavily weighted)
- Edge loss: 0.12 (much lower, correct templates used)
- Topology accuracy: 16.67% after 1 epoch
- Curriculum weight: Started at 3x, will anneal to 1x over 20 epochs
- Teacher forcing: Active (GT filter types used)

---

## Future Work

### Short Term (Next Session)
1. ✅ ~~Implement canonical edge ordering~~ (Complete)
2. ✅ ~~Enable teacher-forced templates~~ (Complete)
3. Run full 200-epoch training with all improvements enabled
4. Compare results with baseline (exp002_optimized_2epochs)
5. Implement remaining improvements (GED loss, topology-conditioned decoder)

### Long Term
1. GED-aware loss with precomputed neighbor matrix
2. Topology-conditioned value decoder with FiLM
3. Variable-length pole/zero prediction (auto-regressive decoder)
4. β-VAE with per-branch KL weights for better disentanglement
5. Attention mechanism for edge features
6. Multi-task learning with circuit property prediction

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
| 1. Curriculum learning | ✅ Complete | `reconstruction.py`, `composite.py`, `train.py`, configs | ~80 |
| 2. Richer edge typing | ✅ Complete | `dataset.py`, all configs | ~30 |
| 3. Canonical ordering | ✅ Complete | `decoder.py`, `reconstruction.py`, `composite.py`, `trainer.py` | ~100 |
| 4. Teacher forcing | ✅ Complete | `decoder.py`, `trainer.py`, `train.py`, configs | ~50 |
| 5. GED-aware loss | ⏳ Pending | `train.py`, `precompute_ged.py`, configs | ~150 (est.) |
| 6. Topo-conditioned decoder | ⏳ Pending | `decoder.py` | ~50 (est.) |

**Total implemented**: 4/6 improvements (~260 LOC)
**Total pending**: 2/6 improvements (~200 LOC estimated)

**Key Achievements**:
- Curriculum learning reduces topology weight gradually (3x → 1x over 20 epochs)
- Richer edge features include binary masks (7D instead of 3D)
- Canonical edge ordering eliminates permutation noise in loss
- Teacher forcing uses correct templates from epoch 0

---

**Last Updated**: December 20, 2025
**Next Review**: After full 200-epoch training run with all 4 improvements enabled
