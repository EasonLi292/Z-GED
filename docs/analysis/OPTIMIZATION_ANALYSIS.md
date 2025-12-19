# Loss Optimization Analysis

## Current Loss Breakdown (Epoch 2)

```
Total Loss:      5.97
├─ Reconstruction: 2.65 (weight: 1.0) → contributes 2.65 (44%)
│  ├─ Topology:    1.73
│  └─ Edge:        0.90
├─ Transfer Func:  6.58 (weight: 0.5) → contributes 3.29 (55%)
│  ├─ Pole Chamfer: 4.85 (from eval)
│  └─ Zero Chamfer: 4.05 (from eval)
└─ KL Divergence:  0.65 (weight: 0.05) → contributes 0.03 (1%)
```

## Problem Identification

### 1. Transfer Function Loss is Too High (6.58)
**Root Causes**:
- Poles/zeros normalized in log-magnitude space, but Chamfer distance computed on normalized values
- Normalized magnitudes are in range ~[0.1, 10] (after exp(normalized_log_mag))
- Chamfer distance on these values yields large errors
- Only 2 poles and 2 zeros predicted (fixed), but circuits have variable numbers

### 2. Normalization Mismatch
**Current approach**:
```python
# Dataset normalization
log_mag = log(|pole|)
normalized_log_mag = (log_mag - mean) / std
normalized_mag = exp(normalized_log_mag)
pole_normalized = normalized_mag * exp(i*phase)
```

**Issue**: Chamfer distance on normalized poles is not meaningful
- Original poles: magnitude ~100 to 1e6 Hz
- After normalization: magnitude ~0.1 to 10
- Chamfer distance: computing L2 in normalized space

**Example**:
- Pole A: -1000 Hz → normalized to -0.5 magnitude
- Pole B: -2000 Hz → normalized to -1.0 magnitude
- Chamfer distance: |(-0.5) - (-1.0)| = 0.5 (seems small)
- But for high-frequency poles (1e6 Hz → mag 50), error is much larger

### 3. Loss Weight Imbalance
- TF loss contributes 55% of total loss
- Reconstruction contributes 44%
- KL contributes only 1% (too low for proper regularization)

### 4. Model Capacity Issues
- Decoder outputs exactly 2 poles and 2 zeros
- Many circuits have only 1 pole (first-order filters)
- Forcing model to predict 2 poles leads to spurious predictions
- Padding with zeros doesn't help - model doesn't learn when to output zero

## Proposed Optimizations

### Optimization 1: Fix Transfer Function Loss Computation ⭐ CRITICAL

**Problem**: Computing Chamfer distance in normalized space

**Solution**: Denormalize before computing loss
```python
# In transfer_function.py
def forward(self, pred_poles, pred_zeros, target_poles_list, target_zeros_list,
            normalization_stats=None):
    if normalization_stats is not None:
        # Denormalize predictions
        pred_poles = denormalize_poles(pred_poles, normalization_stats)
        pred_zeros = denormalize_zeros(pred_zeros, normalization_stats)

    # Now compute Chamfer distance in original scale
    # ...
```

**Expected Impact**: Reduce TF loss from ~6.5 to ~1-2

### Optimization 2: Adaptive Loss Weights

**Current**: Fixed weights (recon=1.0, tf=0.5, kl=0.05)

**Problem**:
- TF loss is large → dominates training
- KL loss is tiny → poor regularization

**Solution 1 - Manual rebalancing**:
```yaml
loss:
  recon_weight: 1.0
  tf_weight: 0.1      # Reduce from 0.5
  kl_weight: 0.1      # Increase from 0.05
```

**Solution 2 - Uncertainty weighting** (more principled):
```python
# Learn loss weights automatically
log_var_recon = nn.Parameter(torch.zeros(1))
log_var_tf = nn.Parameter(torch.zeros(1))
log_var_kl = nn.Parameter(torch.zeros(1))

# Weighted loss with uncertainty
loss = (
    exp(-log_var_recon) * loss_recon + log_var_recon +
    exp(-log_var_tf) * loss_tf + log_var_tf +
    exp(-log_var_kl) * loss_kl + log_var_kl
)
```

**Expected Impact**: Better balance, faster convergence

### Optimization 3: Improved Pole/Zero Prediction

**Problem**: Fixed 2 poles, 2 zeros output

**Solution 1 - Variable-length with attention** (complex):
```python
# Decoder outputs up to N poles/zeros with validity masks
poles = self.pole_decoder(z_pz)  # [B, N, 2]
pole_valid = self.pole_validity(z_pz)  # [B, N]

# Only compute loss on valid poles
for i in range(batch):
    valid_mask = pole_valid[i] > 0.5
    pred_poles_i = poles[i][valid_mask]
    chamfer_distance(pred_poles_i, target_poles[i])
```

**Solution 2 - Separate by filter type** (simpler):
```python
# Different output sizes for different filter types
if filter_type == low_pass:
    n_poles = 1
    n_zeros = 0
elif filter_type == band_pass:
    n_poles = 2
    n_zeros = 0
# ...
```

**Expected Impact**: More accurate pole/zero predictions

### Optimization 4: Curriculum Learning

**Start with easier objectives, gradually increase difficulty**

**Schedule**:
```
Epochs 0-20:   Focus on topology (recon=1.0, tf=0.01, kl=0.01)
Epochs 21-50:  Add edge features (recon=1.0, tf=0.05, kl=0.05)
Epochs 51-100: Add TF matching (recon=1.0, tf=0.2, kl=0.1)
Epochs 101+:   Full objectives (recon=1.0, tf=0.5, kl=0.2)
```

**Expected Impact**: Better convergence, higher final accuracy

### Optimization 5: Architecture Improvements

**Current bottlenecks**:
- GNN: 3 layers, 64 hidden dim
- Latent: 24D (might be too small)
- Decoder: Simple MLPs

**Proposals**:
```yaml
model:
  # Encoder improvements
  gnn_hidden_dim: 128        # Increase from 64
  gnn_num_layers: 4          # Increase from 3
  use_batch_norm: true       # Add normalization

  # Latent space
  latent_dim: 32             # Increase from 24 (10D + 10D + 12D)

  # Decoder improvements
  decoder_hidden_dim: 256    # Increase from 128
  decoder_num_layers: 3      # Add depth
```

**Expected Impact**: +5-10% accuracy

### Optimization 6: Learning Rate and Scheduler

**Current**: lr=1e-3, cosine annealing

**Problem**: Might be too high for fine-grained pole/zero prediction

**Solution**:
```yaml
training:
  learning_rate: 5.0e-4      # Reduce from 1e-3
  warmup_epochs: 10          # Add warmup
  scheduler: cosine_warmup   # Use warmup

  # Or use ReduceLROnPlateau
  scheduler: plateau
  patience: 10
  factor: 0.5
```

**Expected Impact**: More stable training, better final loss

### Optimization 7: Data Augmentation

**Add noise to component values during training**:
```python
# In dataset __getitem__
if self.training and self.augment:
    # Perturb edge features by ±10%
    noise = torch.randn_like(edge_attr) * 0.1
    edge_attr = edge_attr + noise
```

**Expected Impact**: Better generalization, +3-5% test accuracy

## Recommended Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. ✅ **Fix TF loss denormalization** - Will reduce loss by 50-70%
2. ✅ **Rebalance loss weights** - recon=1.0, tf=0.1, kl=0.1
3. ✅ **Reduce learning rate** - 5e-4 with warmup

### Phase 2: Architecture (Next)
4. **Increase model capacity** - 128 hidden, 32 latent
5. **Add batch normalization**
6. **Improve pole/zero decoder** - variable-length or filter-type-specific

### Phase 3: Training Strategy (Later)
7. **Curriculum learning** - progressive loss weights
8. **Data augmentation**
9. **Uncertainty-weighted losses**

## Expected Results After Phase 1

**Current** (2 epochs):
```
Total Loss:  5.97
Recon:       2.65
TF:          6.58
KL:          0.65
Topo Acc:    33%
```

**Expected** (2 epochs with fixes):
```
Total Loss:  2.5-3.0  (↓ 50%)
Recon:       2.5-2.7  (similar)
TF:          1.5-2.0  (↓ 70%)
KL:          0.5-0.8  (similar)
Topo Acc:    40-50%   (↑ 20%)
```

**Expected** (50 epochs with all Phase 1+2):
```
Total Loss:  1.0-1.5
Recon:       0.8-1.0
TF:          0.5-0.8
KL:          0.3-0.5
Topo Acc:    70-85%
Pole/Zero:   Chamfer < 1.0
```

## Implementation Files to Modify

1. **ml/data/dataset.py**: Add denormalization methods
2. **ml/losses/transfer_function.py**: Denormalize before Chamfer
3. **configs/base_config.yaml**: Update loss weights and learning rate
4. **configs/optimized_config.yaml**: NEW - Phase 1 optimizations
5. **ml/models/encoder.py**: (Phase 2) Increase capacity
6. **ml/models/decoder.py**: (Phase 2) Improve pole/zero prediction
