# Latent Space Analysis

**Analysis Date:** 2026-01-27

---

## Latent Space Structure

The encoder produces an 8-dimensional hierarchical latent space:

```
z = [z_topology | z_values | z_transfer_function]
       z[0:2]      z[2:4]         z[4:8]
```

| Dimensions | Name | Intended Purpose |
|------------|------|------------------|
| z[0:2] | Topology | Graph structure, filter type, node count |
| z[2:4] | Values | Component value distributions |
| z[4:8] | Transfer Function | Poles/zeros characteristics |

---

## Filter Type Centroids

Computed from 360 circuits (60 per filter type):

### Full 8D Centroids

| Filter Type | z[0] | z[1] | z[2] | z[3] | z[4] | z[5] | z[6] | z[7] |
|-------------|------|------|------|------|------|------|------|------|
| low_pass | -3.69 | -2.27 | +0.88 | -1.46 | +0.01 | -0.01 | +0.02 | +0.03 |
| high_pass | -3.64 | -2.22 | -1.68 | -0.23 | +0.00 | +0.00 | -0.00 | +0.00 |
| band_pass | -1.54 | +4.13 | -0.06 | -0.11 | +0.00 | -0.00 | +0.00 | -0.00 |
| band_stop | +2.45 | -2.07 | -0.07 | -1.27 | -0.01 | -0.03 | +0.00 | -0.01 |
| rlc_series | +2.51 | -2.05 | +0.23 | +1.22 | +0.00 | -0.00 | +0.00 | -0.01 |
| rlc_parallel | -3.64 | -2.33 | +0.84 | +1.32 | -0.00 | +0.00 | +0.00 | +0.00 |

### Variance by Dimension

| Dimension | Role | Mean | Std | Range | Status |
|-----------|------|------|-----|-------|--------|
| z[0] | Topology | -1.26 | 2.75 | [-3.72, +2.54] | **Active** |
| z[1] | Topology | -1.13 | 2.36 | [-2.43, +4.19] | **Active** |
| z[2] | Values | +0.03 | 0.85 | [-1.71, +0.92] | **Active** |
| z[3] | Values | -0.09 | 1.08 | [-1.52, +1.38] | **Active** |
| z[4] | Transfer func | -0.00 | 0.01 | [-0.02, +0.01] | Collapsed |
| z[5] | Transfer func | -0.01 | 0.01 | [-0.03, +0.01] | Collapsed |
| z[6] | Transfer func | +0.00 | 0.01 | [-0.01, +0.02] | Collapsed |
| z[7] | Transfer func | +0.00 | 0.01 | [-0.04, +0.03] | Collapsed |

**Note:** Both values dimensions are now active: z[2] (std=0.85) distinguishes low_pass/rlc_parallel (+0.84/+0.88) from high_pass (-1.68), while z[3] (std=1.08) separates rlc_parallel/rlc_series (+1.32/+1.22) from low_pass (-1.46). z[4:8] remains collapsed since no loss directly uses transfer function information.

---

## Latent Space Interpretation

### z[0]: Complexity Axis

```
z[0] < -3.5  →  3-node circuits (low_pass, high_pass, rlc_parallel)
z[0] ≈ -1.5  →  4-node circuits (band_pass)
z[0] > +2.4  →  4-5 node circuits (band_stop, rlc_series)
```

### z[1]: Band-pass Axis

```
z[1] > +4.0  →  band_pass (4-node distributed LC)
z[1] ≈ -2.0  →  all other topologies
```

### z[2]: Component Configuration Axis

```
z[2] ≈ +0.9  →  low_pass / rlc_parallel (C or RCL to ground)
z[2] ≈ +0.2  →  rlc_series
z[2] ≈ -0.1  →  band_pass / band_stop
z[2] ≈ -1.7  →  high_pass (C on VIN-VOUT)
```

### z[3]: Component Type Axis

```
z[3] ≈ +1.3  →  rlc_parallel / rlc_series
z[3] ≈ -0.1  →  band_pass
z[3] ≈ -0.2  →  high_pass
z[3] ≈ -1.3  →  band_stop
z[3] ≈ -1.5  →  low_pass
```

### 2D Visualization (z[0] vs z[1])

```
        z[1]
         ^
    +4   |                band_pass
         |                    *
    +2   |
         |
     0   |
         |
    -2   | low_pass  high_pass  rlc_parallel    band_stop  rlc_series
         |    *          *          *               *          *
         +--------------------------------------------------------------> z[0]
            -4        -3                          +2        +3
```

---

## Transfer Function Dimensions (z[4:8]) — Collapsed

### Current State

Dimensions z[4:8] have near-zero variance (~0.01). The KL divergence loss pushes them to the prior N(0,1) since no decoder loss uses that information.

### Root Cause

```
Encoder:
  Branch 1: GNN mean+max pooling → z[0:2] (topology)
  Branch 2: GND/VIN/VOUT node embeddings → z[2:4] (values)
  Branch 3: DeepSets(poles, zeros) → z[4:8] (transfer function)

Decoder:
  z (all 8D) → topology prediction

Loss:
  ✓ Node type loss (cross-entropy)
  ✓ Edge-component loss (8-way classification)
  ✓ Connectivity loss
  ✓ KL divergence (pushes toward N(0,1))

  ✗ NO reconstruction loss for poles/zeros
  ✗ NO auxiliary task using z[4:8]
```

**What happens during training:**

1. Decoder learns to reconstruct topology using z[0:4] (4 active dimensions)
2. z[4:8] provides no useful gradient (nothing depends on it)
3. KL loss pushes all dimensions toward the prior N(0,1)
4. Without competing signal, z[4:8] collapses to near-zero

### Impact

- Transfer function information is **not encoded** in latent space
- Cannot interpolate between different frequency responses
- Cannot generate circuits with specific cutoff/Q by manipulating z[4:8]
- The 4D "transfer function" portion of latent space is wasted

---

## Proposed Solutions for z[4:8] Collapse

### Option 1: Auxiliary Specification Predictor

Add a small MLP that predicts (cutoff, Q) from z[4:8]:

```python
class SpecPredictor(nn.Module):
    def __init__(self, pz_latent_dim=4):
        self.mlp = nn.Sequential(
            nn.Linear(pz_latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [log_cutoff, Q]
        )

    def forward(self, z_pz):
        return self.mlp(z_pz)

# Add to loss:
loss_spec = F.mse_loss(spec_predictor(z[4:8]), target_specs)
```

**Pros:** Simple, directly encourages z[4:8] to encode cutoff/Q
**Cons:** Requires specification labels (already have them)

### Option 2: Poles/Zeros Reconstruction

Add a decoder that reconstructs poles/zeros from z[4:8]:

```python
class PZDecoder(nn.Module):
    def forward(self, z_pz):
        # Predict number of poles/zeros and their values
        return predicted_poles, predicted_zeros

# Add to loss:
loss_pz = pole_zero_reconstruction_loss(predicted, target)
```

**Pros:** Directly uses the information that was encoded
**Cons:** Variable-length output is tricky, need special loss

### Option 3: Per-Branch KL Weights

Use different KL weights for each latent branch:

```python
kl_topo = kl_divergence(mu[:, 0:2], logvar[:, 0:2])
kl_values = kl_divergence(mu[:, 2:4], logvar[:, 2:4])
kl_pz = kl_divergence(mu[:, 4:8], logvar[:, 4:8])

kl_loss = 0.01 * kl_topo + 0.01 * kl_values + 0.001 * kl_pz  # Lower weight for z[4:8]
```

**Pros:** Simple change
**Cons:** Doesn't guarantee z[4:8] encodes useful info, just allows more variance

### Option 4: Contrastive Loss

Push apart circuits with different transfer functions:

```python
# Circuits with similar cutoff/Q should have similar z[4:8]
# Circuits with different cutoff/Q should have different z[4:8]
loss_contrastive = contrastive_loss(z_pz, cutoff_q_labels)
```

**Pros:** Explicitly structures z[4:8] by transfer function similarity
**Cons:** More complex to implement, requires careful batch construction

---

## Recommended Fix

**Option 1 (Auxiliary Predictor)** is recommended because:

1. Simple to implement (small MLP + MSE loss)
2. We already have cutoff/Q labels in the dataset
3. Directly forces z[4:8] to encode transfer function info
4. Can verify success by checking prediction accuracy

### Implementation Plan

1. Add `SpecPredictor` module to `ml/models/`
2. Modify loss function to include `loss_spec = spec_weight * mse(pred_spec, target_spec)`
3. Retrain model with `spec_weight=1.0` (tune as needed)
4. Verify z[4:8] variance increases after training

---

## Files

- **Interpolation script:** `scripts/generation/interpolate_filter_types.py`
- **Encoder:** `ml/models/encoder.py`
- **Loss function:** `ml/losses/circuit_loss.py`
- **This analysis:** `docs/LATENT_SPACE_ANALYSIS.md`
