# Latent Space Analysis

**Analysis Date:** 2026-01-28

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
| low_pass | +1.26 | -2.83 | -3.23 | +1.70 | +0.34 | -0.00 | +0.43 | +0.02 |
| high_pass | +0.89 | -4.19 | -1.61 | -0.64 | -0.09 | +0.22 | +0.66 | +0.41 |
| band_pass | +2.77 | +1.61 | +2.63 | -2.23 | -0.27 | +0.23 | +0.58 | +0.49 |
| band_stop | -2.87 | +0.55 | +0.17 | +2.45 | -0.68 | +0.54 | +1.25 | +1.09 |
| rlc_series | -4.29 | +0.82 | +0.51 | -0.21 | -0.30 | +0.25 | +0.58 | +0.51 |
| rlc_parallel | +1.30 | -2.26 | -3.50 | -0.76 | -0.18 | +0.21 | +0.60 | +0.44 |

### Variance by Dimension

| Dimension | Role | Mean | Std | Range | Status |
|-----------|------|------|-----|-------|--------|
| z[0] | Topology | -0.16 | 2.53 | [-4.39, +2.81] | **Active** |
| z[1] | Topology | -1.05 | 2.15 | [-4.25, +1.66] | **Active** |
| z[2] | Values | -0.84 | 2.17 | [-3.59, +2.67] | **Active** |
| z[3] | Values | 0.05 | 1.57 | [-2.32, +2.49] | **Active** |
| z[4] | Transfer func | -0.19 | 0.33 | [-0.94, +0.41] | Weak |
| z[5] | Transfer func | 0.24 | 0.16 | [-0.01, +0.66] | Weak |
| z[6] | Transfer func | 0.68 | 0.27 | [+0.31, +1.33] | Weak |
| z[7] | Transfer func | 0.49 | 0.32 | [-0.00, +1.36] | Weak |

**Note:** z[0:4] are strongly active with std > 1.5. z[4], z[6], z[7] show weak but non-zero variance (std 0.27–0.33), and z[5] is also non-zero (std 0.16). z[4] separates band_stop (-0.68) from low_pass (+0.34), and z[7] separates band_stop (+1.09) from low_pass (+0.02). The transfer-function dimensions are no longer collapsed, likely because the full 8D latent is used by the node count predictor, providing gradient signal through z[4:8].

---

## Latent Space Interpretation

### z[0]: Filter Complexity Axis

```
z[0] ≈ -4.29  →  rlc_series (4-node)
z[0] ≈ -2.87  →  band_stop (5-node)
z[0] ≈ +0.89  →  high_pass (3-node)
z[0] ≈ +1.26  →  low_pass (3-node)
z[0] ≈ +1.30  →  rlc_parallel (3-node)
z[0] ≈ +2.77  →  band_pass (4-node)
```

### z[1]: 3-node vs Multi-node Axis

```
z[1] < -2.83  →  3-node circuits (low_pass, high_pass, rlc_parallel)
z[1] ≈ +0.55  →  4-5 node circuits (band_pass, band_stop, rlc_series)
```

### z[2]: Component Configuration Axis

```
z[2] ≈ -3.50  →  rlc_parallel (RCL to ground)
z[2] ≈ -3.23  →  low_pass (C to ground)
z[2] ≈ -1.61  →  high_pass (C on VIN-VOUT)
z[2] ≈ +0.17  →  band_stop
z[2] ≈ +0.51  →  rlc_series
z[2] ≈ +2.63  →  band_pass (distributed LC)
```

### z[3]: Component Type Axis

```
z[3] ≈ +2.45  →  band_stop
z[3] ≈ +1.70  →  low_pass
z[3] ≈ -0.21  →  rlc_series
z[3] ≈ -0.76  →  rlc_parallel
z[3] ≈ -0.64  →  high_pass
z[3] ≈ -2.23  →  band_pass
```

### 2D Visualization (z[0] vs z[1])

```
        z[1]
         ^
    +2   |       band_stop   rlc_series   band_pass
         |          *            *            *
    +1   |
         |
     0   |
         |
    -1   |                              rlc_parallel
         |                                   *
    -2   |
         |
    -3   |  low_pass   high_pass
         |     *           *
    -4   |
         +--------------------------------------------------------------> z[0]
            -4        -3        -2        -1         0        +1        +2        +3
```

---

## Transfer Function Dimensions (z[4:8]) — Partially Active

### Current State

Dimensions z[4:8] show weak but non-zero variance (std 0.16–0.33), a significant improvement over earlier models where all four had std ≈ 0.01. The node count predictor now uses the full 8D latent, providing a gradient signal that prevents complete collapse.

| Dimension | Std | Separation Example |
|-----------|-----|-------------------|
| z[4] | 0.33 | band_stop (-0.68) vs low_pass (+0.34) |
| z[5] | 0.16 | Near-collapsed |
| z[6] | 0.27 | band_stop (+1.25) vs low_pass (+0.43) |
| z[7] | 0.32 | band_stop (+1.09) vs low_pass (+0.02) |

### Root Cause of Remaining Weakness

```
Encoder:
  Branch 1: GNN mean+max pooling → z[0:2] (topology)
  Branch 2: GND/VIN/VOUT node embeddings → z[2:4] (values)
  Branch 3: DeepSets(poles, zeros) → z[4:8] (transfer function)

Decoder:
  z (all 8D) → node count prediction (provides gradient to z[4:8])
  z (all 8D) → context encoder → node/edge generation

Loss:
  ✓ Node type loss (cross-entropy)
  ✓ Edge-component loss (8-way classification)
  ✓ Connectivity loss
  ✓ KL divergence (pushes toward N(0,1))
  ~ Node count predictor uses full 8D (weak gradient to z[4:8])

  ✗ NO reconstruction loss for poles/zeros
  ✗ NO auxiliary task specifically targeting z[4:8]
```

### Impact

- z[4:8] encodes some filter-type information but not transfer function details
- Cannot reliably interpolate between different frequency responses using z[4:8] alone
- z[4] and z[7] show the most promise for encoding additional structure
- z[5] remains weak and underutilized

---

## Proposed Solutions for z[4:8] Activation

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
