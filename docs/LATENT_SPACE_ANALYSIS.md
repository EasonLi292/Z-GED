# Latent Space Analysis

**Analysis Date:** 2026-02-05
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`

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
| low_pass | +2.45 | +1.02 | +1.14 | +3.27 | +0.04 | -0.09 | +0.25 | -0.53 |
| high_pass | +3.47 | +1.19 | -0.21 | +2.21 | +0.24 | -0.86 | +1.27 | -0.59 |
| band_pass | -3.03 | +0.40 | -1.49 | +2.01 | +0.26 | -0.87 | +1.30 | -0.65 |
| band_stop | +0.27 | -1.55 | +1.39 | -3.08 | +0.25 | -1.65 | +2.27 | -0.46 |
| rlc_series | -1.26 | -2.05 | +2.88 | -2.09 | +0.27 | -0.88 | +1.30 | -0.66 |
| rlc_parallel | +2.39 | +1.04 | +2.53 | +1.88 | +0.26 | -0.86 | +1.29 | -0.65 |

### Variance by Dimension

| Dimension | Role | Mean | Std | Range | Status |
|-----------|------|------|-----|-------|--------|
| z[0] | Topology | +0.71 | 2.30 | [-3.10, +3.52] | **Active** |
| z[1] | Topology | +0.01 | 1.31 | [-2.12, +1.21] | **Active** |
| z[2] | Values | +1.04 | 1.51 | [-1.57, +2.94] | **Active** |
| z[3] | Values | +0.70 | 2.39 | [-3.15, +3.30] | **Active** |
| z[4] | Transfer func | +0.22 | 0.08 | [+0.03, +0.35] | Weak |
| z[5] | Transfer func | -0.87 | 0.45 | [-1.84, -0.08] | Moderate |
| z[6] | Transfer func | +1.28 | 0.58 | [+0.24, +2.57] | Moderate |
| z[7] | Transfer func | -0.59 | 0.09 | [-0.90, -0.43] | Weak |

**Note:** z[0:4] are strongly active with std > 1.3. z[5] (std=0.45) and z[6] (std=0.58) show moderate variance — z[5] separates band_stop (-1.65) from low_pass (-0.09), and z[6] separates band_stop (+2.27) from low_pass (+0.25). z[4] (std=0.08) and z[7] (std=0.09) remain weak. The transfer-function dimensions are partially active, likely because the full 8D latent is used by the node count predictor, providing gradient signal through z[4:8].

---

## Latent Space Interpretation

### z[0]: Filter Type Separation Axis

```
z[0] ≈ -3.03  →  band_pass (4-node)
z[0] ≈ -1.26  →  rlc_series (5-node)
z[0] ≈ +0.27  →  band_stop (5-node)
z[0] ≈ +2.39  →  rlc_parallel (3-node)
z[0] ≈ +2.45  →  low_pass (3-node)
z[0] ≈ +3.47  →  high_pass (3-node)
```

### z[1]: 3-node vs Multi-node Axis

```
z[1] > +1.0   →  3-node circuits (low_pass, high_pass, rlc_parallel)
z[1] ≈ +0.40  →  band_pass (4-node)
z[1] < -1.5   →  5-node circuits (band_stop, rlc_series)
```

### z[2]: Component Configuration Axis

```
z[2] ≈ -1.49  →  band_pass (distributed LC)
z[2] ≈ -0.21  →  high_pass (C on VIN-VOUT)
z[2] ≈ +1.14  →  low_pass (C to ground)
z[2] ≈ +1.39  →  band_stop
z[2] ≈ +2.53  →  rlc_parallel (RCL to ground)
z[2] ≈ +2.88  →  rlc_series
```

### z[3]: Filter Family Axis

```
z[3] ≈ +3.27  →  low_pass
z[3] ≈ +2.21  →  high_pass
z[3] ≈ +2.01  →  band_pass
z[3] ≈ +1.88  →  rlc_parallel
z[3] ≈ -2.09  →  rlc_series
z[3] ≈ -3.08  →  band_stop
```

### 2D Visualization (z[0] vs z[1])

```
        z[1]
         ^
         |   high_pass   low_pass   rlc_parallel
    +1   |       *           *           *
         |                        band_pass
     0   |                            *
         |
    -1   |              band_stop
         |                  *
    -2   |   rlc_series
         |       *
         +--------------------------------------------------------------> z[0]
            -3        -2        -1         0        +1        +2        +3
```

---

## Transfer Function Dimensions (z[4:8]) — Partially Active

### Current State

Dimensions z[4:8] show mixed variance. z[5] (std=0.45) and z[6] (std=0.58) show moderate activity, while z[4] (std=0.08) and z[7] (std=0.09) remain weak. The node count predictor uses the full 8D latent, providing gradient signal that prevents complete collapse.

| Dimension | Std | Separation Example |
|-----------|-----|-------------------|
| z[4] | 0.08 | Weak — narrow range [+0.03, +0.35] |
| z[5] | 0.45 | band_stop (-1.65) vs low_pass (-0.09) |
| z[6] | 0.58 | band_stop (+2.27) vs low_pass (+0.25) |
| z[7] | 0.09 | Weak — narrow range [-0.90, -0.43] |

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

- z[5] and z[6] encode some filter-type information (especially band_stop separation) but not full transfer function details
- Cannot reliably interpolate between different frequency responses using z[4:8] alone
- z[5] (std=0.45) and z[6] (std=0.58) show the most promise for encoding additional structure
- z[4] and z[7] remain weak and underutilized (std < 0.1)

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
