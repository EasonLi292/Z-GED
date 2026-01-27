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
| low_pass | +0.65 | -3.23 | -2.89 | +1.63 | +0.32 | -0.20 | +0.57 | +0.10 |
| high_pass | +0.28 | -3.82 | -1.80 | -1.29 | -0.24 | -0.05 | +0.67 | +0.54 |
| band_pass | +3.22 | +1.45 | +2.62 | -2.16 | -0.46 | -0.03 | +0.54 | +0.63 |
| band_stop | -2.96 | +1.00 | +0.52 | +2.44 | -0.89 | +0.10 | +0.96 | +1.13 |
| rlc_series | -4.02 | +1.17 | +0.93 | +0.05 | -0.50 | -0.03 | +0.55 | +0.66 |
| rlc_parallel | +0.62 | -1.80 | -4.04 | -0.83 | -0.34 | -0.04 | +0.59 | +0.57 |

### Variance by Dimension

| Dimension | Role | Mean | Std | Range | Status |
|-----------|------|------|-----|-------|--------|
| z[0] | Topology | -0.37 | 2.43 | [-4.12, +3.29] | **Active** |
| z[1] | Topology | -0.87 | 2.17 | [-3.84, +1.48] | **Active** |
| z[2] | Values | -0.78 | 2.32 | [-4.18, +2.67] | **Active** |
| z[3] | Values | -0.03 | 1.62 | [-2.20, +2.50] | **Active** |
| z[4] | Transfer func | -0.35 | 0.40 | [-1.22, +0.41] | Weak |
| z[5] | Transfer func | -0.04 | 0.09 | [-0.21, +0.15] | Collapsed |
| z[6] | Transfer func | +0.65 | 0.17 | [+0.38, +1.00] | Weak |
| z[7] | Transfer func | +0.60 | 0.31 | [+0.07, +1.40] | Weak |

**Note:** z[0:4] are strongly active with std > 1.6. z[4], z[6], z[7] show weak but non-zero variance (std 0.17–0.40), partially encoding transfer function information — z[4] separates band_stop (-0.89) from low_pass (+0.32), and z[7] separates low_pass (+0.10) from band_stop (+1.13). z[5] remains effectively collapsed (std=0.09). The transfer function dimensions are no longer fully collapsed as in earlier models, likely because the full 8D latent is now used by the node count predictor, providing a gradient signal through z[4:8].

---

## Latent Space Interpretation

### z[0]: Filter Complexity Axis

```
z[0] ≈ -4.0  →  rlc_series (4-node)
z[0] ≈ -3.0  →  band_stop (5-node)
z[0] ≈ +0.3  →  high_pass (3-node)
z[0] ≈ +0.6  →  low_pass / rlc_parallel (3-node)
z[0] ≈ +3.2  →  band_pass (4-node)
```

### z[1]: 3-node vs Multi-node Axis

```
z[1] < -3.0  →  3-node circuits (low_pass, high_pass, rlc_parallel)
z[1] ≈ +1.0  →  4-5 node circuits (band_pass, band_stop, rlc_series)
```

### z[2]: Component Configuration Axis

```
z[2] ≈ -4.0  →  rlc_parallel (RCL to ground)
z[2] ≈ -2.9  →  low_pass (C to ground)
z[2] ≈ -1.8  →  high_pass (C on VIN-VOUT)
z[2] ≈ +0.5  →  band_stop
z[2] ≈ +0.9  →  rlc_series
z[2] ≈ +2.6  →  band_pass (distributed LC)
```

### z[3]: Component Type Axis

```
z[3] ≈ +2.4  →  band_stop
z[3] ≈ +1.6  →  low_pass
z[3] ≈ +0.1  →  rlc_series
z[3] ≈ -0.8  →  rlc_parallel
z[3] ≈ -1.3  →  high_pass
z[3] ≈ -2.2  →  band_pass
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

Dimensions z[4:8] show weak but non-zero variance (std 0.09–0.40), a significant improvement over earlier models where all four had std ≈ 0.01. The node count predictor now uses the full 8D latent, providing a gradient signal that prevents complete collapse.

| Dimension | Std | Separation Example |
|-----------|-----|-------------------|
| z[4] | 0.40 | band_stop (-0.89) vs low_pass (+0.32) |
| z[5] | 0.09 | Near-collapsed |
| z[6] | 0.17 | band_stop (+0.96) vs band_pass (+0.54) |
| z[7] | 0.31 | band_stop (+1.13) vs low_pass (+0.10) |

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
- z[5] remains effectively unused

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
