# Latent Space Analysis

**Model:** v5.0 (Latent-Only Decoder)
**Analysis Date:** 2025-01-24

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
| low_pass | -0.64 | -3.97 | 0.21 | 0.40 | 0.03 | 0.01 | 0.07 | -0.01 |
| high_pass | 0.50 | -3.75 | -0.91 | -1.14 | -0.02 | 0.00 | -0.02 | -0.01 |
| band_pass | 3.23 | 0.85 | 0.01 | -0.01 | -0.02 | 0.01 | -0.02 | 0.00 |
| band_stop | -3.03 | 1.91 | -0.01 | 0.01 | 0.04 | -0.03 | 0.06 | -0.02 |
| rlc_series | -1.28 | 3.24 | -0.02 | -0.08 | -0.02 | 0.01 | -0.02 | 0.00 |
| rlc_parallel | -1.80 | -3.52 | 0.63 | 1.11 | -0.02 | 0.01 | -0.02 | 0.00 |

### Variance by Dimension

| Dimension | Mean | Std | Range | Status |
|-----------|------|-----|-------|--------|
| z[0] | -0.50 | 1.99 | [-3.10, 3.27] | **Active** |
| z[1] | -0.87 | 2.96 | [-4.03, 3.30] | **Active** |
| z[2] | -0.02 | 0.46 | [-1.04, 0.78] | Moderate |
| z[3] | 0.05 | 0.68 | [-1.31, 1.40] | Moderate |
| z[4] | -0.00 | 0.03 | [-0.03, 0.05] | **Collapsed** |
| z[5] | -0.00 | 0.01 | [-0.03, 0.01] | **Collapsed** |
| z[6] | 0.01 | 0.04 | [-0.03, 0.08] | **Collapsed** |
| z[7] | -0.01 | 0.01 | [-0.02, 0.03] | **Collapsed** |

---

## Latent Space Interpretation

### z[0]: Filter Type Axis

```
z[0] < -2.5  →  band_stop (5-node notch filter)
z[0] ≈ -1.5  →  rlc_parallel / rlc_series
z[0] ≈ -0.5  →  low_pass (3-node RC)
z[0] ≈ +0.5  →  high_pass (3-node RC)
z[0] > +2.5  →  band_pass (4-node RLC)
```

### z[1]: Complexity Axis

```
z[1] < -3.0  →  Simple 3-node circuits (low_pass, high_pass, rlc_parallel)
z[1] ≈  0.0  →  4-node circuits (band_pass)
z[1] > +2.0  →  Complex 4-5 node circuits (band_stop, rlc_series)
```

### 2D Visualization

```
        z[1]
         ^
    +4   |     rlc_series
         |        *
    +2   |   band_stop
         |      *
     0   |           band_pass
         |              *
    -2   |
         |
    -4   |  low_pass    high_pass    rlc_parallel
         |     *           *             *
         +---------------------------------> z[0]
            -3    -1     0     +1    +3
```

---

## Known Issue: Transfer Function Dimensions Collapsed

### Observation

Dimensions z[4:8] (intended for transfer function encoding) have near-zero variance:
- **Expected:** Encode poles/zeros → different cutoff/Q should produce different z[4:8]
- **Actual:** z[4:8] ≈ 0 for all circuits regardless of transfer function

### Root Cause Analysis

```
Encoder:
  Branch 1: GNN → z[0:2] (topology)
  Branch 2: Edge features → z[2:4] (values)
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

1. Decoder learns to reconstruct topology using only z[0:4]
2. z[4:8] provides no useful gradient (nothing depends on it)
3. KL loss pushes all dimensions toward the prior N(0,1)
4. Without competing signal, z[4:8] collapses to near-zero

### Impact

- Transfer function information is **not encoded** in latent space
- Cannot interpolate between different frequency responses
- Cannot generate circuits with specific cutoff/Q by manipulating z[4:8]
- The 4D "transfer function" portion of latent space is wasted

---

## Proposed Solutions

### Option 1: Auxiliary Specification Predictor (Recommended)

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
- **Loss function:** `ml/losses/gumbel_softmax_loss.py`
- **This analysis:** `docs/LATENT_SPACE_ANALYSIS.md`
