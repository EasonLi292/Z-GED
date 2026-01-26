# Latent Space Analysis

**Model:** v5.1 (Node-Embedding Encoder)
**Analysis Date:** 2026-01-26

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

### Full 8D Centroids (v5.1 with node-embedding encoder)

| Filter Type | z[0] | z[1] | z[2] | z[3] | z[4] | z[5] | z[6] | z[7] |
|-------------|------|------|------|------|------|------|------|------|
| low_pass | +2.71 | +2.55 | -0.17 | -0.07 | +0.01 | +0.00 | +0.03 | +0.05 |
| high_pass | +2.60 | +2.38 | +0.08 | +1.99 | -0.01 | +0.00 | +0.00 | -0.01 |
| band_pass | -3.57 | +1.38 | +0.00 | +0.01 | -0.01 | -0.00 | -0.00 | -0.01 |
| band_stop | +0.85 | -3.10 | -0.06 | -1.41 | -0.01 | +0.00 | +0.01 | -0.00 |
| rlc_series | +0.71 | -3.20 | +0.06 | +0.99 | -0.01 | -0.01 | -0.00 | -0.01 |
| rlc_parallel | +2.69 | +2.14 | +0.11 | -1.94 | -0.01 | +0.00 | -0.00 | -0.01 |

### Variance by Dimension (v5.1)

| Dimension | Role | Mean | Std | Range | Status |
|-----------|------|------|-----|-------|--------|
| z[0] | Topology | +1.00 | 2.22 | [-3.61, 2.73] | **Active** |
| z[1] | Topology | +0.36 | 2.51 | [-3.23, 2.60] | **Active** |
| z[2] | Values | +0.00 | 0.10 | [-0.24, 0.18] | Collapsed |
| z[3] | Values | -0.07 | 1.34 | [-2.02, 2.03] | **Active** |
| z[4] | Transfer func | -0.00 | 0.01 | [-0.01, 0.01] | Collapsed |
| z[5] | Transfer func | -0.00 | 0.01 | [-0.04, 0.01] | Collapsed |
| z[6] | Transfer func | +0.01 | 0.01 | [-0.02, 0.03] | Collapsed |
| z[7] | Transfer func | +0.00 | 0.02 | [-0.03, 0.05] | Collapsed |

**Note:** The node-embedding encoder activates z[3] (std=1.34) which distinguishes component configurations (e.g., high_pass z[3]=+2.0 vs rlc_parallel z[3]=-1.9). z[4:8] remains collapsed as no loss directly uses transfer function information.

---

## Latent Space Interpretation

### z[0]: Filter Type Axis

```
z[0] < -3.0  →  band_pass (4-node RLC)
z[0] ≈ +0.7  →  band_stop / rlc_series (4-5 node)
z[0] > +2.5  →  3-node circuits (low_pass, high_pass, rlc_parallel)
```

### z[1]: Complexity Axis

```
z[1] > +2.0  →  Simple 3-node circuits (low_pass, high_pass, rlc_parallel)
z[1] ≈ +1.4  →  4-node circuits (band_pass)
z[1] < -3.0  →  Complex 4-5 node circuits (band_stop, rlc_series)
```

### z[3]: Component Configuration Axis

```
z[3] ≈ +2.0  →  high_pass (C on VIN-VOUT)
z[3] ≈ +1.0  →  rlc_series
z[3] ≈  0.0  →  low_pass / band_pass
z[3] ≈ -1.4  →  band_stop
z[3] ≈ -1.9  →  rlc_parallel
```

### 2D Visualization (z[0] vs z[1])

```
        z[1]
         ^
    +3   | low_pass  rlc_parallel  high_pass
         |    *          *            *
    +1   |                      band_pass
         |                         *
    -1   |
         |
    -3   |         rlc_series  band_stop
         |            *           *
         +---------------------------------> z[0]
            -4    -2     0     +1    +3
```

---

## Transfer Function Dimensions (z[4:8]) — Collapsed

### Current State (v5.1)

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

1. Decoder learns to reconstruct topology using z[0:4] (3 active dimensions)
2. z[4:8] provides no useful gradient (nothing depends on it)
3. KL loss pushes all dimensions toward the prior N(0,1)
4. Without competing signal, z[4:8] collapses to near-zero

### Impact

- Transfer function information is **not encoded** in latent space
- Cannot interpolate between different frequency responses
- Cannot generate circuits with specific cutoff/Q by manipulating z[4:8]
- The 4D "transfer function" portion of latent space is wasted

### What DID Improve (v5.0 → v5.1)

The node-embedding encoder activated z[3] (std=1.34 vs 0.68), giving the values branch a meaningful dimension that distinguishes component configurations. This is because GND/VIN/VOUT node embeddings from the GNN carry richer position-specific information than the previous edge encoders.

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
- **Loss function:** `ml/losses/gumbel_softmax_loss.py`
- **This analysis:** `docs/LATENT_SPACE_ANALYSIS.md`
