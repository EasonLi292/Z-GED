# Pole/Zero Prediction in Z-GED

## What Are Poles and Zeros?

Every RLC circuit has a transfer function H(s) = Vout(s)/Vin(s) that can be expressed in pole-zero form:

```
        K (s - z₁)(s - z₂)...
H(s) = ────────────────────────
        (s - p₁)(s - p₂)...
```

- **Poles** (pᵢ): Values of s where H(s) → ∞. Determine stability, resonance, and rolloff.
- **Zeros** (zᵢ): Values of s where H(s) = 0. Determine notch frequencies and high-frequency behavior.

For our circuits, poles and zeros are complex numbers: `p = σ + jω`
- **σ (sigma)**: Real part — controls damping (negative = stable)
- **ω (omega)**: Imaginary part — controls oscillation frequency

## Raw Poles/Zeros by Filter Type

| Filter Type | Poles | Zeros | Key Physics |
|---|---|---|---|
| **low_pass** | 1 real pole: `p = -1/(RC)` | None | RC time constant sets cutoff |
| **high_pass** | 1 real pole: `p = -1/(RC)` | 1 zero at `s = 0` | Capacitor blocks DC |
| **band_pass** | 2 complex conjugate: `p = -R/(2L) ± jω₀√(1-ζ²)` | 1 zero at `s = 0` | Series RLC resonance |
| **band_stop** | 2 complex conjugate: `p = -ω₀ζ ± jω₀√(1-ζ²)` | 2 imaginary: `z = ±jω₀` | Notch at ω₀ = 1/√(LC) |
| **rlc_series** | 2 complex conjugate: `p = -R_total/(2L) ± jω₀√(1-ζ²)` | 1 zero at `s = 0` | Series RLC with load |
| **rlc_parallel** | 2 complex conjugate (damped) | 1 zero at `s = 0` | Parallel RLC resonance |
| **lc_lowpass** | 2 purely imaginary: `p = ±jω₀` | None | Undamped, ω₀ = 1/√(LC) |
| **cl_highpass** | 2 purely imaginary: `p = ±jω₀` | 2 zeros at `s = 0` | Undamped, ω₀ = 1/√(LC) |

## What We Predict: 4D Target Vector

We compress the raw poles/zeros into a **4D target**: `[sigma_p, omega_p, sigma_z, omega_z]`

### Dominant Pole/Zero Selection

Circuits can have multiple poles/zeros. We pick the **dominant** one (closest to origin in s-plane) because it has the most influence on the frequency response:

1. **Filter conjugate pairs**: For conjugate pairs `σ ± jω`, keep only the positive-imaginary one
2. **Pick dominant**: Among remaining, choose the one with smallest `|p|` (closest to origin)
3. **Extract**: `sigma_p = Re(dominant_pole)`, `omega_p = |Im(dominant_pole)|`
4. **Same process for zeros**

### Signed-Log Normalization

Raw pole/zero values span many orders of magnitude (10¹ to 10⁶). We compress with:

```
signed_log(x) = sign(x) * log10(|x| + 1) / 7.0
```

The scale factor `PZ_LOG_SCALE = 7.0` maps typical values to roughly [-1, 1].

### Concrete Examples

| Filter Type | Raw Dominant Pole | → sigma_p | → omega_p |
|---|---|---|---|
| low_pass (R=1kΩ, C=1μF) | `-1000 + 0j` | `-0.429` | `0.0` |
| band_pass (R=100Ω, L=10mH, C=1μF) | `-5000 + j9987` | `-0.528` | `0.571` |
| lc_lowpass (L=10mH, C=1μF) | `0 + j10000` | `0.0` | `0.571` |

### What Each Dimension Encodes

| Dimension | Meaning | Range | Which types have nonzero? |
|---|---|---|---|
| **sigma_p** | Damping of dominant pole | Negative (stable) or 0 | All except lc_lowpass, cl_highpass (undamped) |
| **omega_p** | Resonant/natural frequency | Positive or 0 | All except low_pass, high_pass (real poles only) |
| **sigma_z** | Real part of dominant zero | **Always 0** across all circuits | None — all zeros are at origin or on imaginary axis |
| **omega_z** | Imaginary part of dominant zero | Positive or 0 | Only band_stop (imaginary zeros at ±jω₀) |

**Note**: sigma_z is effectively a wasted dimension — it is always 0 because our circuits only produce zeros at the origin (`s = 0`) or on the imaginary axis (`s = ±jω₀`).

## How We Predict: GNN-Based Encoder

### Architecture Overview

The prediction pipeline has two stages:

```
Circuit Graph → [GNN Encoder] → 8D latent z = [z_topo(2D) | z_structure(2D) | z_pz(4D)]
                                                                                ↑
                                                                    This is what predicts pz
```

### Stage 1: Impedance-Aware GNN (ImpedanceConv)

A 3-layer message-passing GNN processes the circuit graph. Each layer uses **component-type-specific MLPs** for message passing:

```
Input: Node features [4D one-hot: GND/VIN/VOUT/INTERNAL]
       Edge features [3D: log10(R), log10(C), log10(L)]  (0 = absent)

Message computation:
  msg_R = MLP_R(cat([x_neighbor, log10(R)]))   × is_R_present
  msg_C = MLP_C(cat([x_neighbor, log10(C)]))   × is_C_present
  msg_L = MLP_L(cat([x_neighbor, log10(L)]))   × is_L_present
  message = msg_R + msg_C + msg_L

Output: 64D node embeddings (per node)
```

Each MLP_R/C/L is a 2-layer network `(65→64, ReLU, 64→64)` that enables **nonlinear** interactions between node features and component values. This is critical for lc_lowpass and cl_highpass where `ω₀ = 1/√(LC)` — a nonlinear function of the component values.

### Stage 2: Pole/Zero Branch (Branch 3 of Hierarchical Encoder)

```
Inputs:
  - Global pooling (mean + max) over all node embeddings → 128D
  - Terminal node embeddings: h_GND + h_VIN + h_VOUT    → 192D (3 × 64D)
                                                         ─────
                                                          320D total

MLP: 320 → 256 → 128 → 64 (ReLU + Dropout 0.2 between layers)

Output: mu_pz [4D], logvar_pz [4D]  →  z_pz [4D] via reparameterization
```

### Supervision: How the Loss Works

The pz prediction is supervised by **directly regressing mu[:, 4:8] against the pz_target**:

```
loss_pz = MSE(mu[:, 4:8], pz_target)   ×   pz_weight (= 5.0)
```

This forces the last 4 dimensions of the encoder's mean to match `[sigma_p, omega_p, sigma_z, omega_z]`, giving the latent space physically meaningful structure.

### Why This Design

1. **GNN sees the full graph**: Poles/zeros are deterministic functions of the circuit topology and component values. The GNN has access to all of this information through message passing.

2. **Terminal embeddings capture signal path**: H(s) = Vout/Vin depends on the path from VIN to VOUT through GND. Including h_GND, h_VIN, h_VOUT gives the MLP direct access to position-specific component information.

3. **Nonlinear MLPs in message passing**: Component values affect poles/zeros nonlinearly (e.g., `ω₀ = 1/√(LC)`). The 2-layer MLPs in ImpedanceConv can learn these nonlinear relationships. With single linear layers, lc_lowpass omega_p R² was only 0.190; with MLPs it improved to 0.878.

## Current Performance (Val Set, 384 Circuits)

| Filter Type | R² sigma_p | R² omega_p | R² omega_z |
|---|---|---|---|
| low_pass | 0.963 | N/A | N/A |
| high_pass | 0.987 | N/A | N/A |
| band_pass | 0.962 | 0.878 | N/A |
| band_stop | 0.851 | 0.928 | 0.912 |
| rlc_series | 0.945 | 0.965 | N/A |
| rlc_parallel | 0.847 | 0.903 | N/A |
| lc_lowpass | N/A | 0.878 | N/A |
| cl_highpass | N/A | 0.997 | N/A |

"N/A" means zero variance for that dimension in the given filter type (nothing to predict).

Overall val MSE: 0.001165, val/train ratio: 1.04x (no overfitting).

## Generalization to Unseen Component Values

The val set tests the model on **the same 8 topologies** with **randomly sampled component values not seen during training**. This section analyzes how robustly the model handles new R, C, L values.

Script: `scripts/eval/eval_pz_component_generalization.py`

### Overall Result

The model generalizes very well to new component values:

- **Val MSE: 0.001165** vs Train MSE: 0.001121 — a ratio of only **1.04x**
- Near-perfect linear fits: sigma_p slope=0.981, omega_p slope=0.989, omega_z slope=1.003
- Median per-sample MAE: 0.009, 95th percentile: 0.024

### Predicted vs Actual (Val Set)

| Dimension | R² | Pearson r | Best-fit slope | Target range | Pred range |
|---|---|---|---|---|---|
| sigma_p | 0.991 | 0.997 | 0.981 | [-0.96, 0.00] | [-0.92, 0.04] |
| omega_p | 0.973 | 0.987 | 0.989 | [0.00, 0.91] | [-0.03, 0.90] |
| sigma_z | N/A | N/A | N/A | constant 0 | mean=-0.0002 |
| omega_z | 0.999 | 0.999 | 1.003 | [0.00, 0.90] | [-0.02, 0.92] |

Slopes near 1.0 and intercepts near 0 confirm the model is not systematically biased.

### Concrete Examples (Best / Median / Worst Per Type)

For each filter type, circuits are ranked by total prediction error. "Best" = lowest error, "worst" = highest.
Values shown in both signed-log scale (what the model outputs) and raw physical units (Hz or rad/s).

**low_pass** — 1 real pole, no zeros. Model predicts damping from RC.

| Case | Components | Raw Pole | sigma_p target | sigma_p pred | Raw target | Raw pred |
|---|---|---|---|---|---|---|
| best | R=470, C=4.3e-7 | -4928 | -0.5275 | -0.5147 | -4928 | -4008 |
| median | R=4419, C=4.0e-8 | -5601 | -0.5355 | -0.5030 | -5601 | -3316 |
| worst | R=28061, C=2.7e-8 | -1306 | -0.4452 | -0.3980 | -1306 | -610 |

The worst case (R=28kΩ) has a slow pole; the model underpredicts its magnitude but stays in the right ballpark. omega_p is correctly near zero for all (real poles).

**band_pass** — 2 poles (real or conjugate), 1 zero at origin. Model must predict both damping and resonant frequency.

| Case | Components | Raw Poles | Pred sigma_p | Pred omega_p | Actual sigma_p | Actual omega_p |
|---|---|---|---|---|---|---|
| best | R=7787, L=9.4e-4, C=6.6e-8 | -1936, -8.3M (real) | -0.4671 | -0.0004 | -0.4696 | 0.0 |
| median | R=1690, L=3.9e-4, C=2.9e-7 | -2038, -4.4M (real) | -0.4511 | 0.0027 | -0.4728 | 0.0 |
| worst | R=154, L=1.0e-3, C=1.1e-7 | -75367 ± 58355j | -0.6613 | 0.0177 | -0.6967 | 0.6809 |

The worst case has conjugate poles (oscillatory), meaning it has a nonzero omega_p. The model predicts sigma_p reasonably (-0.66 vs -0.70) but misses omega_p entirely (0.02 vs 0.68) — it predicted real poles when the circuit actually resonates. This is the hardest band_pass case.

**band_stop** — 2 conjugate poles + 2 imaginary zeros (notch). Model must predict the notch frequency omega_z.

| Case | Components | Raw Poles | Raw Zeros | sigma_p pred/actual | omega_p pred/actual | omega_z pred/actual |
|---|---|---|---|---|---|---|
| best | R=83549, R=2920, C=1.4e-7, L=7.3e-4 | -99441 (real, repeated) | ±99441j | -0.721 / -0.714 | -0.002 / 0.0 | 0.714 / 0.714 |
| median | R=40430, R=115, C=4.5e-8, L=3.3e-4 | -175159 ± 191398j | ±259449j | -0.723 / -0.749 | 0.742 / 0.755 | 0.800 / 0.773 |
| worst | R=34961, R=102, C=6.0e-7, L=1.4e-3 | -34579 (real, repeated) | ±34579j | -0.580 / -0.648 | 0.469 / 0.0 | 0.584 / 0.648 |

The best case nails the notch frequency (omega_z: 99441 vs 99152 in raw units). The worst case again involves repeated real poles where the model incorrectly predicts oscillation (omega_p=0.47 vs 0.0).

**rlc_parallel** — 2 conjugate poles, 1 zero at origin. Complex topology with parallel resonance.

| Case | Components | Raw Poles | sigma_p pred/actual | omega_p pred/actual | Raw sigma_p pred/actual | Raw omega_p pred/actual |
|---|---|---|---|---|---|---|
| best | R=1658, C=7.6e-7, L=3.4e-4, R=187 | -3939 ± 62086j | -0.512 / -0.514 | 0.682 / 0.685 | -3811 / -3939 | 59588 / 62086 |
| median | R=2931, C=6.1e-8, L=1.6e-3, R=154 | -56126 ± 83346j | -0.637 / -0.679 | 0.712 / 0.703 | -28966 / -56126 | 96924 / 83346 |
| worst | R=50617, C=1.4e-7, L=9.0e-3, R=142 | -25447 ± 12324j | -0.535 / -0.629 | 0.643 / 0.584 | -5531 / -25447 | 31801 / 12324 |

Best case is near-perfect (3811 vs 3939 damping, 59588 vs 62086 resonant freq). Worst case: damping is off by ~5x in raw units but only 0.094 in log scale.

**cl_highpass** — 2 purely imaginary poles (undamped). Model only needs to predict omega_p.

| Case | Components | Raw Poles | omega_p pred/actual | Raw omega_p pred/actual |
|---|---|---|---|---|
| best | L=9.7e-4, C=1.8e-7 | ±75874j | 0.6972 / 0.6972 | 75918 / 75874 |
| median | L=2.4e-3, C=9.1e-9 | ±213981j | 0.7649 / 0.7615 | 226014 / 213981 |
| worst | L=1.3e-4, C=2.0e-9 | ±1978806j | 0.8912 / 0.8995 | 1731757 / 1978806 |

Best case is essentially exact (75918 vs 75874). Even the worst case (highest frequency) is within 12% in raw units.

### Error Distribution by Filter Type

| Filter | N | Mean MAE | Median MAE | P95 MAE | Max MAE |
|---|---|---|---|---|---|
| cl_highpass | 48 | 0.005 | 0.004 | 0.012 | 0.013 |
| high_pass | 48 | 0.008 | 0.006 | 0.018 | 0.030 |
| lc_lowpass | 48 | 0.009 | 0.009 | 0.016 | 0.017 |
| low_pass | 48 | 0.010 | 0.010 | 0.015 | 0.018 |
| rlc_series | 48 | 0.013 | 0.009 | 0.026 | 0.128 |
| rlc_parallel | 48 | 0.015 | 0.014 | 0.027 | 0.041 |
| band_pass | 48 | 0.016 | 0.007 | 0.021 | 0.175 |
| band_stop | 48 | 0.020 | 0.015 | 0.041 | 0.150 |

Simpler topologies (3-node: cl_highpass, high_pass, lc_lowpass, low_pass) have lower errors. More complex topologies (4-5 node: band_pass, band_stop, rlc_*) have slightly higher errors but still strong R² values.

### Error vs Component Value Magnitude

Does the model struggle with extreme component values? Correlations between mean log10 component value and prediction error:

| Filter | Correlation | p-value | Interpretation |
|---|---|---|---|
| cl_highpass | r=-0.787 | <0.001 | Larger components (higher freq) are easier |
| lc_lowpass | r=-0.539 | <0.001 | Same pattern |
| high_pass | r=-0.593 | <0.001 | Same pattern |
| low_pass | r=0.287 | 0.048 | Weak: larger components slightly harder |
| rlc_series | r=-0.271 | 0.063 | Not significant |
| band_stop | r=0.182 | 0.215 | Not significant |
| band_pass | r=-0.061 | 0.681 | No correlation |
| rlc_parallel | r=-0.060 | 0.686 | No correlation |

For LC-only filters, smaller component values (lower frequencies, more extreme log10 values) are slightly harder to predict. For most types, there is no meaningful correlation — the model handles the full component range well.

### Error vs Target Magnitude

Prediction error scales modestly with target magnitude:

- **sigma_p**: r=0.371 (larger damping values are harder), mean |error| ranges from 0.017 to 0.030 across quintiles
- **omega_p**: r=0.074 (no meaningful correlation)
- **omega_z**: r=0.625 (but sample is dominated by zeros, so mostly reflects band_stop)

### Interpolation vs Extrapolation

Nearly all val circuits have component values within the training range (random uniform sampling covers the range densely with 192 training samples per type). Only a handful of extrapolation cases exist:

- **high_pass**: 1 extrapolation circuit, 4.0x higher MAE (single sample)
- **rlc_series**: 4 extrapolation circuits, 1.37x higher MAE
- **band_stop**: 9 extrapolation circuits, actually 0.66x *lower* MAE (not a real pattern with this sample size)

With 240 circuits per type sampled uniformly over log10 ranges, and an 80/20 split, the val set almost entirely falls within the convex hull of training component values. The model is primarily interpolating, not extrapolating.

### Summary

The encoder handles unseen component values nearly as well as training values (1.04x MSE ratio). This is expected: once the GNN learns the functional relationship between component values and poles/zeros for a given topology, interpolation to new values within the same range is straightforward. The ImpedanceConv MLPs provide sufficient nonlinear capacity to capture relationships like `ω₀ = 1/√(LC)`.

## Generalization to Unseen Topologies

### Motivation

The in-distribution results above test on val circuits that share the **same topologies** as training circuits — just different component values. Each of the 8 filter types has a single fixed topology with 240 randomized component values, and the train/val split stratifies so every type appears in both sets.

This means the in-distribution eval does NOT test whether the encoder can predict poles/zeros for **topologies it has never seen**. That's the fundamental question for generalization: can the GNN learn the physics of pole/zero computation well enough to transfer to novel circuit structures?

### Methodology: Leave-One-Topology-Out (LOTO) Cross-Validation

For each of 8 filter types (folds):

1. **Train split**: All circuits NOT of this type (7 types × 240 = 1680 circuits, 80% train / 20% val)
2. **Test split**: All 240 circuits OF this type (unseen topology)
3. Build fresh encoder + decoder (same architecture as production)
4. Train end-to-end with same loss function and hyperparameters
5. Evaluate encoder's `mu[:, 4:8]` vs `pz_target` on the held-out type
6. Also evaluate on an in-distribution validation subset (baseline)

Training config: Adam lr=1e-3, 80 epochs, batch size 32, early stopping (patience 20), same loss weights as production.

Script: `scripts/eval/eval_pz_unseen_topology.py`

### Results

| Held-out Type | Unseen MSE | In-Dist MSE | R² sigma_p | R² omega_p | R² omega_z |
|---|---|---|---|---|---|
| low_pass | 0.1143 | 0.0101 | 0.419 | N/A | N/A |
| high_pass | 0.0613 | 0.0033 | -0.501 | N/A | N/A |
| band_pass | 0.0180 | 0.0041 | -0.410 | 0.589 | N/A |
| band_stop | 0.2224 | 0.0028 | -15.025 | -0.615 | -114.263 |
| rlc_series | 0.0254 | 0.0029 | 0.544 | 0.533 | N/A |
| rlc_parallel | 0.1532 | 0.0038 | -2.806 | -94.527 | N/A |
| lc_lowpass | 0.0974 | 0.0108 | N/A | -59.758 | N/A |
| cl_highpass | 0.0243 | 0.0089 | N/A | -3.758 | N/A |
| **Average** | **0.0895** | **0.0058** | | | |

Unseen/In-distribution MSE ratio: **15.35x**

"N/A" = zero variance in that dimension for the held-out type (nothing to predict).

### Interpretation

The LOTO results reveal that **pole/zero prediction does not generalize well to unseen topologies**. Key findings:

1. **Large generalization gap**: Unseen topology MSE is 15x higher than in-distribution MSE on average. The model learns to predict poles/zeros well for known topologies but struggles to transfer this to novel circuit structures.

2. **Positive transfer for structurally similar types**: The best unseen results are for rlc_series (R²=0.54/0.53) and band_pass (R²=0.59 for omega_p). These share structural similarity with other training types (both are series RLC variants), so the GNN can partially leverage learned relationships.

3. **Negative R² is common**: Many unseen folds show negative R², meaning the model predictions are worse than simply predicting the mean. This is expected when the model encounters a topology with fundamentally different pole/zero relationships than anything in training.

4. **Worst cases**: band_stop (omega_z R²=-114) and rlc_parallel (omega_p R²=-95) show extreme negative R². These types have unique structural features (notch zeros, parallel resonance) not represented in the remaining 7 training types.

5. **This is not surprising**: Each filter type has a unique topology, and poles/zeros depend on the specific circuit structure. A GNN trained on 7 topologies has no basis for predicting the transfer function of a fundamentally different 8th topology without seeing at least some examples of it.

**Conclusion**: The current encoder effectively learns a per-topology regression model. True topology-agnostic pole/zero prediction would require either (a) a much larger diversity of training topologies, or (b) explicit physics-informed inductive biases (e.g., impedance-based nodal analysis layers).

Script: `scripts/eval/eval_pz_unseen_topology.py` | Raw results: `scripts/eval/loto_results.json`
