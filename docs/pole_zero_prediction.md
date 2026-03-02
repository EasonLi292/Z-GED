# Pole/Zero Prediction in Z-GED

## Background

Every RLC circuit has a transfer function H(s) = Vout/Vin that can be written in pole-zero form:

```
        K (s - z₁)(s - z₂)...
H(s) = ────────────────────────
        (s - p₁)(s - p₂)...
```

**Poles** (pᵢ) are where H(s) blows up — they control stability, resonance, and rolloff. **Zeros** (zᵢ) are where H(s) vanishes — they control notch frequencies and high-frequency behavior.

Poles and zeros are complex numbers `p = σ + jω`:
- **σ (sigma)**: damping (negative = stable)
- **ω (omega)**: oscillation frequency

### Poles/Zeros by Filter Type

| Filter Type | Poles | Zeros |
|---|---|---|
| low_pass | 1 real: `p = -1/(RC)` | None |
| high_pass | 1 real: `p = -1/(RC)` | 1 at origin |
| band_pass | 2 conjugate: `σ ± jω` | 1 at origin |
| band_stop | 2 conjugate: `σ ± jω` | 2 imaginary: `±jω₀` |
| rlc_series | 2 conjugate: `σ ± jω` | 1 at origin |
| rlc_parallel | 2 conjugate (damped) | 1 at origin |
| lc_lowpass | 2 purely imaginary: `±jω₀` | None |
| cl_highpass | 2 purely imaginary: `±jω₀` | 2 at origin |

## What We Predict

We compress each circuit's poles/zeros into 4 numbers: `[sigma_p, omega_p, sigma_z, omega_z]`.

**How**: pick the dominant pole/zero (closest to origin), extract its real and imaginary parts, then apply signed-log normalization to squash values from the 10¹–10⁶ range into roughly [-1, 1]:

```
signed_log(x) = sign(x) * log10(|x| + 1) / 7.0
```

**Example**: a low_pass with R=1kΩ, C=1μF has pole at -1000. That becomes sigma_p = -0.429, omega_p = 0.0.

| Dimension | What it means | Nonzero for |
|---|---|---|
| sigma_p | Damping rate | All damped types (not lc_lowpass, cl_highpass) |
| omega_p | Resonant frequency | All resonant types (not low_pass, high_pass) |
| sigma_z | Zero real part | Nothing (always 0 in our circuits) |
| omega_z | Zero frequency | Only band_stop |

## How the Encoder Predicts It

```
Circuit Graph → 3-layer ImpedanceConv GNN → 8D latent [topo(2) | structure(2) | pz(4)]
                                                                                 ↑
                                                                        supervised by MSE
```

The GNN uses separate MLPs for R, C, and L edges so it can learn nonlinear relationships like `ω₀ = 1/√(LC)`. The last 4 latent dimensions are trained to match the pz target via `loss_pz = MSE(mu[:, 4:8], target) × 5.0`.

## Evaluation 1: New Component Values, Same Topology

**Question**: if we change R/C/L values but keep the same circuit topology, can the model predict the new poles/zeros?

**Setup**: standard train/val split (1536/384). Val circuits have the same 8 topologies as training but randomly different component values.

**Answer**: yes, very well. Val MSE = 0.001165, only 1.04x the training MSE.

### What This Looks Like in Practice

Below are real val-set predictions, showing best, median, and worst cases per filter type. All values in signed-log scale; raw physical values in parentheses.

**low_pass** — model predicts the RC time constant

| | Components | Actual pole | Predicted sigma_p | Actual sigma_p |
|---|---|---|---|---|
| Best | R=470Ω, C=430nF | -4928 rad/s | -0.515 (-4008) | -0.528 (-4928) |
| Median | R=4.4kΩ, C=40nF | -5601 rad/s | -0.503 (-3316) | -0.536 (-5601) |
| Worst | R=28kΩ, C=27nF | -1306 rad/s | -0.398 (-610) | -0.445 (-1306) |

Worst case underpredicts the pole magnitude by about 2x, but in signed-log space the error is only 0.047. The model correctly predicts omega_p ≈ 0 for all (these are real poles, no oscillation).

**band_pass** — model must predict both damping and resonant frequency

| | Components | Actual poles | Predicted | Actual |
|---|---|---|---|---|
| Best | R=7.8kΩ, L=940μH, C=66nF | -1936 (real) | σ=-0.467, ω≈0 | σ=-0.470, ω=0 |
| Median | R=1.7kΩ, L=385μH, C=290nF | -2038 (real) | σ=-0.451, ω≈0 | σ=-0.473, ω=0 |
| Worst | R=154Ω, L=1mH, C=108nF | -75367 ± 58355j | σ=-0.661, ω=0.018 | σ=-0.697, ω=0.681 |

Best and median cases have overdamped (real) poles — model handles these easily. Worst case is underdamped (conjugate poles with ω≈58kHz). The model gets sigma_p close but misses omega_p entirely — it thinks the circuit is overdamped when it actually resonates.

**band_stop** — model must also predict the notch frequency (omega_z)

| | Components | Actual zeros | omega_z predicted | omega_z actual | Raw predicted | Raw actual |
|---|---|---|---|---|---|---|
| Best | R=84kΩ, R=2.9kΩ, C=140nF, L=730μH | ±99441j | 0.714 | 0.714 | 99,152 | 99,441 |
| Median | R=40kΩ, R=115Ω, C=45nF, L=330μH | ±259449j | 0.800 | 0.773 | 398,311 | 259,449 |
| Worst | R=35kΩ, R=102Ω, C=600nF, L=1.4mH | ±34579j | 0.584 | 0.648 | 12,278 | 34,579 |

Best case nails the notch: 99,152 vs 99,441 rad/s. Worst case is off by ~3x in raw units, but only 0.064 in log scale.

**rlc_parallel** — damped resonance, 2 conjugate poles

| | Components | Actual poles | sigma_p pred / actual | omega_p pred / actual |
|---|---|---|---|---|
| Best | R=1.7kΩ, C=760nF, L=340μH, R=187Ω | -3939 ± 62086j | -0.512 / -0.514 | 0.682 / 0.685 |
| Median | R=2.9kΩ, C=61nF, L=1.6mH, R=154Ω | -56126 ± 83346j | -0.637 / -0.679 | 0.712 / 0.703 |
| Worst | R=51kΩ, C=140nF, L=9mH, R=142Ω | -25447 ± 12324j | -0.535 / -0.629 | 0.643 / 0.584 |

Best case: damping 3,811 vs 3,939 and resonance 59,588 vs 62,086 — within 4%. Worst case is off by ~5x in raw damping, but only 0.094 in log scale.

**cl_highpass** — undamped, model only predicts resonant frequency

| | Components | Actual poles | omega_p predicted | omega_p actual | Raw predicted | Raw actual |
|---|---|---|---|---|---|---|
| Best | L=970μH, C=179nF | ±75874j | 0.6972 | 0.6972 | 75,918 | 75,874 |
| Median | L=2.4mH, C=9.1nF | ±213981j | 0.7649 | 0.7615 | 226,014 | 213,981 |
| Worst | L=131μH, C=1.95nF | ±1978806j | 0.8912 | 0.8995 | 1,731,757 | 1,978,806 |

Best case is essentially exact. Even the worst case (highest frequency in the dataset) is within 12%.

### Where the Model Struggles

The main failure mode is **overdamped vs underdamped confusion**: when a circuit is near the boundary between having real poles and conjugate poles, the model sometimes picks the wrong side. This shows up as omega_p ≈ 0 when it should be large (band_pass worst case), or omega_p > 0 when it should be 0 (band_stop worst case).

For circuits clearly in one regime, predictions are accurate across the full component value range. There is no systematic degradation at extreme component values — the model interpolates well across the log-uniform training distribution.

### Aggregate Statistics

| Filter | Mean MAE | Median MAE | P95 MAE | Max MAE |
|---|---|---|---|---|
| cl_highpass | 0.005 | 0.004 | 0.012 | 0.013 |
| high_pass | 0.008 | 0.006 | 0.018 | 0.030 |
| lc_lowpass | 0.009 | 0.009 | 0.016 | 0.017 |
| low_pass | 0.010 | 0.010 | 0.015 | 0.018 |
| rlc_series | 0.013 | 0.009 | 0.026 | 0.128 |
| rlc_parallel | 0.015 | 0.014 | 0.027 | 0.041 |
| band_pass | 0.016 | 0.007 | 0.021 | 0.175 |
| band_stop | 0.020 | 0.015 | 0.041 | 0.150 |

Simpler 3-node topologies (cl_highpass, high_pass, lc_lowpass, low_pass) are easiest. More complex 4-5 node topologies have slightly higher errors but median MAE stays under 0.015 for all types.

Scripts: `scripts/eval/eval_pz.py`, `scripts/eval/eval_pz_component_generalization.py`

## Evaluation 2: Unseen Topologies (LOTO)

**Question**: if the model has never seen a particular circuit topology during training, can it still predict that topology's poles/zeros?

**Setup**: Leave-One-Topology-Out cross-validation. For each of 8 filter types, train on the other 7 types (1680 circuits), then test on the held-out type (240 circuits). Same architecture and hyperparameters as production.

**Answer**: no. Unseen topology MSE is 15x higher than in-distribution.

### Results

| Held-out Type | Unseen MSE | In-Dist MSE | R² sigma_p | R² omega_p | R² omega_z |
|---|---|---|---|---|---|
| low_pass | 0.1143 | 0.0101 | 0.419 | — | — |
| high_pass | 0.0613 | 0.0033 | -0.501 | — | — |
| band_pass | 0.0180 | 0.0041 | -0.410 | 0.589 | — |
| band_stop | 0.2224 | 0.0028 | -15.0 | -0.615 | -114.3 |
| rlc_series | 0.0254 | 0.0029 | 0.544 | 0.533 | — |
| rlc_parallel | 0.1532 | 0.0038 | -2.806 | -94.5 | — |
| lc_lowpass | 0.0974 | 0.0108 | — | -59.8 | — |
| cl_highpass | 0.0243 | 0.0089 | — | -3.758 | — |
| **Average** | **0.0895** | **0.0058** | | | |

"—" = that dimension is constant for the held-out type (nothing to predict). Negative R² means predictions are worse than just guessing the mean.

### What This Means

**Structurally similar types partially transfer.** rlc_series (R²=0.54/0.53) and band_pass (omega_p R²=0.59) share structural similarity with other training types, so the GNN can leverage some learned relationships.

**Unique topologies don't transfer at all.** band_stop (omega_z R²=-114) has notch zeros that no other type produces. rlc_parallel (omega_p R²=-95) has parallel resonance mechanics unlike anything else in the dataset.

**This is expected.** Each topology has a unique mathematical relationship between component values and poles/zeros. A model trained on 7 topologies has no way to derive the transfer function of a fundamentally different 8th topology. The encoder learns per-topology regression, not general circuit physics.

Improving this would require either (a) training on many more diverse topologies, or (b) building physics-based inductive biases into the architecture (e.g., nodal analysis layers).

Script: `scripts/eval/eval_pz_unseen_topology.py` | Results: `scripts/eval/loto_results.json`
