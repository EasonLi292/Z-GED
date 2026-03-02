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
- **σ (sigma)**: the decay rate — how fast the circuit's response dies out after an impulse. More negative = faster decay = more damping. For a simple RC low-pass, `σ = -1/(RC)`.
- **ω (omega)**: the oscillation frequency — how fast the circuit rings. Zero means no oscillation (overdamped), nonzero means the circuit resonates.

Note: these values are in **rad/s**, not Hz. A pole at -4928 rad/s is 4928/(2π) ≈ 784 Hz. They look large because tiny component values (nF capacitors, μH inductors) produce fast time constants.

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

**How**: pick the dominant pole/zero (closest to origin), extract its real and imaginary parts, then apply signed-log normalization to squash values from the 10¹–10⁶ rad/s range into roughly [-1, 1]:

```
signed_log(x) = sign(x) * log10(|x| + 1) / 7.0
```

The `/7.0` means any value up to ±10 million rad/s maps to ±1.0. Our poles max out around 2M rad/s, which maps to ~0.9.

**Worked example** — low_pass with R=470Ω, C=430nF:

```
Raw pole = -1/(RC) = -1/(470 × 430e-9) = -4928 rad/s   (≈ 784 Hz)

signed_log(-4928):
  sign = -1
  log10(4928 + 1) = 3.693
  3.693 / 7.0 = 0.5275
  → sigma_p = -0.5275

To convert back:  10^(0.5275 × 7.0) - 1 = 4928 rad/s
```

So the model outputs -0.5275 and we know that means "pole at -4928 rad/s, decaying in 0.2ms."

| Dimension | Physical meaning | Nonzero for |
|---|---|---|
| **sigma_p** | Decay rate of dominant pole — how fast the impulse response dies out | All damped types (not lc_lowpass, cl_highpass which ring forever) |
| **omega_p** | Oscillation frequency of dominant pole — how fast the circuit rings | All resonant types (not low_pass, high_pass which just decay) |
| **sigma_z** | Decay rate of dominant zero | Nothing (always 0 in our circuits — all zeros are at the origin or purely imaginary) |
| **omega_z** | Notch frequency — where the circuit completely blocks the signal | Only band_stop (its zeros at ±jω₀ create the notch) |

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

Below are real val-set predictions, showing best, median, and worst cases per filter type.

**low_pass** — model predicts the decay rate from RC

| | Components | Actual pole | Model predicts | Actual |
|---|---|---|---|---|
| Best | R=470Ω, C=430nF | -4928 rad/s (decays in 0.2ms) | -4008 rad/s | -4928 rad/s |
| Median | R=4.4kΩ, C=40nF | -5601 rad/s (decays in 0.18ms) | -3316 rad/s | -5601 rad/s |
| Worst | R=28kΩ, C=27nF | -1306 rad/s (decays in 0.77ms) | -610 rad/s | -1306 rad/s |

Worst case: model says "pole at -610 rad/s" when it's actually -1306 — off by 2x, but it correctly knows this is a slow, low-frequency pole. It also correctly predicts omega_p ≈ 0 for all three (real poles, no oscillation).

**band_pass** — model must predict both the decay rate (sigma_p) and the resonant frequency (omega_p)

| | Components | What actually happens | Model predicts | Actual |
|---|---|---|---|---|
| Best | R=7.8kΩ, L=940μH, C=66nF | Overdamped, pole at -1936 rad/s, no oscillation | σ: -1860 rad/s, ω: ≈0 | σ: -1936 rad/s, ω: 0 |
| Median | R=1.7kΩ, L=385μH, C=290nF | Overdamped, pole at -2038 rad/s, no oscillation | σ: -1436 rad/s, ω: ≈0 | σ: -2038 rad/s, ω: 0 |
| Worst | R=154Ω, L=1mH, C=108nF | Underdamped, oscillates at 58 kHz while decaying at 75 kHz | σ: -42565 rad/s, ω: ≈0 | σ: -75367 rad/s, ω: 58355 rad/s |

Best and median cases are overdamped (no ringing) — model handles these easily. Worst case actually resonates at 58 kHz, but the model predicts ω ≈ 0 — it thinks the circuit just decays without oscillating. This is the hardest failure mode: confusing overdamped with underdamped.

**band_stop** — model must also predict the notch frequency (omega_z), where the circuit completely blocks the signal

| | Components | Actual notch zeros | Model predicts notch at | Actual notch at |
|---|---|---|---|---|
| Best | R=84kΩ, R=2.9kΩ, C=140nF, L=730μH | ±99441j rad/s (15.8 kHz) | 99,152 rad/s | 99,441 rad/s |
| Median | R=40kΩ, R=115Ω, C=45nF, L=330μH | ±259449j rad/s (41.3 kHz) | 398,311 rad/s | 259,449 rad/s |
| Worst | R=35kΩ, R=102Ω, C=600nF, L=1.4mH | ±34579j rad/s (5.5 kHz) | 12,278 rad/s | 34,579 rad/s |

Best case nails it: predicts the signal is blocked at 15.8 kHz, actual is 15.8 kHz. Worst case says the notch is at 2 kHz when it's actually at 5.5 kHz — off by ~3x.

**rlc_parallel** — damped resonance, model predicts both decay and oscillation frequency

| | Components | What actually happens | Model predicts | Actual |
|---|---|---|---|---|
| Best | R=1.7kΩ, C=760nF, L=340μH, R=187Ω | Decays at 3939 rad/s, rings at 62 kHz | decay: 3811, ring: 59,588 | decay: 3939, ring: 62,086 |
| Median | R=2.9kΩ, C=61nF, L=1.6mH, R=154Ω | Decays at 56,126 rad/s, rings at 83 kHz | decay: 28,966, ring: 96,924 | decay: 56,126, ring: 83,346 |
| Worst | R=51kΩ, C=140nF, L=9mH, R=142Ω | Decays at 25,447 rad/s, rings at 12 kHz | decay: 5,531, ring: 31,801 | decay: 25,447, ring: 12,324 |

Best case is within 4% on both decay and oscillation frequency. Worst case gets the decay rate off by ~5x, but in signed-log space this is only 0.094 error.

**cl_highpass** — undamped LC circuit, rings forever at ω₀ = 1/√(LC)

| | Components | Actual resonant freq | Model predicts | Actual |
|---|---|---|---|---|
| Best | L=970μH, C=179nF | 75,874 rad/s (12.1 kHz) | 75,918 rad/s | 75,874 rad/s |
| Median | L=2.4mH, C=9.1nF | 213,981 rad/s (34.1 kHz) | 226,014 rad/s | 213,981 rad/s |
| Worst | L=131μH, C=1.95nF | 1,978,806 rad/s (315 kHz) | 1,731,757 rad/s | 1,978,806 rad/s |

Best case is essentially exact (off by 44 rad/s out of 75,874). Even the worst case — the highest frequency circuit in the dataset — is within 12%.

### Where the Model Struggles

The main failure mode is **overdamped vs underdamped confusion**. When a circuit is near the boundary between having real poles (pure decay) and conjugate poles (decay + oscillation), the model sometimes picks the wrong side:

- **band_pass worst case**: circuit oscillates at 58 kHz, model says no oscillation
- **band_stop worst case**: poles don't oscillate, model predicts they do

For circuits clearly in one regime, predictions are accurate across the full component value range.

### Aggregate Error by Filter Type

| Filter | Median MAE | P95 MAE | Max MAE |
|---|---|---|---|
| cl_highpass | 0.004 | 0.012 | 0.013 |
| high_pass | 0.006 | 0.018 | 0.030 |
| lc_lowpass | 0.009 | 0.016 | 0.017 |
| low_pass | 0.010 | 0.015 | 0.018 |
| rlc_series | 0.009 | 0.026 | 0.128 |
| rlc_parallel | 0.014 | 0.027 | 0.041 |
| band_pass | 0.007 | 0.021 | 0.175 |
| band_stop | 0.015 | 0.041 | 0.150 |

MAE is in signed-log units. An error of 0.01 is roughly 2% in raw rad/s; 0.05 is roughly half an order of magnitude.

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

"—" = that dimension is constant for the held-out type. Negative R² means predictions are worse than just guessing the mean.

### What This Means

**Structurally similar types partially transfer.** rlc_series (R²=0.54/0.53) and band_pass (omega_p R²=0.59) share structural similarity with other training types, so the GNN can leverage some learned relationships.

**Unique topologies don't transfer at all.** band_stop (omega_z R²=-114) has notch zeros that no other type produces. rlc_parallel (omega_p R²=-95) has parallel resonance mechanics unlike anything else in the dataset.

**This is expected.** Each topology has a unique mathematical relationship between component values and poles/zeros. A model trained on 7 topologies has no way to derive the transfer function of a fundamentally different 8th topology. The encoder learns per-topology regression, not general circuit physics.

Improving this would require either (a) training on many more diverse topologies, or (b) building physics-based inductive biases into the architecture (e.g., nodal analysis layers).

Script: `scripts/eval/eval_pz_unseen_topology.py` | Results: `scripts/eval/loto_results.json`
