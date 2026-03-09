# Admittance-Based Gain Prediction: Experiment Report

## Question

Can a GNN with **complex admittance edge features** `[Re(Y), Im(Y)]` predict circuit gain `|H(jω)|` from graph structure alone — replacing the component-specific message paths (`lin_R`, `lin_C`, `lin_L`) of ImpedanceConv with a single unified edge MLP?

## Setup

| | |
|---|---|
| **Edge features** | `[Re(Y), Im(Y)]` where `Y(jω) = G + j(ωC - L_inv/ω)` |
| **Node features** | 4D one-hot `[GND, VIN, VOUT, INTERNAL]` |
| **Target** | `log10(\|H(jω)\|)` clipped to `[-10, 5]` |
| **Architecture** | 3-layer AdmittanceGNN → hierarchical VAE (8D) → gain_head MLP |
| **Samples** | 384 train circuits × 100 freqs = 38,400 train; same for val |
| **Baseline** | Predict mean of all targets → MSE = 6.384 |

## Overall Results

| Metric | Value |
|---|---|
| **Val MSE** | 0.4085 |
| **Val MAE** | 0.284 |
| **Median absolute error** | 0.075 |
| **Baseline ratio** | 0.064× (15.6× better than mean predictor) |
| **Pearson correlation** | 0.968 |
| **% within 0.25 log-decades** | 74.5% |
| **% within 0.50 log-decades** | 84.1% |

**Verdict: admittance features work well for most cases, but fail systematically in two regimes.**

## Where It Works

### High-pass filter (circuit #241): R=4844Ω, C=7.25e-8F

| Freq (Hz) | Target | Predicted | Error |
|---|---|---|---|
| 10.0 | -1.656 | -2.116 | -0.460 |
| 497.7 | -0.131 | -0.129 | +0.002 |
| 29,150 | -0.000 | +0.011 | +0.011 |
| 1,707,353 | -0.000 | +0.010 | +0.010 |
| 100,000,000 | +0.000 | +0.013 | +0.013 |

Errors < 0.02 across the passband. The model captures the transition from attenuation to passband correctly.

### Band-pass filter (circuit #480): R=4748Ω, L=5.15e-4H, C=8.63e-7F

| Freq (Hz) | Target | Predicted | Error |
|---|---|---|---|
| 10.0 | -0.603 | -0.511 | +0.092 |
| 497.7 | -0.001 | -0.008 | -0.007 |
| 29,150 | -0.000 | +0.005 | +0.005 |
| 1,707,353 | -0.186 | -0.133 | +0.053 |
| 100,000,000 | -1.834 | -1.924 | -0.091 |

Peak response near the resonant frequency is captured; rolloff in both directions is tracked.

### Band-stop filter (circuit #720): multi-component

| Freq (Hz) | Target | Predicted | Error |
|---|---|---|---|
| 10.0 | -0.314 | -0.334 | -0.020 |
| 497.7 | -0.314 | -0.322 | -0.008 |
| 29,150 | -0.789 | -0.481 | +0.309 |
| 1,707,353 | -0.314 | -0.330 | -0.016 |
| 100,000,000 | -0.314 | -0.324 | -0.010 |

Flat-band regions are near-perfect. The notch at 29 kHz is detected but underestimated — the model sees the correct direction but not the full depth.

## Where It Fails

### Failure 1: Deep attenuation (target < -3)

| Target range | n | MAE | MSE |
|---|---|---|---|
| `[-10, -3)` | 10,066 | 0.592 | 0.788 |
| `[-3, -1)` | 7,614 | 0.382 | 0.706 |
| `[-1, 0)` | 15,393 | 0.122 | 0.149 |
| `[0, 1)` | 5,259 | 0.022 | 0.003 |

Errors grow dramatically for highly attenuated signals. At -7 dB-decades, the model is off by ~0.6 on average.

### Failure 2: Low-pass filters (MSE = 1.583, worst of all 8 types)

Low-pass circuit #1 (R=322.5Ω, C=2.23e-8F):

| Freq (Hz) | Target | Predicted | Error |
|---|---|---|---|
| 10.0 | -0.000 | -0.056 | -0.056 |
| 497.7 | -0.000 | -0.185 | -0.185 |
| 29,150 | -0.219 | -2.565 | **-2.347** |
| 1,707,353 | -1.888 | -5.972 | **-4.085** |
| 100,000,000 | -3.655 | -6.964 | **-3.309** |

The model massively over-predicts attenuation in the stopband. It sees large admittance values at high frequencies (because `Im(Y) = ωC` grows with `ω`) and incorrectly maps this to deeper attenuation.

### Failure 3: RLC parallel (MSE = 1.140)

Prediction range `[-9.32, -0.58]` vs target range `[-9.94, -3.46]`. These are always-attenuated circuits and the model struggles with the steep resonance features.

### Failure 4: Positive gain is invisible

- Max prediction: **0.87**
- Max target: **4.84**
- 13.9% of samples have positive gain (resonant amplification), but the model never predicts above ~0.9.

The gain head saturates because the VAE's 8D bottleneck has no mechanism to encode the magnitude of resonant peaks.

## Why Admittance Features Are Partially Expressive

### What works

Admittance `Y(jω) = G + j(ωC - L_inv/ω)` is a **physically meaningful** edge feature that naturally encodes:

1. **Frequency dependence**: Different `ω` → different `[Re(Y), Im(Y)]` → the GNN sees different graphs at each frequency.
2. **Component interaction**: Near resonance, `ωC ≈ L_inv/ω`, so `Im(Y) → 0`. The GNN can detect this cancellation, which is why band-pass/band-stop predictions work.
3. **Magnitude information**: The size of `|Y|` indicates how strongly two nodes are coupled at a given frequency.

### What doesn't work

1. **Admittance is a local edge property, but gain is a global transfer function.** Knowing each edge's `Y(jω)` tells you the local coupling, but computing `H(jω) = V_out/V_in` requires solving the full circuit (Kirchhoff's laws across all nodes simultaneously). The 3-layer GNN has a limited receptive field and can't reliably compose these local admittances into a global transfer function for deeper circuits.

2. **The unified MLP conflates different physical roles.** A resistor's `Re(Y) = G` and a capacitor's `Im(Y) = ωC` play fundamentally different roles in the circuit (dissipation vs. energy storage). `ImpedanceConv`'s separate `lin_R/lin_C/lin_L` paths explicitly model these different roles. The single `lin_edge` in `AdmittanceConv` must learn to disentangle them from a 2D input — possible in simple cases, but it loses specificity for complex multi-component circuits.

3. **Scale ambiguity in Im(Y).** `Im(Y) = ωC - L_inv/ω` mixes frequency and component value. At `ω = 10` with `C = 1e-7`, `Im(Y) ≈ 1e-6`. At `ω = 1e6` with `C = 1e-9`, `Im(Y) ≈ 1e-3`. The MLP must learn to factor out the frequency from the component contribution — an unnecessary burden that ImpedanceConv avoids entirely.

## Could One-Hot Encoding Do This?

The main model's `ImpedanceConv` uses edge features `[log10(R), log10(C), log10(L)]` where 0 = absent. This is essentially a **continuous extension of one-hot encoding**:

| | One-hot `[is_R, is_C, is_L]` | Log10 values `[log10(R), log10(C), log10(L)]` |
|---|---|---|
| **Component type** | Yes (which slot is 1) | Yes (which slot is nonzero) |
| **Component value** | No | Yes (the magnitude in each slot) |
| **Frequency info** | No | No |
| **Gains/transfer function** | No — topology only | No — frequency-independent |

### One-hot for gain prediction: no

Pure one-hot `[is_R, is_C, is_L]` encodes **topology only** — which component types exist and where. Two circuits with identical topology but R=100Ω vs R=100kΩ would have identical edge features, despite having completely different frequency responses. One-hot encoding **cannot** predict gain because gain depends on component values and frequency, neither of which one-hot captures.

### Log10 values for gain prediction: also no (without modification)

The current `[log10(R), log10(C), log10(L)]` edge features are **frequency-independent**. The same graph representation is used regardless of whether you're asking about gain at 10 Hz or 10 MHz. To predict gain, the model would need frequency as an additional input — either as a node feature, a global conditioning variable, or baked into the edge features (as this admittance experiment does).

### What admittance adds over both

Admittance **bakes frequency into the edge features**. This is the key insight: `Y(jω)` makes the graph representation frequency-dependent, so the model sees a *different* graph at each frequency. This is why it works at all (correlation 0.97) — it's solving a fundamentally different problem than topology classification.

## Conclusion

Admittance features are **expressive enough for moderate-gain, simple-topology circuits** (high-pass, band-pass, band-stop: MSE 0.03–0.06), but **insufficient for deep-attenuation and resonant-gain regimes** (low-pass, rlc_parallel: MSE 1.1–1.6). The unified edge MLP loses the component-type specificity that ImpedanceConv's separate paths provide, and the 8D VAE bottleneck cannot encode the full dynamic range of |H(jω)|.

The experiment validates that **frequency-dependent edge features are necessary for gain prediction** (one-hot or static log10 values can't do it), but suggests that **admittance alone isn't sufficient** — the model likely needs either (a) component-specific message paths operating on frequency-dependent features, or (b) a direct gain head that bypasses the VAE bottleneck.
