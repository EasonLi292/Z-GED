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

## Current Performance (Val Set, 1920 Circuits)

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
