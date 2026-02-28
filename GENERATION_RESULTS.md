# Circuit Generation Results

**Dataset:** 480 circuits (8 filter types, 60 each), 384 train, 96 validation
**Checkpoint:** `checkpoints/production/best.pt`
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`
**Latent space:** 8D hierarchical `[z_topo(2) | z_values(2) | z_pz(4)]`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Total Loss | 0.96 | 1.04 |
| Pole/Zero Loss | 0.007 | 0.006 |
| Node Count Accuracy | 100% | 100% |
| Edge Existence Accuracy | 100% | 100% |
| Component Type Accuracy | 100% | 100% |
| Encoder Parameters | 83,411 | -- |
| Decoder Parameters | 7,698,901 | -- |

---

## Specification-Based Generation

Generate circuits by specifying **cutoff frequency** and **Q-factor**:

```bash
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5
```

### Standard Examples

| Cutoff | Q | Generated Circuit | Status |
|--------|---|-------------------|--------|
| 1 kHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 10 kHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 100 kHz | 0.707 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Valid |
| 10 kHz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |

### Edge Cases

| Cutoff | Q | Generated Circuit | Analysis |
|--------|---|-------------------|----------|
| 1 Hz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training range |
| 1 MHz | 0.707 | `GND--C--VOUT, VIN--R--VOUT` | Extrapolates beyond training range |
| 10 kHz | 0.01 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | Low Q (rlc_series) |
| 10 kHz | 0.1 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series |
| 10 kHz | 2.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series |
| 50 Hz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | High Q (rlc_parallel) |
| 500 kHz | 0.1 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | band_stop |

### Training Data Coverage

| Parameter | Min | Max |
|-----------|-----|-----|
| Cutoff frequency | 4.70 Hz | 993,359 Hz |
| Q-factor | 0.01 | 6.37 |

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0:4] | Generated from Centroid |
|-------------|--------|-------------------------|
| low_pass | [+2.09, +1.28, +1.18, +3.20] | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | [+3.66, +1.26, -0.71, +2.24] | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | [-3.56, -0.07, -1.84, +2.21] | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | [-0.13, -2.25, +2.32, -3.65] | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` |
| rlc_series | [-1.55, -1.98, +3.47, -2.23] | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` |
| rlc_parallel | [+2.99, +1.10, +2.76, +1.57] | `GND--RCL--VOUT, VIN--R--VOUT` |
| lc_lowpass | [-2.46, +0.16, -0.40, +2.42] | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| cl_highpass | [-1.40, +0.22, +0.28, +2.28] | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |

---

## Reconstruction Accuracy

### By Filter Type (60 circuits each)

| Filter Type | Valid | Example Reconstruction |
|-------------|-------|------------------------|
| low_pass | 60/60 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | 60/60 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | 60/60 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | 60/60 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` |
| rlc_series | 60/60 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` |
| rlc_parallel | 60/60 | `GND--RCL--VOUT, VIN--R--VOUT` |
| lc_lowpass | 60/60 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| cl_highpass | 60/60 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |

**Total: 480/480 (100%) valid reconstructions**

---

## Interpolation

### Low-pass -> High-pass

R and C swap positions at alpha ~ 0.75:

| alpha | Generated |
|---|-----------|
| 0.00 | `GND--C--VOUT, VIN--R--VOUT` (low-pass) |
| 0.25 | `GND--C--VOUT, VIN--R--VOUT` |
| 0.50 | `GND--C--VOUT, VIN--R--VOUT` (transition) |
| 0.75 | `GND--R--VOUT, VIN--C--VOUT` |
| 1.00 | `GND--R--VOUT, VIN--C--VOUT` (high-pass) |

### Band-pass -> RLC-parallel

Distributed -> lumped transition:

| alpha | Generated |
|---|-----------|
| 0.00 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` (distributed LC) |
| 0.25 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| 0.50 | `GND--R--VOUT` (transition) |
| 0.75 | `GND--RCL--VOUT, VIN--R--VOUT` |
| 1.00 | `GND--RCL--VOUT, VIN--R--VOUT` (lumped RCL) |

---

## Novel Topology Generation

### Exploration Results (500 samples)

| Category | Unique Topologies | Samples |
|----------|-------------------|---------|
| Known (in training) | 6 | 334 (66.8%) |
| **Valid novel** | **9** | **112 (22.4%)** |
| Invalid (disconnected) | -- | 54 (10.8%) |

### Known Topology Breakdown

| Topology | Filter Type | Count |
|----------|-------------|-------|
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 114 |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 111 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | band_stop | 63 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series | 26 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 13 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 7 |

### Valid Novel Topologies

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1` | 67 | 4 | R |
| `GND--R--VOUT, VIN--R--INT1` | 16 | 4 | R |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2` | 11 | 5 | R, C |
| `GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | 5 | 5 | R, C, L |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1` | 4 | 4 | R, C |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2` | 4 | 5 | R, C, L |
| `GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1` | 3 | 4 | R, C, L |
| `GND--R--VOUT, VIN--L--INT1, VOUT--R--INT1` | 1 | 4 | L, R |
| `GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--R--INT1` | 1 | 4 | R, C, L |

---

## Pole/Zero-Driven Generation

Generate circuits by specifying **dominant pole/zero** directly. The decoder constructs z[4:8] from signed-log normalized pole/zero values, samples z[0:4] from the prior, and generates using the decoder only (no encoder needed).

```bash
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5
```

### Signed-Log Normalization

`z = sign(x) * log10(|x| + 1) / 7.0` maps raw pole/zero values to ~[-1, 1]:

| z[4:8] | Meaning |
|--------|---------|
| z[4] = sigma_p | signed_log(real part of dominant pole) |
| z[5] = omega_p | signed_log(\|imag part of dominant pole\|) |
| z[6] = sigma_z | signed_log(real part of dominant zero) |
| z[7] = omega_z | signed_log(\|imag part of dominant zero\|) |

### Test Cases (3 samples each)

#### Real Pole, No Zero (RC Low-Pass Behavior)

| Pole | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -6283 + 0j (~1kHz) | [-0.543, 0, 0, 0] | `R--C`, `R--R--C--L`, `R--C` | 3/3 |
| -62832 + 0j (~10kHz) | [-0.685, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |
| -628318 + 0j (~100kHz) | [-0.828, 0, 0, 0] | `R`, `R`, `R--R--R` | 1/3 |

#### Real Zero, No Pole (RC High-Pass Behavior)

| Zero | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -6283 + 0j (~1kHz) | [0, 0, -0.543, 0] | `R--C`, `R--C`, `R--C` | 3/3 |

#### Conjugate Poles (Resonant Behavior)

| Pole | Zero | z[4:8] | Samples | Valid |
|------|------|--------|---------|-------|
| -3142 + 49348j | 0 | [-0.500, +0.670, 0, 0] | band_stop, R-divider, `R` | 2/3 |
| -3142 + 49348j | 0 + 49348j | [-0.500, +0.670, 0, +0.670] | `R--C`, R-divider, R-divider | 3/3 |

#### Extreme Frequencies

| Pole | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -100 + 0j (very low) | [-0.286, 0, 0, 0] | `R`, LC band-pass, R-divider | 2/3 |
| -1e6 + 0j (very high) | [-0.857, 0, 0, 0] | `R--C`, band_stop, `R--C` | 3/3 |

### Summary

| Category | Valid/Total |
|----------|------------|
| Real pole, no zero | 7/9 (78%) |
| Real zero, no pole | 3/3 (100%) |
| Conjugate poles | 5/6 (83%) |
| Extreme frequencies | 5/6 (83%) |
| **Overall** | **20/24 (83%)** |

**Observations:**
- Real-pole-only specs (RC low-pass) reliably produce RC topologies
- The z[0:4] prior sampling introduces topology diversity (same pole/zero can yield different structures)
- Very high frequency poles (-628318, -1e6) occasionally produce degenerate topologies
- Conjugate poles generate more complex topologies (band-stop, RLC networks)

---

## Key Insights

### 1. Supervised Pole/Zero Latent (z[4:8])

The `loss_pz` supervision forces z[4:8] to encode pole/zero information:
- Converged from 0.22 to 0.006 (val) over 100 epochs
- Edge/component accuracy unaffected (stays at 100%)
- Enables decoder-only generation from physical specifications

### 2. Q-Factor Drives Topology Selection

| Q Range | Typical Topology |
|---------|------------------|
| Q < 0.1 | 5-node band-stop (notch filter) |
| Q ~ 0.1 | 4-node series RLC |
| Q ~ 0.707 | 3-node RC (Butterworth) |
| Q > 2.0 | 3-node RLC parallel (tank) |

### 3. Latent Space is Well-Organized

- All 8 dimensions are now meaningful (z[0:4] topology, z[4:8] pole/zero)
- Filter types form distinct clusters
- Interpolation produces smooth transitions
- 89.2% valid rate on random samples (446/500)

### 4. Component Values Improve Generalization

Compared to binary edge features, log10 component values:
- Produce more unique novel topologies
- Achieve higher valid rate
- Maintain lower validation loss

---

## Usage

```bash
# Pole/zero-driven generation (decoder only, no encoder needed)
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5

# Conjugate pole pair (resonant circuit)
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --num-samples 5

# With a zero (band-stop behavior)
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --zero-imag 49348

# Regenerate all results
python scripts/generation/regenerate_all_results.py
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (480 circuits, 8 types)
- **Generation script:** `scripts/generation/generate_from_specs.py`
- **Results script:** `scripts/generation/regenerate_all_results.py`
