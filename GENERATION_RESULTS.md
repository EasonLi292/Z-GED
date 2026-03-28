# Circuit Generation Results

**Dataset:** 1920 circuits (8 filter types, 240 each), 1536 train, 384 validation
**Checkpoint:** `checkpoints/production/best.pt`
**Decoder:** Sequence decoder (GPT-style, Eulerian walk representation)
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`
**Latent space:** 8D hierarchical `[z_topo(2) | z_values(2) | z_pz(4)]`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| CE Loss | 0.035 | 0.021 |
| KL Loss | 0.725 | 0.758 |
| Token Accuracy | 98.2% | 98.6% |
| Topology Match | -- | 100% |
| Valid Walk Rate | -- | 100% |
| Encoder Parameters | 237,907 | -- |
| Decoder Parameters | 3,280,726 | -- |

---

## Specification-Based Generation (Interpolation Analysis)

Exploratory scripts can still map **cutoff frequency** and **Q-factor** to latent codes using K-NN interpolation over encoded dataset points.
For that path, use `scripts/testing/*.py` or `scripts/generation/regenerate_all_results.py`.

For production generation entry point, use pole/zero inputs:

```bash
.venv/bin/python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5
```

### Standard Examples

| Cutoff | Q | Generated Circuit | Status |
|--------|---|-------------------|--------|
| 1 kHz | 0.707 | `VOUT--C--VSS, VIN--R--VOUT` | Valid |
| 10 kHz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Valid |
| 100 kHz | 0.707 | `VIN--C--VOUT, VOUT--L--VSS` | Valid |
| 10 kHz | 5.0 | `VIN--R--VOUT, VOUT--RCL--VSS` | Valid |

### Edge Cases

| Cutoff | Q | Generated Circuit | Analysis |
|--------|---|-------------------|----------|
| 1 Hz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Extrapolates beyond training range |
| 1 MHz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Extrapolates beyond training range |
| 10 kHz | 0.01 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` | Low Q (rlc_series) |
| 10 kHz | 0.1 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` | band_pass |
| 10 kHz | 2.0 | `VIN--C--VOUT, VOUT--R--VSS` | high_pass |
| 50 Hz | 5.0 | `VIN--R--VOUT, VOUT--RCL--VSS` | High Q (rlc_parallel) |
| 500 kHz | 0.1 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` | rlc_series |

### Training Data Coverage

| Parameter | Min | Max |
|-----------|-----|-----|
| Cutoff frequency | 2.05 Hz | 872,085 Hz |
| Q-factor | 0.01 | 6.58 |

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0:4] | Generated from Centroid |
|-------------|--------|-------------------------|
| low_pass | [-0.41, +0.49, +1.76, -0.95] | `VOUT--C--VSS, VIN--R--VOUT` |
| high_pass | [-0.45, +0.41, -1.09, -0.00] | `VIN--C--VOUT, VOUT--R--VSS` |
| band_pass | [+0.08, +0.13, -2.52, +1.90] | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| band_stop | [+0.26, -0.06, +2.32, +2.09] | `INTERNAL_2--C--VSS, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN, INTERNAL_1--R--VOUT` |
| rlc_series | [+0.02, +0.17, -0.23, +3.03] | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` |
| rlc_parallel | [+2.63, -1.01, -0.18, -0.69] | `VIN--R--VOUT, VOUT--RCL--VSS` |
| lc_lowpass | [-1.12, -0.06, +1.66, -2.14] | `VOUT--C--VSS, VIN--L--VOUT` |
| cl_highpass | [-0.93, -0.20, -1.75, -1.99] | `VIN--C--VOUT, VOUT--L--VSS` |

---

## Reconstruction Accuracy

### By Filter Type (240 circuits each)

| Filter Type | Valid | Example Reconstruction |
|-------------|-------|------------------------|
| low_pass | 240/240 | `VOUT--C--VSS, VIN--R--VOUT` |
| high_pass | 240/240 | `VIN--C--VOUT, VOUT--R--VSS` |
| band_pass | 240/240 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| band_stop | 240/240 | `INTERNAL_2--C--VSS, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN, INTERNAL_1--R--VOUT` |
| rlc_series | 240/240 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` |
| rlc_parallel | 240/240 | `VIN--R--VOUT, VOUT--RCL--VSS` |
| lc_lowpass | 240/240 | `VOUT--C--VSS, VIN--L--VOUT` |
| cl_highpass | 240/240 | `VIN--C--VOUT, VOUT--L--VSS` |

**Total: 1920/1920 (100%) valid reconstructions**

---

## Interpolation

### Low-pass -> High-pass

R and C swap positions at alpha ~ 0.50:

| alpha | Generated |
|---|-----------|
| 0.00 | `VOUT--C--VSS, VIN--R--VOUT` (low-pass) |
| 0.25 | `VOUT--C--VSS, VIN--R--VOUT` |
| 0.50 | `VIN--C--VOUT, VOUT--R--VSS` (transition) |
| 0.75 | `VIN--C--VOUT, VOUT--R--VSS` |
| 1.00 | `VIN--C--VOUT, VOUT--R--VSS` (high-pass) |

### Band-pass -> RLC-parallel

Distributed -> lumped transition:

| alpha | Generated |
|---|-----------|
| 0.00 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` (distributed LC) |
| 0.25 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| 0.50 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` (transition) |
| 0.75 | `VIN--R--VOUT, VOUT--RCL--VSS` |
| 1.00 | `VIN--R--VOUT, VOUT--RCL--VSS` (lumped RCL) |

---

## Novel Topology Generation

### Exploration Results (500 samples)

| Category | Unique Topologies | Samples |
|----------|-------------------|---------|
| Known (in training) | 8 | 498 (99.6%) |
| **Valid novel** | **1** | **1 (0.2%)** |
| Invalid | -- | 1 (0.2%) |

### Known Topology Breakdown

| Topology | Count |
|----------|-------|
| `C(VIN-VOUT)\|R(VOUT-VSS)` | 196 |
| `C(VOUT-VSS)\|R(VIN-VOUT)` | 70 |
| `C(VIN-VOUT)\|L(VOUT-VSS)` | 62 |
| `C(INTERNAL_2-VOUT)\|L(INTERNAL_1-INTERNAL_2)\|R(VOUT-VSS)\|R(INTERNAL_1-VIN)` | 42 |
| `R(VIN-VOUT)\|RCL(VOUT-VSS)` | 41 |
| `C(INTERNAL_1-VOUT)\|L(INTERNAL_1-VIN)\|R(VOUT-VSS)` | 41 |
| `C(VOUT-VSS)\|L(VIN-VOUT)` | 27 |
| `C(INTERNAL_2-VSS)\|L(INTERNAL_1-INTERNAL_2)\|R(VOUT-VSS)\|R(INTERNAL_1-VIN)\|R(INTERNAL_1-VOUT)` | 19 |

### Valid Novel Topologies

| Topology | Count |
|----------|-------|
| `L(VIN-VOUT)\|RCL(VOUT-VSS)` | 1 |

The sequence decoder generates almost exclusively known training topologies (99.6% valid rate, 99.8% known). This reflects the decoder's high-fidelity reconstruction -- it strongly prefers the 8 canonical filter topologies over novel structures.

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
| -6283 + 0j (~1kHz) | [-0.543, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |
| -62832 + 0j (~10kHz) | [-0.685, 0, 0, 0] | band_pass, `R--C`, `R--C` | 3/3 |
| -628318 + 0j (~100kHz) | [-0.828, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |

#### Real Zero, No Pole (RC High-Pass Behavior)

| Zero | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -6283 + 0j (~1kHz) | [0, 0, -0.543, 0] | `R--C`, `R--C`, rlc_parallel | 3/3 |

#### Conjugate Poles (Resonant Behavior)

| Pole | Zero | z[4:8] | Samples | Valid |
|------|------|--------|---------|-------|
| -3142 + 49348j | 0 | [-0.500, +0.670, 0, 0] | `R--C`, band_pass, `R--C` | 3/3 |
| -3142 + 49348j | 0 + 49348j | [-0.500, +0.670, 0, +0.670] | rlc_series, `R--C`, `R--C` | 3/3 |

#### Extreme Frequencies

| Pole | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -100 + 0j (very low) | [-0.286, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |
| -1e6 + 0j (very high) | [-0.857, 0, 0, 0] | rlc_parallel, `R--C`, `R--C` | 3/3 |

### Summary

| Category | Valid/Total |
|----------|------------|
| Real pole, no zero | 9/9 (100%) |
| Real zero, no pole | 3/3 (100%) |
| Conjugate poles | 6/6 (100%) |
| Extreme frequencies | 6/6 (100%) |
| **Overall** | **24/24 (100%)** |

**Observations:**
- 100% valid rate across all pole/zero test cases (up from 83% with adjacency decoder)
- Real-pole-only specs reliably produce RC topologies
- The z[0:4] prior sampling introduces topology diversity (same pole/zero can yield different structures)
- Extreme frequency poles now reliably produce valid topologies

---

## Key Insights

### 1. Sequence Decoder vs Adjacency Decoder

| Metric | Adjacency Decoder | Sequence Decoder |
|--------|-------------------|------------------|
| Reconstruction accuracy | 100% | 100% |
| Valid rate (random samples) | 89.2% | 99.8% |
| P/Z generation valid rate | 83% | 100% |
| Novel topologies (500 samples) | 9 unique, 112 samples | 1 unique, 1 sample |
| Decoder parameters | 7,698,901 | 3,280,726 |

### 2. Q-Factor Drives Topology Selection

| Q Range | Typical Topology |
|---------|------------------|
| Q < 0.1 | 4-node rlc_series |
| Q ~ 0.1 | 3-node band_pass (distributed LC) |
| Q ~ 0.707 | 2-component RC (Butterworth) |
| Q > 2.0 | 2-component RCL parallel (tank) |

### 3. Latent Space is Well-Organized

- All 8 dimensions are meaningful (z[0:4] topology, z[4:8] pole/zero)
- Filter types form distinct clusters
- Interpolation produces smooth transitions
- 99.8% valid rate on random samples (499/500)

### 4. High Fidelity = Few Novel Topologies

The sequence decoder's near-perfect valid rate means it strongly prefers known training topologies. The adjacency decoder's lower fidelity paradoxically produced more novel structures (including some physically meaningful ones). This is a precision-vs-exploration tradeoff.

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
- **Archived model:** `checkpoints/production/best_adjacency.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (1920 circuits, 8 types)
- **Generation script:** `scripts/generation/generate_from_specs.py`
- **Results script:** `scripts/generation/regenerate_all_results.py`
