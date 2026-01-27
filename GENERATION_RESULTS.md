# Circuit Generation Results

**Dataset:** 360 circuits (288 train, 72 validation)
**Checkpoint:** `checkpoints/production/best.pt`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Total Loss | 0.96 | 1.03 |
| Node Count Accuracy | 100% | 100% |
| Edge Existence Accuracy | 100% | 100% |
| Component Type Accuracy | 100% | 100% |

---

## Specification-Based Generation

Generate circuits by specifying **cutoff frequency** and **Q-factor**:

```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

### Standard Examples

| Cutoff | Q | Generated Circuit | Status |
|--------|---|-------------------|--------|
| 1 kHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 10 kHz | 0.707 | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |
| 100 kHz | 0.707 | `GND--C--VOUT, VIN--R--VOUT` | Valid |
| 10 kHz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |

### Edge Cases

| Cutoff | Q | Generated Circuit | Analysis |
|--------|---|-------------------|----------|
| 1 Hz | 0.707 | `GND--C--VOUT, VIN--R--VOUT` | Extrapolates beyond training (min: 9 Hz) |
| 1 MHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (max: 540 kHz) |
| 10 kHz | 0.01 | `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | Low Q → notch filter |
| 10 kHz | 0.1 | `GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2` | Low Q → higher-order filter |
| 10 kHz | 2.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Moderate Q → tank circuit |
| 50 Hz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Low freq + high Q → RLC parallel |
| 500 kHz | 0.1 | `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | High freq + low Q → RLC series |

### Training Data Coverage

**Overall Range:**

| Parameter | Min | Max |
|-----------|-----|-----|
| Cutoff frequency | 8.77 Hz | 539,651 Hz |
| Q-factor | 0.01 | 6.50 |

**By Filter Type:**

| Filter Type | Cutoff Range (Hz) | Q-factor Range |
|-------------|-------------------|----------------|
| low_pass | 15.0 - 194,302 | 0.707 (fixed) |
| high_pass | 8.8 - 539,651 | 0.707 (fixed) |
| band_pass | 1,959 - 429,727 | 0.01 - 6.22 |
| band_stop | 4,447 - 402,533 | 0.01 (fixed) |
| rlc_series | 2,090 - 358,219 | 0.01 - 6.01 |
| rlc_parallel | 2,060 - 263,391 | 0.13 - 6.50 |

**Notes:**
- Simple RC filters (low_pass, high_pass) have fixed Q=0.707 (Butterworth response)
- Band-stop filters have fixed Q=0.01 (wide notch)
- RLC circuits have variable Q depending on component ratios

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0] | z[1] | z[2] | z[3] | Generated from Centroid |
|-------------|------|------|------|------|-------------------------|
| low_pass | +0.65 | -3.23 | -2.89 | +1.63 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | +0.28 | -3.82 | -1.80 | -1.29 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | +3.22 | +1.45 | +2.62 | -2.16 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | -2.96 | +1.00 | +0.52 | +2.44 | `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` |
| rlc_series | -4.02 | +1.17 | +0.93 | +0.05 | `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` |
| rlc_parallel | +0.62 | -1.80 | -4.04 | -0.83 | `GND--RCL--VOUT, VIN--R--VOUT` |

**Observations:**
- z[0] separates band_pass (+3.22) from rlc_series (-4.02) and band_stop (-2.96)
- z[1] separates low_pass/high_pass/rlc_parallel (negative) from band_pass/rlc_series/band_stop (positive)
- z[2] separates rlc_parallel (-4.04) and low_pass (-2.89) from band_pass (+2.62)
- z[3] separates band_stop (+2.44) from band_pass (-2.16) and high_pass (-1.29)

---

## Reconstruction Accuracy

### By Filter Type (60 circuits each)

| Filter Type | Valid | Example Reconstruction |
|-------------|-------|------------------------|
| low_pass | 60/60 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | 60/60 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | 60/60 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | 60/60 | `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` |
| rlc_series | 60/60 | `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` |
| rlc_parallel | 60/60 | `GND--RCL--VOUT, VIN--R--VOUT` |

**Total: 360/360 (100%) valid reconstructions**

---

## Interpolation

### Low-pass → High-pass

R and C swap positions at α ≈ 0.75:

| α | Generated |
|---|-----------|
| 0.00 | `GND--C--VOUT, VIN--R--VOUT` (low-pass) |
| 0.25 | `GND--C--VOUT, VIN--R--VOUT` |
| 0.50 | `GND--C--VOUT, VIN--R--VOUT` (transition) |
| 0.75 | `GND--R--VOUT, VIN--C--VOUT` |
| 1.00 | `GND--R--VOUT, VIN--C--VOUT` (high-pass) |

### Band-pass → RLC-parallel

Distributed → lumped transition:

| α | Generated |
|---|-----------|
| 0.00 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` (distributed LC) |
| 0.25 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| 0.50 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` (transition) |
| 0.75 | `GND--RCL--VOUT, VIN--R--VOUT` |
| 1.00 | `GND--RCL--VOUT, VIN--R--VOUT` (lumped RCL) |

---

## Novel Topology Generation

The model can generate **novel topologies not seen in training** through latent space sampling.

### Exploration Results (500 samples)

| Category | Unique Topologies | Samples |
|----------|-------------------|---------|
| Known (in training) | 6 | 385 (77%) |
| **Valid novel** | **17** | **69 (14%)** |
| Invalid (disconnected) | - | 46 (9%) |

### Top Valid Novel Topologies Discovered

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2` | 14 | 5 | R, L, C |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2` | 11 | 5 | R, L, C |
| `GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1` | 7 | 5 | R, L, C |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1` | 7 | 5 | R, L, C |
| `GND--RCL--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1` | 6 | 4 | R, L, C, RCL |

### Generalization Capability

**Strengths:**
- Compositional generalization (recombines learned components)
- 17 unique valid novel topologies discovered
- Maintains graph connectivity in 91% of samples
- Novel topologies are predominantly higher-order RLC filters (5-node)

**Limitations:**
- Most samples (77%) reproduce training topologies
- Novel circuits are variations, not fundamentally new architectures
- No guarantee of electrical validity (only structural validity)

---

## Key Insights

### 1. Q-Factor Drives Topology Selection

| Q Range | Typical Topology |
|---------|------------------|
| Q < 0.1 | 5-node band-stop (notch filter) |
| Q ≈ 0.5 | 4-node series RLC |
| Q ≈ 0.707 | 3-node RC (Butterworth) |
| Q > 2.0 | 3-node RLC parallel (tank) |

### 2. Frequency Has Less Structural Impact

Within training range, frequency mainly affects K-NN neighbor selection but not topology.

### 3. Graceful Extrapolation

Specs beyond training distribution still produce valid circuits by finding nearest neighbors.

### 4. Latent Space is Well-Organized

- All 4 active dimensions (z[0:4]) contribute to filter type separation
- Filter types form distinct clusters
- Interpolation produces smooth transitions

---

## Usage

```bash
# Spec-based generation
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707

# With GED weighting
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 5.0 --ged-weight 0.5

# Multiple samples
python scripts/generation/generate_from_specs.py --cutoff 5000 --q-factor 2.0 --num-samples 5

# Filter type interpolation
python scripts/generation/interpolate_filter_types.py --from low_pass --to high_pass --steps 5
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (360 circuits)
- **GED Matrix:** `analysis_results/ged_matrix_360.npy`
- **Generation script:** `scripts/generation/generate_from_specs.py`
- **Interpolation script:** `scripts/generation/interpolate_filter_types.py`
