# Circuit Generation Results

**Dataset:** 360 circuits (288 train, 72 validation)
**Checkpoint:** `checkpoints/production/best.pt`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Total Loss | 0.98 | 1.03 |
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
| 1 Hz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (min: 8.77 Hz) |
| 1 MHz | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (max: 539,651 Hz) |
| 10 kHz | 0.01 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Low Q |
| 10 kHz | 0.1 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Within training range |
| 10 kHz | 2.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Within training range |
| 50 Hz | 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | High Q |
| 500 kHz | 0.1 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Within training range |

### Training Data Coverage

**Overall Range:**

| Parameter | Min | Max |
|-----------|-----|-----|
| Cutoff frequency | 8.77 Hz | 539,651 Hz |
| Q-factor | 0.01 | 6.5 |

**By Filter Type:**

| Filter Type | Cutoff Range (Hz) | Q-factor Range |
|-------------|-------------------|----------------|
| low_pass | 15 - 194,302 | 0.707 |
| high_pass | 8.77 - 539,651 | 0.707 |
| band_pass | 1,959 - 429,727 | 0.01 - 6.2 |
| band_stop | 4,447 - 402,533 | 0.01 |
| rlc_series | 2,090 - 358,219 | 0.01 - 6.0 |
| rlc_parallel | 2,060 - 263,391 | 0.128 - 6.5 |

**Notes:**
- Simple RC filters (low_pass, high_pass) have fixed Q=0.707 (Butterworth response)
- Band-stop filters have fixed Q=0.01 (wide notch)
- RLC circuits have variable Q depending on component ratios

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0] | z[1] | z[2] | z[3] | Generated from Centroid |
|-------------|------|------|------|------|-------------------------|
| low_pass | +1.26 | -2.83 | -3.23 | +1.70 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | +0.89 | -4.19 | -1.61 | -0.64 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | +2.77 | +1.61 | +2.63 | -2.23 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | -2.87 | +0.55 | +0.17 | +2.45 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` |
| rlc_series | -4.29 | +0.82 | +0.51 | -0.21 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` |
| rlc_parallel | +1.30 | -2.26 | -3.50 | -0.76 | `GND--RCL--VOUT, VIN--R--VOUT` |

**Observations:**
- z[0] separates band_pass (+2.77) from rlc_series (-4.29) and band_stop (-2.87)
- z[1] separates low_pass/high_pass/rlc_parallel (negative) from band_pass/rlc_series/band_stop (positive)
- z[2] separates rlc_parallel (-3.50) and low_pass (-3.23) from band_pass (+2.63)
- z[3] separates band_stop (+2.45) from band_pass (-2.23) and high_pass (-0.64)

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
| 0.50 | `GND--R--VOUT` (transition) |
| 0.75 | `GND--RCL--VOUT, VIN--R--VOUT` |
| 1.00 | `GND--RCL--VOUT, VIN--R--VOUT` (lumped RCL) |

---

## Novel Topology Generation

The model can generate **novel topologies not seen in training** through latent space sampling.

### Exploration Results (500 samples)

| Category | Unique Topologies | Samples |
|----------|-------------------|---------|
| Known (in training) | 6 | 350 (70%) |
| **Valid novel** | **15** | **75 (15%)** |
| Invalid (disconnected) | - | 75 (15%) |

### Top Valid Novel Topologies Discovered

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1` | 21 | 4 | R |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2` | 17 | 5 | R, L, C |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2` | 7 | 5 | R, C |
| `GND--C--VOUT, VIN--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2` | 6 | 5 | R, L, C |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1` | 4 | 5 | R, C |

### Generalization Capability

**Strengths:**
- Compositional generalization (recombines learned components)
- 15 unique valid novel topologies discovered
- Maintains graph connectivity in 85% of samples
- Novel topologies are predominantly higher-order filters

**Limitations:**
- Most samples (70%) reproduce training topologies
- Novel circuits are variations, not fundamentally new architectures
- 75 / 500 (15%) invalid generations when sampling randomly

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
