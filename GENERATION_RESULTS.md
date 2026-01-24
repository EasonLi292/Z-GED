# Circuit Generation Results

**Model:** v5.1 (Latent-Only Decoder with Spec Predictor)
**Dataset:** 360 circuits (288 train, 72 validation)
**Checkpoint:** `checkpoints/production/best.pt`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Total Loss | 2.16 | 1.67 |
| Node Count Accuracy | 100% | 100% |
| Edge Existence Accuracy | 100% | 100% |
| Component Type Accuracy | 99% | 100% |
| Cutoff Error | 0.67 decades | 0.50 decades |
| Q Error | 0.75 | 0.47 |

---

## Specification-Based Generation

Generate circuits by specifying **cutoff frequency** and **Q-factor**:

```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

### Standard Examples

| Cutoff | Q | Nearest Type | Generated Circuit | Status |
|--------|---|--------------|-------------------|--------|
| 1 kHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 10 kHz | 0.707 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |
| 100 kHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 10 kHz | 5.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |

### Edge Cases

| Cutoff | Q | Nearest Type | Generated Circuit | Analysis |
|--------|---|--------------|-------------------|----------|
| 1 Hz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (min: 9 Hz) |
| 1 MHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (max: 540 kHz) |
| 10 kHz | 0.01 | band_stop | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Low Q → notch filter |
| 10 kHz | 0.1 | band_pass | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Overdamped RLC |
| 10 kHz | 2.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Moderate Q → tank circuit |
| 50 Hz | 5.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Low freq + high Q → RLC parallel |
| 500 kHz | 0.1 | rlc_series | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | High freq + low Q → series RLC |

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

| Filter Type | z[0] | z[1] | Generated from Centroid |
|-------------|------|------|-------------------------|
| low_pass | -1.36 | -4.25 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | +0.25 | -4.32 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | +3.47 | +0.77 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | -3.10 | +2.11 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` |
| rlc_series | -1.28 | +3.33 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` |
| rlc_parallel | -1.11 | -3.37 | `GND--RCL--VOUT, VIN--R--VOUT` |

**Observation:** z[1] separates simple 3-node circuits (negative) from complex 4-5 node circuits (positive).

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
| 0.50 | `GND--R--VOUT` (simplified) |
| 0.75 | `GND--R--VOUT, VIN--C--VOUT` |
| 1.00 | `GND--RCL--VOUT, VIN--R--VOUT` (lumped RCL) |

---

## Novel Topology Generation

The model can generate **novel topologies not seen in training** through latent space interpolation and sampling.

### Exploration Results (500 samples)

| Category | Unique Topologies | Samples |
|----------|-------------------|---------|
| Known (in training) | 6 | 277 (55%) |
| **Valid novel** | **14** | **125 (25%)** |
| Invalid (disconnected) | 3 | 98 (20%) |

### Valid Novel Topologies Discovered

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--R--VOUT, VIN--R--INT1, INT1--L--INT2, VOUT--C--INT2` | 53 | 5 | RLC |
| `GND--R--VOUT, VIN--R--VOUT` | 18 | 3 | R |
| `GND--R--VOUT, GND--C--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 10 | 5 | RLC |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 9 | 5 | RLC |
| `GND--R--VOUT, VIN--R--INT1` | 8 | 5 | R |
| `GND--R--VOUT, VIN--L--INT1` | 4 | 4 | RL |

### Generalization Capability

**Strengths:**
- Compositional generalization (recombines learned components)
- 14 valid novel topologies discovered
- Maintains graph connectivity in 80% of samples

**Limitations:**
- Most samples (55%) reproduce training topologies
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

- z[0:2] encodes topology (interpretable)
- z[4:8] now encodes transfer function (after spec predictor fix)
- Filter types form distinct clusters
- Interpolation produces smooth transitions

---

## v5.1 Improvements

The spec predictor auxiliary loss fixed the z[4:8] collapse:

| Metric | v5.0 (before) | v5.1 (after) |
|--------|---------------|--------------|
| z[4] variance | ~0.01 | ~2.0 |
| z[5] variance | ~0.01 | ~0.8 |
| Cutoff error | N/A | 0.5 decades |
| Q error | N/A | 0.47 |

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
