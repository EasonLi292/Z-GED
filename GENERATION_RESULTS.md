# Circuit Generation Results

**Model:** v5.0 (Latent-Only Decoder)
**Dataset:** 360 circuits (288 train, 72 validation)
**Checkpoint:** `checkpoints/production/best.pt`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Total Loss | 0.92 | 1.00 |
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

| Cutoff | Q | Nearest Type | Generated Circuit | Status |
|--------|---|--------------|-------------------|--------|
| 1 kHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Valid |
| 10 kHz | 0.707 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |
| 100 kHz | 0.707 | low_pass | `GND--C--VOUT, VIN--R--VOUT` | Valid |
| 10 kHz | 5.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Valid |

### Edge Cases

| Cutoff | Q | Nearest Type | Generated Circuit | Analysis |
|--------|---|--------------|-------------------|----------|
| 1 Hz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (min: 9 Hz) |
| 1 MHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates beyond training (max: 540 kHz) |
| 10 kHz | 0.01 | band_stop | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Low Q → notch filter |
| 10 kHz | 0.1 | rlc_series | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Overdamped RLC |
| 10 kHz | 2.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Moderate Q → tank circuit |
| 50 Hz | 5.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Low freq + high Q → RLC parallel |
| 500 kHz | 0.1 | band_pass | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | High freq + low Q → series RLC |

### Training Data Coverage

| Parameter | Range |
|-----------|-------|
| Cutoff frequency | 9 Hz - 540 kHz |
| Q-factor | 0.01 - 6.5 |

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0] | z[1] | Generated from Centroid |
|-------------|------|------|-------------------------|
| low_pass | -0.64 | -3.96 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | +0.50 | -3.75 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | +3.23 | +0.85 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | -3.03 | +1.91 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` |
| rlc_series | -1.28 | +3.23 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` |
| rlc_parallel | -1.80 | -3.52 | `GND--RCL--VOUT, VIN--R--VOUT` |

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

## Latent Space Exploration

### z[0] Axis (topology complexity)

| z[0] | Generated | Interpretation |
|------|-----------|----------------|
| -3.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Band-stop (5-node) |
| -1.5 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | Band-stop (5-node) |
| 0.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass (4-node) |
| +1.5 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass (4-node) |
| +3.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass (4-node) |

### z[1] Axis (node count)

| z[1] | Generated | Interpretation |
|------|-----------|----------------|
| -4.0 | `GND--R--VOUT, VIN--C--VOUT` | Simple 3-node (high-pass) |
| -2.0 | `GND--R--VOUT, VIN--C--VOUT` | Simple 3-node (high-pass) |
| 0.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | 4-node (band-pass) |
| +2.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | 4-node (RLC series) |
| +4.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | 4-node (RLC series) |

---

## Interpolation

### Low-pass → High-pass

R and C swap positions at α ≈ 0.5:

| α | Generated |
|---|-----------|
| 0.00 | `GND--C--VOUT, VIN--R--VOUT` (low-pass) |
| 0.25 | `GND--C--VOUT, VIN--R--VOUT` |
| 0.50 | `GND--R--VOUT, VIN--C--VOUT` (transition) |
| 0.75 | `GND--R--VOUT, VIN--C--VOUT` |
| 1.00 | `GND--R--VOUT, VIN--C--VOUT` (high-pass) |

### Band-pass → RLC-parallel

Distributed → lumped transition:

| α | Generated |
|---|-----------|
| 0.00 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` (distributed LC) |
| 0.25 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| 0.50 | `GND--R--VOUT` (simplified) |
| 0.75 | `GND--C--VOUT, VIN--R--VOUT` |
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
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (360 circuits)
- **GED Matrix:** `analysis_results/ged_matrix_360.npy`
- **Generation script:** `scripts/generation/generate_from_specs.py`
