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
| Component Type Accuracy | 99% | 100% |

---

## Specification-Based Generation

Generate circuits by specifying **cutoff frequency** and **Q-factor**:

```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

### Standard Examples

| Cutoff | Q | Nearest Match | Generated Circuit | Analysis |
|--------|---|---------------|-------------------|----------|
| 1 kHz | 0.707 | high_pass | `GND--R--VOUT, VIN--C--VOUT` | Simple RC high-pass |
| 10 kHz | 0.707 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Parallel tank circuit |
| 100 kHz | 0.707 | low_pass | `GND--C--VOUT, VIN--R--VOUT` | Simple RC low-pass |
| 10 kHz | 5.0 | rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | High-Q resonant |

### Edge Cases and Outliers

#### Extreme Frequencies

| Cutoff | Q | Generated | Analysis |
|--------|---|-----------|----------|
| **1 Hz** | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Extrapolates to RC high-pass (nearest training: 9 Hz) |
| **10 Hz** | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Low-frequency region dominated by high-pass |
| **1 MHz** | 0.707 | `GND--R--VOUT, VIN--C--VOUT` | Beyond training max (540 kHz), defaults to high-pass |

#### Extreme Q Values

| Cutoff | Q | Generated | Analysis |
|--------|---|-----------|----------|
| 10 kHz | **0.01** | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT1, INT1--L--INT2` | Matches band_stop (Q≈0.01), complex 5-node notch |
| 10 kHz | **0.1** | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Series LC with load resistor |
| 10 kHz | **10.0** | `GND--RCL--VOUT, VIN--R--VOUT` | High-Q parallel tank |
| 10 kHz | **20.0** | `GND--RCL--VOUT, VIN--R--VOUT` | Beyond training max (Q=6.5), still produces valid RLC parallel |

#### Hybrid/Boundary Regions

| Cutoff | Q | Nearest Types | Generated | Analysis |
|--------|---|---------------|-----------|----------|
| 10 kHz | **1.0** | band_pass, rlc_parallel | `GND--R--VOUT, VIN--C--VOUT` | Critical damping boundary - simple RC |
| 10 kHz | **1.5** | rlc_parallel, rlc_series | `GND--R--VOUT` | Transition region - minimal output |
| 5 kHz | **0.5** | high_pass, rlc_parallel, low_pass | `GND--C--VOUT, VIN--R--VOUT` | Mixed neighbors → RC low-pass |

#### Unusual Combinations

| Cutoff | Q | Generated | Analysis |
|--------|---|-----------|----------|
| **50 Hz** + 5.0 | `GND--RCL--VOUT, VIN--R--VOUT` | Low freq but high Q → RLC parallel (Q dominates) |
| **500 kHz** + 0.1 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1` | High freq + low Q → series RLC |
| **200 kHz** + 3.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | High freq + moderate Q → band-pass |

### Training Data Coverage

| Parameter | Range | Notes |
|-----------|-------|-------|
| Cutoff frequency | 9 Hz - 540 kHz | Log-uniform distribution |
| Q-factor | 0.01 - 6.5 | Varies by filter type |

**Q-Factor by Filter Type:**

| Filter Type | Q Range | Typical Use |
|-------------|---------|-------------|
| low_pass | 0.707 (fixed) | Butterworth response |
| high_pass | 0.707 (fixed) | Butterworth response |
| band_pass | 0.01 - 6.2 | Narrowband selection |
| band_stop | 0.01 (fixed) | Notch filter |
| rlc_series | 0.01 - 6.0 | Series resonance |
| rlc_parallel | 0.13 - 6.5 | Parallel tank circuits |

---

## Latent Space Geometry

The 8D latent space has interpretable structure. The first two dimensions encode topology:

### Filter Type Centroids

| Filter Type | z[0] | z[1] | Generated from Centroid |
|-------------|------|------|-------------------------|
| low_pass | -0.64 | -3.96 | `GND--C--VOUT, VIN--R--VOUT` |
| high_pass | +0.50 | -3.75 | `GND--R--VOUT, VIN--C--VOUT` |
| band_pass | +3.23 | +0.85 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` |
| band_stop | -3.03 | +1.91 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT1, INT1--L--INT2` |
| rlc_series | -1.28 | +3.23 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1` |
| rlc_parallel | -1.80 | -3.52 | `GND--RCL--VOUT, VIN--R--VOUT` |

**Observation:** z[1] roughly separates simple 3-node circuits (negative) from complex 4-5 node circuits (positive).

### Principal Axis Exploration

**z[0] axis (topology complexity):**

| z[0] | Generated | Interpretation |
|------|-----------|----------------|
| -3.0 | `GND--R--VOUT, GND--C--INT1, INT1--L--INT1` | Complex with LC trap |
| -1.5 | `GND--R--VOUT, VIN--R--VOUT` | Simple resistive |
| 0.0 | `GND--R--VOUT` | Minimal |
| +1.5 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass-like |
| +3.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass dominant |

**z[1] axis (node count):**

| z[1] | Generated | Interpretation |
|------|-----------|----------------|
| -4.0 | `GND--C--VOUT, VIN--R--VOUT` | Simple 3-node RC |
| -2.0 | `GND--R--VOUT, VIN--R--VOUT` | 3-node resistive |
| 0.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, ...` | 5-node with internals |
| +2.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1` | 4-node series RLC |
| +3.0 | `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1` | 4-node series RLC |

### Inter-Cluster Interpolation

**Low-pass ↔ High-pass:** R and C swap positions at α=0.5
```
α=0.0: GND--C--VOUT, VIN--R--VOUT  (low-pass)
α=0.5: GND--R--VOUT, VIN--C--VOUT  (transition)
α=1.0: GND--R--VOUT, VIN--C--VOUT  (high-pass)
```

**Band-pass ↔ RLC-parallel:** Distributed → lumped transition
```
α=0.0: GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1  (distributed LC)
α=0.5: GND--R--VOUT                               (simplified)
α=1.0: GND--RCL--VOUT, VIN--R--VOUT               (lumped RCL)
```

### Extrapolation Behavior

Moving 2× beyond cluster centroids still produces valid circuits:

| Direction | Generated | Stability |
|-----------|-----------|-----------|
| 2× low_pass | `GND--C--VOUT, VIN--R--VOUT` | Stable |
| 2× high_pass | `GND--R--VOUT, VIN--C--VOUT` | Stable |
| 2× rlc_parallel | `GND--RCL--VOUT, VIN--R--VOUT` | Stable |

### Hybrid Regions

**Midpoint of 3 filter types:**

| Midpoint | Generated | Analysis |
|----------|-----------|----------|
| (low_pass + high_pass + band_pass) / 3 | `GND--R--VOUT, VIN--C--VOUT` | Defaults to simple RC |
| (band_stop + rlc_series + rlc_parallel) / 3 | `GND--R--VOUT, VIN--R--INT1, ...` | Maintains complex topology |

### Corner Cases

| Latent | Generated | Analysis |
|--------|-----------|----------|
| All z = +2.0 | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Band-pass-like |
| All z = -2.0 | `GND--R--VOUT, VIN--R--VOUT` | Simple resistive |
| z = [5, -5, 0...] | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Extreme topology dims |
| z = [0, 0, 0...] | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | Zero latent → band-pass |

---

## Reconstruction by Filter Type

### Low-Pass (60 circuits)

All correctly generate: `GND--C--VOUT, VIN--R--VOUT`

| Index | Frequency | Reconstruction |
|-------|-----------|----------------|
| 0 | 3,049 Hz | `GND--C--VOUT, VIN--R--VOUT` ✓ |
| 1 | 275 Hz | `GND--C--VOUT, VIN--R--VOUT` ✓ |
| 2 | 5,708 Hz | `GND--C--VOUT, VIN--R--VOUT` ✓ |

### High-Pass (60 circuits)

All correctly generate: `GND--R--VOUT, VIN--C--VOUT`

| Index | Frequency | Reconstruction |
|-------|-----------|----------------|
| 60 | 18,256 Hz | `GND--R--VOUT, VIN--C--VOUT` ✓ |
| 61 | 3,271 Hz | `GND--R--VOUT, VIN--C--VOUT` ✓ |

### Band-Pass (60 circuits)

All correctly generate: `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1`

| Index | Frequency | Reconstruction |
|-------|-----------|----------------|
| 120 | 157,646 Hz | `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` ✓ |

### Band-Stop (60 circuits)

All correctly generate 5-node notch: `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT1, INT1--L--INT2`

### RLC-Series (60 circuits)

All correctly generate: `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT1`

### RLC-Parallel (60 circuits)

All correctly generate: `GND--RCL--VOUT, VIN--R--VOUT`

---

## Key Insights

### 1. Q-Factor Drives Topology Selection

| Q Range | Typical Topology | Explanation |
|---------|------------------|-------------|
| Q < 0.1 | 5-node band-stop | Very overdamped → notch filter |
| Q ≈ 0.5 | Series RLC (4-node) | Moderately damped |
| Q ≈ 0.707 | Simple RC (3-node) | Butterworth standard |
| Q > 2.0 | RLC parallel (3-node) | High selectivity tank |

### 2. Frequency Has Less Structural Impact

Within training range, frequency mainly affects K-NN neighbor selection but not topology. A 100 Hz and 100 kHz circuit with same Q produce similar structures.

### 3. Graceful Extrapolation

Specs beyond training distribution still produce valid circuits by finding nearest neighbors. However, extreme values may produce simplified outputs.

### 4. Latent Space is Well-Organized

- z[0:2] encodes topology (interpretable)
- Filter types form distinct clusters
- Interpolation produces smooth transitions
- Extrapolation remains stable

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

**Most common novel topology (53x):**
```
GND--R--VOUT, VIN--R--INT1, INT1--L--INT2, VOUT--C--INT2
```
A 4-node series R-L-C filter with different arrangement than training data.

**All valid novel topologies:**

| Topology | Count | Nodes | Edges | Components |
|----------|-------|-------|-------|------------|
| `GND--R--VOUT, VIN--R--INT1, INT1--L--INT2, VOUT--C--INT2` | 53 | 5 | 4 | RLC |
| `GND--R--VOUT, VIN--R--VOUT` | 18 | 3 | 2 | R |
| `GND--R--VOUT, GND--C--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 10 | 5 | 6 | RLC |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 9 | 5 | 5 | RLC |
| `GND--R--VOUT, VIN--R--INT1` | 8 | 5 | 2 | R |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 4 | 5 | 5 | RLC |
| `GND--R--VOUT, VIN--L--INT1` | 4 | 4 | 2 | RL |
| `GND--R--VOUT, GND--C--INT2, VIN--R--INT1, VOUT--R--INT1, VOUT--C--INT2, INT1--L--INT2` | 4 | 5 | 6 | RLC |
| `GND--R--VOUT, GND--C--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 3 | 5 | 6 | RLC |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT2, INT1--L--INT2` | 3 | 5 | 4 | RLC |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1` | 3 | 5 | 3 | R |
| `GND--R--VOUT, GND--C--INT2, VIN--R--INT1, VIN--C--INT2, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 2 | 5 | 7 | RLC |
| `GND--R--VOUT, GND--C--INT2, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | 2 | 5 | 5 | RLC |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1` | 2 | 5 | 3 | RC |

### How Novel Topologies Emerge

1. **Interpolation regions**: Midpoints between filter type clusters produce hybrid structures
2. **Component recombination**: Model combines learned R, L, C patterns in new arrangements
3. **Edge variations**: Same nodes connected with different component types

### Invalid Novel Topologies

Some samples produce degenerate circuits:

| Topology | Count | Issue |
|----------|-------|-------|
| `GND--R--VOUT` | 89 | VIN disconnected |
| `GND--RCL--VOUT` | 6 | VIN disconnected |
| `GND--R--VOUT, GND--C--INT2, INT1--L--INT2` | 3 | VIN disconnected |

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

## Usage

```python
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
- **Generation script:** `scripts/generation/generate_from_specs.py`
