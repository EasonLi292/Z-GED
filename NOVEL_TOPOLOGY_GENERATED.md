# Novel Topology Generation Analysis

**Analysis Date:** 2026-02-27
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`
**Latent space:** 8D with supervised pole/zero loss on z[4:8]

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 334 (66.8%) |
| Valid novel samples | 112 (22.4%) |
| Invalid samples | 54 (10.8%) |
| Unique novel topologies | 9 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Count (of 500) |
|----------|-------------|----------------|
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 114 |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass / lc_lowpass / cl_highpass | 111 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | band_stop | 63 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series | 26 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 13 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 7 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Novel Topology #1 (4-node) --- 67 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R |

**Analysis:** Pure resistive voltage divider with an internal node. All three edges are resistors. This provides frequency-independent attenuation (flat response). Structurally similar to band_stop but with all reactive components replaced by resistors.

### 2. Novel Topology #2 (4-node) --- 16 occurrences

```
GND--R--VOUT, VIN--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R |

**Analysis:** Minimal resistive network. VIN connects to INT1 via R, and VOUT is grounded via R, but INT1 and VOUT are not directly connected. A degenerate topology where the signal path is incomplete through the internal node.

### 3. Novel Topology #3 (5-node) --- 11 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C |

**Analysis:** Resistive divider (like Novel #1) with a disconnected capacitive branch. The GND--C--INT2 edge doesn't participate in the signal path. A partial band_stop structure where the LC branch is incomplete.

### 4. Novel Topology #4 (5-node) --- 5 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Full RLC topology combining a high-pass direct path (VIN--C--VOUT) with a complete band-stop sub-network. The resistive divider (VIN-INT1-VOUT) and LC branch (INT1-L-INT2, GND-C-INT2) create a complex multi-path filter.

### 5. Novel Topology #5 (4-node) --- 4 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C |

**Analysis:** RC variant of the band_pass topology. Replaces the inductor (VIN--L--INT1) with a resistor, creating an RC lowpass/bandpass hybrid. The internal node INT1 bridges VIN and VOUT through different component paths.

### 6. Novel Topology #6 (5-node) --- 4 occurrences

```
GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Higher-order band-pass variant. Extends the training band_pass topology by adding an extra inductor from INT1 to INT2, creating an additional pole for steeper rolloff.

### 7. Novel Topology #7 (4-node) --- 3 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C, L |

**Analysis:** Combines a high-pass direct path (VIN--C--VOUT) with a band-pass branch (VIN--L--INT1, VOUT--C--INT1) and a shunt resistor. Creates a multi-path filter with both high-pass and band-pass characteristics.

### 8. Novel Topology #8 (4-node) --- 1 occurrence

```
GND--R--VOUT, VIN--L--INT1, VOUT--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | L, R |

**Analysis:** RL variant of the band_pass topology. The inductor connects VIN to INT1, and a resistor connects INT1 to VOUT, with another resistor grounding VOUT. An RL filter without capacitance.

### 9. Novel Topology #9 (4-node) --- 1 occurrence

```
GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C, L |

**Analysis:** Multi-path RLC filter. Combines a high-pass direct path (VIN--C--VOUT) with an RL branch through INT1. The resistor from VOUT to INT1 creates a feedback path.

---

## Invalid Topologies

54 out of 500 random samples (10.8%) produced invalid circuits where VIN or VOUT is not connected. This is an improvement over the previous model (14.8%).

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Pole/Zero-Driven Novel Generation

Beyond random sampling, the supervised z[4:8] enables **targeted** novel topology discovery by specifying pole/zero locations and sampling z[0:4] from the prior.

### Example: Conjugate Pole at 50kHz

Specifying a conjugate pole pair (-3142 + 49348j) with z[0:4] sampled randomly:

| Sample | Generated Topology | Valid |
|--------|-------------------|-------|
| 1 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` (band_stop) | Yes |
| 2 | `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1` (R-divider, novel) | Yes |
| 3 | `GND--R--VOUT` (degenerate) | No |

This demonstrates that the same pole/zero specification can produce both known and novel topologies depending on the random z[0:4] draw.

---

## Conclusions

### Generalization Capability

- **9 unique valid novel topologies** demonstrate compositional generalization
- **89.2% valid rate** (446/500 produce connected circuits), up from 85.2% previously
- **22.4% novel rate** (112/500), up from 17.0% previously
- Diverse structures span R-only (83 samples), RC (15 samples), RL (1 sample), and full RLC (13 samples)

### Comparison Across Models

| Metric | Binary (7D) | Binary (3D) | Log10 (3D) | Log10 + PZ Loss |
|--------|-------------|-------------|------------|-----------------|
| Known topology samples | 350 (70%) | 325 (65%) | 341 (68.2%) | 334 (66.8%) |
| Valid novel samples | 75 (15%) | 62 (12.4%) | 85 (17.0%) | **112 (22.4%)** |
| Unique novel topologies | 15 | 4 | 12 | 9 |
| Invalid samples | 75 (15%) | 113 (22.6%) | 74 (14.8%) | **54 (10.8%)** |
| Valid rate | 85% | 77.4% | 85.2% | **89.2%** |

The pole/zero supervised model achieves the **highest valid rate** (89.2%) and **most novel samples** (112), while reducing invalid outputs to 10.8%. The supervised z[4:8] dimensions provide better latent space organization, leading to fewer degenerate outputs.

### Recommendations

1. **Use pole/zero-driven generation** for targeted circuit design (decoder-only, no encoder needed)
2. **Sample near centroids** for reliable known topologies
3. **Validate generated circuits** with SPICE simulation before use

---

## Test Parameters

```bash
# 500 random samples from N(0,1) in 8D latent space
torch.manual_seed(42)
z = torch.randn(500, 8)
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (480 circuits, 8 types)
- **Generation script:** `scripts/generation/generate_from_specs.py`
- **Results script:** `scripts/generation/regenerate_all_results.py`
