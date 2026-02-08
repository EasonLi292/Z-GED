# Novel Topology Generation Analysis

**Analysis Date:** 2026-02-05
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 341 (68.2%) |
| Valid novel samples | 85 (17.0%) |
| Invalid samples | 74 (14.8%) |
| Unique novel topologies | 12 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Count (of 500) |
|----------|-------------|----------------|
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 120 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 88 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series | 46 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | band_stop | 42 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 39 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 6 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Novel Topology #1 (4-node) --- 22 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C |

**Analysis:** RC variant of the band_pass topology. Replaces the inductor (VIN--L--INT1) with a resistor, creating an RC lowpass/bandpass hybrid. The internal node INT1 bridges VIN and VOUT through different component paths.

### 2. Novel Topology #2 (4-node) --- 20 occurrences

```
GND--R--VOUT, VIN--R--INT1
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R |

**Analysis:** Minimal resistive network. VIN connects to INT1 via R, and VOUT is grounded via R, but INT1 and VOUT are not directly connected. This is a degenerate topology where signal path is incomplete through the internal node.

### 3. Novel Topology #3 (4-node) --- 12 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
VOUT--R--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R |

**Analysis:** Pure resistive voltage divider with an internal node. All three edges are resistors. This provides frequency-independent attenuation (flat response). Structurally similar to band_stop but with all reactive components replaced by resistors.

### 4. Novel Topology #4 (5-node) --- 7 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
VOUT--C--INT1
INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Higher-order variant of Novel #1. Extends the RC topology by adding an inductor from INT1 to INT2, creating an additional resonant pole. This is a hybrid between the band_pass and rlc_series training topologies.

### 5. Novel Topology #5 (5-node) --- 6 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
VOUT--R--INT1
GND--C--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C |

**Analysis:** Resistive divider (like Novel #3) with a disconnected capacitive branch. The GND--C--INT2 edge doesn't participate in the signal path.

### 6. Novel Topology #6 (5-node) --- 5 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--C--VOUT
VIN--R--INT1
VOUT--R--INT1
GND--C--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C |

**Analysis:** Combines a direct high-pass path (VIN--C--VOUT) with a resistive divider through INT1. The GND--C--INT2 branch adds a grounded capacitor on a separate internal node. Multi-path filter structure.

### 7. Novel Topology #7 (4-node) --- 4 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1
```

**Structure:**
```
GND--R--VOUT
VIN--C--VOUT
VIN--L--INT1
VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C, L |

**Analysis:** Combines a high-pass direct path (VIN--C--VOUT) with a band-pass branch (VIN--L--INT1, VOUT--C--INT1) and a shunt resistor. Creates a multi-path filter with both high-pass and band-pass characteristics.

### 8. Novel Topology #8 (5-node) --- 3 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--C--VOUT
VIN--R--INT1
VOUT--R--INT1
GND--C--INT2
INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Full RLC topology combining a high-pass direct path with a complete band-stop sub-network. The resistive divider (VIN-INT1-VOUT) and LC branch (INT1-L-INT2, GND-C-INT2) create a complex multi-path filter.

### 9. Novel Topology #9 (4-node) --- 2 occurrences

```
GND--C--VOUT, VIN--R--VOUT, VIN--L--INT1, VOUT--C--INT1
```

**Structure:**
```
GND--C--VOUT
VIN--R--VOUT
VIN--L--INT1
VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C, L |

**Analysis:** Low-pass core (GND--C--VOUT, VIN--R--VOUT) augmented with a band-pass branch (VIN--L--INT1, VOUT--C--INT1). Combines low-pass and band-pass characteristics.

### 10. Novel Topology #10 (5-node) --- 2 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, GND--C--INT2, INT1--L--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--R--INT1
VOUT--C--INT1
GND--C--INT2
INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Full RLC topology. Combines Novel #1 (GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1) with an LC branch to ground (INT1--L--INT2, GND--C--INT2). Creates a series resonant trap from INT1.

### 11. Novel Topology #11 (4-node) --- 1 occurrence

```
GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1, VOUT--C--INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 |
| Components | R, C |

**Analysis:** High-pass direct path (VIN--C--VOUT) with RC branches through INT1. Dual-path RC filter.

### 12. Novel Topology #12 (5-node) --- 1 occurrence

```
GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, C, L |

**Analysis:** Higher-order band-pass variant. Extends the training band_pass topology by adding an extra inductor from INT1 to INT2, creating an additional pole for steeper rolloff.

---

## Invalid Topologies

74 out of 500 random samples (14.8%) produced invalid circuits where VIN or VOUT is not connected. These occur when random latent codes fall in regions between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **12 unique valid novel topologies** demonstrate strong compositional generalization
- Diverse novel structures span R-only (12+20 samples), RC (22+6+5+1 samples), and full RLC (7+4+3+2+2+1 samples)
- 426/500 (85.2%) of random samples produce valid connected circuits

### Comparison Across Edge Feature Models

| Metric | Binary (7D) | Binary (3D) | Log10 Values (3D) |
|--------|-------------|-------------|-------------------|
| Known topology samples | 350 (70%) | 325 (65%) | 341 (68.2%) |
| Valid novel samples | 75 (15%) | 62 (12.4%) | 85 (17.0%) |
| Unique novel topologies | 15 | 4 | 12 |
| Invalid samples | 75 (15%) | 113 (22.6%) | 74 (14.8%) |
| Valid rate | 85% | 77.4% | 85.2% |

The log10 value model recovers the generalization diversity lost when simplifying from 7D to 3D binary features. It achieves the highest valid rate (85.2%) while producing 12 unique novel topologies with a smooth distribution (no single dominant topology, unlike the 3D binary model where 93.5% of novel samples were one topology).

### Recommendations

1. **Validate generated circuits** with SPICE simulation before use
2. **Sample near centroids** for reliable generation (use filter type centroids as starting points)
3. **Use nearest-neighbor method** (`--method nearest`) for critical applications to stay closer to training data

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
- **Dataset:** `rlc_dataset/filter_dataset.pkl`
- **Generation script:** `scripts/generation/generate_from_specs.py`
