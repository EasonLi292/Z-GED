# Novel Topology Generation Analysis

**Model:** v5.1 (Node-Embedding Encoder)
**Analysis Date:** 2026-01-26

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 326 (65%) |
| Valid novel samples | 97 (19%) |
| Invalid samples | 77 (15%) |
| Unique novel topologies | 18 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type |
|----------|-------------|
| `GND--C--VOUT, VIN--R--VOUT` | low_pass |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | band_stop |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | rlc_series |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Double-C Band-Pass Variant (5-node) — 17 occurrences

```
GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2
```

**Structure:**
```
VIN ──L── INT1 ──L── INT2
           │         │
           C         C
           │         │
         VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 5 |
| Components | R, L, C |

**Analysis:** A valid RLC filter with two capacitors providing additional filtering. The dual-L, dual-C structure suggests a higher-order filter response. Electrically plausible as a 4th-order band-pass or notch filter variant.

---

### 2. Extended Double-C with Ground Capacitor (5-node) — 13 occurrences

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2
```

**Structure:**
```
VIN ──L── INT1 ──L── INT2 ──C── GND
           │         │
           C         C
           │         │
         VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 6 |
| Components | R, L, C |

**Analysis:** An extension of topology #1 with an additional ground capacitor at INT2. Creates a more complex LC ladder structure with 3 capacitors and 2 inductors, suggesting a higher-order filter with sharper rolloff.

---

### 3. Resistor T-Network (4-node) — 11 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1
```

**Structure:**
```
VIN ──R── INT1 ──R── VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 3 |
| Components | R only |

**Analysis:** A T-network resistor attenuator. Structurally valid but has no frequency-dependent behavior (no L or C). Functions as a fixed voltage divider with improved impedance matching compared to a simple 2-resistor divider.

---

### 4. Modified RLC Series with Dual-C (5-node) — 10 occurrences

```
GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2
```

**Structure:**
```
VIN ──R── INT1 ──L── INT2
           │         │
           C         C
           │         │
         VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 5 |
| Components | R, L, C |

**Analysis:** Similar to topology #1 but with R instead of L at input. The dual capacitors suggest this could function as a notch or band-reject filter variant with different input impedance.

---

### 5. Pure Resistor Divider (3-node) — 6 occurrences

```
GND--R--VOUT, VIN--R--VOUT
```

**Structure:**
```
VIN ──R── VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 3 (GND, VIN, VOUT) |
| Edges | 2 |
| Components | R only |

**Analysis:** A simple resistor voltage divider. Structurally valid but has no frequency-dependent behavior. Not a true filter — attenuates all frequencies equally.

---

### 6. RL Band-Pass Variant (4-node) — 6 occurrences

```
GND--R--VOUT, VIN--L--INT1, VOUT--R--INT1
```

**Structure:**
```
VIN ──L── INT1 ──R── VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 3 |
| Components | R, L |

**Analysis:** An RL filter variant. Without capacitance, this provides inductive high-pass characteristics. The inductor blocks low frequencies while the resistor network sets the output level.

---

## Invalid Topologies

77 out of 500 random samples (15%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin (e.g., low_pass at z[0:2]=[2.7, 2.6]), so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **18 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Most novel topologies are variations of training topologies (e.g., adding extra C or L)
- The model does NOT invent fundamentally new architectures
- 85% of random samples produce valid connected circuits

### Limitations

- **77/500 (15%) invalid generations** when sampling randomly from N(0,1)
- Novel topologies are not guaranteed to be electrically optimal
- Some novel topologies lack reactive components (R-only networks)

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
