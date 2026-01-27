# Novel Topology Generation Analysis

**Analysis Date:** 2026-01-27

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 385 (77%) |
| Valid novel samples | 69 (14%) |
| Invalid samples | 46 (9%) |
| Unique novel topologies | 17 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Count (of 500) |
|----------|-------------|----------------|
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 112 |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | rlc_series | 75 |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | band_stop | 60 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 55 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 47 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 36 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Dual-L Dual-C Higher-Order Filter (5-node) — 14 occurrences

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

**Analysis:** A higher-order RLC filter with two inductors in series and two capacitors coupling INT1/INT2 to VOUT. Creates a more complex frequency response than training topologies, potentially a 4th-order low-pass or band-pass filter. The most frequently generated novel topology.

---

### 2. RLC Series Variant with Dual Capacitor (5-node) — 11 occurrences

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

**Analysis:** Similar to topology #1 but with R instead of L on the VIN-INT1 path. Resembles the rlc_series training topology with an extra capacitive coupling from VOUT to INT1. Provides additional frequency shaping beyond the standard 4-node RLC series.

---

### 3. Reduced Dual-L Filter (5-node) — 7 occurrences

```
GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1
```

**Structure:**
```
VIN ──L── INT1 ──L── INT2
           │
           C
           │
         VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 4 |
| Components | R, L, C |

**Analysis:** A variant of topology #1 with only one capacitor (INT1-VOUT). INT2 is a dangling inductor node, creating an open-ended inductor branch. Functions as a modified band-pass with inductive loading.

---

### 4. Extended Band-pass with Ground Capacitor (5-node) — 7 occurrences

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1
```

**Structure:**
```
VIN ──L── INT1 ──L── INT2 ──C── GND
           │
           C
           │
         VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 5 |
| Components | R, L, C |

**Analysis:** Extends topology #3 by adding a capacitor from INT2 to GND, creating a complete signal path to ground through the inductor chain. Forms a higher-order filter with both shunt and series reactive elements.

---

### 5. RCL-Parallel with Distributed LC (4-node) — 6 occurrences

```
GND--RCL--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1
```

**Structure:**
```
VIN ──R── VOUT ──RCL── GND
 │         │
 L         C
 │         │
INT1 ─────INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 4 |
| Components | R, L, C, RCL |

**Analysis:** Combines a parallel RCL tank to ground with distributed LC coupling through INT1. A hybrid of the band_pass and rlc_parallel training topologies, providing both resonant grounding and distributed frequency selection.

---

### 6. Band-pass with Direct Capacitive Path (4-node) — 5 occurrences

```
GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1
```

**Structure:**
```
VIN ──C── VOUT ──R── GND
 │         │
 L         C
 │         │
INT1 ─────INT1
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 4 |
| Components | R, L, C |

**Analysis:** A band-pass variant with an additional direct capacitive path from VIN to VOUT. Combines the distributed LC topology of band_pass with a direct high-pass coupling, creating a modified frequency response.

---

### Remaining Novel Topologies (11 unique, 30 total)

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--C--INT2, GND--RCL--VOUT, INT1--L--INT2, VIN--R--INT1, VIN--R--VOUT, VOUT--R--INT1` | 3 | 5 | R, L, C, RCL |
| `GND--R--VOUT, INT1--L--INT2, VIN--C--VOUT, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2` | 3 | 5 | R, L, C |
| `GND--RCL--VOUT, INT1--L--INT2, VIN--R--VOUT, VOUT--C--INT2` | 3 | 5 | R, L, C, RCL |
| `GND--R--VOUT, VIN--R--INT1` | 2 | 4 | R |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1` | 2 | 5 | R, L, C |
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1` | 1 | 4 | R |
| `GND--RCL--VOUT, INT1--L--INT2, VIN--R--INT1, VIN--R--VOUT, VOUT--C--INT2` | 1 | 5 | R, L, C, RCL |
| `GND--R--VOUT, VIN--RCL--INT1, VOUT--R--INT1` | 1 | 4 | R, RCL |
| `GND--C--INT2, GND--C--VOUT, INT1--L--INT2, VIN--R--INT1, VIN--R--VOUT, VOUT--R--INT1` | 1 | 5 | R, L, C |
| `GND--RCL--VOUT, INT1--L--INT2, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1, VOUT--C--INT2` | 1 | 5 | R, L, C, RCL |
| `GND--RCL--VOUT, INT1--L--INT2, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1` | 1 | 5 | R, L, C, RCL |

---

## Invalid Topologies

46 out of 500 random samples (9%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **17 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Novel topologies are predominantly higher-order 5-node RLC filters (dual inductors, dual capacitors)
- The model favors electrically meaningful recombinations over R-only networks
- 91% of random samples produce valid connected circuits

### Limitations

- **46/500 (9%) invalid generations** when sampling randomly from N(0,1)
- Novel topologies are not guaranteed to be electrically optimal
- Some novel topologies have dangling nodes (connected but no complete signal path)
- R-only networks still appear but are much rarer (3 samples vs 64 in previous models)

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
