# Novel Topology Generation Analysis

**Model:** v5.2 (Dynamic Node Count)
**Analysis Date:** 2026-01-26

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 331 (66%) |
| Valid novel samples | 90 (18%) |
| Invalid samples | 79 (16%) |
| Unique novel topologies | 14 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Samples |
|----------|-------------|---------|
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 107 |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | rlc_series | 95 |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | band_stop | 81 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 25 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 17 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 6 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Pure Resistor Divider (3-node) — 26 occurrences

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

### 2. Modified Band-Stop Variant (5-node) — 10 occurrences

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2
```

**Structure:**
```
VIN ──R── INT1 ──L── INT2 ──C── GND
                      │
                      C
                      │
GND ──R── VOUT ──────┘
```

| Property | Value |
|----------|-------|
| Nodes | 5 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 5 |
| Components | R, L, C |

**Analysis:** A variant of the band-stop topology where VOUT connects to INT2 via capacitor instead of INT1 via resistor. Maintains the LC tank (INT1--L--INT2, INT2--C--GND) but couples the output differently, potentially altering the notch depth and bandwidth.

---

### 3. Disconnected R-Network (4-node) — 9 occurrences

```
GND--R--VOUT, VIN--R--INT1
```

**Structure:**
```
VIN ──R── INT1        VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 2 |
| Components | R only |

**Analysis:** Passes the validity check (both VIN and VOUT have edges) but has no path from VIN to VOUT — the two subgraphs are disconnected. Not a functional circuit. Represents a boundary case in the latent space.

---

### 4. Resistor T-Network (4-node) — 9 occurrences

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

### 5. Extended RLC with Dual-C (5-node) — 9 occurrences

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2
```

**Structure:**
```
VIN ──R── INT1 ──L── INT2 ──C── GND
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

**Analysis:** An extension of the band-stop topology with additional capacitors from VOUT to both INT1 and INT2. The dual-C coupling creates a more complex frequency response, potentially a higher-order notch or band-reject filter.

---

### 6. RC Series Variant (4-node) — 6 occurrences

```
GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1
```

**Structure:**
```
VIN ──R── INT1 ──C── VOUT ──R── GND
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 3 |
| Components | R, C |

**Analysis:** A 4-node RC filter with the capacitor between INT1 and VOUT. Functionally similar to the 3-node high-pass but with an explicit internal node splitting the resistive path.

---

### Remaining Novel Topologies (8 unique, 21 total)

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1` | 5 | 5 | R, L, C |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2` | 4 | 5 | R, L, C |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2, VOUT--R--INT1` | 3 | 5 | R, L, C |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2` | 3 | 5 | R, L, C |
| `GND--R--VOUT, VIN--L--INT1` | 2 | 4 | R, L |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1` | 2 | 5 | R, L, C |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1` | 1 | 5 | R, L, C |
| `GND--R--VOUT, VIN--L--INT1, VOUT--R--INT1` | 1 | 4 | R, L |

---

## Invalid Topologies

79 out of 500 random samples (16%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin (e.g., low_pass at z[0:2]=[2.7, 2.6]), so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **14 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Most novel topologies are variations of training topologies (e.g., adding extra C or L)
- The model does NOT invent fundamentally new architectures
- 84% of random samples produce valid connected circuits

### Limitations

- **79/500 (16%) invalid generations** when sampling randomly from N(0,1)
- Novel topologies are not guaranteed to be electrically optimal
- Some novel topologies lack reactive components (R-only networks)
- One novel topology (#3) passes validity check but has disconnected VIN/VOUT paths

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
