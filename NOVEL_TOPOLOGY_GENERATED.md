# Novel Topology Generation Analysis

**Analysis Date:** 2026-01-27

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 390 (78%) |
| Valid novel samples | 65 (13%) |
| Invalid samples | 45 (9%) |
| Unique novel topologies | 10 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Samples |
|----------|-------------|---------|
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 139 |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | rlc_series | 101 |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | band_stop | 97 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 20 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 17 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 16 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Resistor T-Network (4-node) — 33 occurrences

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

### 2. Disconnected R-Network (4-node) — 10 occurrences

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

### 3. Extended RLC with Dual Inductor + Dual Capacitor (5-node) — 5 occurrences

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

**Analysis:** A higher-order RLC filter with two inductors in series and two capacitors coupling INT1/INT2 to VOUT. Creates a more complex frequency response than training topologies, potentially a higher-order low-pass or band-pass filter.

---

### 4. RC Network with Feedback (4-node) — 5 occurrences

```
GND--C--VOUT, VIN--R--INT1, VIN--R--VOUT, VOUT--R--INT1
```

**Structure:**
```
VIN ──R── INT1
 │         │
 R         R
 │         │
VOUT ──C── GND
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1) |
| Edges | 4 |
| Components | R, C |

**Analysis:** An RC network where VIN connects to both VOUT and INT1 via resistors, with additional R between VOUT and INT1, and a capacitor from VOUT to GND. Functions as a low-pass filter with a more complex resistive divider network.

---

### 5. RL Series Variant (4-node) — 4 occurrences

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

**Analysis:** A 4-node RL filter with the inductor between VIN and INT1. Provides inductive low-pass behavior with the resistive path providing DC bias. Similar to a training topology but with L instead of C in one branch.

---

### Remaining Novel Topologies (5 unique, 8 total)

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--RCL--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1` | 2 | 4 | R, L, C |
| `GND--C--INT2, GND--C--VOUT, INT1--L--INT2, VIN--R--INT1, VIN--R--VOUT, VOUT--R--INT1` | 2 | 5 | R, L, C |
| `GND--C--VOUT, VIN--R--VOUT, VOUT--R--INT1` | 2 | 4 | R, C |
| `GND--C--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1` | 1 | 4 | R, L, C |
| `GND--R--VOUT, VIN--L--INT1` | 1 | 4 | R, L |

---

## Invalid Topologies

45 out of 500 random samples (9%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **10 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Most novel topologies are variations of training topologies (e.g., substituting L for C, adding extra R branches)
- The model does NOT invent fundamentally new architectures
- 91% of random samples produce valid connected circuits

### Limitations

- **45/500 (9%) invalid generations** when sampling randomly from N(0,1)
- Novel topologies are not guaranteed to be electrically optimal
- Some novel topologies lack reactive components (R-only networks)
- One novel topology (#2) passes validity check but has disconnected VIN/VOUT paths

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
