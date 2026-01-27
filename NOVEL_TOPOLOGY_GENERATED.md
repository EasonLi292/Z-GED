# Novel Topology Generation Analysis

**Analysis Date:** 2026-01-27

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 350 (70%) |
| Valid novel samples | 98 (20%) |
| Invalid samples | 52 (10%) |
| Unique novel topologies | 14 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Samples |
|----------|-------------|---------|
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 101 |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT2` | rlc_series | 71 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 48 |
| `GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1` | band_stop | 44 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 43 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 43 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Resistor T-Network (4-node) — 49 occurrences

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

### 2. Disconnected R-Network (4-node) — 15 occurrences

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

### 3. RCL-Parallel with Distributed LC (4-node) — 8 occurrences

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

**Analysis:** Combines a parallel RCL tank to ground with distributed LC coupling through INT1. The RCL parallel element provides a resonant path to ground while the L and C branches through INT1 create additional frequency shaping. A hybrid of the band_pass and rlc_parallel training topologies.

---

### 4. Extended RLC with Dual Inductor + Capacitor (5-node) — 8 occurrences

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

**Analysis:** A higher-order filter with two inductors in series (VIN→INT1→INT2) and two capacitors coupling to ground through different paths. Creates a more complex frequency response than training topologies, potentially a higher-order low-pass or band-pass filter.

---

### 5. RL Series Variant (4-node) — 5 occurrences

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

**Analysis:** A 4-node RL filter with the inductor between VIN and INT1. Provides inductive low-pass behavior with the resistive path providing DC bias. Similar to the band_pass training topology but with R instead of C in one branch.

---

### 6. Band-pass with Direct RC Path (4-node) — 4 occurrences

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

### Remaining Novel Topologies (8 unique, 9 total)

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--R--VOUT, VIN--RCL--INT1, VOUT--R--INT1` | 2 | 4 | R, RCL |
| `GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT1, VOUT--C--INT2` | 1 | 5 | R, L, C |
| `GND--C--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--C--INT1` | 1 | 4 | R, L, C |
| `GND--RCL--VOUT, VIN--R--VOUT, VOUT--R--INT1` | 1 | 4 | R, RCL |
| `GND--RCL--VOUT, VIN--L--INT1, VIN--R--VOUT, VOUT--R--INT1` | 1 | 4 | R, L, RCL |
| `GND--R--VOUT, VIN--C--VOUT, VIN--R--INT1` | 1 | 4 | R, C |
| `GND--C--INT2, GND--RCL--VOUT, INT1--L--INT2, VIN--R--VOUT` | 1 | 5 | R, L, C, RCL |
| `GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2` | 1 | 5 | R, L, C |

---

## Invalid Topologies

52 out of 500 random samples (10%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **14 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Most novel topologies are variations of training topologies (e.g., substituting L for C, adding extra R branches, combining RCL parallel with distributed components)
- The model does NOT invent fundamentally new architectures
- 90% of random samples produce valid connected circuits

### Limitations

- **52/500 (10%) invalid generations** when sampling randomly from N(0,1)
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
