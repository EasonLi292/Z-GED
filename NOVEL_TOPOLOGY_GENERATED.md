# Novel Topology Generation Analysis

**Analysis Date:** 2026-01-28

---

## Summary

| Metric | Value |
|--------|-------|
| Random samples | 500 |
| Training topologies | 6 |
| Known topology samples | 350 (70%) |
| Valid novel samples | 75 (15%) |
| Invalid samples | 75 (15%) |
| Unique novel topologies | 15 |

---

## Training Topologies (Known)

These 6 topologies appear in the training data:

| Topology | Filter Type | Count (of 500) |
|----------|-------------|----------------|
| `GND--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | band_stop | 106 |
| `GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | band_pass | 101 |
| `GND--R--VOUT, VIN--C--VOUT` | high_pass | 62 |
| `GND--C--VOUT, VIN--R--VOUT` | low_pass | 32 |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT2, INT1--L--INT2` | rlc_series | 29 |
| `GND--RCL--VOUT, VIN--R--VOUT` | rlc_parallel | 20 |

**Method:** 500 random samples from N(0,1) in 8D latent space, decoded and classified.

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Novel Topology #1 (4-node) — 21 occurrences

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

**Analysis:** Resistive-only topology; no reactive components, so frequency response is flat.

### 2. Novel Topology #2 (5-node) — 17 occurrences

```
GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2
```

**Structure:**
```
GND--R--VOUT
VIN--L--INT1
VOUT--C--INT1
INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, L, C |

**Analysis:** Higher-order RLC topology combining series and shunt reactances.

### 3. Novel Topology #3 (5-node) — 7 occurrences

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

**Analysis:** RC network with shunt/series branches; likely low-pass or notch-like behavior.

### 4. Novel Topology #4 (5-node) — 6 occurrences

```
GND--C--VOUT, VIN--R--VOUT, VIN--L--INT1, VOUT--C--INT1, INT1--L--INT2
```

**Structure:**
```
GND--C--VOUT
VIN--R--VOUT
VIN--L--INT1
VOUT--C--INT1
INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Nodes | 5 |
| Components | R, L, C |

**Analysis:** Higher-order RLC topology combining series and shunt reactances.

### 5. Novel Topology #5 (5-node) — 4 occurrences

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
| Nodes | 5 |
| Components | R, C |

**Analysis:** RC network with shunt/series branches; likely low-pass or notch-like behavior.

### 6. Novel Topology #6 (5-node) — 4 occurrences

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
| Components | R, L, C |

**Analysis:** Higher-order RLC topology combining series and shunt reactances.


### Remaining Novel Topologies (9 unique, 16 total)

| Topology | Count | Nodes | Components |
|----------|-------|-------|------------|
| `GND--C--VOUT, VIN--R--VOUT, VIN--R--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | 3 | 5 | R, L, C |
| `GND--R--VOUT, VIN--R--INT1` | 3 | 5 | R |
| `GND--C--VOUT, VIN--R--VOUT, VIN--R--INT1, VOUT--C--INT1, INT1--L--INT2` | 3 | 5 | R, L, C |
| `GND--C--VOUT, VIN--R--VOUT, VIN--C--INT1, VOUT--R--INT1, GND--C--INT2, INT1--L--INT2` | 2 | 5 | R, L, C |
| `GND--RCL--VOUT, VIN--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | 1 | 4 | R, L, C, RCL |
| `GND--RCL--VOUT, VIN--R--VOUT, VIN--R--INT1` | 1 | 5 | R, L, C, RCL |
| `GND--C--VOUT, VIN--R--VOUT, VIN--L--INT1, VOUT--C--INT1` | 1 | 4 | R, L, C |
| `GND--R--VOUT, VIN--C--VOUT, VIN--L--INT1, VOUT--C--INT1` | 1 | 4 | R, L, C |
| `GND--R--VOUT, VIN--R--INT1, VOUT--C--INT1, VOUT--C--INT2, INT1--L--INT2` | 1 | 5 | R, L, C |

---

## Invalid Topologies

75 out of 500 random samples (15%) produced invalid circuits where VIN or VOUT is not connected. The most common failure mode is the isolated resistor (`GND--R--VOUT` with VIN disconnected). These occur when random latent codes fall in "dead zones" between training clusters.

### Root Causes

1. **Latent space gaps:** Random samples from N(0,1) may land between training clusters where no clear topology exists
2. **Boundary regions:** Latent codes near cluster boundaries can produce incomplete circuits
3. **Standard normal mismatch:** Training data centroids are not centered at origin, so standard random samples may fall outside the training distribution

---

## Conclusions

### Generalization Capability

- **15 unique valid novel topologies** demonstrate the model can recombine learned components in new ways
- Novel topologies are predominantly higher-order filters and resistive variants
- The model favors electrically meaningful recombinations over arbitrary graphs
- 425/500 (85%) of random samples produce valid connected circuits

### Limitations

- **75/500 (15%) invalid generations** when sampling randomly from N(0,1)
- Novel topologies are not guaranteed to be electrically optimal
- Some novel topologies include resistive-only networks
- Validity does not guarantee desired filter response

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
