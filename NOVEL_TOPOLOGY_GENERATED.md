# Novel Topology Generation Analysis

**Model:** v5.0 (Latent-Only Decoder)
**Analysis Date:** 2025-01-24

---

## Summary

| Metric | Value |
|--------|-------|
| Specs tested | 247 (13 cutoffs × 19 Q values) |
| Training topologies | 6 |
| Novel topologies found | 8 |
| Valid novel | 5 |
| Invalid novel | 3 |

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

---

## Valid Novel Topologies

These topologies were **not seen during training** but are structurally valid (VIN and VOUT both connected).

### 1. Double-C Band-Pass Variant (5-node)

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
| Occurrences | 7 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 2,000 | 0.05 |
| 20,000 | 0.1 |
| 50,000 | 0.1 |
| 200,000 | 0.3 |
| 200,000 | 4.0 |

**Analysis:** This is a valid RLC filter with two capacitors providing additional filtering. The dual-L, dual-C structure suggests a higher-order filter response. Electrically plausible as a 4th-order band-pass or notch filter variant.

---

### 2. Pure Resistor Divider (3-node)

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
| Occurrences | 3 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 200,000 | 5.0 |
| 500,000 | 5.0 |
| 500,000 | 6.0 |

**Analysis:** This is a simple resistor voltage divider. While structurally valid, it has **no frequency-dependent behavior** (no L or C). It's not a true filter - just attenuates all frequencies equally. Generated at high frequency + high Q specs where the model may be extrapolating beyond training data.

---

### 3. Modified RLC Series with Dual-C (5-node)

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
| Occurrences | 2 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 100,000 | 0.3 |
| 100,000 | 0.9 |

**Analysis:** Similar to topology #1 but with R instead of L at input. This creates a different impedance characteristic. The dual capacitors suggest this could function as a notch or band-reject filter with sharper rolloff.

---

### 4. Asymmetric LC Filter (4-node)

```
GND--R--VOUT, INT1--L--INT2, VIN--L--INT1, VOUT--C--INT2
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
| Nodes | 4 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 4 |
| Components | R, L, C |
| Occurrences | 1 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 100,000 | 4.0 |

**Analysis:** An asymmetric LC ladder filter. The single capacitor at INT2 combined with dual inductors creates a low-pass characteristic with potential resonance. Valid for high-frequency filtering applications.

---

### 5. T-Network with Inductor (4-node)

```
GND--R--VOUT, INT1--L--INT2, VIN--R--INT1, VOUT--R--INT1
```

**Structure:**
```
VIN ──R── INT1 ──R── VOUT ──R── GND
           │
           L
           │
          INT2
```

| Property | Value |
|----------|-------|
| Nodes | 4 (GND, VIN, VOUT, INT1, INT2) |
| Edges | 4 |
| Components | R, L |
| Occurrences | 1 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 200,000 | 0.05 |

**Analysis:** A T-network with an inductor branch. Without capacitance, this provides inductive impedance characteristics. The dangling INT2 node only connects via inductor, which may cause issues in practical circuits. Marginal validity - depends on application.

---

## Invalid Novel Topologies

These topologies have structural issues (VIN or VOUT not connected to the main circuit).

### 1. Isolated Resistor

```
GND--R--VOUT
```

| Property | Value |
|----------|-------|
| Issue | VIN not connected |
| Occurrences | 18 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 500 | 0.01 |
| 500 | 0.05 |
| 2,000 | 1.2 |
| 10,000 | 1.5 |
| 10,000 | 4.0 |

**Analysis:** The model generates this incomplete circuit when interpolating between dissimilar filter types or at edge cases. This is the most common failure mode.

---

### 2. Disconnected LC Section

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2
```

| Property | Value |
|----------|-------|
| Issue | VIN not connected; INT1 floating |
| Occurrences | 2 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 50,000 | 0.9 |
| 100,000 | 0.1 |

**Analysis:** The LC section (INT1-L-INT2-C-GND) is disconnected from the input. This suggests the model partially generated a complex filter but failed to complete the input connection.

---

### 3. Partially Connected LC

```
GND--C--INT2, GND--R--VOUT, INT1--L--INT2, VOUT--C--INT2
```

| Property | Value |
|----------|-------|
| Issue | VIN not connected |
| Occurrences | 1 |

**Input specs that generated this:**
| Cutoff (Hz) | Q |
|-------------|---|
| 50,000 | 0.4 |

**Analysis:** VOUT connects to the LC section, but VIN has no path. Another incomplete generation.

---

## Failure Analysis

### Where Invalid Circuits Occur

| Q Range | Cutoff Range | Failure Rate |
|---------|--------------|--------------|
| 0.01 - 0.1 | < 1 kHz | High |
| 1.0 - 1.5 | All | Moderate |
| 4.0 - 6.0 | > 100 kHz | Moderate |

### Root Causes

1. **Spec interpolation artifacts:** When target specs fall between training clusters, K-NN averaging can produce latent codes in "dead zones"

2. **Training data gaps:** Certain Q ranges (especially 1.0-1.5) have fewer training examples, leading to poor generalization

3. **Edge case extrapolation:** Very low/high frequencies combined with extreme Q values push the model outside its training distribution

---

## Conclusions

### Generalization Capability

- **5 valid novel topologies** demonstrate the model can recombine learned components in new ways
- Most novel topologies are variations of training topologies (e.g., adding extra C or L)
- The model does NOT invent fundamentally new architectures

### Limitations

- **18/247 (7.3%) invalid generations** - primarily the isolated resistor failure mode
- Invalid circuits cluster around specific spec combinations (Q ≈ 1.2-1.5, very low frequencies)
- Novel topologies are not guaranteed to be electrically optimal

### Recommendations

1. **Validate generated circuits** with SPICE simulation before use
2. **Avoid spec ranges** with known high failure rates
3. **Use nearest-neighbor method** (`--method nearest`) for critical applications to stay closer to training data

---

## Test Parameters

```bash
# Cutoffs tested (Hz)
10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000

# Q factors tested
0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0

# Total combinations: 13 × 19 = 247
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl`
- **Generation script:** `scripts/generation/generate_from_specs.py`
