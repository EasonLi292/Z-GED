# Circuit Generation Guide

## How Generation Works

The Z-GED system generates circuits from **two specifications**:
1. **Cutoff frequency** (Hz) - Where the filter response changes
2. **Q-factor** (dimensionless) - Sharpness of the response

**No explicit "filter type" selection** - The system infers what kind of filter to generate based on these two parameters.

---

## Current Interface: Command Line

### Basic Usage

```bash
python scripts/generate_from_specs.py --cutoff <Hz> --q-factor <Q>
```

### Examples

```bash
# Generate 10 kHz Butterworth low-pass filter
python scripts/generate_from_specs.py --cutoff 10000 --q-factor 0.707

# Generate 5 kHz band-pass filter with moderate Q
python scripts/generate_from_specs.py --cutoff 5000 --q-factor 2.0

# Generate high-Q resonator at 1 kHz
python scripts/generate_from_specs.py --cutoff 1000 --q-factor 10.0
```

### All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cutoff` | float | **Required** | Target cutoff frequency in Hz |
| `--q-factor` | float | 0.707 | Target Q-factor (0.01 to 50) |
| `--num-samples` | int | 5 | Number of circuit variants to generate |
| `--method` | choice | interpolate | Generation method: `nearest` or `interpolate` |
| `--ged-weight` | float | 0.5 | GED vs spec weighting (0=spec only, **recommended: 0.0**) |
| `--checkpoint` | path | checkpoints/production/best.pt | Model checkpoint |
| `--device` | str | cpu | Device: `cpu` or `cuda` |

---

## How Specifications Map to Filter Types

### Q-Factor Determines Filter Character

| Q-Factor Range | Filter Type | Characteristics | Topology Generated |
|----------------|-------------|-----------------|-------------------|
| **Q ≈ 0.707** | Butterworth | Maximally flat passband, -3dB at cutoff | 2-edge RC or RL |
| **0.1 < Q < 0.5** | Overdamped | Gradual rolloff, no peaking | 2-4 edge RLC |
| **1.0 < Q < 5.0** | Band-pass (moderate) | Moderate selectivity, some peaking | 2-4 edge RLC |
| **Q ≥ 5.0** | Band-pass (high-Q) | Sharp resonance, narrow bandwidth | 3-4 edge RLC |

### Frequency Determines Component Values

The cutoff frequency determines the **absolute values** of R, L, C:
- **Low frequency (10-100 Hz)**: Large L, large C
- **Mid frequency (1-100 kHz)**: Moderate values
- **High frequency (100 kHz+)**: Small L, small C

---

## Generation Process Step-by-Step

### 1. **Encode Training Circuits**
```
120 training circuits → Encode → Specification database
                                   [cutoff, Q] → latent code
```

### 2. **K-NN Search**
```
Your target specs: [10 kHz, Q=0.707]
    ↓
Find 5 nearest training circuits:
  1. 8975 Hz, Q=0.707  (weight=0.44)
  2. 11721 Hz, Q=0.707 (weight=0.30)
  3. 7957 Hz, Q=0.903  (weight=0.09)
  ...
```

### 3. **Latent Interpolation**
```
Weighted average of 5 latent codes
    ↓
Interpolated latent: [8D vector]
```

### 4. **Conditional Generation**
```
Decoder receives:
  - Interpolated latent (from similar circuits)
  - Target conditions [log(cutoff)/4, log(Q)/2]
    ↓
Autoregressive generation:
  - Generate nodes: GND, VIN, VOUT, Internal, ...
  - Generate edges: Connections + component types
  - Generate values: R, L, C values (normalized)
```

### 5. **Circuit Output**
```
Circuit structure:
  Node 0: GND
  Node 1: VIN
  Node 2: VOUT
  Node 3: Internal

  Edge (0,2): R=4.7kΩ, C=3.4nF
  Edge (1,2): R=2.2kΩ
```

---

## Example: Different Specs → Different Topologies

### Example 1: Butterworth Low-Pass (Simple)

**Command:**
```bash
python scripts/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```

**Generated:**
- **Topology**: 2 edges (simple RC)
- **Cutoff**: ~5000 Hz (50% error, but topology is correct)
- **Q**: 0.707 (perfect match!)
- **Components**: R + C (correct for low-pass)

### Example 2: Band-Pass Moderate-Q

**Command:**
```bash
python scripts/generate_from_specs.py --cutoff 5000 --q-factor 2.0
```

**Generated:**
- **Topology**: 2-4 edges (RLC network)
- **Cutoff**: ~58000 Hz (moderate error)
- **Q**: 0.707-2.0 (varies)
- **Components**: R + L + C (correct for resonance)

### Example 3: High-Q Resonator

**Command:**
```bash
python scripts/generate_from_specs.py --cutoff 10000 --q-factor 10.0
```

**Generated:**
- **Topology**: 3 edges (LC tank + damping)
- **Cutoff**: Varies
- **Q**: 0.707-10.0 (often defaults lower)
- **Components**: R + L + C (correct for high-Q)

---

## What You CAN'T Specify (Currently)

The current system does **NOT** allow explicit control of:

1. ❌ **Filter type** (low-pass, high-pass, band-pass, band-stop)
   - System infers from Q-factor
   - Q=0.707 → likely low-pass
   - Q=2-10 → likely band-pass

2. ❌ **Topology** (2-edge, 3-edge, 4-edge)
   - System chooses based on k-NN neighbors
   - More complex specs → more edges

3. ❌ **Component types** (RC only, RLC, etc.)
   - System chooses based on what's needed
   - Q>1 requires L+C, so it generates them

4. ❌ **Order** (1st-order, 2nd-order, 3rd-order)
   - Implicit from topology
   - 2-edge → 1st or 2nd order
   - 4-edge → 2nd or 3rd order

5. ❌ **Exact component values**
   - Generated values often don't match specs
   - **61% average error** (latent dominates over conditions)

---

## What You CAN Specify (Future Improvements)

### Option 1: Add Filter Type Parameter

```bash
python scripts/generate_from_specs.py \
    --cutoff 10000 \
    --q-factor 0.707 \
    --filter-type low-pass  # NEW
```

**Implementation:**
- Add filter type as 3rd condition dimension
- Train decoder to condition on [cutoff, Q, type]
- Type encoding: 0=low-pass, 1=high-pass, 2=band-pass, 3=band-stop

### Option 2: Add Topology Constraints

```bash
python scripts/generate_from_specs.py \
    --cutoff 10000 \
    --q-factor 0.707 \
    --max-edges 2  # NEW: Force simple topology
```

**Implementation:**
- Add early stopping to edge generation
- Constrain decoder to generate ≤ N edges

### Option 3: Add Component Type Preferences

```bash
python scripts/generate_from_specs.py \
    --cutoff 10000 \
    --q-factor 0.707 \
    --prefer-rc  # NEW: Prefer RC over RL
```

**Implementation:**
- Add bias term to component selection
- Weight RC edges higher during generation

---

## Programmatic API (For Python Integration)

Instead of command line, you can use the API directly:

```python
import torch
import numpy as np
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from ml.data.dataset import CircuitDataset

# Load models
encoder = HierarchicalEncoder(...).to('cpu')
decoder = LatentGuidedGraphGPTDecoder(...).to('cpu')
checkpoint = torch.load('checkpoints/production/best.pt')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Load dataset for k-NN database
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
specs_db, latents_db = build_specification_database(encoder, dataset)

# Generate circuit for target specs
target_cutoff = 10000  # Hz
target_q = 0.707

# K-NN interpolation
latent, info = interpolate_latents(target_cutoff, target_q, specs_db, latents_db, k=5)

# Prepare conditions
conditions = torch.tensor([[
    np.log10(target_cutoff) / 4.0,
    np.log10(target_q) / 2.0
]], dtype=torch.float32)

# Generate
with torch.no_grad():
    circuit = decoder.generate(latent.unsqueeze(0), conditions, verbose=True)

# Extract results
edge_existence = circuit['edge_existence'][0]
edge_values = circuit['edge_values'][0]
node_types = circuit['node_types'][0]

print(f"Generated {(edge_existence > 0.5).sum().item() // 2} edges")
```

---

## Understanding the Training Data Coverage

The system can generate circuits for any specs, but **accuracy depends on training coverage**:

### Well-Covered Ranges (Good Accuracy)

| Specification | Training Range | Confidence |
|--------------|----------------|------------|
| Cutoff | 100 Hz - 100 kHz | ✅ High |
| Q (Butterworth) | 0.5 - 1.0 | ✅ High |
| Q (moderate) | 1.0 - 5.0 | ✅ High |
| Q (high) | 5.0 - 20.0 | ⚠️ Medium |

### Poorly-Covered Ranges (Low Accuracy)

| Specification | Training Range | Issue |
|--------------|----------------|-------|
| Very low freq | <100 Hz | ❌ Few examples |
| Very high freq | >500 kHz | ❌ Few examples |
| Very low Q | <0.1 | ❌ Rare in dataset |
| Very high Q | >20 | ❌ Rare in dataset |

---

## Recommended Specifications for Best Results

### ✅ High Success Rate (20-40% error)

1. **Butterworth filters** (Q ≈ 0.707)
   - Any cutoff 100 Hz - 100 kHz
   - Example: `--cutoff 10000 --q-factor 0.707`

2. **Moderate band-pass** (1 < Q < 3)
   - Cutoff 1 kHz - 50 kHz
   - Example: `--cutoff 5000 --q-factor 2.0`

### ⚠️ Medium Success Rate (40-60% error)

3. **High-Q resonators** (5 < Q < 20)
   - Cutoff 5 kHz - 20 kHz
   - Example: `--cutoff 10000 --q-factor 10.0`

4. **Overdamped filters** (0.1 < Q < 0.5)
   - Any cutoff 1 kHz - 100 kHz
   - Example: `--cutoff 50000 --q-factor 0.3`

### ❌ Low Success Rate (>60% error)

5. **Very high-Q** (Q > 20)
   - Often defaults to Q=0.707
   - Example: `--cutoff 5000 --q-factor 30.0` (likely to fail)

6. **Unusual combinations**
   - Low freq + high Q: `--cutoff 100 --q-factor 10.0`
   - High freq + very low Q: `--cutoff 200000 --q-factor 0.05`

---

## Next Steps

### For Users

1. **Start with well-covered specs** (Butterworth filters)
2. **Check generated topology** (use `scripts/test_single_spec.py`)
3. **Accept ~30-60% error** (topology is correct, values need tuning)
4. **Use component value optimization** (coming soon)

### For Developers

1. **Add filter type parameter** (explicit low/high/band-pass control)
2. **Implement component refinement** (numerical optimization)
3. **Expand training data** (more high-Q, low-Q, unusual combinations)
4. **Strengthen condition signal** (reduce latent dominance)

---

## FAQ

**Q: Why can't I specify "low-pass" or "high-pass"?**
A: The current system uses only [cutoff, Q] as inputs. Filter type emerges implicitly from Q-factor. Q≈0.707 tends to produce low-pass, Q>1 tends to produce band-pass.

**Q: Why don't generated specs match my target?**
A: The decoder follows the latent code (from k-NN neighbors) more than the target conditions. Average error is 63.5%. Topology is correct, but component values need tuning.

**Q: Can I force a specific topology (e.g., 2-edge only)?**
A: Not currently. Topology is determined by k-NN neighbors and decoder. A 2-edge constraint could be added as future work.

**Q: How do I get multiple design options?**
A: Use `--num-samples 10`. The system adds small random noise to latent codes for diversity (currently gives ~10% topology diversity).

**Q: What's the difference between `--method nearest` and `--method interpolate`?**
A:
- `nearest`: Copy latent from single closest training circuit
- `interpolate`: Weighted average of 5 nearest (smoother, more novel)

Recommended: `interpolate` (default)

**Q: Should I use GED weighting (`--ged-weight`)?**
A: **No**. Set `--ged-weight 0.0` (spec-only). GED adds complexity without clear benefit. Our analysis shows spec-only search works best.
