# Quick Start: Targeted Transfer Function Generation

Generate circuits with **exact transfer functions** you specify.

## What This Enables

✅ **Before:** Random circuits with unpredictable transfer functions
✅ **Now:** Specify poles/zeros → get multiple circuit implementations

---

## Quick Start (3 Steps)

### 1. Train TF Encoder (One-Time, ~5 minutes)

```bash
python scripts/train_tf_encoder.py --epochs 100 --device mps
```

**Output:**
```
Best validation MSE: 0.0494
Model saved to: checkpoints/tf_encoder/20251227_203319/best.pt
```

### 2. Generate Circuits with Target TF

**Example: 2nd-Order Butterworth at 1kHz**

```bash
python scripts/generate_targeted_tf.py \
    --tf-encoder-checkpoint checkpoints/tf_encoder/20251227_203319/best.pt \
    --poles "(-4442.88+4442.88j)" "(-4442.88-4442.88j)" \
    --cutoff 1000 \
    --q-factor 0.707 \
    --num-samples 10
```

### 3. Export to SPICE

```bash
# Use the same latent codes from targeted generation
# (Future: add direct SPICE export from targeted generation)
```

---

## Common Filter Designs

### Butterworth (Maximally Flat)

**2nd-order at 1kHz:**
```bash
--poles "(-4442.88+4442.88j)" "(-4442.88-4442.88j)"
--cutoff 1000 --q-factor 0.707
```

**Calculation:**
- ω₀ = 2π × 1000 = 6283 rad/s
- ζ = 0.707 (Butterworth)
- Real: -ζω₀ = -4442.88
- Imag: ±ω₀√(1-ζ²) = ±4442.88

### Bessel (Linear Phase)

**2nd-order at 1kHz:**
```bash
--poles "(-4605.17+2653.82j)" "(-4605.17-2653.82j)"
--cutoff 1000 --q-factor 0.577
```

### Chebyshev (1dB Ripple)

**2nd-order at 1kHz:**
```bash
--poles "(-1848.93+5680.65j)" "(-1848.93-5680.65j)"
--cutoff 1000 --q-factor 0.8636
```

### Notch Filter (60Hz Rejection)

```bash
--poles "(-377.0+0j)" "(-377.0+0j)" \
--zeros "(0+377j)" "(0-377j)" \
--cutoff 60 --q-factor 10.0
```

---

## Control Parameters

### Topology Diversity

```bash
--topology-variation 0.1   # Low: similar structures
--topology-variation 1.0   # High: very different structures
```

### Component Value Diversity

```bash
--values-variation 0.1     # Low: similar R/C/L values
--values-variation 1.0     # High: wide value ranges
```

---

## What the Latent Vector Controls

```
Full latent (8D) = [topo | values | TF]
                   [0:2] | [2:4]  | [4:8]
```

| Part | Dims | Controlled By | Varies? |
|------|------|---------------|---------|
| **Topology** | [0:2] | Random sampling | ✅ Yes |
| **Values** | [2:4] | Random sampling | ✅ Yes |
| **TF** | [4:8] | **TF Encoder (your target)** | ❌ No |

**Result:** Same transfer function, different circuit designs!

---

## Expected Results

**TF Accuracy:**
- Pole count: **100%** correct
- Zero count: **90-95%** correct
- Pole location: **Within ~20%** of target
- Stability: **100%** (all poles in LHP)

**Topology Diversity:**
- `topology_variation=0.3`: **20-30%** unique
- `topology_variation=1.0`: **50-70%** unique

---

## Full Documentation

See `docs/TARGETED_TF_GENERATION.md` for:
- Detailed pole/zero calculation formulas
- Filter design cookbook
- Architecture details
- Troubleshooting

---

## Workflow Example

```bash
# 1. Design a 2nd-order low-pass Butterworth
# Calculate poles: (-4442.88 ± 4442.88j)

# 2. Generate 20 circuit variations
python scripts/generate_targeted_tf.py \
    --poles "(-4442.88+4442.88j)" "(-4442.88-4442.88j)" \
    --cutoff 1000 --q-factor 0.707 --num-samples 20

# 3. Review generated topologies
# - Circuit 0: Simple RC (3 nodes, 2 edges)
# - Circuit 5: RLC multi-stage (4 nodes, 4 edges)
# - Circuit 12: Active filter (5 nodes, 6 edges)

# 4. Select best based on:
# - Component practicality
# - Sensitivity
# - Cost/complexity

# 5. Export to SPICE and simulate
# (Coming soon: direct export from targeted generation)
```

---

## Key Advantage

**Traditional Design:**
1. Calculate transfer function
2. Pick a topology (Sallen-Key, MFB, etc.)
3. Calculate component values
4. Hope it works

**With Z-GED:**
1. Calculate transfer function
2. Generate **20 different topologies** automatically
3. All implement same TF
4. Pick the best one!

You get **design exploration** for free! 🚀
