# Z-GED: Circuit Generation Made Simple

Generate analog filter circuits from high-level specifications.

## What You Need to Know

**Input:** Filter type + cutoff frequency + Q-factor
**Output:** Multiple novel circuit designs implementing your specs

That's it!

---

## Quick Start (3 Commands)

### 1. Train the TF Encoder (One-Time, 5 minutes)

```bash
python scripts/train_tf_encoder.py --epochs 100 --device mps
```

### 2. Generate Circuits

**2nd-Order Butterworth at 1kHz:**
```bash
python scripts/generate_circuit.py \
    --filter-type butterworth \
    --order 2 \
    --cutoff 1000 \
    --num-samples 10
```

**3rd-Order Bessel at 5kHz:**
```bash
python scripts/generate_circuit.py \
    --filter-type bessel \
    --order 3 \
    --cutoff 5000 \
    --num-samples 10
```

**60Hz Notch Filter:**
```bash
python scripts/generate_circuit.py \
    --filter-type notch \
    --order 2 \
    --cutoff 60 \
    --q-factor 15 \
    --num-samples 10
```

### 3. (Optional) Export to SPICE

```bash
# Coming soon: direct export from generate_circuit.py
# For now, use:
python scripts/export_spice_netlists.py \
    --cutoff 1000 \
    --q-factor 0.707 \
    --num-samples 10
```

---

## Supported Filter Types

| Filter | What It Does | Command |
|--------|--------------|---------|
| **Butterworth** | Maximally flat response | `--filter-type butterworth` |
| **Bessel** | Linear phase (constant group delay) | `--filter-type bessel` |
| **Chebyshev** | Steep rolloff, passband ripple | `--filter-type chebyshev` |
| **Notch** | Rejects specific frequency | `--filter-type notch` |

---

## Common Use Cases

### Design a 2nd-Order Lowpass Filter

```bash
# Butterworth (maximally flat)
python scripts/generate_circuit.py \
    --filter-type butterworth \
    --order 2 \
    --cutoff 1000

# Bessel (linear phase)
python scripts/generate_circuit.py \
    --filter-type bessel \
    --order 2 \
    --cutoff 1000

# Chebyshev (steeper rolloff, 1dB ripple)
python scripts/generate_circuit.py \
    --filter-type chebyshev \
    --order 2 \
    --cutoff 1000 \
    --ripple-db 1.0
```

### Remove 60Hz Hum

```bash
python scripts/generate_circuit.py \
    --filter-type notch \
    --order 2 \
    --cutoff 60 \
    --q-factor 15
```

### Audio Crossover at 3kHz

```bash
# Low-pass for woofer
python scripts/generate_circuit.py \
    --filter-type butterworth \
    --order 2 \
    --cutoff 3000

# High-pass for tweeter (coming soon)
# python scripts/generate_circuit.py \
#     --filter-type butterworth-highpass \
#     --order 2 \
#     --cutoff 3000
```

---

## How It Works (Behind the Scenes)

```
Your Input                    System Does
─────────────────────────────────────────────────
Filter type: Butterworth  →   Calculate poles/zeros
Order: 2                       mathematically
Cutoff: 1kHz
                          ↓
                              Encode to latent space
                              (TF Encoder)
                          ↓
                              Generate 10 different
                              circuit topologies
                              (GraphGPT Decoder)
                          ↓
Your Output
─────────────────────────────
10 novel circuit designs
All implementing Butterworth
Different R/C/L combinations
Ready for SPICE simulation
```

**You don't need to:**
- Calculate poles/zeros manually ✅
- Know complex transfer function math ✅
- Pick a specific topology (Sallen-Key, MFB, etc.) ✅

**System handles all of that automatically!**

---

## What You Get

For each filter spec, you get **10 different circuit designs**:

```
Butterworth 2nd-order @ 1kHz:

Circuit 0: Simple RC (3 nodes, 2 edges)
  - 2 poles @ -4442.88 ± 4442.88j
  - Components: R1=10kΩ, C1=100pF

Circuit 1: RC + RL (4 nodes, 4 edges)
  - 2 poles @ -4455.12 ± 4430.21j
  - Components: R1=5kΩ, R2=8kΩ, C1=220pF, L1=2mH

Circuit 2: Multi-stage (5 nodes, 6 edges)
  - 2 poles @ -4438.90 ± 4448.33j
  - Components: R1=12kΩ, R2=15kΩ, R3=3kΩ, C1=150pF, C2=100pF

...and 7 more variations!
```

**All have same transfer function, different implementations.**

Pick the best based on:
- Component availability
- Sensitivity to tolerances
- Cost
- PCB space

---

## Parameters Explained

### Required

- `--filter-type`: butterworth, bessel, chebyshev, notch
- `--order`: Filter order (1-4)
- `--cutoff`: Cutoff frequency in Hz

### Optional

- `--q-factor`: Quality factor (default: 0.707 for Butterworth)
  - Higher Q → sharper cutoff, more ringing
  - Butterworth: 0.707
  - Bessel: 0.577
  - Notch: 10-50 (higher = sharper rejection)

- `--num-samples`: How many circuit variations (default: 10)

- `--topology-variation`: How different topologies are (0-1, default: 0.5)
  - 0.0 = very similar structures
  - 1.0 = very different structures

- `--values-variation`: How different component values are (0-1, default: 0.3)

- `--gain-db`: DC gain in dB (default: 0)
- `--ripple-db`: Passband ripple for Chebyshev (default: 1.0 dB)

---

## Example Output

```
======================================================================
Filter Design
======================================================================

Type: Butterworth
Order: 2
Cutoff: 1000.0 Hz
Q-factor: 0.707

Calculated Transfer Function:
  Poles (2):
    Pole 1: -4442.88 +4442.88j
    Pole 2: -4442.88 -4442.88j
  Zeros: None (all-pole filter)

======================================================================
Generation Summary
======================================================================

Generated: 10 circuits
Topology diversity: 3 unique structures
  • 3 nodes, 2 edges: 6 circuits
  • 4 nodes, 4 edges: 3 circuits
  • 5 nodes, 6 edges: 1 circuits

Transfer Function Accuracy:
  Pole count match: 10/10
  Zero count match: 10/10
  Stable circuits: 10/10 ✓
```

---

## Troubleshooting

**Q: I get "checkpoint not found"**
A: Run training first: `python scripts/train_tf_encoder.py --epochs 100`

**Q: Circuits have weird component values (22MΩ resistors, 80kH inductors)**
A: This is a known issue. Use `--values-variation 0.1` for more practical values. Post-processing rescaling coming soon.

**Q: Can I get higher-order filters (3rd, 4th order)?**
A: Yes! Use `--order 3` or `--order 4`

**Q: Can I get highpass or bandpass?**
A: Not yet directly. Coming in next update. For now, design as lowpass then transform manually.

**Q: How do I simulate the circuits?**
A: Export to SPICE with `export_spice_netlists.py`, then: `ngspice circuit_0.cir`

---

## Advanced Usage

See full documentation:
- `QUICKSTART_TARGETED_TF.md` - Direct pole/zero specification
- `docs/TARGETED_TF_GENERATION.md` - Technical details
- `GRAPHGPT_GENERATION_ANALYSIS.md` - Model accuracy analysis

---

## What Makes This Special

**Traditional Filter Design:**
1. Pick a filter type (Butterworth)
2. Calculate component values for Sallen-Key topology
3. Build it
4. Hope it works

**With Z-GED:**
1. Specify filter type + cutoff
2. Get 10 different topologies automatically
3. Pick the best one
4. Done!

**Advantage:** You get design exploration for free! 🚀

---

## Citation

If you use Z-GED in your work:

```bibtex
@software{zged2024,
  title={Z-GED: Graph-Based Circuit Generation with Transfer Function Control},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Z-GED}
}
```

---

## Need Help?

1. Check `QUICKSTART_TARGETED_TF.md` for examples
2. Read `docs/TARGETED_TF_GENERATION.md` for technical details
3. Open an issue on GitHub

Happy circuit designing! ⚡
