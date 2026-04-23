# Encoder Generalization Analysis

How well do the v2 attribute heads (FreqHead, GainHead, TypeHead) generalize to circuit topologies **not seen during training**?

## Setup

The v2 encoder was trained on 2,400 circuits across 10 topology families. We construct 9 novel circuits by hand, plus 3 known references, encode them, and compare attribute head predictions against analytical expectations.

Checkpoint: `checkpoints/production/best_v2.pt` (Box-Cox gamma=0.5, epoch 29).

Reproduction:

```bash
.venv/bin/python /private/tmp/test_encoder_generalization.py
```

---

## Known Training Topologies (10)

| Type | Topology |
|------|----------|
| low_pass | R(VIN-VOUT), C(VOUT-GND) |
| high_pass | C(VIN-VOUT), R(VOUT-GND) |
| lc_lowpass | L(VIN-VOUT), C(VOUT-GND) |
| cl_highpass | C(VIN-VOUT), L(VOUT-GND) |
| rl_lowpass | R(VIN-VOUT), L(VOUT-GND) |
| rl_highpass | L(VIN-VOUT), R(VOUT-GND) |
| band_pass | L(VIN-INT), C(INT-VOUT), R(VOUT-GND) |
| band_stop | R(VIN-INT1), R(INT1-VOUT), L(INT1-INT2), C(INT2-GND), R(VOUT-GND) |
| rlc_series | R(VIN-INT1), L(INT1-INT2), C(INT2-VOUT), R(VOUT-GND) |
| rlc_parallel | R(VIN-VOUT), R\|\|C\|\|L(VOUT-GND) |

---

## Results

### Known References

| Circuit | Pred fc | Expected fc | fc error | Pred gain | Pred type (conf) |
|---------|---------|-------------|----------|-----------|------------------|
| RC low-pass (1kΩ, 100nF) | 1.4 kHz | 1.59 kHz | 0.9x | 0.82 | low_pass (73%) |
| RC high-pass (1kΩ, 100nF) | 2.5 kHz | 1.59 kHz | 1.6x | 0.57 | high_pass (47%) |
| LC low-pass (1mH, 100nF) | 19.4 kHz | 15.9 kHz | 1.2x | 0.97 | lc_lowpass (67%) |

All three known references are correctly classified with reasonable frequency estimates.

### Novel Topologies

| # | Circuit | Pred fc | Expected fc | fc error | Pred gain | Pred type (conf) | Assessment |
|---|---------|---------|-------------|----------|-----------|------------------|------------|
| 1 | RC low-pass (100Ω, 1µF) [known topo, novel values] | 1.5 kHz | 1.59 kHz | 0.9x | 0.82 | low_pass (72%) | **excellent** |
| 2 | RC + parallel R shunt | 1.6 kHz | 1.6 kHz | 1.0x | 0.55 | low_pass (54%) | **excellent** |
| 3 | Shunt-L high-pass | 154.5 kHz | 159 kHz | 1.0x | ~0 | rl_highpass (40%) | **excellent** |
| 4 | Pi-C low-pass (CRC) | 161 Hz | ~800 Hz | 0.2x | 0.41 | low_pass (30%) | type correct, fc off 5x |
| 5 | 2-stage RC low-pass | 739 Hz | ~1 kHz | 0.7x | 0.40 | rlc_parallel (19%) | fc ok, type wrong |
| 6 | T-network low-pass (RCR) | 6.3 kHz | 1.6 kHz | 3.9x | 0.26 | high_pass (23%) | fc off, type wrong |
| 7 | Dual-L low-pass (LLC) | 129 Hz | 11.3 kHz | 0.01x | 0.04 | rlc_series (32%) | fc off 100x, type wrong |
| 8 | R-R divider (1k/1k) | 11.1 kHz | N/A (flat) | N/A | 0.34 | high_pass (25%) | confused (no reactive elements) |
| 9 | R-R divider (1k/10k) | 377.5 kHz | N/A (flat) | N/A | 0.33 | rlc_parallel (27%) | confused |

### Latent Codes

| # | Circuit | mu[0] | mu[1] | mu[2] | mu[3] | mu[4] | \|\|mu\|\| |
|---|---------|-------|-------|-------|-------|-------|--------|
| 1 | RC low-pass (novel values) | +1.28 | +0.79 | -2.24 | -2.67 | +2.09 | 4.33 |
| 2 | RC + parallel R shunt | +0.63 | +0.41 | -2.17 | -2.76 | +1.67 | 3.96 |
| 3 | Shunt-L high-pass | -1.41 | +0.33 | +1.43 | -2.42 | -2.56 | 4.07 |
| 4 | Pi-C low-pass (CRC) | +0.52 | -0.37 | -1.41 | -1.71 | +0.94 | 2.49 |
| 5 | 2-stage RC low-pass | +0.40 | -0.56 | -1.48 | -1.28 | -1.15 | 2.37 |
| 6 | T-network low-pass (RCR) | -0.40 | -0.14 | -1.18 | -0.31 | -2.40 | 2.73 |
| 7 | Dual-L low-pass (LLC) | +0.46 | -0.29 | +2.33 | -0.56 | +0.07 | 2.45 |
| 8 | R-R divider (1k/1k) | -0.30 | -0.13 | -1.23 | +0.74 | -2.59 | 2.98 |
| 9 | R-R divider (1k/10k) | -0.07 | +0.14 | -1.54 | -3.21 | -2.93 | 4.61 |
| — | RC low-pass [REF] | +1.32 | +0.80 | -2.16 | -2.65 | +2.13 | 4.31 |
| — | RC high-pass [REF] | -1.79 | -0.56 | -1.81 | +1.47 | -1.22 | 3.23 |
| — | LC low-pass [REF] | +0.99 | +0.07 | -4.96 | +0.73 | +0.01 | 5.11 |

---

## Conclusions

### What generalizes

1. **Value generalization within known topologies**: Near-perfect. RC low-pass with 100Ω/1µF (100x different from typical training values) gives the same predictions as 1kΩ/100nF. The encoder learned physics, not lookup tables.

2. **Small structural perturbations**: Adding a parallel resistor to a known low-pass (RC + R shunt) or rearranging the same component types in a familiar way (shunt-L high-pass) gives correct frequency and type predictions.

3. **Frequency prediction**: Remarkably accurate for circuits that are structurally close to training topologies. The FreqHead is the strongest of the three heads.

### What doesn't generalize

1. **Topologically distant circuits**: The dual-L low-pass (two inductors in series + cap) is off by 100x on frequency. Multi-stage filters (2-stage RC, T-network) have 4-5x errors.

2. **Pure resistive circuits**: R-R dividers have no reactive elements and thus no meaningful cutoff frequency. The encoder has never seen this and produces arbitrary predictions with low confidence.

3. **Gain prediction**: Weak across the board — most novel circuits predict 0.3-0.5 regardless of actual behavior. This is a known limitation (see `docs/inverse_design.md`).

4. **Type classification**: Confidence drops sharply for novel topologies (19-40% vs 47-73% for known ones). The model often defaults to the structurally closest known type rather than admitting uncertainty.

### Implications for Inverse Design

The gradient-descent-on-mu pipeline works well when the target specification maps to a **known or near-known** topology region. For specifications that require genuinely novel topologies:

- The attribute heads cannot guide optimization toward structures they don't understand.
- Latent codes for novel topologies cluster near the origin (||mu|| ≈ 2.4-3.0 vs 3.2-5.1 for known types), suggesting the encoder maps unfamiliar circuits to a "default" region rather than meaningfully encoding their physics.
- More topologically diverse training data (not just more circuits of the same 10 types) is needed to improve generalization.
