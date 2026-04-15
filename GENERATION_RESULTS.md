# Circuit Generation Results

**Dataset:** 1920 circuits (8 filter types, 240 each), 1536 train, 384 validation
**Checkpoint:** `checkpoints/production/best.pt`
**Decoder:** Sequence decoder (GPT-style, Eulerian walk representation)
**Edge features:** 3D log10 values `[log10(R), log10(C), log10(L)]`
**Latent space:** 8D hierarchical `[z_topo(2) | z_values(2) | z_pz(4)]`

---

## Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| CE Loss | 0.035 | 0.021 |
| KL Loss | 0.725 | 0.758 |
| Token Accuracy | 98.2% | 98.6% |
| Topology Match | -- | 100% |
| Valid Walk Rate | -- | 100% |
| Encoder Parameters | 237,907 | -- |
| Decoder Parameters | 3,280,726 | -- |

---

## Specification-Based Generation (Interpolation Analysis)

Exploratory scripts can still map **cutoff frequency** and **Q-factor** to latent codes using K-NN interpolation over encoded dataset points.
For that path, use `scripts/testing/*.py` or `scripts/generation/regenerate_all_results.py`.

For production generation entry point, use pole/zero inputs:

```bash
.venv/bin/python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5
```

### Standard Examples

| Cutoff | Q | Generated Circuit | Status |
|--------|---|-------------------|--------|
| 1 kHz | 0.707 | `VOUT--C--VSS, VIN--R--VOUT` | Valid |
| 10 kHz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Valid |
| 100 kHz | 0.707 | `VIN--C--VOUT, VOUT--L--VSS` | Valid |
| 10 kHz | 5.0 | `VIN--R--VOUT, VOUT--RCL--VSS` | Valid |

### Edge Cases

| Cutoff | Q | Generated Circuit | Analysis |
|--------|---|-------------------|----------|
| 1 Hz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Extrapolates beyond training range |
| 1 MHz | 0.707 | `VIN--C--VOUT, VOUT--R--VSS` | Extrapolates beyond training range |
| 10 kHz | 0.01 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` | Low Q (rlc_series) |
| 10 kHz | 0.1 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` | band_pass |
| 10 kHz | 2.0 | `VIN--C--VOUT, VOUT--R--VSS` | high_pass |
| 50 Hz | 5.0 | `VIN--R--VOUT, VOUT--RCL--VSS` | High Q (rlc_parallel) |
| 500 kHz | 0.1 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` | rlc_series |

### Training Data Coverage

| Parameter | Min | Max |
|-----------|-----|-----|
| Cutoff frequency | 2.05 Hz | 872,085 Hz |
| Q-factor | 0.01 | 6.58 |

---

## Filter Type Centroids

The 8D latent space clusters by filter type. Generating from cluster centroids:

| Filter Type | z[0:4] | Generated from Centroid |
|-------------|--------|-------------------------|
| low_pass | [-0.41, +0.49, +1.76, -0.95] | `VOUT--C--VSS, VIN--R--VOUT` |
| high_pass | [-0.45, +0.41, -1.09, -0.00] | `VIN--C--VOUT, VOUT--R--VSS` |
| band_pass | [+0.08, +0.13, -2.52, +1.90] | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| band_stop | [+0.26, -0.06, +2.32, +2.09] | `INTERNAL_2--C--VSS, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN, INTERNAL_1--R--VOUT` |
| rlc_series | [+0.02, +0.17, -0.23, +3.03] | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` |
| rlc_parallel | [+2.63, -1.01, -0.18, -0.69] | `VIN--R--VOUT, VOUT--RCL--VSS` |
| lc_lowpass | [-1.12, -0.06, +1.66, -2.14] | `VOUT--C--VSS, VIN--L--VOUT` |
| cl_highpass | [-0.93, -0.20, -1.75, -1.99] | `VIN--C--VOUT, VOUT--L--VSS` |

---

## Reconstruction Accuracy

### By Filter Type (240 circuits each)

| Filter Type | Valid | Example Reconstruction |
|-------------|-------|------------------------|
| low_pass | 240/240 | `VOUT--C--VSS, VIN--R--VOUT` |
| high_pass | 240/240 | `VIN--C--VOUT, VOUT--R--VSS` |
| band_pass | 240/240 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| band_stop | 240/240 | `INTERNAL_2--C--VSS, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN, INTERNAL_1--R--VOUT` |
| rlc_series | 240/240 | `INTERNAL_2--C--VOUT, INTERNAL_1--L--INTERNAL_2, VOUT--R--VSS, INTERNAL_1--R--VIN` |
| rlc_parallel | 240/240 | `VIN--R--VOUT, VOUT--RCL--VSS` |
| lc_lowpass | 240/240 | `VOUT--C--VSS, VIN--L--VOUT` |
| cl_highpass | 240/240 | `VIN--C--VOUT, VOUT--L--VSS` |

**Total: 1920/1920 (100%) valid reconstructions**

---

## Interpolation

### Low-pass -> High-pass

R and C swap positions at alpha ~ 0.50:

| alpha | Generated |
|---|-----------|
| 0.00 | `VOUT--C--VSS, VIN--R--VOUT` (low-pass) |
| 0.25 | `VOUT--C--VSS, VIN--R--VOUT` |
| 0.50 | `VIN--C--VOUT, VOUT--R--VSS` (transition) |
| 0.75 | `VIN--C--VOUT, VOUT--R--VSS` |
| 1.00 | `VIN--C--VOUT, VOUT--R--VSS` (high-pass) |

### Band-pass -> RLC-parallel

Distributed -> lumped transition:

| alpha | Generated |
|---|-----------|
| 0.00 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` (distributed LC) |
| 0.25 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` |
| 0.50 | `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` (transition) |
| 0.75 | `VIN--R--VOUT, VOUT--RCL--VSS` |
| 1.00 | `VIN--R--VOUT, VOUT--RCL--VSS` (lumped RCL) |

---

## Novel Topology Generation

### Multi-Target Temperature Sweep (V2)

Comprehensive sweep across 4 target specs, 5 temperatures, and 5 random seeds.
For each target, the V2 pipeline computes one optimized latent via K-NN interpolation
+ gradient descent (no filter-type constraint), then decodes 2,000 samples per
(temperature, seed) combination. Total: 200,000 decoded samples.

Setup:

- Checkpoint: `checkpoints/production/best_v2.pt`
- Dataset: `rlc_dataset/filter_dataset.pkl` + `rlc_dataset/rl_dataset.pkl`
- Training set: 2,400 circuits, **10 known topology signatures**
- Seeds: 0, 1, 2, 3, 4
- Samples per (target, temperature, seed): 2,000

| Target | Temperature | Valid (mean +/- std) | Rate | Unique Topos | Novel / seed | Novel total |
|--------|-------------|----------------------|------|--------------|-------------|-------------|
| fc=1kHz | 0.5 | 2000 +/- 0.0 | 100.0% | 6.0 | 0.00 | 0 |
| fc=1kHz | 0.7 | 2000 +/- 0.5 | 100.0% | 6.0 | 0.00 | 0 |
| fc=1kHz | 1.0 | 1986 +/- 3.1 | 99.3% | 6.0 | 0.00 | 0 |
| fc=1kHz | 1.2 | 1944 +/- 5.5 | 97.2% | 6.2 | 0.00 | 0 |
| fc=1kHz | 1.5 | 1742 +/- 8.5 | 87.1% | 8.4 | 1.00 | 5 |
| | | | | | | |
| fc=10kHz, g=0.5 | 0.5 | 1998 +/- 1.7 | 99.9% | 1.2 | 0.00 | 0 |
| fc=10kHz, g=0.5 | 0.7 | 1990 +/- 4.1 | 99.5% | 3.0 | 0.00 | 0 |
| fc=10kHz, g=0.5 | 1.0 | 1939 +/- 9.0 | 96.9% | 6.6 | 0.20 | 1 |
| fc=10kHz, g=0.5 | 1.2 | 1814 +/- 13.7 | 90.7% | 8.0 | 0.40 | 2 |
| fc=10kHz, g=0.5 | 1.5 | 1310 +/- 16.0 | 65.5% | 9.2 | 0.60 | 3 |
| | | | | | | |
| fc=50kHz | 0.5 | 2000 +/- 0.0 | 100.0% | 2.0 | 0.00 | 0 |
| fc=50kHz | 0.7 | 1999 +/- 1.4 | 99.9% | 3.2 | 0.00 | 0 |
| fc=50kHz | 1.0 | 1983 +/- 2.9 | 99.2% | 3.8 | 0.00 | 0 |
| fc=50kHz | 1.2 | 1937 +/- 5.3 | 96.8% | 5.4 | 0.20 | 1 |
| fc=50kHz | 1.5 | 1730 +/- 3.9 | 86.5% | 6.8 | 1.00 | 5 |
| | | | | | | |
| gain=0.1 | 0.5 | 2000 +/- 0.0 | 100.0% | 4.4 | 0.00 | 0 |
| gain=0.1 | 0.7 | 1999 +/- 1.0 | 99.9% | 5.6 | 0.00 | 0 |
| gain=0.1 | 1.0 | 1989 +/- 2.5 | 99.4% | 7.6 | 0.20 | 1 |
| gain=0.1 | 1.2 | 1947 +/- 5.9 | 97.3% | 9.2 | 0.20 | 1 |
| gain=0.1 | 1.5 | 1728 +/- 9.2 | 86.4% | 10.4 | 0.80 | 4 |

Across all 200,000 samples, **4 unique novel topologies** were discovered:

| # | Topology | Count | Source |
|---|----------|-------|--------|
| 1 | `C1(VIN--VOUT), RCL1(VOUT--VSS)` [1C+1RCL] | x4 | fc=1kHz / T=1.5 |
| 2 | `R1(VIN--VOUT), R3(VOUT--VSS)` [2R] | x2 | fc=10kHz,g=0.5 / T=1.2 |
| 3 | `C1(INT2--VSS), L1(INT1--INT2), R1(VOUT--VSS), R2(INT1--VIN)` [1C+1L+2R] | x1 | fc=10kHz,g=0.5 / T=1.2 |
| 4 | `R1(VOUT--VSS), R2(INT1--VIN), R3(INT1--VOUT)` [3R] | x1 | gain=0.1 / T=1.5 |

Key observations:

- **T <= 0.7**: Zero novel topologies across all targets and seeds. Validity near 100%.
- **T = 1.0-1.2**: Occasional novel topologies (0-2 total across 5 seeds). Validity 91-99%.
- **T = 1.5**: ~1 novel topology per 2,000 samples on average. Validity drops to 65-87%.
- The training set has only 10 known topology signatures. The decoder has effectively memorized these and strongly resists generating anything else.
- Novel topologies require temperatures high enough to partially break the learned distribution, at which point validity also collapses.

### Random Latent Temperature Sweep

Random latent samples were decoded with sampling enabled (`greedy=False`) at different decoder temperatures.
Validity here means an electrically valid Eulerian walk: well-formed sequence, no self-loop component, VIN/VOUT/VSS each connected, and no dangling internal net.
Novelty is counted only among valid samples, using topology signatures that are invariant to traversal order and internal-net renaming.
This is stricter than the older `sequence_to_topology_key` helper, which can assign topology strings to walks that contain self-loops, repeated malformed components, or dangling internal nets.

Reproduction:

```bash
.venv/bin/python scripts/analysis/temperature_topology_sweep.py \
  --samples 500 \
  --out analysis_results/temperature_topology_sweep_500.json
```

| Temperature | Valid | Invalid | Valid Novel | Novel % of Valid | Unique Valid Topologies | Dominant Valid Topology |
|-------------|-------|---------|-------------|------------------|-------------------------|-------------------------|
| 0.1 | 497/500 (99.4%) | 3/500 (0.6%) | 0/497 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (184) |
| 0.3 | 498/500 (99.6%) | 2/500 (0.4%) | 0/498 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (180) |
| 0.7 | 498/500 (99.6%) | 2/500 (0.4%) | 0/498 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (176) |
| 1.0 | 494/500 (98.8%) | 6/500 (1.2%) | 0/494 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (172) |
| 1.3 | 454/500 (90.8%) | 46/500 (9.2%) | 0/454 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (160) |
| 1.7 | 306/500 (61.2%) | 194/500 (38.8%) | 0/306 | 0.0% | 8 | `C(VIN-VOUT)\|R(VOUT-VSS)` (102) |
| 2.0 | 131/500 (26.2%) | 369/500 (73.8%) | 0/131 | 0.0% | 7 | `C(VIN-VOUT)\|R(VOUT-VSS)` (53) |
| 2.5 | 13/500 (2.6%) | 487/500 (97.4%) | 0/13 | 0.0% | 5 | `C(VIN-VOUT)\|R(VOUT-VSS)` (6) |
| 3.0 | 1/500 (0.2%) | 499/500 (99.8%) | 0/1 | 0.0% | 1 | `C(VIN-VOUT)\|R(VOUT-VSS)` (1) |

### Frequent Topologies

The most common valid outputs remain known training topologies at every temperature. Higher temperature reduces validity before it creates valid novel structures.

| Temperature | Frequent Valid Topologies |
|-------------|---------------------------|
| 0.1 | `C(VIN-VOUT)\|R(VOUT-VSS)` (184); `C(VOUT-VSS)\|R(VIN-VOUT)` (61); `C(INTERNAL_2-VOUT)\|L(INTERNAL_1-INTERNAL_2)\|R(VOUT-VSS)\|R(INTERNAL_1-VIN)` (59); `C(VIN-VOUT)\|L(VOUT-VSS)` (56); `R(VIN-VOUT)\|RCL(VOUT-VSS)` (43) |
| 1.0 | `C(VIN-VOUT)\|R(VOUT-VSS)` (172); `C(VIN-VOUT)\|L(VOUT-VSS)` (62); `C(INTERNAL_2-VOUT)\|L(INTERNAL_1-INTERNAL_2)\|R(VOUT-VSS)\|R(INTERNAL_1-VIN)` (60); `C(VOUT-VSS)\|R(VIN-VOUT)` (60); `R(VIN-VOUT)\|RCL(VOUT-VSS)` (47) |
| 1.7 | `C(VIN-VOUT)\|R(VOUT-VSS)` (102); `C(VIN-VOUT)\|L(VOUT-VSS)` (48); `R(VIN-VOUT)\|RCL(VOUT-VSS)` (40); `C(VOUT-VSS)\|R(VIN-VOUT)` (35); `C(VOUT-VSS)\|L(VIN-VOUT)` (33) |
| 2.5 | `C(VIN-VOUT)\|R(VOUT-VSS)` (6); `C(VOUT-VSS)\|R(VIN-VOUT)` (3); `R(VIN-VOUT)\|RCL(VOUT-VSS)` (2); `C(INTERNAL_1-VOUT)\|L(INTERNAL_1-VIN)\|R(VOUT-VSS)` (1); `C(VIN-VOUT)\|L(VOUT-VSS)` (1) |

Across 4,500 random latent samples, the decoder produced **0 electrically valid novel topologies** under this stricter validity definition. Temperature mainly trades valid known topologies for malformed or electrically invalid walks.

### Validator Sensitivity

The older topology-key metric explains why some earlier or external-looking novelty numbers can look much higher. It only asks whether a generated walk can be converted into a topology string; it does not reject every electrically invalid construction.

| Decode Mode | Metric | Valid | Valid Novel | Novel % of Valid | Example Novel Outputs |
|-------------|--------|-------|-------------|------------------|-----------------------|
| Greedy | strict electrical walk | 499/500 (99.8%) | 1/499 | 0.2% | `L(VIN-VOUT)\|RCL(VOUT-VSS)` |
| Greedy | topology-key only | 499/500 (99.8%) | 1/499 | 0.2% | `L(VIN-VOUT)\|RCL(VOUT-VSS)` |
| T=1.0 sampling | strict electrical walk | 494/500 (98.8%) | 0/494 | 0.0% | -- |
| T=1.0 sampling | topology-key only | 496/500 (99.2%) | 2/496 | 0.4% | includes dangling/renamed internal variants |
| T=1.7 sampling | strict electrical walk | 306/500 (61.2%) | 0/306 | 0.0% | -- |
| T=1.7 sampling | topology-key only | 431/500 (86.2%) | 99/431 | 23.0% | includes self-loops and malformed multi-component variants |
| T=2.5 sampling | strict electrical walk | 13/500 (2.6%) | 0/13 | 0.0% | -- |
| T=2.5 sampling | topology-key only | 195/500 (39.0%) | 172/195 | 88.2% | many contain `VOUT-VOUT`, `VSS-VSS`, or dangling internals |
| T=3.0 sampling | strict electrical walk | 1/500 (0.2%) | 0/1 | 0.0% | -- |
| T=3.0 sampling | topology-key only | 85/500 (17.0%) | 83/85 | 97.6% | mostly malformed self-loop-heavy strings |

The stricter metric is the one used in the temperature sweep because it better matches whether a topology is usable as a circuit graph. The topology-key-only metric is useful as a debugging signal: it shows that temperature increases structural variety, but most of that variety is not valid circuit structure.

### Comparison To AnalogGenie Paper

The AnalogGenie paper reports much higher novelty because it studies a different task and model scale:

- Dataset: 3350 distinct analog IC topologies across 11 circuit families, expanded to 227,766 augmented sequences.
- Model: decoder-only transformer with 11.825M parameters, 1029-token vocabulary, and maximum sequence length 1024.
- Validity: SPICE-simulatable with default sizing, checking for floating and shorting nodes.
- Novelty: different from every topology in their full dataset.

This repo's result is from a much smaller sequence decoder trained on 8 RLC filter topology families. Its random-latent samples mostly recover those known families. The numbers are therefore not directly comparable to AnalogGenie's Table 1.

---

## Pole/Zero-Driven Generation

Generate circuits by specifying **dominant pole/zero** directly. The decoder constructs z[4:8] from signed-log normalized pole/zero values, samples z[0:4] from the prior, and generates using the decoder only (no encoder needed).

```bash
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5
```

### Signed-Log Normalization

`z = sign(x) * log10(|x| + 1) / 7.0` maps raw pole/zero values to ~[-1, 1]:

| z[4:8] | Meaning |
|--------|---------|
| z[4] = sigma_p | signed_log(real part of dominant pole) |
| z[5] = omega_p | signed_log(\|imag part of dominant pole\|) |
| z[6] = sigma_z | signed_log(real part of dominant zero) |
| z[7] = omega_z | signed_log(\|imag part of dominant zero\|) |

### Test Cases (3 samples each)

#### Real Pole, No Zero (RC Low-Pass Behavior)

| Pole | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -6283 + 0j (~1kHz) | [-0.543, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |
| -62832 + 0j (~10kHz) | [-0.685, 0, 0, 0] | band_pass, `R--C`, `R--C` | 3/3 |
| -628318 + 0j (~100kHz) | [-0.828, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |

#### Real Zero, No Pole (RC High-Pass Behavior)

| Zero | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -6283 + 0j (~1kHz) | [0, 0, -0.543, 0] | `R--C`, `R--C`, rlc_parallel | 3/3 |

#### Conjugate Poles (Resonant Behavior)

| Pole | Zero | z[4:8] | Samples | Valid |
|------|------|--------|---------|-------|
| -3142 + 49348j | 0 | [-0.500, +0.670, 0, 0] | `R--C`, band_pass, `R--C` | 3/3 |
| -3142 + 49348j | 0 + 49348j | [-0.500, +0.670, 0, +0.670] | rlc_series, `R--C`, `R--C` | 3/3 |

#### Extreme Frequencies

| Pole | z[4:8] | Samples | Valid |
|------|--------|---------|-------|
| -100 + 0j (very low) | [-0.286, 0, 0, 0] | `R--C`, `R--C`, `R--C` | 3/3 |
| -1e6 + 0j (very high) | [-0.857, 0, 0, 0] | rlc_parallel, `R--C`, `R--C` | 3/3 |

### Summary

| Category | Valid/Total |
|----------|------------|
| Real pole, no zero | 9/9 (100%) |
| Real zero, no pole | 3/3 (100%) |
| Conjugate poles | 6/6 (100%) |
| Extreme frequencies | 6/6 (100%) |
| **Overall** | **24/24 (100%)** |

**Observations:**
- 100% valid rate across all pole/zero test cases (up from 83% with adjacency decoder)
- Real-pole-only specs reliably produce RC topologies
- The z[0:4] prior sampling introduces topology diversity (same pole/zero can yield different structures)
- Extreme frequency poles now reliably produce valid topologies

---

## Key Insights

### 1. Sequence Decoder vs Adjacency Decoder

| Metric | Adjacency Decoder | Sequence Decoder |
|--------|-------------------|------------------|
| Reconstruction accuracy | 100% | 100% |
| Valid rate (random samples) | 89.2% | 98.8% at T=1.0; 99.6% at T=0.3/0.7 |
| P/Z generation valid rate | 83% | 100% |
| Novel topologies (500 samples) | 9 unique, 112 samples | 0 electrically valid novel samples at T=0.1-3.0 |
| Decoder parameters | 7,698,901 | 3,280,726 |

### 2. Q-Factor Drives Topology Selection

| Q Range | Typical Topology |
|---------|------------------|
| Q < 0.1 | 4-node rlc_series |
| Q ~ 0.1 | 3-node band_pass (distributed LC) |
| Q ~ 0.707 | 2-component RC (Butterworth) |
| Q > 2.0 | 2-component RCL parallel (tank) |

### 3. Latent Space is Well-Organized

- All 8 dimensions are meaningful (z[0:4] topology, z[4:8] pole/zero)
- Filter types form distinct clusters
- Interpolation produces smooth transitions
- Random-latent validity remains high through T=1.0, then falls quickly at hotter temperatures

### 4. High Fidelity = Few Novel Topologies

The sequence decoder strongly prefers known training topologies. Raising temperature reduces validity before it creates electrically valid novel circuits, so exploration needs a stronger intervention than temperature alone. The adjacency decoder's lower fidelity paradoxically produced more novel structures, including some physically meaningful ones. This is a precision-vs-exploration tradeoff.

---

## Usage

```bash
# Pole/zero-driven generation (decoder only, no encoder needed)
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5

# Conjugate pole pair (resonant circuit)
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --num-samples 5

# With a zero (band-stop behavior)
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --zero-imag 49348

# Regenerate all results
python scripts/generation/regenerate_all_results.py
```

---

## Files

- **Model:** `checkpoints/production/best.pt`
- **Archived model:** `checkpoints/production/best_adjacency.pt`
- **Dataset:** `rlc_dataset/filter_dataset.pkl` (1920 circuits, 8 types)
- **Generation script:** `scripts/generation/generate_from_specs.py`
- **Results script:** `scripts/generation/regenerate_all_results.py`
