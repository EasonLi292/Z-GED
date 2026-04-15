# Inverse Design Model (v2): Admittance-Polynomial Encoder

## Overview

The v2 model replaces the v1 impedance-GNN encoder with a physics-informed **admittance-polynomial encoder** and adds **attribute prediction heads** for frequency, gain, and filter type. Together with the sequence decoder, this enables **spec-driven inverse design**: specify a target cutoff frequency, gain, and/or filter type, and the model generates valid RLC circuit topologies that match.

**Key differences from v1:**

| | v1 (HierarchicalEncoder) | v2 (AdmittanceEncoder) |
|---|---|---|
| Edge features | `[log10(R), log10(C), log10(L)]` | `[G/G_REF, C/C_REF, L_inv/L_INV_REF]` (admittance polynomials) |
| Latent dim | 8D `[topo(2) \| values(2) \| pz(4)]` | 5D `[topo(2) \| VIN(1) \| VOUT(1) \| GND(1)]` |
| Physics prior | Presence masks + value-conditioned linear | Parallel additivity + learned coefficient scaling |
| Conditioning | Pole/zero DeepSets branch | Attribute heads (freq, gain, type) on mu |
| Decoder | 4-layer, d=256 (3.3M params) | 2-layer, d=128 (441K params) |
| Training data | 1920 circuits, 8 types | 2400 circuits, 10 types |
| Generation | Random z or pole/zero spec | K-NN interpolation + gradient descent on mu |

---

## Architecture

### AdmittanceEncoder (84,828 parameters)

The encoder uses **admittance-polynomial edge features** that respect how passive components combine in parallel: admittances add. Each edge between two nets carries:

```
edge_attr = [G / G_REF,  C / C_REF,  L_inv / L_INV_REF]
```

where:
- `G = 1/R` (conductance), `G_REF = 1e-3` (1 kOhm reference)
- `C` (capacitance), `C_REF = 10^-7.5` (~31.6 nF reference)
- `L_inv = 1/L` (inverse inductance), `L_INV_REF = 1e3` (1 mH reference)

All three quantities are **parallel-additive**: two resistors in parallel have `G_total = G1 + G2`. Normalizing around 1.0 ensures the GNN processes values in a numerically stable range.

#### AdmittanceConv (message passing layer)

Each `AdmittanceConv` layer processes the three component channels with a fixed Box-Cox transform:

```
f(x) = 2 * (sqrt(1 + x) - 1)    [Box-Cox with gamma = 0.5]
```

- `f(0) = 0`: absent components contribute nothing
- `f'(0) = 1`: small admittances are treated linearly
- `f(1) = 0.83`, `f(10) = 4.63`, `f(100) = 18.1`: moderate compression of outliers
- Each channel has a 2-layer MLP phi function: `Linear(h, h) -> ReLU -> Linear(h, h, bias=False)`
- Messages from all three channels are summed

Gamma = 0.5 was selected by grid search over the Box-Cox family (see `ARCHITECTURE.md`). It is not learnable — a fixed transform avoids the init-bias problem where learnable scaling parameters stay near their initialization regardless of the true optimum.

#### Structured 5D latent space

After 3 layers of AdmittanceConv + global mean pooling, the encoder maps to a 5D latent:

```
mu = [z_topo(2) | z_VIN(1) | z_VOUT(1) | z_GND(1)]
```

- `z_topo[0:2]`: From global graph pooling — captures overall topology
- `z_VIN[2]`: From VIN node embedding — captures input-side structure
- `z_VOUT[3]`: From VOUT node embedding — captures output-side structure
- `z_GND[4]`: From GND node embedding — captures grounding structure

VAE mode: outputs `(z, mu, logvar)` where `z = mu + eps * exp(0.5 * logvar)`.

### SequenceDecoder (440,662 parameters)

Same architecture as v1 but smaller:
- 2 transformer layers, 4 attention heads, d_model=128
- Latent prefix conditioning (5D projected to 128D prefix token)
- 86-token vocabulary (same as v1)
- max_seq_len = 33 (32 walk tokens + 1 prefix)

### Attribute Heads (5,118 parameters total)

Three lightweight heads predict circuit attributes from `mu` (not sampled `z`):

| Head | Architecture | Output | Loss |
|------|-------------|--------|------|
| **FreqHead** | Linear(5,64) -> ReLU -> Dropout -> Linear(64,1) | log10(fc) | MSE |
| **GainHead** | Linear(5,64) -> ReLU -> Dropout -> Linear(64,64) -> ReLU -> Dropout -> Linear(64,1) | \|H(1kHz)\| | MSE |
| **TypeHead** | Linear(5,10) | 10-way logits | Cross-entropy |

---

## Training

### Dataset

2400 circuits from two dataset files:
- `rlc_dataset/filter_dataset.pkl` — 1920 circuits, 8 types (240 each): low_pass, high_pass, band_pass, band_stop, rlc_series, rlc_parallel, lc_lowpass, cl_highpass
- `rlc_dataset/rl_dataset.pkl` — 480 circuits, 2 types (240 each): rl_lowpass, rl_highpass

All 10 types: `band_pass, band_stop, cl_highpass, high_pass, lc_lowpass, low_pass, rl_highpass, rl_lowpass, rlc_parallel, rlc_series`

### Loss function

```
total_loss = ce_loss + kl_weight * kl_loss + w_freq * freq_mse + w_gain * gain_mse + w_type * type_ce
```

- **ce_loss**: Next-token cross-entropy (teacher forcing on Eulerian walk)
- **kl_loss**: KL divergence `D_KL(q(z|x) || N(0,I))`
- **kl_weight**: Linear warmup over first 30 epochs, max 0.005
- **freq_mse**: MSE on log10(fc) prediction from mu (weight 0.1)
- **gain_mse**: MSE on |H(1kHz)| prediction from mu (weight 0.1)
- **type_ce**: Cross-entropy on filter type classification from mu (weight 0.1)

Optimizer: AdamW, lr=3e-4, weight_decay=1e-4. Scheduler: ReduceLROnPlateau (factor=0.5, patience=15).

### Training command

```bash
.venv/bin/python scripts/training/train_inverse_design.py
```

Checkpoint saved to `checkpoints/production/best_v2.pt`.

---

## Generation Pipeline

The generation pipeline (`scripts/generation/generate_inverse_design.py`) uses a three-step process:

### Step 1: K-NN Interpolation

Given target attributes (type, frequency, gain), find the K nearest training circuits by attribute distance and compute a weighted average of their latent codes:

```
mu_init = sum(w_i * mu_i)  where w_i = 1/d_i / sum(1/d_j)
```

Distance includes frequency distance (normalized), gain distance (normalized), and a large penalty (10x) for type mismatch.

### Step 2: Gradient Descent on mu

Starting from `mu_init`, optimize mu through the frozen attribute heads to minimize:

```
L = w_freq * (freq_pred - target_freq)^2
  + w_gain * (gain_pred - target_gain)^2
  + w_type * CE(type_logits, target_type)
  + w_prior * ||mu||^2
```

The prior term keeps mu close to N(0,1) to stay in-distribution. Default: 200 Adam steps at lr=0.05.

### Step 3: Decode

Sample walks from the decoder conditioned on the optimized mu. Validate for well-formedness and electrical validity.

### Usage

```bash
# Generate a band_pass filter at 10 kHz
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type band_pass --fc 10000

# Generate with gain target
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type band_pass --fc 10000 --gain 0.5

# More samples, higher temperature
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type low_pass --fc 1000 --samples 20 --temperature 1.5

# Frequency only (no type constraint)
.venv/bin/python scripts/generation/generate_inverse_design.py --fc 50000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--type` | None | Target filter type (one of 10 types) |
| `--fc` | None | Target cutoff frequency in Hz |
| `--gain` | None | Target \|H(1kHz)\| gain |
| `--samples` | 10 | Number of circuits to generate |
| `--temperature` | 0.1 | Decoder sampling temperature |
| `--ckpt` | `checkpoints/production/best_v2.pt` | Model checkpoint |
| `--k` | 10 | K for K-NN interpolation |
| `--opt-steps` | 200 | Gradient descent steps |

---

## Behavioral Inverse Design

A second script (`scripts/generation/inverse_design.py`) tests whether the decoder can recover topologies from behavioral signatures alone. It:

1. Encodes training circuits to get reference latent codes
2. Tests three target types:
   - **Hold-out**: Real `h` from a training circuit — can the decoder rediscover the source topology?
   - **Pairwise centroid blends**: `(h_a + h_b)/2` — forces the decoder to interpolate between behaviors
   - **Centroid + Gaussian noise**: Robustness probe

```bash
.venv/bin/python scripts/generation/inverse_design.py \
  --n-samples 1000 --temperature 1.0
```

---

## Analysis

The latent space analysis script (`scripts/analysis/analyze_latent.py`) provides:

- Per-type centroid and variance analysis
- Separation ratio (inter-type / intra-type distance)
- Attribute head accuracy evaluation
- Topology signature analysis
- Nearest-neighbor hit rates

```bash
.venv/bin/python scripts/analysis/analyze_latent.py
```

---

## Results

### Latent Space Quality

- **Separation ratio**: 7.4x (inter-type distance / intra-type distance)
- **Attribute prediction**: FreqHead and TypeHead achieve high accuracy; GainHead is weaker due to gain variation within types

### Generation Quality

| Temperature | Validity Rate | Novel Topologies (per 2000 samples) |
|-------------|--------------|-------------------------------------|
| 0.1 | ~100% | 0 |
| 0.7 | ~100% | 0 |
| 1.0 | ~97% | 0 |
| 1.5 | ~88% | 1-3 |
| 2.0 | ~60% | more, but low validity |

The decoder is conservative: it reliably generates known topologies with near-perfect validity at low temperatures, but rarely proposes novel structures.

---

## Limitations

1. **Novel topology generation**: The decoder strongly favors training topologies. Novel structures only appear at high temperatures where validity drops.
2. **Gain prediction**: The GainHead is less accurate than FreqHead/TypeHead because gain varies significantly within each filter type.
3. **Component values**: The model generates topology only (which components connect where), not component values. Values must be determined separately (e.g., via optimization or lookup).
4. **Dataset scope**: Limited to passive RLC filters. No active components, no multi-stage filters.

---

## File Reference

| File | Description |
|------|-------------|
| `ml/models/admittance_encoder.py` | AdmittanceConv + AdmittanceEncoder |
| `ml/models/attribute_heads.py` | FreqHead, GainHead, TypeHead, kl_divergence |
| `ml/models/constants.py` | G_REF, C_REF, L_INV_REF, FILTER_TYPES_V2, TYPE_TO_IDX |
| `ml/data/cross_topo_dataset.py` | CrossTopoSequenceDataset + collate_fn |
| `ml/utils/runtime.py` | build_v2_encoder, build_v2_decoder, load_v2_model |
| `scripts/generation/generate_inverse_design.py` | Spec-driven generation CLI |
| `scripts/generation/inverse_design.py` | Behavioral inverse design pipeline |
| `scripts/training/train_inverse_design.py` | v2 training script |
| `scripts/analysis/analyze_latent.py` | Latent space analysis |
| `tests/unit/test_admittance_encoder.py` | Physics sanity tests |
| `checkpoints/production/best_v2.pt` | Trained v2 checkpoint |
