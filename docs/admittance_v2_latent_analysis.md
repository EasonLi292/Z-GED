# V2 Admittance Encoder: 5D Structured Latent Space Analysis

**Checkpoint:** `eason/latent_5d/best_inverse_design_v2.pt` (epoch 64)
**Dataset:** 2400 circuits, 10 filter types (240 each), admittance-polynomial edge features

---

## 0. Model Architecture

### Encoder: AdmittanceEncoder v2

Physics-informed GNN with VAE bottleneck. 3 message-passing layers, hidden_dim=64. **84,828 parameters.**

Each layer computes messages via the admittance polynomial prior:

```
msg = g_eff * phi_G(x_j) + c_eff * phi_C(x_j) + l_eff * phi_L(x_j)
```

where:
- `phi_G/C/L` are 2-layer MLPs: `Linear(64,64) -> ReLU -> Linear(64,64, bias=False)`. The `bias=False` on the outer layer preserves the parallel-additivity prior: `coeff * phi(x_j)` sums correctly over parallel edges.
- Coefficient scaling is learned per layer: `g_eff = alpha_G * G + beta_G * log1p(G)` (init: alpha=1, beta=0 -> exact v1 at init). 6 scalars per layer (18 total).
- Aggregation: SUM (computes exact parallel admittance of incident branches).
- Post-aggregation: bias -> LayerNorm -> ReLU -> residual connection.

Edge features are admittance-polynomial coefficients `[G/G_ref, C/C_ref, L_inv/L_inv_ref]` where `Y(s) = G + sC + L_inv/s`.

**Structured 5D VAE latent:** `[z_topo(2D) | z_VIN(1D) | z_VOUT(1D) | z_GND(1D)]`
- Topology branch (2D): `mu_topo = Linear(192, 2)(cat(h_VIN, h_VOUT, h_GND))` — global terminal view
- Terminal branches (1D each): `mu_vin = Linear(64, 1)(h_VIN)` — each terminal's own GNN embedding
- Logvar init to -2.0 (initial std ~ 0.37)
- Reparameterisation: `z = mu + exp(0.5*logvar) * eps` at train; `z = mu` at eval

### Decoder: SequenceDecoder

GPT-style autoregressive transformer generating Eulerian walk tokens conditioned on the 5D latent. **440,662 parameters.**
- d_model=128, n_heads=4, n_layers=2, max_seq_len=33
- Vocabulary: CircuitVocabulary (net names + component tokens + BOS/EOS/PAD)

### Attribute Heads (training-only)

All heads take **mu** `[B, 5]` as input (deterministic posterior mean, not stochastic z). They exist only to shape the latent space during training and are discarded at inference.

| Head | Architecture | Parameters | Target | Loss |
|------|-------------|------------|--------|------|
| **FreqHead** | `Linear(5,64) -> ReLU -> Drop(0.2) -> Linear(64,1)` | 449 | `log10(fc)` | MSE |
| **GainHead** | `Linear(5,64) -> ReLU -> Drop(0.2) -> Linear(64,64) -> ReLU -> Drop(0.2) -> Linear(64,1)` | 4,609 | `\|H(1kHz)\|` | MSE |
| **TypeHead** | `Linear(5,10)` | 60 | filter type (10 classes) | Cross-entropy |

FreqHead is a 1-hidden-layer MLP. GainHead is 2 hidden layers (deeper because gain depends on component values more nonlinearly). TypeHead is a single linear layer — topology separation is already clean in mu, no hidden layer needed.

### Training Loss

```
loss = CE_walk
     + beta_topo * KL(mu[:,:2], logvar[:,:2])
     + beta_term * KL(mu[:,2:], logvar[:,2:])
     + 0.5 * MSE(freq_pred, log10_fc)
     + 0.5 * MSE(gain_pred, |H(1kHz)|)
     + 0.5 * CE(type_logits, type_id)
```

| Weight | Value | Notes |
|--------|-------|-------|
| CE_walk | 1.0 | primary reconstruction objective |
| beta_topo | 0.1 | annealed from 0 over 20 epochs |
| beta_term | 0.02 | annealed from 0 over 20 epochs |
| alpha_freq | 0.5 | |
| alpha_gain | 0.5 | |
| alpha_type | 0.5 | |

KL annealing: `ramp = min(1.0, epoch / 20)`. Both betas are scaled by this ramp so the encoder first learns good reconstruction, then gradually regularises the latent.

### Parameter Summary

| Component | Parameters |
|-----------|------------|
| Encoder | 84,828 |
| Decoder | 440,662 |
| Attribute heads (training-only) | 5,118 |
| **Total** | **530,608** |

---

## Training Results

| Metric | Value |
|--------|-------|
| Val CE loss | 0.0138 |
| Val token accuracy | 98.4% |
| KL (topo + terminal) | 0.96 + 8.68 |
| Freq prediction MSE | 0.039 |
| Gain prediction MSE | 0.0026 |
| Type classification | 100% |

---

## 1. Latent Space Structure

### 1.1 Per-Type Centroids

The 5D latent cleanly separates all 10 filter types:

| Type | z_topo1 | z_topo2 | z_VIN | z_VOUT | z_GND | intra-sigma |
|------|---------|---------|-------|--------|-------|-------------|
| band_pass | -1.51 | -0.56 | +2.62 | +2.71 | -0.32 | 0.80 |
| band_stop | +0.77 | -0.38 | -0.63 | +0.48 | -3.36 | 0.43 |
| cl_highpass | +0.53 | -1.21 | +2.41 | -2.10 | -2.25 | 0.55 |
| high_pass | -1.69 | -0.55 | -1.92 | +1.52 | -0.57 | 1.01 |
| lc_lowpass | +0.95 | -0.03 | -4.85 | +0.14 | -0.30 | 0.78 |
| low_pass | +1.18 | +0.72 | -1.28 | -2.52 | +2.07 | 1.33 |
| rl_highpass | -0.92 | +1.35 | +2.04 | -3.36 | -1.22 | 1.00 |
| rl_lowpass | +0.41 | +0.80 | -1.59 | +4.87 | -1.33 | 0.87 |
| rlc_parallel | -1.15 | -0.65 | -2.06 | -3.72 | +0.07 | 0.67 |
| rlc_series | +1.28 | +0.55 | +3.20 | +2.53 | -0.42 | 0.77 |

**Separation ratio** (mean inter-class distance / mean intra-class distance): **7.4x**

This is a strong result for only 5 dimensions. The minimum inter-class distance (3.06, between band_pass and rlc_series) is still ~3.8x the intra-class spread of those types.

### 1.2 Nearest Neighbour Relationships

The latent organises filter types by physical similarity:

| Type | Nearest Neighbour | Distance |
|------|-------------------|----------|
| band_pass | rlc_series | 3.06 |
| band_stop | high_pass | 4.08 |
| cl_highpass | rl_highpass | 3.38 |
| high_pass | band_stop | 4.08 |
| lc_lowpass | high_pass | 4.22 |
| low_pass | rlc_parallel | 3.66 |
| rl_highpass | cl_highpass | 3.38 |
| rl_lowpass | high_pass | 4.26 |
| rlc_parallel | low_pass | 3.66 |
| rlc_series | band_pass | 3.06 |

These pairings are physically meaningful:
- **band_pass ↔ rlc_series**: both have series LC resonance, differ only in load configuration
- **cl_highpass ↔ rl_highpass**: both are high-pass, differ in reactive element (C vs L)
- **low_pass ↔ rlc_parallel**: both attenuate high frequencies, rlc_parallel adds resonance
- **band_stop ↔ high_pass**: both have notch/rejection character at low frequencies

### 1.3 1-NN Classification

Using only centroid distance in 5D: **2399/2400 = 99.96% accuracy**. The single misclassified sample is from rlc_parallel (239/240). Every other type achieves 100%.

---

## 2. Dimension-by-Dimension Analysis

### 2.1 Fisher Discriminant Ratios

| Dimension | Fisher Ratio | Role |
|-----------|-------------|------|
| z_VIN | **76.8** | Strongest discriminator — separates types by what VIN "sees" |
| z_topo1 | **30.3** | Primary topology axis |
| z_VOUT | **25.2** | Strong secondary — separates by output load configuration |
| z_topo2 | **11.7** | Secondary topology axis (complements topo1) |
| z_GND | **4.4** | Weakest discriminator, still meaningful |

**z_VIN (dim 2)** is the most discriminative single dimension, with Fisher ratio 76.8. This makes physical sense: the VIN terminal's local admittance view directly encodes what impedance the source drives into, which is the primary distinguishing feature between filter topologies.

### 2.2 What Each Dimension Encodes

**z_topo1 (dim 0)**: Separates filters along a "complexity/reactance" axis.
- Negative: simple high-pass types (high_pass, band_pass, rlc_parallel)
- Positive: series/lowpass types (rlc_series, low_pass, lc_lowpass, band_stop)
- Interpretation: distinguishes whether the dominant reactive path is in shunt (negative) vs series (positive) configuration

**z_topo2 (dim 1)**: Separates within the groups created by topo1.
- Negative: capacitor-coupled types (cl_highpass, rlc_parallel)
- Positive: inductor-coupled types (rl_highpass, rl_lowpass)
- The two topology dims together create a 2D map where each type occupies a distinct quadrant

**z_VIN (dim 2)**: Encodes the input-port impedance character.
- Strongly negative: VIN sees an inductor (lc_lowpass at -4.85) or resistor to ground (rlc_parallel at -2.06)
- Strongly positive: VIN sees a series reactive element (rlc_series at +3.20, band_pass at +2.62)
- Range spans ~8 units with minimal overlap — this dimension alone could classify most types

**z_VOUT (dim 3)**: Encodes the output-port load configuration.
- Negative: VOUT drives into low-impedance / shunt paths (rlc_parallel at -3.72, rl_highpass at -3.36)
- Positive: VOUT sees series reactive paths (rl_lowpass at +4.87, band_pass at +2.71)
- rl_lowpass is the extreme positive outlier — its output terminal sees a pure inductive load

**z_GND (dim 4)**: Encodes the ground-return path character. Weakest discriminator but still meaningful.
- Strongly negative: band_stop (-3.36) and cl_highpass (-2.25) — both have complex ground-return paths
- Strongly positive: low_pass (+2.07) — simple capacitor-to-ground return
- Broader intra-type spread (std 0.4-0.9) means this dimension carries more value-dependent information

### 2.3 Dimension-Attribute Correlations

| Dimension | corr(log10 fc) | corr(\|H(1kHz)\|) |
|-----------|----------------|-------------------|
| z_topo1 | -0.03 | +0.34 |
| z_topo2 | **+0.43** | +0.14 |
| z_VIN | +0.11 | **-0.52** |
| z_VOUT | +0.10 | **+0.48** |
| z_GND | **-0.56** | +0.23 |

Key findings:
- **z_GND** has the strongest frequency correlation (-0.56): circuits with simple ground returns (low_pass) tend to have higher cutoff frequencies in the dataset
- **z_VIN** and **z_VOUT** are the primary gain predictors (|r| ~ 0.5): the terminal admittance views directly determine the voltage transfer function at 1 kHz
- **z_topo1/topo2** have weak single-variable frequency correlation but strong *type* separation — frequency depends on component values within a type, while topology dims encode the structural class
- No single dimension dominates frequency prediction — the MLP head combines all 5 to achieve r=0.994

---

## 3. Attribute Prediction Quality

### 3.1 Type Classification: 100% (2400/2400)

All 10 filter types are perfectly classified from the 5D mu vector. Zero confusion on the full dataset.

### 3.2 Frequency Prediction

**Overall**: MSE = 0.079, correlation = 0.994 (on log10 Hz scale)

| Type | GT mean | Pred mean | MSE | corr |
|------|---------|-----------|-----|------|
| band_pass | 3.13 | 3.32 | 0.056 | 0.994 |
| band_stop | 3.12 | 3.32 | 0.079 | 0.990 |
| cl_highpass | 4.39 | 4.63 | 0.069 | 0.981 |
| high_pass | 4.49 | 4.78 | 0.099 | 0.975 |
| lc_lowpass | 4.47 | 4.63 | 0.032 | 0.987 |
| low_pass | 4.47 | 4.74 | 0.090 | 0.974 |
| rl_highpass | 4.43 | 4.69 | 0.071 | 0.996 |
| rl_lowpass | 4.45 | 4.73 | 0.088 | 0.988 |
| rlc_parallel | 5.90 | 6.19 | 0.097 | 0.996 |
| rlc_series | 5.67 | 5.98 | 0.106 | 0.995 |

There is a consistent positive bias of ~0.2-0.3 in log10 Hz (predicts slightly higher than GT), but within-type correlation is excellent (0.974-0.996). The head predicts cutoff frequency from 5D mu with sub-decade accuracy across the full range (0.31 to 8.02 log10 Hz, i.e., ~2 Hz to 100 MHz).

### 3.3 Gain Prediction

**Overall**: MSE = 0.0035, correlation = 0.993 (on |H(1kHz)| scale)

| Type | GT mean | Pred mean | MSE | corr |
|------|---------|-----------|-----|------|
| band_pass | 0.630 | 0.610 | 0.006 | 0.981 |
| band_stop | 0.570 | 0.553 | 0.001 | 0.997 |
| cl_highpass | 0.414 | 0.399 | 0.001 | 0.997 |
| lc_lowpass | 0.346 | 0.307 | 0.004 | 0.997 |
| rl_highpass | 1.016 | 0.960 | 0.005 | 0.641 |
| rlc_parallel | 0.998 | 0.952 | 0.005 | 0.481 |

Two types have noticeably weaker within-type gain correlation:
- **rlc_parallel** (r=0.481) and **rl_highpass** (r=0.641): these types have near-unity passband gain with little variation, making correlation noisy. The absolute MSE is still excellent (0.005).
- Types with large gain dynamic range (band_stop, cl_highpass, lc_lowpass) achieve r>0.99.

---

## 4. Posterior Uncertainty

### 4.1 Mean Posterior Std Per Dimension

| Dimension | Global Mean Std |
|-----------|----------------|
| z_topo1 | 0.947 |
| z_topo2 | 0.988 |
| z_VIN | 0.757 |
| z_VOUT | 0.746 |
| z_GND | 0.845 |

The topology dimensions have posterior std near 1.0 (close to the N(0,1) prior), while terminal dimensions are more certain (std ~0.75). This matches the information content: terminals encode specific admittance values that the encoder is confident about, while topology is more global/uncertain.

### 4.2 Per-Type Uncertainty

The most certain types are **band_stop** (avg std=0.80) and **high_pass** (0.80) — these have the most distinctive impedance signatures. The least certain are **lc_lowpass** (0.89) and **rl_lowpass** (0.89) — these have broader component value ranges in the training data.

---

## 5. Interpolation Smoothness

Interpolating between type centroids at alpha = {0, 0.25, 0.5, 0.75, 1.0} with T=1.5, 1000 samples:

### 5.1 low_pass → high_pass

| alpha | Valid % | Dominant | Count | #Unique | Novel |
|-------|---------|----------|-------|---------|-------|
| 0.00 | 92.6% | low_pass | 905 | 3 | 0 |
| 0.25 | 95.6% | low_pass | 888 | 5 | 0 |
| 0.50 | 93.6% | low_pass | 639 | 8 | 1 |
| 0.75 | 94.2% | high_pass | 909 | 8 | 0 |
| 1.00 | 95.3% | high_pass | 945 | 6 | 0 |

Clean transition at alpha ~0.5-0.75. The midpoint is topologically rich (8 unique topologies) — the decoder explores alternatives when the latent is equidistant between two attractors.

### 5.2 band_pass → band_stop

| alpha | Valid % | Dominant | Count | #Unique | Novel |
|-------|---------|----------|-------|---------|-------|
| 0.00 | 89.3% | band_pass | 875 | 4 | 0 |
| 0.25 | 90.9% | band_pass | 864 | 5 | 0 |
| 0.50 | 71.9% | band_stop | 389 | 7 | 1 |
| 0.75 | 61.2% | band_stop | 594 | 10 | 0 |
| 1.00 | 62.5% | band_stop | 610 | 8 | 1 |

The validity drop at midpoint (71.9%) indicates the decoder is genuinely uncertain in the intermediate region — it's trying to produce novel structures, some of which are malformed. The 10 unique topologies at alpha=0.75 is the highest diversity observed.

### 5.3 rl_lowpass → rl_highpass

| alpha | Valid % | Dominant | Count | #Unique |
|-------|---------|----------|-------|---------|
| 0.00 | 94.0% | rl_lowpass | 938 | 3 |
| 0.50 | 71.3% | **band_stop** | 456 | 9 |
| 1.00 | 94.1% | rl_highpass | 935 | 2 |

**Notable**: the midpoint between rl_lowpass and rl_highpass produces band_stop as the dominant topology. This is physically meaningful — the midpoint between "pass low, block high" and "pass high, block low" is "block a band" (notch filter). The VAE latent has learned a frequency-domain interpolation, not just a structural one.

### 5.4 lc_lowpass → cl_highpass

| alpha | Valid % | Dominant | Count | #Unique |
|-------|---------|----------|-------|---------|
| 0.00 | 93.4% | lc_lowpass | 921 | 5 |
| 0.50 | 67.9% | **band_stop** | 454 | 8 |
| 1.00 | 94.6% | cl_highpass | 932 | 2 |

Same pattern — low-pass ↔ high-pass midpoint produces band_stop. The encoder has captured that the "opposite" of a low-pass response is not structural inversion, but frequency-domain complementarity.

### 5.5 band_pass → rlc_parallel

| alpha | Valid % | Dominant | Count | #Unique |
|-------|---------|----------|-------|---------|
| 0.00 | 87.8% | band_pass | 855 | 4 |
| 0.50 | 91.9% | **high_pass** | 502 | 7 |
| 1.00 | 92.0% | rlc_parallel | 912 | 4 |

The midpoint produces high_pass — between a distributed LC band-pass and a lumped RLC parallel, the decoder finds an RC high-pass as the compromise.

### 5.6 Key Interpolation Findings

1. **Transitions are sharp but physically meaningful** — the dominant type flips between alpha=0.25 and alpha=0.75, not gradually
2. **Midpoints produce intermediate filter types**, not noise. rl_lowpass ↔ rl_highpass → band_stop is a genuine frequency-domain interpolation
3. **Diversity peaks at midpoints** — up to 10 unique topologies when the latent is equidistant between clusters
4. **Validity remains high** even at midpoints (67-94%), much better than random latent sampling
5. **Novel topologies appear at transition boundaries** — the decoder is most creative when the latent is ambiguous

---

## 6. Centroid-Conditioned Generation

Generating 1000 samples from each type's centroid at T=1.5:

| Type | Valid % | Self-type % | #Unique | #Novel | Top Non-Self |
|------|---------|-------------|---------|--------|--------------|
| band_pass | 87.7% | 97.8% | 3 | 0 | rlc_series (18) |
| band_stop | 59.6% | 98.7% | 6 | 0 | rl_lowpass (3) |
| cl_highpass | 93.2% | 99.2% | 2 | 0 | rl_highpass (7) |
| high_pass | 93.6% | 99.0% | 6 | 0 | rl_lowpass (4) |
| lc_lowpass | 93.7% | 98.9% | 7 | 1 | rl_lowpass (4) |
| low_pass | 94.2% | 96.6% | 3 | 0 | rlc_parallel (31) |
| rl_highpass | 94.7% | 98.6% | 3 | 0 | cl_highpass (11) |
| rl_lowpass | 93.4% | 99.5% | 2 | 0 | high_pass (5) |
| rlc_parallel | 91.4% | 98.8% | 4 | 1 | low_pass (9) |
| rlc_series | 82.9% | 97.0% | 5 | 0 | band_pass (21) |

**Self-type recovery rate: 96.6–99.5%** across all types. The decoder overwhelmingly produces the correct topology when conditioned on the type centroid.

Observations:
- **band_stop has lowest validity (59.6%)** — its topology (5 components, 3 internal nets) is the most complex in the dataset, so the decoder makes more structural errors
- **rl_lowpass has highest self-recovery (99.5%)** — the simplest RL topology with the most extreme z_VOUT value (+4.87), making it a strong attractor
- **Non-self leakage is to nearest neighbours**: low_pass→rlc_parallel, cl_highpass→rl_highpass, band_pass→rlc_series — matching the nearest-neighbour table
- **2 novel topologies** found even from centroid generation (lc_lowpass, rlc_parallel) — the high temperature enables occasional exploration

---

## 7. Perturbation Robustness

Self-type recovery (%) when adding N(0, sigma) noise to centroids:

| Type | sigma=0 | sigma=0.1 | sigma=0.25 | sigma=0.5 | sigma=1.0 | sigma=2.0 |
|------|---------|-----------|------------|-----------|-----------|-----------|
| band_pass | 98.4 | 98.1 | 98.1 | 98.1 | 96.8 | 92.6 |
| band_stop | 98.8 | 97.4 | 97.7 | 97.7 | 98.1 | 97.7 |
| cl_highpass | 98.9 | 99.1 | 99.1 | 99.1 | 98.7 | 97.6 |
| high_pass | 99.3 | 99.4 | 98.9 | 98.7 | 96.7 | **43.3** |
| lc_lowpass | 99.8 | 99.4 | 99.4 | 99.6 | 99.4 | 97.9 |
| low_pass | 97.5 | 98.9 | 98.9 | 98.9 | 98.9 | 97.9 |
| rl_highpass | 99.6 | 99.1 | 98.9 | 98.9 | 97.6 | 94.6 |
| rl_lowpass | 99.6 | 99.8 | 99.8 | 99.8 | 99.8 | 99.4 |
| rlc_parallel | 99.3 | 99.3 | 99.3 | 99.6 | 99.8 | 99.1 |
| rlc_series | 96.9 | 96.4 | 96.7 | 96.4 | 96.9 | 97.1 |

**Most types are robust up to sigma=1.0** (>96% self-recovery). The standout exception:

- **high_pass drops to 43.3% at sigma=2.0** — high_pass has the largest intra-class spread (intra-sigma=1.01) and sits near band_stop (d=4.08) and lc_lowpass (d=4.22). A perturbation of sigma=2.0 in 5D has expected L2 norm ~4.5, which exceeds these inter-class distances.

The most robust types are **rl_lowpass** and **rlc_parallel** (>99% even at sigma=2.0), which have the most extreme z_VOUT values (+4.87 and -3.72 respectively) — they are deep attractors in the latent space.

---

## 8. Physical Interpretation of the Latent

### 8.1 The VIN/VOUT Axes as Impedance Maps

The terminal dimensions form a natural 2D impedance map:

```
               z_VOUT
                 +5   rl_lowpass (series L to output)
                 |
                 +3   rlc_series, band_pass (series RLC)
                 |
                 +1   high_pass, band_stop
                 |
    z_VIN  -5---+0---+5  z_VIN
                 |
                 -1   cl_highpass, lc_lowpass
                 |
                 -3   rl_highpass, rlc_parallel (shunt to output)
                 |
                 -5
```

- **z_VIN positive**: VIN drives through a reactive element (L or C in series)
- **z_VIN negative**: VIN drives directly into a resistive/lumped load
- **z_VOUT positive**: VOUT has high output impedance (series-path topology)
- **z_VOUT negative**: VOUT has low output impedance (shunt-path topology)

This directly corresponds to the Thevenin equivalent impedance seen from each terminal.

### 8.2 The GND Axis as Ground Return Character

z_GND separates:
- **Positive** (low_pass +2.07): simple capacitor-to-ground path
- **Negative** (band_stop -3.36): complex multi-element ground return through internal nodes

This axis captures the "complexity of the ground network" — how many components sit between the active circuit and the return path.

### 8.3 Topology Dims as High-Level Structure

The 2D topology subspace encodes the overall filter architecture without reference to specific terminals:
- **topo1**: series vs shunt dominant reactive element
- **topo2**: inductor-primary vs capacitor-primary design

Together, the 5D latent provides a physically complete parameterisation: *what* the circuit does (topology dims) and *how* each terminal connects to it (terminal dims).

---

## 9. Comparison with V1

| Metric | V1 (64D deterministic) | V2 (5D structured VAE) |
|--------|----------------------|----------------------|
| Latent dimensions | 64 | **5** (12.8x compression) |
| Val CE | 0.016 | **0.014** |
| Val accuracy | 99.2% | 98.4% |
| Hold-out recovery | 10/10 | **10/10** |
| Novel topologies (T=1.5) | 11 | 10 |
| Type classification (from latent) | N/A | **100%** |
| Frequency prediction (from latent) | N/A | **r=0.994** |
| Gain prediction (from latent) | N/A | **r=0.993** |
| 1-NN centroid classification | N/A | **99.96%** |
| KL regularisation | None | 0.96 + 8.68 |
| Interpolation | Snaps to nearest template | **Physically meaningful midpoints** |
| Perturbation robustness (sigma=1) | Not tested | **>96% for 9/10 types** |

The v2 model achieves comparable reconstruction quality with 12.8x fewer latent dimensions while gaining physics-interpretable structure, smooth interpolation, and attribute prediction from the latent vector alone.

---

## 10. Limitations and Next Steps

1. **band_stop validity (59.6%)** is the weakest — its 5-component topology is at the complexity limit of the current decoder. Targeted augmentation or longer sequence training could help.

2. **Frequency prediction has systematic positive bias** (~0.2-0.3 in log10 Hz). The head slightly overestimates cutoff frequencies. This doesn't affect generation (the head is training-only) but suggests the latent doesn't fully capture the lowest-frequency circuits.

3. **high_pass perturbation sensitivity** — the largest intra-class spread (sigma=1.01) and proximity to multiple types makes high_pass the most fragile cluster. More training data variety for this type could tighten it.

4. **Novel topology rate is modest** (~10 at T=1.5) — the decoder is conservative, strongly preferring the 10 training topologies. To increase novelty, consider: higher temperatures (T=2.0+), targeted sampling from inter-cluster regions, or training with topology-diverse augmentation.

5. **Terminal KL is high (8.36)** relative to topology KL (0.96) — the encoder puts most information into the terminal dimensions. Increasing beta_term could force more compression, potentially improving interpolation smoothness at the cost of reconstruction quality.

---

## Files

- **Analysis script:** `eason/latent_5d/analyze_latent_v2.py`
- **Analysis log:** `eason/latent_5d/analyze_latent_v2.log`
- **Checkpoint:** `eason/latent_5d/best_inverse_design_v2.pt`
- **Encoder:** `eason/latent_5d/admittance_encoder.py`
- **Training:** `eason/latent_5d/train_inverse_design.py`
- **Inverse design:** `eason/latent_5d/inverse_design.py`
