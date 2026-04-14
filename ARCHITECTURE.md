# Z-GED V2 Architecture

This document describes the maintained V2 inverse-design path under `ml/` and
`scripts/`. It intentionally excludes older exploratory and archived paths.

V2 generates passive RLC filter topologies from target attributes:

- filter type
- characteristic frequency `fc`
- gain magnitude at 1 kHz, `|H(1kHz)|`

The model does not generate component values directly. It generates topology as
an Eulerian walk over a bipartite net/component graph.

---

## Source Of Truth

The active V2 code path is:

| Area | File |
|---|---|
| Encoder | `ml/models/admittance_encoder.py` |
| Attribute heads | `ml/models/attribute_heads.py` |
| Sequence decoder | `ml/models/decoder.py` |
| Vocabulary | `ml/models/vocabulary.py` |
| V2 constants | `ml/models/constants.py` |
| Dataset | `ml/data/cross_topo_dataset.py` |
| Runtime loaders/builders | `ml/utils/runtime.py` |
| Training | `scripts/training/train_inverse_design.py` |
| Spec-driven generation | `scripts/generation/generate_inverse_design.py` |
| Latent analysis | `scripts/analysis/analyze_latent.py` |
| Checkpoint | `checkpoints/production/best_v2.pt` |

When this document discusses architecture-level implementation, it follows the
code above. Hyperparameters such as dropout rates and loss weights are training
choices unless changing them would alter the model interface or latent contract.

---

## End-To-End Pipeline

```text
Circuit topology + component values
  -> CrossTopoSequenceDataset
  -> PyG net graph with admittance-polynomial edge features
  -> AdmittanceEncoder
  -> 5D VAE latent z, mu, logvar
  -> Attribute heads trained on mu
  -> SequenceDecoder trained to reconstruct Eulerian walk tokens

At generation time:

Target attributes (type, fc, gain)
  -> KNN initialization from encoded training latents
  -> gradient descent on mu through frozen attribute heads
  -> SequenceDecoder samples topology walks
  -> graph-signature validation and reporting
```

The encoder and attribute heads are used during training and for KNN reference
encoding. The decoder is the final topology generator.

---

## Dataset And Representation

V2 trains on 2,400 circuits across 10 filter types:

- `rlc_dataset/filter_dataset.pkl`: 1,920 circuits, 8 types, 240 each
- `rlc_dataset/rl_dataset.pkl`: 480 circuits, 2 RL types, 240 each

The type set is defined by `FILTER_TYPES_V2` in `ml/models/constants.py`:

```text
band_pass, band_stop, cl_highpass, high_pass, lc_lowpass,
low_pass, rl_highpass, rl_lowpass, rlc_parallel, rlc_series
```

### Node Features

Each circuit graph is represented as a PyTorch Geometric graph over electrical
nets. Node features are 4D one-hot vectors:

```text
[is_GND, is_VIN, is_VOUT, is_INTERNAL]
```

The encoder reads the GND, VIN, and VOUT terminal embeddings explicitly, so the
terminal node roles are part of the latent contract.

### Edge Features

V2 uses admittance-polynomial edge features:

```text
edge_attr = [G / G_REF, C / C_REF, L_inv / L_INV_REF]
```

where:

- `G = 1 / R`
- `C = capacitance`
- `L_inv = 1 / L`
- `G_REF = 1e-3`
- `C_REF = 10 ** -7.5`
- `L_INV_REF = 1e3`

These coefficients correspond to:

```text
Y(s) = G + sC + L_inv / s
```

This representation is an architecture choice: admittance coefficients add in
parallel, so sum aggregation in message passing can preserve a useful circuit
prior. Normalization keeps typical values near order 1.

### Sequence Targets

The decoder target is an Eulerian walk through a bipartite graph where:

- net nodes are tokens such as `VSS`, `VIN`, `VOUT`, `INTERNAL_1`
- component nodes are tokens such as `R1`, `C1`, `L1`, `RC1`, `RCL1`
- each walk is terminated with `EOS`
- sequences are padded to `max_seq_len = 32` walk tokens during training

The vocabulary is deterministic:

- `PAD`, `EOS`
- fixed nets: `VSS`, `VIN`, `VOUT`, `VDD`
- `INTERNAL_1` through `INTERNAL_10`
- 10 tokens each for `R`, `C`, `L`, `RC`, `RL`, `CL`, and `RCL`

With default `max_internal=10` and `max_components=10`, the vocabulary size is
86.

---

## AdmittanceEncoder

`AdmittanceEncoder` is a physics-informed GNN with an optional VAE bottleneck.
The production V2 path uses `vae=True`.

Default training construction:

```python
AdmittanceEncoder(
    node_feature_dim=4,
    hidden_dim=64,
    latent_dim=5,
    num_layers=3,
    dropout=0.1,
    vae=True,
)
```

The encoder has 84,828 parameters with these settings.

### AdmittanceConv

Each `AdmittanceConv` layer:

1. Projects node features into the layer hidden width.
2. Reads normalized edge coefficients `g_raw`, `c_raw`, and `l_raw`.
3. Applies learned coefficient scaling:

   ```text
   g_eff = alpha_G * g_raw + beta_G * log1p(g_raw)
   c_eff = alpha_C * c_raw + beta_C * log1p(c_raw)
   l_eff = alpha_L * l_raw + beta_L * log1p(l_raw)
   ```

4. Applies separate neighbor transforms:

   ```text
   phi_G(x_j), phi_C(x_j), phi_L(x_j)
   ```

5. Sums the channel messages:

   ```text
   msg = g_eff * phi_G(x_j)
       + c_eff * phi_C(x_j)
       + l_eff * phi_L(x_j)
   ```

6. Aggregates messages with sum aggregation.
7. Applies bias, dropout, layer norm, ReLU, and residual projection in the
   surrounding `AdmittanceEncoder` stack.

The `alpha` parameters initialize to 1 and `beta` parameters initialize to 0.
At initialization, coefficient scaling is identity. This is deliberate: the
model starts from the linear admittance-additivity prior and learns deviations
only if training supports them.

Each `phi_*` transform is:

```text
Linear(hidden_dim, hidden_dim)
ReLU
Linear(hidden_dim, hidden_dim, bias=False)
```

The final `bias=False` keeps a zero coefficient from contributing through that
channel.

### Structured 5D Latent

After the 3 GNN layers, the encoder extracts terminal embeddings:

```text
h_VIN, h_VOUT, h_GND
```

The 5D posterior mean is:

```text
mu = [z_topo(2) | z_VIN(1) | z_VOUT(1) | z_GND(1)]
```

where:

- `z_topo`: linear readout from `concat(h_VIN, h_VOUT, h_GND)`
- `z_VIN`: linear readout from `h_VIN`
- `z_VOUT`: linear readout from `h_VOUT`
- `z_GND`: linear readout from `h_GND`

The encoder also predicts matching `logvar` values. In training mode:

```text
z = mu + eps * exp(0.5 * logvar)
```

In eval mode, `z = mu`.

The initial log-variance bias is `-2.0`, so the initial posterior standard
deviation is approximately 0.37.

---

## SequenceDecoder

The decoder is a GPT-style causal transformer conditioned only on the latent
vector.

Default V2 training construction:

```python
SequenceDecoder(
    vocab_size=vocab.vocab_size,
    latent_dim=5,
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_seq_len=33,
    dropout=0.15,
    pad_id=vocab.pad_id,
)
```

The decoder has 440,662 parameters with these settings.

### Latent Prefix Conditioning

The latent vector is projected into a transformer prefix token:

```text
position 0: latent_proj(z)
position 1..L: shifted walk token embeddings
```

During teacher forcing, the input is:

```text
[latent_prefix, seq[0], seq[1], ..., seq[L-2]]
```

The target is:

```text
[seq[0], seq[1], ..., seq[L-1]]
```

The transformer uses a causal mask, and loss ignores padded positions.

### Generation

During generation, the decoder starts with only the latent prefix and samples
tokens autoregressively until `EOS` or `max_length`.

`generate_inverse_design.py` uses stochastic sampling:

```python
decoder.generate(
    latents,
    max_length=32,
    temperature=temperature,
    greedy=False,
    eos_id=vocab.eos_id,
)
```

The default CLI temperature is `0.1`, which is close to greedy but still samples.

---

## Attribute Heads

Attribute heads predict target circuit properties from `mu`, not sampled `z`.
This keeps attribute predictions deterministic and avoids training the heads on
VAE sampling noise.

| Head | Architecture | Output | Parameters |
|---|---|---|---|
| `FreqHead` | `Linear(5,64) -> ReLU -> Dropout -> Linear(64,1)` | `log10(fc)` | 449 |
| `GainHead` | `Linear(5,64) -> ReLU -> Dropout -> Linear(64,64) -> ReLU -> Dropout -> Linear(64,1)` | `|H(1kHz)|` | 4,609 |
| `TypeHead` | `Linear(5,10)` | filter-type logits | 60 |

Total attribute-head parameters: 5,118.

These heads are architecture-level because generation optimizes `mu` through
them. Their loss weights are training choices.

---

## Training

The maintained V2 training entry point is:

```bash
.venv/bin/python scripts/training/train_inverse_design.py
```

The script trains encoder, decoder, and attribute heads jointly.

### Training Hyperparameters

Current script values:

| Parameter | Value |
|---|---|
| epochs | 80 |
| batch size | 64 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| scheduler | ReduceLROnPlateau |
| scheduler factor | 0.5 |
| scheduler patience | 8 |
| max walk tokens | 32 |
| augmentation walks | 32 |
| encoder dropout | 0.1 |
| decoder dropout | 0.15 |

The runtime V2 builders in `ml/utils/runtime.py` use `dropout=0.0` defaults.
That is a construction default for loading/evaluation convenience, not a change
to the architecture contract. The trained checkpoint stores weights, and modules
are put in eval mode when loaded.

### Loss

For each batch:

```text
loss = CE_walk
     + beta_topo * KL(mu[:, :2], logvar[:, :2])
     + beta_term * KL(mu[:, 2:], logvar[:, 2:])
     + alpha_freq * MSE(freq_head(mu), log10_fc)
     + alpha_gain * MSE(gain_head(mu), |H(1kHz)|)
     + alpha_type * CE(type_head(mu), type_id)
```

Current script values:

| Weight | Value |
|---|---|
| `beta_topo` | 0.1 |
| `beta_term` | 0.02 |
| `beta_warmup` | 20 epochs |
| `alpha_freq` | 0.5 |
| `alpha_gain` | 0.5 |
| `alpha_type` | 0.5 |

The KL terms are warmed up with:

```text
ramp = min(1.0, epoch / beta_warmup)
cur_beta_topo = beta_topo * ramp
cur_beta_term = beta_term * ramp
```

The topology and terminal KL terms are weighted separately because the latent is
structured. This is an architecture-informed training choice: it lets the 2D
topology branch and 3D terminal branch be regularized with different strength.

### Checkpoint

The maintained checkpoint path is:

```text
checkpoints/production/best_v2.pt
```

Current checkpoint metadata:

| Field | Value |
|---|---|
| epoch | 64 |
| validation CE | 0.013788633512376691 |
| latent dim | 5 |
| vocab config | `max_internal=10`, `max_components=10` |

---

## Spec-Driven Generation

The main user-facing V2 generation entry point is:

```bash
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type band_pass --fc 10000 --gain 0.5
```

At least one target must be specified:

- `--type`
- `--fc`
- `--gain`

### Step 1: Build Reference Latents

The script loads the V2 checkpoint, builds the 2,400-circuit V2 dataset, and
encodes every circuit:

```text
training circuit -> AdmittanceEncoder -> mu
```

It stores:

- `mu`
- `log10(fc)`
- `|H(1kHz)|`
- filter type

### Step 2: KNN Initialization

Given target attributes, the script computes an attribute-space distance to all
reference circuits:

- normalized squared distance in `log10(fc)` when `--fc` is present
- normalized squared distance in gain when `--gain` is present
- large type-mismatch penalty when `--type` is present

It picks the top `k` neighbors and computes an inverse-distance weighted average:

```text
mu_init = sum(w_i * mu_i)
```

This keeps generation initialized near the training manifold.

### Step 3: Optimize mu

Starting from `mu_init`, the script runs Adam on `mu` through frozen attribute
heads:

```text
loss = w_freq * (freq_head(mu) - target_log_fc)^2
     + w_gain * (gain_head(mu) - target_gain)^2
     + w_type * CE(type_head(mu), target_type)
     + w_prior * ||mu||^2
```

Default values:

| Parameter | Value |
|---|---|
| optimization steps | 200 |
| optimization lr | 0.05 |
| KNN k | 10 |
| prior weight | 0.01 |

The prior term is a generation-time design choice to discourage optimized
latents from moving too far away from the VAE prior.

### Step 4: Decode And Validate

The optimized `mu` is expanded to the requested sample count and decoded into
walks. Generated walks are checked with graph-signature utilities for:

- well-formedness
- electrical validity
- unique topology signatures

The CLI reports predicted attributes at the optimized `mu`, generated validity,
and a netlist-style summary of the top generated topologies.

---

## Analysis And Exploratory Tools

`scripts/analysis/analyze_latent.py` is the maintained latent-analysis script
for the V2 path. It loads `checkpoints/production/best_v2.pt` and analyzes:

- per-type centroids
- latent dimension separation
- attribute-head behavior
- interpolation behavior
- centroid generation
- perturbation behavior
- topology signature census

`scripts/generation/inverse_design.py` is best understood as an exploratory
sampling harness rather than the main product CLI. It samples from supplied
latent targets, centroid blends, and perturbations to probe how the decoder
responds to latent-space behavior.

---

## Current Scope And Limitations

V2 is a topology generator, not a complete circuit synthesizer.

Important limitations:

- It generates topology only, not numerical component values.
- It is scoped to passive RLC filters in the 10 known V2 filter types.
- The decoder is conservative at low temperature and strongly favors known
  training topologies.
- Novel topologies require higher-temperature sampling or latent perturbation,
  which lowers validity.
- The gain target is `|H(1kHz)|`, not a full transfer-function specification.

---

## Implementation Notes

- The architecture identity is the 5D admittance VAE encoder plus latent-prefix
  sequence decoder.
- Dropout rates, loss weights, KNN `k`, optimization steps, and sampling
  temperature are tunable choices. They should be documented with current script
  defaults, but they are not separate architectures.
- If future code changes alter the latent split, edge feature semantics,
  attribute-head targets, or decoder token contract, this document should be
  updated as an architecture change.
