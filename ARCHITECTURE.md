# Circuit Generation Model Architecture

## Overview

This model generates RLC filter circuit topologies from an 8-dimensional latent space.
The decoder is latent-only: no external condition tensor is required at decode time.

**Dataset:** 1920 circuits across 8 filter types (240 each):
`low_pass`, `high_pass`, `band_pass`, `band_stop`, `rlc_series`, `rlc_parallel`, `lc_lowpass`, `cl_highpass`

Current production representation uses **3 edge features**:

```text
edge_attr = [log10(R), log10(C), log10(L)]
```

with `0` meaning the component is absent on that edge.

---

## End-to-End Pipeline

```text
Circuit graph + poles/zeros
  -> HierarchicalEncoder (VAE)
  -> z in R^8
  -> SequenceDecoder (GPT-style autoregressive transformer)
  -> Eulerian walk token sequence (e.g. VSS, RCL1, VOUT, L1, VIN, L1, VOUT, RCL1, VSS)
```

---

## 1) Data Representation

### Node features

- 4D one-hot: `[is_GND, is_VIN, is_VOUT, is_INTERNAL]`

### Edge features

- 3D values: `[log10(R), log10(C), log10(L)]`
- `0` means absent component
- Nonzero means present component

### Pole/zero features

- Variable-length lists of `[real, imag]`
- Log-scale magnitude normalization is applied for the DeepSets encoder input
- A separate **pz_target** `[4]` is computed per circuit for supervised loss on z[4:8]:
  - `sigma_p = signed_log(real(dominant_pole))`
  - `omega_p = signed_log(|imag(dominant_pole)|)`
  - `sigma_z = signed_log(real(dominant_zero))`
  - `omega_z = signed_log(|imag(dominant_zero)|)`
  - `signed_log(x) = sign(x) * log10(|x| + 1) / 7.0` (normalizes to ~[-1, 1])
  - Dominant = pole/zero with positive imag part (from conjugate pair) closest to origin
  - No poles/zeros → 0.0 (unambiguous since signed_log never produces exact 0 for nonzero input)

### Important preprocessing change

- Edge feature normalization/clipping used in older versions is removed.
- Edge features are now raw `log10` values.

---

## 2) Encoder (`HierarchicalEncoder`)

### Stage A: Impedance-aware GNN (3 layers)

- Module: `ImpedanceGNN` using `ImpedanceConv`
- Input: node 4D + edge 3D
- Hidden size: 64

`ImpedanceConv` behavior:

1. Extracts `val_R`, `val_C`, `val_L` from edge features
2. Derives masks internally:
   - `is_R = |val_R| > 0.01`
   - `is_C = |val_C| > 0.01`
   - `is_L = |val_L| > 0.01`
3. Applies value-conditioned transforms:
   - `lin_R([x_j, val_R])`
   - `lin_C([x_j, val_C])`
   - `lin_L([x_j, val_L])`
4. Combines by masks, then applies attention weighting

### Stage B: Hierarchical latent branches

The encoder outputs 8D latent split as:

```text
z = [z_topology(2) | z_structure(2) | z_pz(4)]
```

- Branch 1 (`z[0:2]`): topology from global mean+max pooling
- Branch 2 (`z[2:4]`): structure/values from GND/VIN/VOUT node embeddings
- Branch 3 (`z[4:8]`): transfer-function descriptors from DeepSets poles/zeros

VAE outputs:

- `mu`, `logvar`, and sampled `z`

---

## 3) Decoder (`SequenceDecoder`)

The decoder represents circuits as **bipartite Eulerian walks** — alternating sequences of net tokens (VSS, VIN, VOUT, INTERNAL_N) and component tokens (R1, C1, L1, RCL1, etc.). Each walk starts and ends at VSS.

### Circuit representation

A circuit is converted to a bipartite graph where nets and components are both nodes, connected by edges. An Eulerian circuit through this graph produces a token sequence that fully describes the topology.

Example (RC low-pass): `VSS, C1, VOUT, R1, VIN, R1, VOUT, C1, VSS`

### Vocabulary (`CircuitVocabulary`)

86 tokens total:
- Special: `PAD`, `EOS`
- Net tokens: `VSS`, `VIN`, `VOUT`, `VDD`, `INTERNAL_1`..`INTERNAL_10`
- Component tokens: `R1`..`R10`, `C1`..`C10`, `L1`..`L10`, `RC1`..`RC10`, `RL1`..`RL10`, `CL1`..`CL10`, `RCL1`..`RCL10`

### Architecture

- GPT-style transformer encoder with causal masking
- Latent prefix conditioning: z is projected to a prefix token prepended to the sequence
- 4 layers, 4 attention heads, d_model=256, max_seq_len=33
- Training: teacher forcing with cross-entropy next-token prediction
- Inference: greedy autoregressive generation

### Data augmentation

Each circuit has multiple valid Eulerian walks (different starting edges, different traversal orders). During training, a random valid walk is sampled each epoch, providing natural data augmentation.

---

## 4) Training Objective

Total loss:

```python
total_loss = ce_loss + kl_weight * kl_loss
```

- **CE loss**: Cross-entropy next-token prediction on the walk sequence (teacher forcing)
- **KL loss**: KL divergence between encoder posterior and N(0,I) prior
- **kl_weight**: 0.01 (with linear warmup over first 20 epochs)

Optimizer: AdamW, lr=3e-4, with ReduceLROnPlateau scheduler (factor=0.5, patience=10).

Auxiliary heads (trained jointly):
- Regression MLP: predicts dominant pole from latent (MSE loss, weight 0.1)
- Classification MLP: predicts filter type from latent (CE loss, weight 0.1)

---

## 5) Current Model Scale

- Encoder parameters: **237,907**
- Decoder parameters: **3,280,726**

---

## 6) Current Training/Checkpoint Status

From `checkpoints/production/best.pt`:

- Best epoch: **21**
- Validation loss: **0.0229**
- Token accuracy: **98.6%**

From current reported results (`GENERATION_RESULTS.md`):

- Topology match: 100% (384/384 validation circuits)
- Valid walk rate: 100%
- Reconstruction validity: 1920/1920 (100%)

---

## 7) Generation Modes

1. **Random sampling:** `z ~ N(0, I)` — 99.8% valid rate
2. **Reconstruction:** encode circuit → decode `mu` → Eulerian walk
3. **Centroid generation:** average latent per filter type → decode
4. **Latent interpolation:** linear blend between two latent codes
5. **Pole/zero-driven generation** (decoder-only):
   - User specifies dominant pole/zero (real + imag parts)
   - `z[4:8]` is set via signed-log normalization of those values
   - `z[0:4]` is sampled from N(0, I) (random topology)
   - Decoder generates Eulerian walk — no encoder or dataset needed
   - 100% valid rate across test cases

```bash
# Example: RC low-pass with pole at -6283 rad/s (~1kHz)
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5

# Resonant circuit with conjugate pole pair
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --num-samples 5
```

---

## Core Files

### Models

- `ml/models/encoder.py` — `HierarchicalEncoder` (impedance-aware GNN + hierarchical latent)
- `ml/models/gnn_layers.py` — `ImpedanceConv`, `ImpedanceGNN`, pooling/deepsets
- `ml/models/decoder.py` — `SequenceDecoder` (GPT-style autoregressive walk generator)
- `ml/models/vocabulary.py` — `CircuitVocabulary` (86-token vocabulary for Eulerian walks)
- `ml/models/constants.py` — `PZ_LOG_SCALE`, `FILTER_TYPES`, `CIRCUIT_TEMPLATES`

### Data

- `ml/data/dataset.py` — `CircuitDataset` (graph-based, used by encoder)
- `ml/data/sequence_dataset.py` — `SequenceDataset` (walk-based, used by decoder training)
- `ml/data/bipartite_graph.py` — `BipartiteCircuitGraph`, `from_pickle_circuit`
- `ml/data/traversal.py` — Hierholzer's algorithm, Euler walk enumeration

### Utilities

- `ml/utils/runtime.py` — model construction and checkpoint loading
- `ml/utils/circuit_ops.py` — `walk_to_string`, `is_valid_walk`, `generate_walk`
- `ml/utils/evaluate.py` — `sequence_to_topology_key`, `evaluate_reconstruction`

---

## v2: Admittance-Polynomial Encoder (Inverse Design)

The v2 model adds a second encoder path optimized for **spec-driven inverse design**. The v1 model above remains the production topology generator; v2 focuses on generating circuits that match target electrical specifications.

See [`docs/inverse_design.md`](docs/inverse_design.md) for full details.

### Key differences from v1

| | v1 (HierarchicalEncoder) | v2 (AdmittanceEncoder) |
|---|---|---|
| Encoder params | 237,907 | 84,828 |
| Decoder params | 3,280,726 | 440,662 |
| Latent dim | 8D | 5D |
| Edge features | log10 values | Admittance polynomials (G/G_REF, C/C_REF, L_inv/L_INV_REF) |
| Physics prior | Presence masks | Parallel additivity + learned coefficient scaling |
| Dataset | 1920 circuits, 8 types | 2400 circuits, 10 types |

### AdmittanceEncoder

- **Edge features**: `[G/G_REF, C/C_REF, L_inv/L_INV_REF]` — admittance quantities that add in parallel
- **AdmittanceConv**: Separate MLP channels per component type with learnable `alpha*x + beta*log1p(x)` scaling
- **5D latent**: `[z_topo(2) | z_VIN(1) | z_VOUT(1) | z_GND(1)]`
- **Attribute heads**: FreqHead (log10 fc), GainHead (|H(1kHz)|), TypeHead (10-way classification) — all predict from mu

### Generation pipeline

1. **K-NN interpolation** in latent space using target attributes
2. **Gradient descent** on mu through frozen attribute heads
3. **Decode** walks from optimized mu

### v2 files

- `ml/models/admittance_encoder.py` — AdmittanceConv + AdmittanceEncoder
- `ml/models/attribute_heads.py` — FreqHead, GainHead, TypeHead
- `ml/data/cross_topo_dataset.py` — CrossTopoSequenceDataset (2400 circuits, 10 types)
- `scripts/generation/generate_inverse_design.py` — Spec-driven generation CLI
- `scripts/training/train_inverse_design.py` — v2 training script
- `checkpoints/production/best_v2.pt` — Trained v2 checkpoint

---

### Archived (old adjacency decoder)

- `ml/models/archive/decoder_adjacency.py` — `SimplifiedCircuitDecoder`
- `ml/models/archive/node_decoder.py` — `AutoregressiveNodeDecoder`
- `ml/models/archive/decoder_components.py` — `LatentGuidedEdgeDecoder`
- `ml/models/archive/circuit_loss.py` — adjacency-based loss function

### Training and generation

- `scripts/training/train.py`
- `scripts/generation/generate_from_specs.py` — pole/zero-driven generation (decoder-only)
- `scripts/generation/regenerate_all_results.py` — all result sections
