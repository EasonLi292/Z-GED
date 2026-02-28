# Circuit Generation Model Architecture

## Overview

This model generates RLC filter circuit topologies from an 8-dimensional latent space.
The decoder is latent-only: no external condition tensor is required at decode time.

**Dataset:** 480 circuits across 8 filter types (60 each):
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
  -> SimplifiedCircuitDecoder (autoregressive nodes + edges)
  -> node_types, edge_existence, component_types
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

## 3) Decoder (`SimplifiedCircuitDecoder`)

### Node count prediction

- Predicts node count class for nodes in `[3, ..., max_nodes]`
- Production `max_nodes = 10`

### Node generation

- `AutoregressiveNodeDecoder` (Transformer decoder)
- Generates node types sequentially
- First 3 nodes are fixed semantic terminals: `GND, VIN, VOUT`

### Edge/component generation

- `LatentGuidedEdgeDecoder` (Transformer encoder with causal mask)
- Predicts 8-way joint class per edge pair:

```text
0: no edge
1: R
2: C
3: L
4: RC
5: RL
6: CL
7: RCL
```

Training uses teacher forcing on prior edge tokens.
Inference feeds back predicted edge tokens autoregressively.

---

## 4) Training Objective

Total loss:

```python
total_loss = (
    1.0  * node_type_loss
  + 5.0  * node_count_loss
  + 2.0  * edge_component_loss
  + 5.0  * connectivity_loss
  + 0.01 * kl_loss
  + 1.0  * pz_loss
)
```

Connectivity loss penalizes:

- disconnected VIN
- disconnected VOUT
- weak global connectivity
- isolated non-mask nodes

**Pole/zero supervision loss** (`pz_loss`):

- `F.mse_loss(mu[:, 4:], pz_target)` — forces z[4:8] to encode specific pole/zero values
- `pz_target` is computed from raw complex poles/zeros using signed-log normalization
- Converges from ~0.22 to ~0.006 (val) over 100 epochs without degrading other metrics

KL warmup is applied over the first 20 epochs.

---

## 5) Current Model Scale

Measured from current code/config (`edge_feature_dim=3`, `latent_dim=8`):

- Encoder parameters: **83,411**
- Decoder parameters: **7,698,901**

---

## 6) Current Training/Checkpoint Status

From `checkpoints/production/best.pt` (100 epochs, with pz supervision):

- Best epoch: **99**
- Validation loss: **1.04** (includes pz_loss component)
- Validation pz_loss: **0.006**

From current reported results (`GENERATION_RESULTS.md`):

- Node count accuracy: 100%
- Edge existence accuracy: 100%
- Component type accuracy: 100%
- Reconstruction validity: 480/480 (100%)

---

## 7) Generation Modes

1. **Random sampling:** `z ~ N(0, I)` — 89.2% valid rate
2. **Reconstruction:** encode circuit → decode `mu`
3. **Centroid generation:** average latent per filter type → decode
4. **Latent interpolation:** linear blend between two latent codes
5. **Pole/zero-driven generation** (new, decoder-only):
   - User specifies dominant pole/zero (real + imag parts)
   - `z[4:8]` is set via signed-log normalization of those values
   - `z[0:4]` is sampled from N(0, I) (random topology)
   - Decoder generates circuit — no encoder or dataset needed
   - 83% valid rate across test cases

```bash
# Example: RC low-pass with pole at -6283 rad/s (~1kHz)
python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 5

# Resonant circuit with conjugate pole pair
python scripts/generation/generate_from_specs.py --pole-real -3142 --pole-imag 49348 --num-samples 5
```

---

## Core Files

### Models

- `ml/models/encoder.py`
- `ml/models/gnn_layers.py`
- `ml/models/decoder.py`
- `ml/models/decoder_components.py`
- `ml/models/node_decoder.py`

### Constants

- `ml/models/constants.py` — `PZ_LOG_SCALE`, `FILTER_TYPES`, `CIRCUIT_TEMPLATES`

### Data and loss

- `ml/data/dataset.py` — dataset with `pz_target` computation
- `ml/losses/circuit_loss.py` — includes `pz_loss` term
- `ml/losses/connectivity_loss.py`

### Training and generation

- `scripts/training/train.py`
- `scripts/generation/generate_from_specs.py` — pole/zero-driven generation (decoder-only)
- `scripts/generation/regenerate_all_results.py` — all 6 result sections

---

## Notes on Changes vs Older Docs

- If you see references to 7D edge features (`[log(C), log(G), log(L_inv), is_R, is_C, is_L, is_parallel]`), those are outdated. The current architecture uses 3D `log10` component values with internally derived presence masks.
- If you see 6 filter types or 360 circuits, those are outdated. The current dataset has 8 filter types and 480 circuits.
- If you see `generate_from_specs.py` with `--cutoff`/`--q-factor` args, that is outdated. It now uses `--pole-real`/`--pole-imag`/`--zero-real`/`--zero-imag`.
