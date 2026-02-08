# Circuit Generation Model Architecture

## Overview

This model generates RLC filter circuit topologies from an 8-dimensional latent space.
The decoder is latent-only: no external condition tensor is required at decode time.

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
- Log-scale magnitude normalization is applied for poles/zeros

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
)
```

Connectivity loss penalizes:

- disconnected VIN
- disconnected VOUT
- weak global connectivity
- isolated non-mask nodes

KL warmup is applied in early epochs.

---

## 5) Current Model Scale

Measured from current code/config (`edge_feature_dim=3`, `latent_dim=8`):

- Encoder parameters: **83,411**
- Decoder parameters: **7,698,901**

---

## 6) Current Training/Checkpoint Status

From `checkpoints/production/best.pt`:

- Best epoch: **95**
- Validation loss: **1.0291**

From current reported results (`GENERATION_RESULTS.md`):

- Node count accuracy: 100%
- Edge existence accuracy: 100%
- Component type accuracy: 100%

---

## 7) Generation Modes

1. Random latent sampling: `z ~ N(0, I)`
2. Encode existing circuits, decode `mu`
3. Latent interpolation
4. Specification-driven generation via K-NN latent interpolation (`scripts/generation/generate_from_specs.py`)

---

## Core Files

### Models

- `ml/models/encoder.py`
- `ml/models/gnn_layers.py`
- `ml/models/decoder.py`
- `ml/models/decoder_components.py`
- `ml/models/node_decoder.py`

### Data and loss

- `ml/data/dataset.py`
- `ml/losses/circuit_loss.py`
- `ml/losses/connectivity_loss.py`

### Training and generation

- `scripts/training/train.py`
- `scripts/training/validate.py`
- `scripts/generation/generate_from_specs.py`

---

## Notes on Changes vs Older Docs

If you see references to 7D edge features (`[log(C), log(G), log(L_inv), is_R, is_C, is_L, is_parallel]`), those are outdated.
The current architecture uses 3D `log10` component values with internally derived presence masks.
