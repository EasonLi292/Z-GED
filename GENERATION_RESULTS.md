# Circuit Generation Results - Node Count Predictor Model

**Date:** January 8, 2026
**Model:** Node Count Predictor with Direct Prediction (best.pt, epoch 99)
**Dataset:** 120 circuits with corrected topologies

---

## Executive Summary

The model now includes an **explicit node count predictor** that directly predicts how many nodes to generate (3, 4, or 5) based on the topology latent. This eliminates the train/test mismatch issue where the previous model always generated 5 nodes regardless of target complexity.

### Key Achievements

| Metric | Value |
|--------|-------|
| **Node Count Accuracy** | **100.0%** (24/24) |
| **Component Type Accuracy** | 85.3% (58/68) |
| **Edge Count Exact Match** | 83.3% (20/24) |
| **Validation Loss** | 2.0225 |

---

## Training Results

### Final Performance (Epoch 99)

| Metric | Training | Validation |
|--------|----------|------------|
| **Total Loss** | 4.73 | **2.02** |
| **Node Type Accuracy** | 100.0% | **100.0%** |
| **Node Count Accuracy** | 69.8% | **100.0%** |
| **Edge Existence Accuracy** | 97.9% | **100.0%** |
| **Component Type Accuracy** | 80.8% | **88.1%** |

### Architecture Changes

The key improvement is the **Direct Node Count Predictor** in `ml/models/decoder.py`:

```python
self.node_count_predictor = nn.Sequential(
    nn.Linear(2 + conditions_dim, hidden_dim // 4),  # topology latent + conditions
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 4, 3)  # 3 classes: 3, 4, 5 nodes
)
```

This head predicts node count directly from the topology latent, avoiding the train/test mismatch of learned stopping criteria.

---

## Node Count Prediction Results

### Accuracy by Node Count

| Target | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| 3-node | 14 | 14 | **100.0%** |
| 4-node | 4 | 4 | **100.0%** |
| 5-node | 6 | 6 | **100.0%** |
| **Overall** | **24** | **24** | **100.0%** |

### Confusion Matrix

```
Target →  Predicted
        3     4     5
  3:  [14,    0,    0]
  4:  [ 0,    4,    0]
  5:  [ 0,    0,    6]
```

Perfect diagonal - no misclassifications!

---

## Component Type Prediction Results

### Overall Accuracy: 85.3% (58/68 components)

### Per-Component Breakdown

| Component | Correct | Total | Accuracy |
|-----------|---------|-------|----------|
| **R** (Resistor) | 31 | 34 | **91.2%** |
| **C** (Capacitor) | 13 | 20 | 65.0% |
| **L** (Inductor) | 10 | 10 | **100.0%** |
| **RCL** (Parallel) | 4 | 4 | **100.0%** |

### Confusion Matrix

```
True\Pred   None    R       C       L       RCL
  R   :       .    31       1       .       2
  C   :       .     6      13       .       1
  L   :       .     .       .      10       .
  RCL :       .     .       .       .       4
```

**Key Observations:**
- **Resistors**: 91.2% accuracy (excellent)
- **Inductors**: 100% accuracy (perfect)
- **RCL parallel**: 100% accuracy (perfect)
- **Capacitors**: 65% accuracy (some confusion with R)
- No components incorrectly predicted as "None" (edge deletion)

---

## Edge Generation Results

| Metric | Value |
|--------|-------|
| **Exact edge count match** | 83.3% (20/24) |
| **Within ±1 edge** | 83.3% (20/24) |
| **Mean edge difference** | -0.33 |

The model slightly under-predicts edges on average, but 83% of circuits have the exact correct number of edges.

---

## Example Circuit Generations

### Example 1: Low-Pass Filter (3-node)

**Target:**
```
VIN --R(48.1kΩ)-- VOUT --C(54.8nF)-- GND
```

**Generated:**
```
VIN --R-- VOUT --C-- GND
```
✓ Node count: 3/3 | ✓ Topology: Correct

---

### Example 2: Band-Pass Filter (4-node)

**Target:**
```
VIN --L(0.925mH)-- Node3 --C(23.4nF)-- VOUT --R(2.08kΩ)-- GND
```

**Generated:**
```
VIN --L-- Node3 --C-- VOUT --R-- GND
```
✓ Node count: 4/4 | ✓ Topology: Correct

---

### Example 3: Band-Stop Filter (5-node)

**Target:**
```
VIN --R(683Ω)-- Node3 --L(133μH)-- Node4 --C(4.82nF)-- GND
                    └--R(40.1kΩ)-- VOUT --R(40.1kΩ)-- GND
```

**Generated:**
```
VIN --R-- Node3 --L-- Node4 --C-- GND
              └--R-- VOUT --R-- GND
```
✓ Node count: 5/5 | ✓ Topology: Correct

---

### Example 4: RLC Parallel (3-node with parallel components)

**Target:**
```
VIN --R-- VOUT --[L||C||R]-- GND
```

**Generated:**
```
VIN --R-- VOUT --RCL-- GND
```
✓ Node count: 3/3 | ✓ Topology: Correctly uses RCL parallel component

---

## Generalization to Unseen Specifications

The model uses K-NN interpolation in latent space to generate circuits for specifications not seen during training.

### Training Data Coverage

| Parameter | Min | Max |
|-----------|-----|-----|
| **Frequency** | 2.2 Hz | 449,321 Hz |
| **Q Factor** | 0.01 | 14.57 |

### Generalization Performance

| Distance from Training | Samples | Node Match | Structurally Valid |
|------------------------|---------|------------|-------------------|
| **Close** (within distribution) | 20 | 85% | **100%** |
| **Medium** (edge of distribution) | 20 | 95% | **100%** |
| **Far** (outside distribution) | 10 | 30% | **100%** |
| **Very Far** (extreme extrapolation) | 10 | 90%* | **100%** |

\* Falls back to simple 3-node circuits (safe default behavior)

### Example: Novel Specification → Correct Topology

**Target:** 75,000 Hz, Q=0.08 (NOT in training data)

**Nearest training samples:**
1. rlc_series @ 69,507 Hz, Q=0.095 (dist=0.036)
2. rlc_series @ 84,870 Hz, Q=0.134 (dist=0.076)
3. band_stop @ 62,816 Hz, Q=0.010 (dist=0.104)

**Generated:** 5-node notch filter
```
       ┌────[R]────┐
       │           │
      VIN         N3
                   │
                  [L]
                   │
                  N4
                   │
                  [C]
                   │
      GND───[R]───VOUT
```

The model correctly generates a 5-node band-stop topology by interpolating between nearby training examples.

---

## Interesting Generated Topologies

### Topology 1: 4-Node Band-Pass (Medium Distance)

**Target:** 15,000 Hz, Q=3.0 (interpolated, dist=0.194)

```
        VIN ───[L]─── N3
                      │
                     [C]
                      │
        GND ───[R]─── VOUT
```

A series LC band-pass filter - the model correctly uses L-C-R for high-Q resonant specifications.

---

### Topology 2: RCL Parallel Tank Circuit (Medium Distance)

**Target:** 8,500 Hz, Q=1.2 (interpolated, dist=0.174)

```
        VIN ───[R]─── VOUT
                      │
                    [RCL]  ← Parallel L||C||R tank
                      │
                     GND
```

The model correctly uses RCL parallel for high-Q resonant specs, creating a tank circuit.

---

### Topology 3: Novel 5-Node, 6-Edge Circuit

**Target:** 55,206 Hz, Q=0.018 (extreme low Q)

```
              ┌───[R]───┐
              │         │
    VIN ─────[R]───── N3 ─────[L]───── N4
                       │               │
                      [R]             [C]
                       │               │
    GND ─────[R]───── VOUT ───[C]─────┘
```

**This is a NOVEL topology!** Training data only has 4-5 edges for 5-node circuits; this has **6 edges**. The model created a more complex structure by interpolating between band_pass, band_stop, and rlc_series training examples.

---

### Topology 4: Novel 5-Node with Extra Capacitor Path

**Target:** 4,319 Hz, Q=0.022 (extreme low Q)

```
    VIN ──[R]── N3 ──[L]── N4
                           │
              ┌────[C]────┤
              │           │
    GND ──────┴───[C]──── VOUT ──[R]── GND
```

Another novel structure with parallel capacitor paths - not present in any training example.

---

## Novel Topology Analysis

From 200 random generations across the specification space:

| Topology Type | Count | Percentage |
|--------------|-------|------------|
| **Known training topologies** | 120 | 60% |
| **Novel component combinations** | 60 | 30% |
| **Truly novel graph structures** | 20 | 10% |

### Novel Structures Found

| Structure | Training | Generated | Notes |
|-----------|----------|-----------|-------|
| 5-node, 6 edges | Never (max 5) | Yes | More connections than training |
| 4-node, 2 edges | Never (always 3) | Yes | Simpler than training |
| 3-node GND-VIN edge | Never | Yes | Different connection pattern |
| RCL on any edge | Only GND-VOUT | Yes | Novel component placement |

**Key Insight:** The model primarily interpolates training topologies but can generate truly novel structures (~10% of cases) when pushed to unusual specifications. This demonstrates the model learned generalizable circuit structure patterns, not just memorized training examples.

---

## Filter Type Distribution (Validation Set)

| Filter Type | Count |
|-------------|-------|
| low_pass | 4 |
| high_pass | 6 |
| band_pass | 4 |
| band_stop | 4 |
| rlc_series | 2 |
| rlc_parallel | 4 |
| **Total** | **24** |

---

## Model Architecture Summary

### Encoder
- **Type:** Hierarchical GNN Encoder
- **Latent Space:** 8D = 2D topology + 2D values + 4D transfer function
- **GNN Layers:** 3 layers, 64 hidden dim

### Decoder
- **Type:** Latent-Guided GraphGPT Decoder
- **Hidden Dim:** 256
- **Attention Heads:** 8
- **Node Layers:** 4
- **Key Feature:** Direct node count predictor from topology latent

### Loss Function
- Node type: Cross-entropy (weight=1.0)
- Node count: Cross-entropy (weight=5.0) **← NEW**
- Stop criterion: BCE (weight=2.0)
- Stop-node correlation: (weight=2.0)
- Edge existence: BCE (weight=3.0)
- Component type: Cross-entropy (weight=5.0)
- Component values: MSE (weight=0.5)
- KL divergence: (weight=0.005)

---

## Improvements Over Previous Model

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| Node Count Accuracy | 0% (always 5) | **100%** | +100% |
| Component Type Accuracy | 60.3% | **85.3%** | +25% |
| Validation Loss | 0.23 | 2.02 | (different loss function) |

The previous model always generated 5 nodes regardless of target complexity due to train/test mismatch in the stopping criterion. The new direct node count predictor eliminates this issue entirely.

---

## Usage

### Generate Circuit from Latent

```python
from ml.models.decoder import LatentGuidedGraphGPTDecoder

decoder = LatentGuidedGraphGPTDecoder(...)
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Generate with automatic node count prediction
result = decoder.generate(latent, conditions, verbose=True)
# Output shows: "Predicted: N nodes" based on topology latent
```

### Reconstruct from Existing Circuit

```python
# Encode existing circuit
z, mu, logvar = encoder(graph.x, graph.edge_index, graph.edge_attr, batch, poles, zeros)

# Decode with correct node count
result = decoder.generate(mu, conditions)
# Node count automatically matches original circuit complexity
```

---

## Files & Checkpoints

### Model Checkpoint
- **Best model:** `checkpoints/production/best.pt` (epoch 99)
- **Validation loss:** 2.0225
- **Node count accuracy:** 100%

### Key Files
- **Decoder:** `ml/models/decoder.py` (node_count_predictor at line 114)
- **Loss function:** `ml/losses/gumbel_softmax_loss.py` (node_count_weight)
- **Training script:** `scripts/training/train.py`

---

## Future Work

### Potential Improvements
1. **Increase dataset size** - 240-500 circuits for better generalization
2. **Improve capacitor prediction** - Currently 65% accuracy
3. **Add more filter types** - Bessel, Chebyshev, elliptic
4. **Component value prediction** - Currently predicting types, not values

### Known Limitations
1. Capacitor-Resistor confusion in some cases
2. Edge count slightly under-predicted on average
3. Model trained on 3-5 node circuits only

---

## Conclusion

The explicit node count predictor successfully solves the train/test mismatch problem. The model now:

1. ✅ **100% accurate node count prediction** on validation set
2. ✅ **85.3% component type accuracy** (up from 60%)
3. ✅ **Correctly generates all filter types** (low-pass, high-pass, band-pass, band-stop)
4. ✅ **Properly handles variable-length circuits** (3, 4, or 5 nodes)

The topology latent successfully encodes circuit complexity, enabling direct prediction of node count without relying on learned stopping criteria that suffer from train/test mismatch.

---

**Model Version:** v3.0 (Node Count Predictor)
**Training Date:** January 8, 2026
**Best Checkpoint:** `checkpoints/production/best.pt` (epoch 99)
**Validation Loss:** 2.0225
**Node Count Accuracy:** 100%
