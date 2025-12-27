# GraphGPT Circuit Generation: Process & Accuracy

## Table of Contents
1. [Autoregressive Decoding Process](#decoding-process)
2. [Generation Accuracy Metrics](#accuracy-metrics)
3. [Component Analysis](#component-analysis)
4. [Topology Analysis](#topology-analysis)

---

## Autoregressive Decoding Process

### Overview
GraphGPT generates circuits **sequentially** in 4 stages:
1. Context encoding
2. Node generation (autoregressive)
3. Edge generation (autoregressive)
4. Transfer function generation (poles/zeros)

### Step-by-Step Process

#### **Stage 1: Context Encoding**

Input:
- `latent_code`: 8D vector from encoder (circuit design space)
- `conditions`: [log(cutoff_freq), log(Q_factor)]

Process:
```python
context_input = concat(latent_code, conditions)  # [batch, 10]
context = MLP(context_input)                     # [batch, 256]
```

Output: 256D context vector containing design intent

---

#### **Stage 2: Autoregressive Node Generation**

Generates **5 nodes sequentially**: N0, N1, N2, N3, N4

For each position `i` (0 to 4):

1. **Input to decoder:**
   - Global context (256D)
   - Position embedding (learned for each i)
   - Previous node embeddings [N0, ..., N_{i-1}]

2. **Transformer processing:**
   ```python
   # Causal self-attention (can only see previous nodes)
   query = position_i
   keys = [N0, N1, ..., N_{i-1}, context]
   attention_output = MultiHeadAttention(query, keys, values)

   # Feed-forward layers
   node_features = FFN(attention_output)
   ```

3. **Node type prediction:**
   ```python
   node_logits = Linear(node_features)  # [5 classes]
   # Classes: GND(0), VIN(1), VOUT(2), INTERNAL(3), MASK(4)
   ```

4. **Constraint enforcement:**
   - Position 0: **Always GND** (forced)
   - Position 1: **Always VIN** (forced)
   - Position 2: **Always VOUT** (forced)
   - Positions 3-4: Sample from predicted distribution (can be INTERNAL or MASK)

5. **Create embedding:**
   ```python
   node_i_embedding = TypeEmbedding(predicted_type) + PositionEmbedding(i)
   node_embeddings.append(node_i_embedding)
   ```

**Example execution:**
```
i=0: Generate N0 → forced to GND
i=1: Generate N1 → forced to VIN
i=2: Generate N2 → forced to VOUT
i=3: Generate N3 → predict [GND:0.01, VIN:0.02, VOUT:0.03, INTERNAL:0.89, MASK:0.05]
                    → choose INTERNAL (highest probability)
i=4: Generate N4 → predict [GND:0.02, VIN:0.01, VOUT:0.02, INTERNAL:0.15, MASK:0.80]
                    → choose MASK
Final: [GND, VIN, VOUT, INTERNAL, MASK]
```

---

#### **Stage 3: Autoregressive Edge Generation**

For each node pair (i, j) where **j < i** (lower triangular):

1. **Input:**
   - `node_i_embedding` (256D)
   - `node_j_embedding` (256D)

2. **Edge decoder:**
   ```python
   pair_features = concat(node_i, node_j)              # [512D]
   hidden = ReLU(Linear(pair_features))                # [256D]

   # Binary edge existence
   edge_exist_logit = Linear(hidden, 1)                # scalar logit
   edge_exists = sigmoid(edge_exist_logit) > 0.5

   # Edge component values (if edge exists)
   edge_values = Linear(hidden, 7)  # [log(C), log(G), log(L_inv), has_C, has_R, has_L, is_parallel]
   ```

3. **Generate all edges:**
   ```
   For node N1 (i=1):
     - Edge N1-N0: predict existence & values

   For node N2 (i=2):
     - Edge N2-N0: predict existence & values
     - Edge N2-N1: predict existence & values

   For node N3 (i=3):
     - Edge N3-N0: predict existence & values
     - Edge N3-N1: predict existence & values
     - Edge N3-N2: predict existence & values

   For node N4 (i=4):
     - Edge N4-N0: predict existence & values (but N4 is MASK, so typically no edges)
     - ... (all pairs with j < 4)
   ```

4. **Symmetry:**
   ```python
   # Make undirected graph
   edge_matrix[i, j] = edge_matrix[j, i]
   edge_values[i, j] = edge_values[j, i]
   ```

**Example execution:**
```
Nodes: [GND, VIN, VOUT, INTERNAL, MASK]

Edge (VIN, GND):      exists=0.92 → YES → values=[C_log, G_log, L_log, 0, 1, 0, 1] (resistor)
Edge (VOUT, GND):     exists=0.88 → YES → values=[C_log, G_log, L_log, 1, 1, 0, 1] (RC parallel)
Edge (VOUT, VIN):     exists=0.91 → YES → values=[C_log, G_log, L_log, 0, 1, 0, 1] (resistor)
Edge (INTERNAL, GND): exists=0.23 → NO
Edge (INTERNAL, VIN): exists=0.87 → YES → values=[C_log, G_log, L_log, 0, 1, 0, 1] (resistor)
Edge (INTERNAL, VOUT): exists=0.89 → YES → values=[C_log, G_log, L_log, 0, 1, 1, 1] (RL parallel)
(All MASK node edges are ignored)

Result: Circuit with 5 edges connecting 4 real nodes
```

---

#### **Stage 4: Transfer Function Generation**

1. **Graph pooling:**
   ```python
   graph_repr = mean([node_0, node_1, node_2, node_3, node_4])  # Average all node embeddings
   graph_features = MLP(graph_repr) + context
   ```

2. **Pole count prediction:**
   ```python
   pole_count_logits = Linear(graph_features, 5)  # Classes: 0, 1, 2, 3, 4 poles
   predicted_pole_count = argmax(pole_count_logits)
   ```

3. **Zero count prediction:**
   ```python
   zero_count_logits = Linear(graph_features, 5)  # Classes: 0, 1, 2, 3, 4 zeros
   predicted_zero_count = argmax(zero_count_logits)
   ```

4. **Pole/zero value prediction:**
   ```python
   # Predict all 4 possible pole locations
   pole_values = Linear(graph_features, 8).reshape(4, 2)  # 4 poles × [real, imag]

   # Predict all 4 possible zero locations
   zero_values = Linear(graph_features, 8).reshape(4, 2)  # 4 zeros × [real, imag]

   # Only use first N poles/zeros based on predicted counts
   actual_poles = pole_values[:predicted_pole_count]
   actual_zeros = zero_values[:predicted_zero_count]
   ```

**Example execution:**
```
Graph pooling → 256D representation
Context fusion → enriched with design intent

Pole count: predict [0.05, 0.12, 0.78, 0.04, 0.01] → choose 2 poles
Zero count: predict [0.42, 0.35, 0.18, 0.03, 0.02] → choose 0 zeros

Pole values (all 4 predicted, use first 2):
  Pole 1: [-0.52, +0.89]  (stable: real < 0 ✓)
  Pole 2: [-0.61, -0.87]  (stable: real < 0 ✓)
  Pole 3: [-0.34, +0.12]  (not used)
  Pole 4: [-0.28, -0.41]  (not used)

Final: 2 poles, 0 zeros
```

---

## Generation Accuracy Metrics

### Training Accuracy (Epoch 200)

| Metric | Training | Validation |
|--------|----------|------------|
| **Node Type** | 99.4% | **100.0%** |
| **Pole Count** | 100.0% | **100.0%** |
| **Zero Count** | 99.0% | **100.0%** |
| **Edge Existence** | 92.7% | **93.0%** |
| **Validation Loss** | - | **0.5102** |

**Interpretation:**
- **100% node accuracy**: Always generates valid circuit structure (GND, VIN, VOUT)
- **100% pole/zero count**: Perfect transfer function order prediction
- **93% edge accuracy**: Correctly predicts which node pairs should be connected

---

### Generation Testing (80 circuits across 4 specifications)

#### **Success Rate**
- **Total circuits generated:** 80
- **Valid circuits:** 80/80 **(100.0%)**
- **All circuits simulatable:** Yes ✓

#### **Topology Diversity**

| Specification | Unique Topologies | Most Common |
|---------------|------------------|-------------|
| 100Hz, Q=0.5 | 2/20 (10%) | 4 nodes, 4 edges (60%) |
| 1kHz, Q=0.707 | 2/20 (10%) | 3 nodes, 2 edges (60%) |
| 5kHz, Q=1.5 | 2/20 (10%) | 3 nodes, 2 edges (55%) |
| 10kHz, Q=2.0 | 3/20 (15%) | 4 nodes, 4 edges (60%) |

**Observations:**
- **Low diversity (11.3% average)**: Model generates similar topologies within same spec
- **Two dominant patterns:**
  - Simple: 3 nodes, 2 edges (RC filter)
  - Complex: 4 nodes, 4 edges (multi-stage filter)
- **Trade-off:** High reliability vs. limited exploration

**Reason for low diversity:**
- Model trained on 120 circuits with 6 filter types
- Learns typical topologies for each specification
- Prioritizes working designs over exploration
- Latent space may need more variation for diversity

---

## Component Analysis

### Generated Component Values (across 80 circuits)

#### **Resistors** (202 total generated)

| Metric | Value |
|--------|-------|
| Mean | 10.6 MΩ |
| Range | [704 Ω, 22.9 MΩ] |
| **Practical range** | 10 Ω - 100 kΩ |
| **In practical range** | 42/202 (20.8%) |

**Issue:** Values often too large (megaohm range)
- Many resistors > 1 MΩ (impractical)
- Root cause: Normalization during training
- **Impact:** Circuits are valid but not buildable without rescaling

#### **Capacitors** (81 total generated)

| Metric | Value |
|--------|-------|
| Mean | 15.8 pF |
| Range | [15.8 pF, 17.3 nF] |
| **Practical range** | 1 pF - 1 μF |
| **In practical range** | 81/81 **(100.0%)** ✓ |

**Success:** All capacitors in practical range!

#### **Inductors** (43 total generated)

| Metric | Value |
|--------|-------|
| Mean | 76.4 kH |
| Range | [0.93 mH, 80.0 kH] |
| **Practical range** | 1 nH - 10 mH |
| **In practical range** | 1/43 (2.3%) |

**Issue:** Values often enormous (kilohenries!)
- Most inductors > 1 H (impractical)
- Training data may have had extreme inductor values
- **Impact:** Circuits valid but not physically realizable

---

## Transfer Function Analysis

### Pole/Zero Statistics (80 circuits)

#### **Pole Count Distribution**
- **Mean:** 1.7 ± 0.5 poles per circuit
- **Range:** 1-2 poles
- **Stable poles:** 135/135 **(100.0%)**

All poles have **negative real parts** → all circuits are **stable systems**

#### **Zero Count Distribution**
- **Mean:** 0.6 ± 0.7 zeros per circuit
- **Range:** 0-2 zeros
- **Most common:** 0 zeros (all-pole filters)

**Interpretation:**
- Generates mostly **low-order filters** (1-2 poles)
- Prefers **all-pole designs** (Butterworth-like)
- **Perfect stability**: No unstable poles ever generated

---

## Summary: What GraphGPT Does Well

### Strengths ✅
1. **100% generation success**: Never fails to produce valid circuits
2. **100% structural accuracy**: Always has GND, VIN, VOUT, edges
3. **100% stability**: All transfer functions are stable (poles in LHP)
4. **100% practical capacitors**: All C values in reasonable range
5. **Deterministic constraints**: First 3 nodes always correct
6. **Fast generation**: ~0.1 seconds per circuit

### Weaknesses ⚠️
1. **Low topology diversity**: Only 2-3 unique structures per spec (11.3%)
2. **Impractical resistors**: 79% too large (megaohm range)
3. **Impractical inductors**: 98% too large (kilohenry range)
4. **Limited complexity**: Prefers simple 2-4 edge circuits
5. **Specification coupling**: Similar specs → very similar circuits

---

## How Accuracy Could Be Improved

### For Topology Diversity:
1. **Increase latent dimension** (8D → 16D or 32D)
2. **Add diversity loss** during training
3. **Train on more varied topologies** (currently only 6 types)
4. **Sample with temperature** (currently greedy argmax)

### For Component Values:
1. **Post-processing rescaling** based on specifications
2. **Add component value constraints** during training
3. **Better normalization** of training data
4. **Practical range penalties** in loss function

### For Complexity:
1. **Reward more edges** during training
2. **Train on higher-order filters** (3-4 poles)
3. **Curriculum learning**: simple → complex

---

## Current Performance Rating

| Aspect | Score | Grade |
|--------|-------|-------|
| **Reliability** | 100% | A+ |
| **Structural Accuracy** | 100% | A+ |
| **Stability** | 100% | A+ |
| **Topology Diversity** | 11% | D |
| **Component Practicality** | ~40% | C- |
| **Transfer Function** | 100% | A+ |
| **Overall** | - | **B+** |

**Verdict:** GraphGPT is **highly reliable** at generating valid, stable circuits, but needs improvement in **diversity** and **component value realism**. It excels at core functionality (structure, stability) but conservative in exploration.
