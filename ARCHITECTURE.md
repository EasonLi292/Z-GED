# Circuit Generation Model Architecture

**Model Version:** v5.0 (Latent-Only Generation)

## Overview

This model generates RLC filter circuit topologies from an 8-dimensional latent space. Unlike conditional VAEs, no external specifications are required - the latent code alone determines the circuit structure.

**Key Change from v4.7:** Removed condition inputs (frequency, Q-factor). The decoder now uses only the latent code.

---

## Model Components

### 1. Hierarchical Encoder

**Purpose:** Encode circuit graphs into a structured 8D latent space

**Architecture:**
```
Input: Circuit graph (nodes, edges, transfer function)
  ↓
3-layer Component-Aware GNN (ImpedanceConv)
  ↓
Hierarchical latent: 8D = [topology(2) | values(2) | transfer_function(4)]
  ↓
Output: μ, σ for VAE sampling
```

**Parameters:** ~95,000

**Component-Aware Message Passing:**
```python
class ImpedanceConv:
    def forward(x, edge_index, edge_attr):
        # Separate transformations for each component type
        msg_R = self.lin_R(x_j, edge_feat)
        msg_C = self.lin_C(x_j, edge_feat)
        msg_L = self.lin_L(x_j, edge_feat)

        # Weighted combination based on component masks
        is_R, is_C, is_L = edge_attr[:, 3:6]
        message = is_R * msg_R + is_C * msg_C + is_L * msg_L
```

This ensures R, C, and L edges are processed with different learned transformations.

---

### 2. Latent Space (8D)

**Hierarchical Structure:**
```
z = [z_topology | z_values | z_transfer_function]
     [   2D     |   2D    |        4D           ]
```

| Dimensions | Name | Encodes |
|------------|------|---------|
| z[0:2] | Topology | Graph structure, filter type, node count |
| z[2:4] | Values | Component value distributions |
| z[4:8] | Transfer Function | Poles/zeros characteristics |

**Statistics (from 360 training circuits):**
```
z[0]: mean=-0.50, std=1.99, range=[-3.10, 3.27]
z[1]: mean=-0.87, std=2.96, range=[-4.03, 3.30]
z[2]: mean=-0.02, std=0.46, range=[-1.04, 0.78]
z[3]: mean= 0.05, std=0.68, range=[-1.31, 1.40]
z[4]: mean=-0.00, std=0.03, range=[-0.03, 0.05]
z[5]: mean=-0.00, std=0.01, range=[-0.03, 0.01]
z[6]: mean= 0.01, std=0.04, range=[-0.03, 0.08]
z[7]: mean=-0.01, std=0.01, range=[-0.02, 0.03]
```

The topology dimensions (z[0:2]) have the highest variance, as they encode the most structural information.

---

### 3. Simplified Circuit Decoder

**Purpose:** Generate circuit topology from latent code only (no conditions)

**Architecture:**
```
Input: Latent code z (8D)
  ↓
Context Encoder: Linear(8 → 256) + LayerNorm + ReLU
  ↓
Node Count Predictor: z[0:2] → 3-way classification (3, 4, or 5 nodes)
  ↓
Autoregressive Node Generation (GND, VIN, VOUT, INTERNAL...)
  ↓
Joint Edge-Component Prediction (for each node pair)
  ↓
Output: node_types, edge_existence, component_types
```

**Parameters:** ~4.9M

**Key Design: No Conditions**

Previous versions concatenated conditions (frequency, Q) with latent:
```python
# Old (v4.7):
context = encoder(torch.cat([latent, conditions], dim=-1))

# New (v5.0):
context = encoder(latent)  # Latent alone determines everything
```

---

### 4. Joint Edge-Component Prediction

**8-way Classification:**
```
Class 0: No edge
Class 1: Edge with R (resistor)
Class 2: Edge with C (capacitor)
Class 3: Edge with L (inductor)
Class 4: Edge with RC (parallel)
Class 5: Edge with RL (parallel)
Class 6: Edge with CL (parallel)
Class 7: Edge with RCL (parallel)
```

**Edge Decoder:**
```python
class LatentGuidedEdgeDecoder:
    def forward(self, node_i, node_j, latent):
        # Encode node pair
        edge = self.edge_encoder(concat([node_i, node_j]))

        # Cross-attention to latent context
        context = self.context_proj(latent)  # No conditions!
        attended = self.cross_attention(edge, context)

        # 8-way classification
        return self.output_head(attended)  # → [batch, 8]
```

---

## Loss Function

```python
total_loss = (
    node_type_loss      # Cross-entropy on node types
  + node_count_loss     # Cross-entropy on node count (3/4/5)
  + edge_component_loss # Cross-entropy on 8-way edge-component
  + connectivity_loss   # Ensure VIN/VOUT connected
  + kl_divergence       # VAE regularization
)
```

**Weights:**
- Node type: 1.0
- Node count: 5.0
- Edge-component: 2.0
- Connectivity: 5.0
- KL divergence: 0.01

---

## Generation Process

### Step 1: Obtain Latent Code

```python
# Option A: Random sampling
z = torch.randn(1, 8)

# Option B: Encode existing circuit
z, mu, logvar = encoder(graph)

# Option C: Interpolation
z = (1 - alpha) * z1 + alpha * z2
```

### Step 2: Generate Circuit

```python
with torch.no_grad():
    result = decoder.generate(z)

# Returns:
#   node_types: [batch, num_nodes] - indices (0=GND, 1=VIN, 2=VOUT, 3=INT)
#   edge_existence: [batch, num_nodes, num_nodes] - binary
#   component_types: [batch, num_nodes, num_nodes] - indices (0-7)
```

### Step 3: Interpret Results

```python
# Base node names and component types
BASE_NAMES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INT', 4: 'INT'}
COMP_NAMES = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']

# Build unique names (number internal nodes as INT1, INT2, etc.)
node_names = []
int_counter = 1
for idx in range(num_nodes):
    nt = node_types[idx]
    if nt >= 3:  # Internal node
        node_names.append(f'INT{int_counter}')
        int_counter += 1
    else:
        node_names.append(BASE_NAMES[nt])

# Extract edges
for i in range(num_nodes):
    for j in range(i):
        if edge_existence[i, j] > 0.5:
            comp = COMP_NAMES[component_types[i, j]]
            print(f"{node_names[j]}--{comp}--{node_names[i]}")
```

---

## Performance

### Validation Results (72 circuits)

| Metric | Accuracy |
|--------|----------|
| Node Count | 100% |
| Edge Existence | 100% |
| Component Type | 100% |
| VIN Connectivity | 100% |
| VOUT Connectivity | 100% |

### Training

- **Dataset:** 360 circuits (288 train, 72 val)
- **Epochs:** 100
- **Best Val Loss:** 1.0044 (epoch 95)
- **Training Time:** ~12 minutes on CPU

---

## Files

### Core Model
- `ml/models/encoder.py` - HierarchicalEncoder
- `ml/models/decoder.py` - SimplifiedCircuitDecoder
- `ml/models/decoder_components.py` - LatentGuidedEdgeDecoder
- `ml/models/gnn_layers.py` - ImpedanceConv, ImpedanceGNN
- `ml/models/node_decoder.py` - AutoregressiveNodeDecoder

### Training
- `ml/losses/gumbel_softmax_loss.py` - GumbelSoftmaxCircuitLoss
- `scripts/training/train.py` - Training script
- `scripts/training/validate.py` - Validation

### Checkpoints
- `checkpoints/production/best.pt` - Best model (val_loss=1.0044)

---

## Design Decisions

### Why Remove Conditions?

1. **Simplicity** - Latent space alone encodes all circuit information
2. **Flexibility** - Can sample, interpolate, or modify latents freely
3. **Decoupling** - No need to specify frequency/Q for generation
4. **Better Reconstruction** - 100% accuracy vs 90% with conditions

### Why Joint Edge-Component Prediction?

**Problem with separate heads:**
```python
edge_exists = binary_classifier(...)      # Independent decision
component_type = 7_way_classifier(...)    # Independent decision
# Issue: No coordination between decisions
```

**Solution with joint prediction:**
```python
edge_component = 8_way_classifier(...)    # Single unified decision
# Class 0 = no edge, Classes 1-7 = edge with specific component
```

### Why Hierarchical Latent Space?

Enables semantic manipulation:
```python
# Modify only topology
z_new = z.clone()
z_new[0:2] = target_topology

# Modify only values
z_new[2:4] = target_values
```

---

**Status:** Production ready

**Checkpoint:** `checkpoints/production/best.pt`
