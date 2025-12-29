# Usage Guide

This guide shows how to use the circuit generation model for various tasks.

---

## Prerequisites

```bash
# Install dependencies
pip install torch torch-geometric numpy scipy

# Ensure you have the trained model
ls checkpoints/production/best.pt  # Should exist
```

---

## 1. Training the Model

### Basic Training

```bash
python scripts/train.py
```

**Output:**
- Checkpoints saved to `checkpoints/production/`
- Best model selected based on validation loss
- Training progress printed every epoch

**Configuration:**
- Modify hyperparameters in the script directly
- Default: 100 epochs, batch_size=16, lr=1e-4

**Duration:** ~2 hours on CPU, ~30 minutes on GPU

---

## 2. Validating the Model

### Run Comprehensive Validation

```bash
python scripts/validate.py
```

**Output:**
```
===============================================================
Circuit Generation Model Validation
===============================================================

Loaded checkpoint from epoch 98
Best validation loss: 0.2142

===============================================================
Validating on Full Validation Set
===============================================================

Overall Accuracy: 100.0% (128/128 edges correct)

Component Type Accuracy:
  R:   100.0% (68/68)
  C:   100.0% (32/32)
  L:   100.0% (12/12)
  RCL: 100.0% (16/16)

Confusion Matrix:
       None    R    C    L   RC   RL   CL  RCL
None     0    0    0    0    0    0    0    0
R        0   68    0    0    0    0    0    0
C        0    0   32    0    0    0    0    0
L        0    0    0   12    0    0    0    0
RC       0    0    0    0    0    0    0    0
RL       0    0    0    0    0    0    0    0
CL       0    0    0    0    0    0    0    0
RCL      0    0    0    0    0    0    0   16
```

---

## 3. Generating Circuits

### Generate from Random Latent Codes

```python
import torch
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder

# Load model
device = 'cpu'
encoder = HierarchicalEncoder(...).to(device)
decoder = LatentGuidedGraphGPTDecoder(...).to(device)

checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

# Generate circuit
latent = torch.randn(1, 8, device=device)  # Random latent code
conditions = torch.randn(1, 2, device=device)  # Random conditions

circuit = decoder.generate(latent, conditions, verbose=True)

# Access results
node_types = circuit['node_types']         # [batch, max_nodes]
edge_existence = circuit['edge_existence'] # [batch, max_nodes, max_nodes]
edge_values = circuit['edge_values']       # [batch, max_nodes, max_nodes, 3]
```

### Generate from Specific Circuit (Reconstruction)

```python
from ml.data.dataset import CircuitDataset
from torch_geometric.data import Batch

# Load dataset
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
sample = dataset[0]  # Get first circuit

# Encode circuit
graph = sample['graph']
poles = [sample['poles']]
zeros = [sample['zeros']]

with torch.no_grad():
    z, mu, logvar = encoder(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        torch.zeros(graph.x.size(0), dtype=torch.long),  # batch
        poles,
        zeros
    )

    # Reconstruct from mean latent
    conditions = torch.randn(1, 2, device=device)
    reconstructed = decoder.generate(mu, conditions, verbose=True)

# Compare original vs reconstructed
print(f"Original edges: {graph.edge_index.shape[1] // 2}")
print(f"Reconstructed edges: {(reconstructed['edge_existence'][0] > 0.5).sum().item() // 2}")
```

---

## 4. Evaluating Transfer Function Accuracy

```bash
python scripts/evaluate_tf.py
```

**What it does:**
- Loads validation set
- Generates circuits from encoder latent codes
- Compares predicted vs ground truth transfer functions
- Reports pole/zero count and value accuracy

**Output:**
```
Transfer Function Evaluation
=============================

Pole Count Accuracy: 83.3%
Zero Count Accuracy: 100.0%

Pole Value MAE: 12.5 Hz
Zero Value MAE: 8.3 Hz
```

---

## 5. Analyzing Generated Circuits

### Extract Circuit Netlist

```python
def circuit_to_netlist(circuit):
    """Convert generated circuit to human-readable netlist."""
    node_types = circuit['node_types'][0]
    edge_exist = circuit['edge_existence'][0]
    edge_vals = circuit['edge_values'][0]

    # Node mapping
    node_names = ['GND', 'VIN', 'VOUT', 'INT1', 'INT2']

    # Extract edges
    edges = []
    for i in range(5):
        for j in range(i):
            if edge_exist[i, j] > 0.5:
                # Decode component values
                log_C, G, log_L_inv = edge_vals[i, j, :3]
                C = torch.exp(log_C).item()  # Farads
                R = 1.0 / G.item() if G > 0 else 0  # Ohms
                L = 1.0 / torch.exp(log_L_inv).item() if log_L_inv > -10 else 0  # Henries

                edges.append({
                    'nodes': (node_names[j], node_names[i]),
                    'C': C,
                    'R': R,
                    'L': L
                })

    return edges

# Usage
circuit = decoder.generate(latent, conditions)
netlist = circuit_to_netlist(circuit)

for edge in netlist:
    print(f"{edge['nodes'][0]} -- {edge['nodes'][1]}")
    if edge['R'] > 0:
        print(f"  R = {edge['R']:.2e} Ω")
    if edge['C'] > 0:
        print(f"  C = {edge['C']:.2e} F")
    if edge['L'] > 0:
        print(f"  L = {edge['L']:.2e} H")
```

**Example Output:**
```
GND -- VOUT
  C = 3.30e-08 F  (33 nF)

VIN -- VOUT
  R = 1.00e+03 Ω  (1 kΩ)
```

---

## 6. Circuit Visualization

### ASCII Circuit Diagram

```python
def visualize_circuit(circuit):
    """Print ASCII circuit diagram."""
    node_types = circuit['node_types'][0]
    edge_exist = circuit['edge_existence'][0]

    print("\nCircuit Topology:")
    print("=================")

    nodes = ['GND', 'VIN', 'VOUT', 'INT1', 'INT2']

    for i in range(5):
        for j in range(i):
            if edge_exist[i, j] > 0.5:
                print(f"{nodes[j]:>4s} ──────── {nodes[i]:<4s}")

# Usage
circuit = decoder.generate(latent, conditions)
visualize_circuit(circuit)
```

**Output:**
```
Circuit Topology:
=================
 GND ──────── VOUT
 VIN ──────── VOUT
 VIN ──────── INT1
INT1 ──────── VOUT
```

---

## 7. Batch Generation

### Generate Multiple Circuits

```python
def generate_batch(decoder, num_circuits=10):
    """Generate multiple circuits at once."""
    latents = torch.randn(num_circuits, 8, device=device)
    conditions = torch.randn(num_circuits, 2, device=device)

    circuits = []
    for i in range(num_circuits):
        circuit = decoder.generate(
            latents[i:i+1],
            conditions[i:i+1],
            verbose=False
        )
        circuits.append(circuit)

    return circuits

# Generate 50 circuits
circuits = generate_batch(decoder, num_circuits=50)

# Analyze distribution
edge_counts = []
for circuit in circuits:
    num_edges = (circuit['edge_existence'][0] > 0.5).sum().item() // 2
    edge_counts.append(num_edges)

print(f"Mean edges: {np.mean(edge_counts):.2f}")
print(f"Edge range: {min(edge_counts)}-{max(edge_counts)}")
```

---

## 8. Latent Space Exploration

### Interpolate Between Circuits

```python
def interpolate_circuits(circuit1_idx, circuit2_idx, steps=10):
    """Generate interpolation between two circuits."""
    # Encode both circuits
    sample1 = dataset[circuit1_idx]
    sample2 = dataset[circuit2_idx]

    with torch.no_grad():
        z1, mu1, _ = encoder(...)  # Encode circuit 1
        z2, mu2, _ = encoder(...)  # Encode circuit 2

    # Interpolate latent codes
    interpolated = []
    for alpha in torch.linspace(0, 1, steps):
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        conditions = torch.randn(1, 2)

        circuit = decoder.generate(z_interp, conditions, verbose=False)
        interpolated.append(circuit)

    return interpolated

# Interpolate between low-pass and band-pass filter
circuits = interpolate_circuits(0, 10, steps=20)
```

### Latent Space Arithmetic

```python
# "R filter" - "C filter" + "L filter" = ?
z_R = encode(dataset[0])   # R-dominant circuit
z_C = encode(dataset[5])   # C-dominant circuit
z_L = encode(dataset[10])  # L-dominant circuit

z_new = z_R - z_C + z_L
circuit = decoder.generate(z_new, conditions)
```

---

## 9. Common Patterns

### Pattern 1: Check Generation Quality

```python
# Generate 100 circuits and check validity
valid_count = 0
total_count = 100

for _ in range(total_count):
    latent = torch.randn(1, 8)
    conditions = torch.randn(1, 2)

    circuit = decoder.generate(latent, conditions, verbose=False)

    # Check VIN and VOUT connectivity
    edge_exist = circuit['edge_existence'][0]
    vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
    vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()

    if vin_connected and vout_connected:
        valid_count += 1

print(f"Validity: {100 * valid_count / total_count:.1f}%")
```

### Pattern 2: Save Generated Circuit

```python
def save_circuit(circuit, filename):
    """Save circuit to file."""
    torch.save({
        'node_types': circuit['node_types'],
        'edge_existence': circuit['edge_existence'],
        'edge_values': circuit['edge_values']
    }, filename)

def load_circuit(filename):
    """Load circuit from file."""
    return torch.load(filename)

# Usage
circuit = decoder.generate(latent, conditions)
save_circuit(circuit, 'generated_circuit.pt')
loaded = load_circuit('generated_circuit.pt')
```

---

## Troubleshooting

### Issue: Model not generating valid circuits

**Check:**
1. Loaded correct checkpoint: `checkpoints/production/best.pt`
2. Using eval mode: `decoder.eval()`
3. Latent code in valid range: `z ~ N(0, 1)`

**Fix:**
```python
decoder.eval()  # Disable dropout
with torch.no_grad():  # Disable gradient computation
    circuit = decoder.generate(latent, conditions)
```

### Issue: Component type always "R"

**Check:** Loss weights in training
**Fix:** Retrain with balanced loss (should be fixed in production model)

### Issue: Too few edges generated

**Check:** Dataset distribution
**Fix:** Model generates 2-4 edges (mean 2.67) matching training data - this is correct behavior

---

## Additional Scripts

### Analyze Dataset Distribution

```bash
python scripts/explore_dataset_specs.py
```

Shows distribution of:
- Edge counts
- Component types
- Transfer function characteristics

### Precompute GED Metrics

```bash
python scripts/precompute_ged.py
```

Computes Graph Edit Distance between generated and target circuits.

---

## References

- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Examples:** See [GENERATION_EXAMPLES.md](GENERATION_EXAMPLES.md)
- **Training Details:** See [scripts/train.py](scripts/train.py)
