# Usage Guide

This guide shows how to use the circuit generation model for various tasks.

---

## Prerequisites

```bash
# Install dependencies
pip install torch torch-geometric numpy scipy networkx pyyaml tqdm

# Ensure you have the trained model
ls checkpoints/production/best.pt  # Should exist
```

---

## 1. Generate from Specifications (Primary Usage)

The primary way to generate circuits is by specifying **cutoff frequency** and **Q-factor**:

### Command Line

```bash
# Standard Butterworth (Q=0.707)
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707

# High-Q resonant
python scripts/generation/generate_from_specs.py --cutoff 5000 --q-factor 5.0

# Generate multiple samples
python scripts/generation/generate_from_specs.py --cutoff 1000 --q-factor 0.707 --num-samples 5
```

### Example Output

```
======================================================================
Specification-Driven Circuit Generation
======================================================================

Target specifications:
  Cutoff frequency: 10000.0 Hz
  Q-factor: 0.707
  Generation method: interpolate

Building specification database...
Built database with 360 circuits
  Cutoff range: 8.8 - 539651.2 Hz
  Q-factor range: 0.010 - 6.496

Generating 1 circuits...
======================================================================

Sample 1: Interpolated from 5 nearest
  Top neighbor: cutoff=9789.0 Hz, Q=0.707, weight=0.312
  Nearest match: cutoff=9789.0 Hz, Q=0.707
  Generated: GND--RCL--VOUT, VIN--R--VOUT
  Status: Valid (3 nodes, 2 edges)

======================================================================
Generation complete!
======================================================================
```

### How It Works

1. **Input:** Cutoff frequency (Hz) and Q-factor
2. **K-NN Search:** Find 5 nearest circuits in training database by spec similarity
3. **Latent Interpolation:** Weighted average of nearest latent codes
4. **Decode:** Generate circuit from interpolated latent

---

## 2. Loading the Model

```python
import torch
from ml.models.encoder import HierarchicalEncoder
from ml.models.decoder import SimplifiedCircuitDecoder

device = 'cpu'

# Initialize encoder
encoder = HierarchicalEncoder(
    node_feature_dim=4,
    edge_feature_dim=7,
    gnn_hidden_dim=64,
    gnn_num_layers=3,
    latent_dim=8,
    topo_latent_dim=2,
    values_latent_dim=2,
    pz_latent_dim=4,
    dropout=0.1
).to(device)

# Initialize decoder
decoder = SimplifiedCircuitDecoder(
    latent_dim=8,
    hidden_dim=256,
    num_heads=8,
    num_node_layers=4,
    max_nodes=10,
    dropout=0.1
).to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints/production/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print(f"Loaded from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
```

---

## 2. Generating Circuits

### Generate from Random Latent

```python
# Sample random latent code
z = torch.randn(1, 8)

# Generate circuit from latent code
with torch.no_grad():
    result = decoder.generate(z)

# Result contains:
#   node_types: [1, num_nodes] - node type indices
#   edge_existence: [1, num_nodes, num_nodes] - binary edge matrix
#   component_types: [1, num_nodes, num_nodes] - component type indices

print(f"Nodes: {result['node_types'].shape[1]}")
print(f"Edges: {int(result['edge_existence'].sum().item() // 2)}")
```

### Interpret Generated Circuit

```python
BASE_NAMES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INT', 4: 'INT'}
COMP_NAMES = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']

def describe_circuit(result):
    """Convert generation result to human-readable string."""
    node_types = result['node_types'][0]
    edge_exist = result['edge_existence'][0]
    comp_types = result['component_types'][0]
    num_nodes = node_types.shape[0]

    # Build unique names for each node (numbering internal nodes)
    node_names = []
    int_counter = 1
    for idx in range(num_nodes):
        nt = node_types[idx].item()
        if nt >= 3:  # Internal node
            node_names.append(f'INT{int_counter}')
            int_counter += 1
        else:
            node_names.append(BASE_NAMES[nt])

    edges = []
    for i in range(num_nodes):
        for j in range(i):
            if edge_exist[i, j] > 0.5:
                n1 = node_names[j]
                n2 = node_names[i]
                comp = COMP_NAMES[comp_types[i, j].item()]
                edges.append(f"{n1}--{comp}--{n2}")

    return ', '.join(edges) if edges else '(no edges)'

# Example
z = torch.randn(1, 8)
result = decoder.generate(z)
print(describe_circuit(result))
# Output: GND--R--VOUT, VIN--L--INT1, VOUT--C--INT1
```

### Generate Multiple Circuits

```python
circuits = []
for i in range(10):
    z = torch.randn(1, 8)
    with torch.no_grad():
        result = decoder.generate(z)
    circuits.append(result)
    print(f"Circuit {i+1}: {describe_circuit(result)}")
```

---

## 3. Encoding Existing Circuits

### Encode from Dataset

```python
from ml.data.dataset import CircuitDataset
from torch_geometric.data import Batch

dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

# Get a circuit
sample = dataset[0]
graph = sample['graph']
poles = [sample['poles']]
zeros = [sample['zeros']]

# Encode
with torch.no_grad():
    z, mu, logvar = encoder(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        torch.zeros(graph.x.size(0), dtype=torch.long),  # batch indices
        poles,
        zeros
    )

print(f"Latent code: {mu[0].numpy()}")
```

### Reconstruct Circuit

```python
# Use the mean latent (mu) for reconstruction
with torch.no_grad():
    reconstructed = decoder.generate(mu)

print(f"Original filter type: {dataset.circuits[0]['filter_type']}")
print(f"Reconstructed: {describe_circuit(reconstructed)}")
```

---

## 4. Latent Space Interpolation

### Interpolate Between Two Circuits

```python
# Encode two circuits
sample1 = dataset[0]   # low_pass
sample2 = dataset[60]  # high_pass

with torch.no_grad():
    _, mu1, _ = encoder(sample1['graph'].x, ...)
    _, mu2, _ = encoder(sample2['graph'].x, ...)

# Interpolate
print("Interpolation from low-pass to high-pass:")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z = (1 - alpha) * mu1 + alpha * mu2
    result = decoder.generate(z)
    print(f"  alpha={alpha:.2f}: {describe_circuit(result)}")
```

**Example Output:**
```
Interpolation from low-pass to high-pass:
  alpha=0.00: GND--C--VOUT, VIN--R--VOUT
  alpha=0.25: GND--C--VOUT, VIN--R--VOUT
  alpha=0.50: GND--R--VOUT, VIN--C--VOUT
  alpha=0.75: GND--R--VOUT, VIN--C--VOUT
  alpha=1.00: GND--R--VOUT, VIN--C--VOUT
```

### Latent Space Arithmetic

```python
# Get latents for different filter types
z_lowpass = encode(dataset[0])
z_highpass = encode(dataset[60])
z_bandpass = encode(dataset[120])

# "lowpass" + ("bandpass" - "highpass") = ?
z_new = z_lowpass + (z_bandpass - z_highpass)
result = decoder.generate(z_new)
print(describe_circuit(result))
```

---

## 5. Training

### Train from Scratch

```bash
python scripts/training/train.py
```

**Output:**
- Checkpoints saved to `checkpoints/production/`
- Best model selected by validation loss
- Progress printed every epoch

**Duration:** ~12 minutes on CPU for 100 epochs

### Training Code Overview

```python
# Key parts of train.py

# Create models
encoder = HierarchicalEncoder(...)
decoder = SimplifiedCircuitDecoder(latent_dim=8, hidden_dim=256, ...)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        # Encode
        z, mu, logvar = encoder(graph.x, graph.edge_index, ...)

        # Sample latent
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std

        # Unified edge-component target for teacher forcing (0=no edge, 1-7=type)
        target_edge_components = torch.where(
            targets['edge_existence'] > 0.5,
            targets['component_types'],
            torch.zeros_like(targets['component_types'])
        ).long()

        # Decode (teacher forcing: ground-truth edges fed back autoregressively)
        predictions = decoder(
            latent_code=latent,
            target_node_types=targets['node_types'],
            target_edges=target_edge_components
        )

        # Compute loss and backprop
        loss = criterion(predictions, targets)
        loss.backward()
```

---

## 6. Validation

```bash
python scripts/training/validate.py
```

**Output:**
```
Validation Results
==================
Node Count Accuracy: 100.0%
Edge Existence Accuracy: 100.0%
Component Type Accuracy: 100.0%
VIN Connectivity: 100.0%
VOUT Connectivity: 100.0%
```

---

## 7. Checking Circuit Validity

```python
def is_valid_circuit(result):
    """Check if generated circuit has VIN and VOUT connected."""
    edge_exist = result['edge_existence'][0]
    num_nodes = result['node_types'].shape[1]

    # Node 1 is VIN, Node 2 is VOUT
    vin_connected = (edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any()
    vout_connected = (edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any()

    return vin_connected and vout_connected

# Test validity rate
valid_count = 0
for _ in range(100):
    z = torch.randn(1, 8)
    result = decoder.generate(z)
    if is_valid_circuit(result):
        valid_count += 1

print(f"Validity rate: {valid_count}%")
```

---

## 8. Saving and Loading Circuits

### Save Circuit

```python
def save_circuit(result, filename):
    """Save generated circuit to file."""
    torch.save({
        'node_types': result['node_types'],
        'edge_existence': result['edge_existence'],
        'component_types': result['component_types']
    }, filename)

# Usage
result = decoder.generate(z)
save_circuit(result, 'my_circuit.pt')
```

### Load Circuit

```python
def load_circuit(filename):
    """Load circuit from file."""
    return torch.load(filename)

circuit = load_circuit('my_circuit.pt')
print(describe_circuit(circuit))
```

---

## 9. Building a Latent Database

For K-NN based generation or exploration:

```python
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def collate_fn(batch_list):
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    return {
        'graph': Batch.from_data_list(graphs),
        'poles': poles,
        'zeros': zeros
    }

# Build database
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

latent_db = []
with torch.no_grad():
    for batch in loader:
        z, mu, _ = encoder(
            batch['graph'].x,
            batch['graph'].edge_index,
            batch['graph'].edge_attr,
            batch['graph'].batch,
            batch['poles'],
            batch['zeros']
        )
        latent_db.append(mu[0])

latent_db = torch.stack(latent_db)
print(f"Built database of {len(latent_db)} latent codes")

# Find nearest neighbor to a random latent
z = torch.randn(8)
distances = ((latent_db - z) ** 2).sum(dim=1).sqrt()
nearest_idx = distances.argmin().item()
print(f"Nearest circuit: idx={nearest_idx}, filter_type={dataset.circuits[nearest_idx]['filter_type']}")
```

---

## 10. Troubleshooting

### Issue: Model not generating valid circuits

**Check:**
1. Loaded correct checkpoint
2. Using `eval()` mode
3. Using `torch.no_grad()` context

**Fix:**
```python
decoder.eval()
with torch.no_grad():
    result = decoder.generate(z)
```

### Issue: Import errors

**Fix:**
```bash
# Make sure you're in the project root
cd /path/to/Z-GED

# Run from there
python scripts/training/train.py
```

### Issue: Old checkpoint incompatible

**Problem:** Checkpoints from older models won't load with the current encoder.

**Fix:** Retrain the model:
```bash
python scripts/training/train.py
```

---

## API Reference

### SimplifiedCircuitDecoder

```python
class SimplifiedCircuitDecoder:
    def __init__(
        self,
        latent_dim: int = 8,      # Latent code dimension
        hidden_dim: int = 256,    # Hidden layer size
        num_heads: int = 8,       # Attention heads
        num_node_layers: int = 4, # Transformer layers
        max_nodes: int = 10,      # Maximum nodes
        dropout: float = 0.1
    )

    def forward(
        self,
        latent_code: Tensor,                    # [batch, latent_dim]
        target_node_types: Optional[Tensor],     # [batch, num_nodes] teacher forcing
        target_edges: Optional[Tensor] = None    # [batch, num_nodes, num_nodes] 0-7 teacher forcing
    ) -> Dict[str, Tensor]:
        """Training forward pass with teacher forcing for edges."""
        # Returns:
        #   node_types: [batch, num_nodes, 5] logits
        #   node_count_logits: [batch, max_nodes-2] logits
        #   edge_component_logits: [batch, num_nodes, num_nodes, 8] logits

    def generate(
        self,
        latent_code: Tensor,      # [batch, latent_dim]
        edge_threshold: float = 0.5,
        verbose: bool = False
    ) -> Dict[str, Tensor]:
        """Generate circuit from latent code (autoregressive edge decoding)."""
        # Returns:
        #   node_types: [batch, num_nodes]
        #   edge_existence: [batch, num_nodes, num_nodes]
        #   component_types: [batch, num_nodes, num_nodes]
```

### HierarchicalEncoder

```python
class HierarchicalEncoder:
    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 8,
        topo_latent_dim: int = 2,
        values_latent_dim: int = 2,
        pz_latent_dim: int = 4,
        dropout: float = 0.1
    )

    def forward(
        self,
        x: Tensor,           # Node features [num_nodes, node_feature_dim]
        edge_index: Tensor,  # Edge indices [2, num_edges]
        edge_attr: Tensor,   # Edge features [num_edges, edge_feature_dim]
        batch: Tensor,       # Batch indices [num_nodes]
        poles: List,         # List of pole arrays
        zeros: List          # List of zero arrays
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode circuit to latent space."""
        # Returns: (z, mu, logvar)
```

---

## References

- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Results:** See [GENERATION_RESULTS.md](GENERATION_RESULTS.md)
- **Training:** See `scripts/training/train.py`
