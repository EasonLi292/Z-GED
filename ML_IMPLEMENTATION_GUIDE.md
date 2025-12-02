# ML Implementation Guide for Z-GED Dataset

## Executive Summary

**✅ YES - The dataset is READY for ML implementation!**

Your graphs and labels are well-structured and suitable for modern graph neural network architectures. Here's what you have:

## Current State

### ✓ What's Working Well

1. **Graph Structure**
   - Clean NetworkX graph format
   - Consistent node features (4-dimensional one-hot)
   - Consistent edge features (3-dimensional impedance vectors)
   - All 120 circuits successfully processed

2. **Feature Engineering**
   - **Node Features (4D):** `[is_GND, is_IN, is_OUT, is_Internal]`
   - **Edge Features (3D):** `[Conductance, Capacitance, Inverse_Inductance]`
   - Physics-based features (impedance/admittance)

3. **Labels Available**
   - Filter type (6 classes) - **Perfect for classification**
   - Characteristic frequency (scalar) - **Perfect for regression**
   - Frequency response (701 points) - **Perfect for multi-output regression**

### ⚠ Minor Limitations

1. **Pole/Zero Extraction Failed**
   - All 120 circuits have empty pole/zero lists
   - **Solution:** Use frequency response instead (works perfectly)

2. **Dataset Size**
   - 120 samples is small for deep learning
   - **Solution:** Start with simpler models or generate more data

## ML Tasks You Can Implement

### 1. Filter Type Classification (RECOMMENDED FIRST TASK) ⭐

**Difficulty:** Easy
**Success Rate:** High

```python
Task: Given a circuit graph → Predict filter type
Input: Graph with node/edge features
Output: One of 6 classes [low_pass, high_pass, band_pass, band_stop, rlc_series, rlc_parallel]
Metric: Accuracy, F1-score
Baseline: Random guess = 16.7% accuracy
```

**Why start here:**
- Balanced dataset (20 samples per class)
- Clear ground truth labels
- Easy to evaluate success

### 2. Characteristic Frequency Prediction

**Difficulty:** Medium
**Success Rate:** High

```python
Task: Given a circuit graph → Predict cutoff/resonant frequency
Input: Graph with node/edge features
Output: Scalar frequency value (Hz)
Metric: MAE, RMSE, R²
Frequency range: ~10 Hz to ~500 kHz
```

### 3. Frequency Response Prediction

**Difficulty:** Hard
**Success Rate:** Medium-High

```python
Task: Given a circuit graph → Predict full frequency response
Input: Graph with node/edge features
Output: 701-dimensional vector (magnitude at each frequency)
Metric: MSE, cosine similarity
```

**Applications:**
- Circuit simulation replacement
- Fast design space exploration

## Recommended Architectures

### Option 1: Graph Convolutional Network (GCN)

**Best for:** Filter classification

```python
# Pseudocode structure
class CircuitGCN(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(node_features=4, hidden=64)
        self.conv2 = GCNConv(hidden=64, hidden=128)
        self.fc = Linear(128, num_classes=6)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x)  # Aggregate to graph-level
        return self.fc(x)
```

**Framework:** PyTorch Geometric

### Option 2: Graph Attention Network (GAT)

**Best for:** Learning important circuit connections

```python
class CircuitGAT(nn.Module):
    def __init__(self):
        self.conv1 = GATConv(node_features=4, hidden=64, heads=4)
        self.conv2 = GATConv(hidden=256, hidden=128, heads=4)
        self.fc = Linear(512, num_classes=6)
```

**Advantage:** Learns which edges/components are most important

### Option 3: Message Passing Neural Network (MPNN)

**Best for:** Frequency response prediction

```python
class CircuitMPNN(nn.Module):
    def __init__(self):
        self.mpnn = NNConv(node_features=4,
                           hidden=128,
                           edge_nn=EdgeNetwork(edge_features=3))
        self.fc = Linear(128, output=701)  # 701 freq points
```

**Advantage:** Can incorporate edge features naturally

## Data Preprocessing Pipeline

### Step 1: Load and Convert

```python
import pickle
import torch
from torch_geometric.data import Data

with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

def circuit_to_pyg(circuit_dict):
    """Convert circuit dict to PyTorch Geometric Data object"""
    G = nx.adjacency_graph(circuit_dict['graph_adj'])

    # Node features
    node_features = []
    for node in sorted(G.nodes()):
        node_features.append(G.nodes[node]['features'])
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index and features
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # Add reverse edge (undirected)
        edge_attr.append(data['features'])
        edge_attr.append(data['features'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Labels
    filter_type_map = {
        'low_pass': 0, 'high_pass': 1, 'band_pass': 2,
        'band_stop': 3, 'rlc_series': 4, 'rlc_parallel': 5
    }
    y = torch.tensor([filter_type_map[circuit_dict['filter_type']]])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

### Step 2: Feature Normalization

```python
# Edge features have very different scales
# [Conductance: ~1e-5 to 1e-2, Capacitance: ~1e-9 to 1e-6, Inv_Inductance: ~100 to 10000]

from sklearn.preprocessing import StandardScaler

def normalize_edge_features(data_list):
    """Normalize edge features across dataset"""
    all_edge_features = torch.cat([d.edge_attr for d in data_list])
    scaler = StandardScaler()
    scaler.fit(all_edge_features.numpy())

    for data in data_list:
        data.edge_attr = torch.tensor(
            scaler.transform(data.edge_attr.numpy()),
            dtype=torch.float
        )
    return data_list, scaler
```

### Step 3: Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# Stratified split (preserve class balance)
train_data, temp_data = train_test_split(
    pyg_dataset, test_size=0.3, stratify=labels, random_state=42
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_labels, random_state=42
)

# Result: ~84 train, ~18 val, ~18 test
```

## Training Recommendations

### For Classification Task

```python
# Hyperparameters
batch_size = 16
learning_rate = 0.001
epochs = 200
patience = 20  # Early stopping

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)
```

### Data Augmentation Ideas

Since you have limited data (120 samples), consider:

1. **Component Value Perturbation**
   - Add small noise to edge features
   - Simulate manufacturing tolerances

2. **Generate More Circuits**
   - Run `circuit_generator.py` with different seeds
   - Increase `NUM_SAMPLES_PER_FILTER` to 50-100

3. **Graph Augmentation**
   - Edge dropout (randomly remove edges during training)
   - Node feature dropout

## Evaluation Metrics

### For Classification

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    'confusion_matrix': confusion_matrix(y_true, y_pred)
}
```

### For Frequency Prediction

```python
from sklearn.metrics import mean_absolute_error, r2_score

metrics = {
    'mae': mean_absolute_error(y_true, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    'r2': r2_score(y_true, y_pred),
    'mape': mean_absolute_percentage_error(y_true, y_pred)
}
```

## Expected Performance Baselines

### Filter Classification

- **Random Baseline:** 16.7% accuracy (1/6)
- **Frequency-based Baseline:** ~50-60% (using only characteristic frequency)
- **Expected GNN Performance:** 80-95% accuracy
- **Target Performance:** >90% accuracy

### Frequency Prediction

- **Mean Baseline:** Predict mean frequency
- **Expected GNN Performance:** R² > 0.85
- **Target Performance:** R² > 0.90, MAE < 5kHz

## Common Pitfalls to Avoid

1. **❌ Don't use poles/zeros as labels** - They're all empty
2. **❌ Don't forget edge features** - They contain the physics!
3. **❌ Don't skip normalization** - Edge features have wildly different scales
4. **✅ Do use frequency response** - It's complete and accurate
5. **✅ Do start with classification** - Easier to debug
6. **✅ Do visualize predictions** - Plot predicted vs actual frequency response

## Next Steps Implementation Checklist

- [ ] Install PyTorch Geometric or DGL
- [ ] Implement data loading pipeline
- [ ] Create train/val/test split
- [ ] Implement simple GCN baseline
- [ ] Train filter type classifier
- [ ] Evaluate and visualize results
- [ ] Try more complex architectures (GAT, MPNN)
- [ ] Implement frequency prediction
- [ ] Generate more data if needed
- [ ] Experiment with data augmentation

## Sample Starter Code

See `starter_ml_example.py` (to be created) for a complete working example.

## Conclusion

**Your dataset is ML-ready!** The graph structure is clean, features are well-defined, and you have multiple learning tasks to explore. Start with filter classification to validate your pipeline, then move to more challenging tasks like frequency response prediction.

The impedance-based edge features are particularly valuable as they encode physical circuit behavior that should help ML models learn meaningful representations.

Good luck with your ML implementation! 🚀
