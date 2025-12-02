import pickle
import numpy as np
import networkx as nx

# Load dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

print("="*70)
print("ML READINESS ANALYSIS")
print("="*70)

# Sample circuit
sample = dataset[0]

print("\n1. GRAPH STRUCTURE")
print("-" * 70)
print(f"Graph format: {type(sample['graph_adj'])}")
print(f"Keys: {sample['graph_adj'].keys()}")

# Reconstruct graph to analyze
G = nx.adjacency_graph(sample['graph_adj'])
print(f"\nReconstructed graph type: {type(G)}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

print("\n2. NODE FEATURES")
print("-" * 70)
for node in sorted(G.nodes()):
    features = G.nodes[node].get('features', None)
    print(f"Node {node}: {features} (type: {type(features)})")

print("\n3. EDGE FEATURES")
print("-" * 70)
for u, v, data in list(G.edges(data=True))[:5]:  # Show first 5 edges
    features = data.get('features', None)
    print(f"Edge ({u},{v}): {features} (type: {type(features)})")

print("\n4. LABELS")
print("-" * 70)
print(f"Poles: {len(sample['label']['poles'])} (type: {type(sample['label']['poles'])})")
print(f"Zeros: {len(sample['label']['zeros'])} (type: {type(sample['label']['zeros'])})")
print(f"Gain: {sample['label']['gain']} (type: {type(sample['label']['gain'])})")

print("\n5. FREQUENCY RESPONSE (Alternative Label)")
print("-" * 70)
freq_resp = sample['frequency_response']
print(f"Frequency points: {len(freq_resp['freqs'])}")
print(f"H_magnitude shape: {freq_resp['H_magnitude'].shape}")
print(f"H_phase shape: {freq_resp['H_phase'].shape}")
print(f"Sample magnitudes (first 5): {freq_resp['H_magnitude'][:5]}")

print("\n6. FEATURE VECTOR DIMENSIONS")
print("-" * 70)
print(f"Node feature dimension: {len(G.nodes[0]['features'])}")
print(f"Edge feature dimension: {len(list(G.edges(data=True))[0][2]['features'])}")

print("\n7. DATASET STATISTICS")
print("-" * 70)
filter_types = {}
for circuit in dataset:
    ftype = circuit['filter_type']
    filter_types[ftype] = filter_types.get(ftype, 0) + 1

for ftype, count in sorted(filter_types.items()):
    print(f"{ftype}: {count} samples")

print("\n8. ML TASK POSSIBILITIES")
print("-" * 70)

# Check if we have valid labels
has_poles_zeros = any(len(d['label']['poles']) > 0 for d in dataset)
has_freq_response = all('frequency_response' in d for d in dataset)

print("\nSupervised Learning Tasks:")
if has_freq_response:
    print("✓ Frequency Response Prediction (regression)")
    print("  - Input: Graph structure + component values")
    print("  - Output: 701-dimensional frequency response vector")

print("✓ Filter Type Classification")
print("  - Input: Graph structure + component values")
print("  - Output: 6 classes (filter types)")

print("✓ Characteristic Frequency Prediction (regression)")
print("  - Input: Graph structure + component values")
print("  - Output: Cutoff/resonant frequency")

if has_poles_zeros:
    print("✓ Pole/Zero Prediction (if poles/zeros available)")
else:
    print("⚠ Pole/Zero Prediction (currently no poles/zeros extracted)")

print("\nGraph-based Learning Tasks:")
print("✓ Graph Neural Network (GNN) compatible")
print("✓ Message Passing Neural Networks (MPNN)")
print("✓ Graph Attention Networks (GAT)")

print("\n9. POTENTIAL ISSUES")
print("-" * 70)

# Check for consistency
node_dims = set()
edge_dims = set()
for circuit in dataset[:10]:  # Sample first 10
    G = nx.adjacency_graph(circuit['graph_adj'])
    for node in G.nodes():
        node_dims.add(len(G.nodes[node]['features']))
    for u, v, data in G.edges(data=True):
        edge_dims.add(len(data['features']))

if len(node_dims) == 1:
    print(f"✓ Consistent node feature dimensions: {node_dims.pop()}")
else:
    print(f"⚠ Inconsistent node feature dimensions: {node_dims}")

if len(edge_dims) == 1:
    print(f"✓ Consistent edge feature dimensions: {edge_dims.pop()}")
else:
    print(f"⚠ Inconsistent edge feature dimensions: {edge_dims}")

# Check for poles/zeros
circuits_with_poles = sum(1 for d in dataset if len(d['label']['poles']) > 0)
print(f"\n⚠ Circuits with poles extracted: {circuits_with_poles}/{len(dataset)}")
print(f"⚠ Circuits without poles: {len(dataset) - circuits_with_poles}/{len(dataset)}")

print("\n10. RECOMMENDATIONS FOR ML")
print("-" * 70)
print("\n✓ READY FOR:")
print("  - Filter type classification (6 classes)")
print("  - Frequency prediction (regression)")
print("  - Frequency response prediction (multi-output regression)")
print("  - Graph-based learning with PyTorch Geometric or DGL")
print("  - Transfer learning between filter types")

print("\n⚠ NEEDS IMPROVEMENT:")
print("  - Pole/zero extraction failing (use frequency response instead)")
print("  - Consider adding more samples per filter type for better ML performance")
print("  - May want to normalize/standardize edge features (wide value ranges)")

print("\n📊 SUGGESTED NEXT STEPS:")
print("  1. Use frequency response as primary label (works for all circuits)")
print("  2. Implement GNN architecture (e.g., GraphSAGE, GAT)")
print("  3. Try filter type classification as first ML task")
print("  4. Use characteristic frequency for regression task")
print("  5. Consider data augmentation (more samples)")

print("\n" + "="*70)
