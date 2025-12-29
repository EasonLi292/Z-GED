"""Create stratified train/val split based on component type distribution."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from ml.models.gumbel_softmax_utils import masks_to_component_type
from collections import defaultdict

print("="*70)
print("Creating Stratified Train/Val Split")
print("="*70)

dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

# Analyze each circuit's component type distribution
circuit_component_profiles = []

for idx in range(len(dataset)):
    sample = dataset[idx]
    graph = sample['graph']

    # Count component types in this circuit
    component_counts = defaultdict(int)

    for edge_idx in range(graph.edge_attr.shape[0]):
        edge_attr = graph.edge_attr[edge_idx]
        masks = edge_attr[3:6]
        comp_type = masks_to_component_type(masks.unsqueeze(0))[0].item()
        component_counts[comp_type] += 1

    # Create a signature for this circuit
    # Use the majority component type as the class label
    majority_type = max(component_counts.items(), key=lambda x: x[1])[0]

    circuit_component_profiles.append({
        'idx': idx,
        'majority_type': majority_type,
        'counts': dict(component_counts)
    })

# Group circuits by majority type
circuits_by_type = defaultdict(list)
for profile in circuit_component_profiles:
    circuits_by_type[profile['majority_type']].append(profile['idx'])

print(f"\nCircuit Distribution by Majority Component Type:")
print("-" * 50)
comp_names = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']
for comp_type in sorted(circuits_by_type.keys()):
    count = len(circuits_by_type[comp_type])
    print(f"  Type {comp_type} ({comp_names[comp_type]:8s}): {count} circuits")

# Stratified split: 80/20 split within each component type
train_indices = []
val_indices = []

np.random.seed(42)

for comp_type, indices in circuits_by_type.items():
    np.random.shuffle(indices)

    n = len(indices)
    n_train = int(0.8 * n)

    train_indices.extend(indices[:n_train])
    val_indices.extend(indices[n_train:])

# Sort indices for consistency
train_indices.sort()
val_indices.sort()

print(f"\n{'='*70}")
print("Stratified Split Results")
print("="*70)
print(f"Train size: {len(train_indices)}")
print(f"Val size: {len(val_indices)}")

# Verify distribution in each split
def analyze_split(indices, name):
    print(f"\n{name} Set Component Distribution:")
    print("-" * 50)

    component_counts = defaultdict(int)
    total_edges = 0

    for idx in indices:
        sample = dataset[idx]
        graph = sample['graph']

        for edge_idx in range(graph.edge_attr.shape[0]):
            edge_attr = graph.edge_attr[edge_idx]
            masks = edge_attr[3:6]
            comp_type = masks_to_component_type(masks.unsqueeze(0))[0].item()
            component_counts[comp_type] += 1
            total_edges += 1

    for comp_type in sorted(component_counts.keys()):
        count = component_counts[comp_type]
        percentage = (count / total_edges) * 100
        print(f"  Type {comp_type} ({comp_names[comp_type]:8s}): {count:4d} edges ({percentage:5.1f}%)")

    return component_counts

train_dist = analyze_split(train_indices, "Train")
val_dist = analyze_split(val_indices, "Val")

# Save indices
output = {
    'train_indices': train_indices,
    'val_indices': val_indices,
    'train_distribution': dict(train_dist),
    'val_distribution': dict(val_dist)
}

torch.save(output, 'stratified_split.pt')
print(f"\n{'='*70}")
print("Saved stratified split to: stratified_split.pt")
print("="*70)
