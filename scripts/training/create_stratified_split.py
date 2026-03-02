"""Create stratified train/val split based on filter type distribution."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pickle
import numpy as np
from ml.data.dataset import CircuitDataset
from collections import defaultdict

print("="*70)
print("Creating Stratified Train/Val Split")
print("="*70)

# Load raw dataset for filter_type labels
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    raw_data = pickle.load(f)

dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

# Group circuits by filter type
circuits_by_type = defaultdict(list)
for idx, circuit in enumerate(raw_data):
    circuits_by_type[circuit['filter_type']].append(idx)

print(f"\nCircuit Distribution by Filter Type:")
print("-" * 50)
for ft in sorted(circuits_by_type.keys()):
    count = len(circuits_by_type[ft])
    print(f"  {ft:20s}: {count} circuits")

# Stratified split: 80/20 split within each filter type
train_indices = []
val_indices = []

np.random.seed(42)

for ft, indices in circuits_by_type.items():
    indices = list(indices)
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
    print(f"\n{name} Set Filter Type Distribution:")
    print("-" * 50)

    type_counts = defaultdict(int)
    for idx in indices:
        ft = raw_data[idx]['filter_type']
        type_counts[ft] += 1

    for ft in sorted(type_counts.keys()):
        count = type_counts[ft]
        print(f"  {ft:20s}: {count:4d} circuits")

    return dict(type_counts)

train_dist = analyze_split(train_indices, "Train")
val_dist = analyze_split(val_indices, "Val")

# Save indices
output = {
    'train_indices': train_indices,
    'val_indices': val_indices,
    'train_distribution': train_dist,
    'val_distribution': val_dist
}

torch.save(output, 'rlc_dataset/stratified_split.pt')
print(f"\n{'='*70}")
print("Saved stratified split to: rlc_dataset/stratified_split.pt")
print("="*70)
