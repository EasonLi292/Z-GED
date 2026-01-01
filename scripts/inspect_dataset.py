"""Inspect the structure of the existing dataset."""

import pickle

with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    circuits = pickle.load(f)

print(f"Total circuits: {len(circuits)}")
print(f"\nFirst circuit keys: {circuits[0].keys()}")
print(f"\nFirst circuit structure:")
for key, value in circuits[0].items():
    print(f"  {key}: {type(value)}")
    if hasattr(value, 'shape'):
        print(f"         shape: {value.shape}")
