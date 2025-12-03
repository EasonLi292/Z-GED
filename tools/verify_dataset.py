import pickle
import numpy as np

# Load the dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Total circuits: {len(dataset)}")
print(f"\nFilter type distribution:")

# Count by filter type
from collections import Counter
filter_counts = Counter([d['filter_type'] for d in dataset])
for ftype, count in sorted(filter_counts.items()):
    print(f"  {ftype}: {count}")

# Show details of first circuit
print(f"\n{'='*60}")
print("First Circuit Details:")
print(f"{'='*60}")
first = dataset[0]
print(f"ID: {first['id']}")
print(f"Filter Type: {first['filter_type']}")
print(f"Characteristic Frequency: {first['characteristic_frequency']:.2f} Hz")
print(f"\nComponents:")
for comp in first['components']:
    print(f"  {comp['name']}: {comp['type']} = {comp['value']:.6e} ({comp['node1']} -> {comp['node2']})")

print(f"\nFrequency Response:")
print(f"  Frequency points: {len(first['frequency_response']['freqs'])}")
print(f"  Freq range: {first['frequency_response']['freqs'][0]:.2f} Hz to {first['frequency_response']['freqs'][-1]:.2e} Hz")
print(f"  Max magnitude: {np.max(first['frequency_response']['H_magnitude']):.4f}")
print(f"  Min magnitude: {np.min(first['frequency_response']['H_magnitude']):.4f}")

print(f"\nGraph Structure:")
print(f"  Nodes: {len(first['graph_adj']['nodes'])}")
print(f"  Edges: {sum(len(adj) for adj in first['graph_adj']['adjacency'])}")

print(f"\n{'='*60}")
print("Sample of different filter types:")
print(f"{'='*60}")
for ftype in ['low_pass', 'high_pass', 'band_pass']:
    sample = next((d for d in dataset if d['filter_type'] == ftype), None)
    if sample:
        print(f"\n{ftype.upper()}:")
        print(f"  fc = {sample['characteristic_frequency']:.2f} Hz")
        print(f"  Components: {', '.join([c['name'] for c in sample['components']])}")
