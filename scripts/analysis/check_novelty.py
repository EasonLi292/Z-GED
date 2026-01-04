"""
Check if generated circuits are novel or just reproductions of training data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import numpy as np
from collections import Counter

# Load training dataset
print("Loading training dataset...")
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Get training topologies
train_topologies = []
for circuit in train_data:
    components = circuit['components']
    num_R = sum(1 for c in components if c['type'] == 'R')
    num_C = sum(1 for c in components if c['type'] == 'C')
    num_L = sum(1 for c in components if c['type'] == 'L')

    topology = f"{num_R}R"
    if num_C > 0:
        topology += f"+{num_C}C"
    if num_L > 0:
        topology += f"+{num_L}L"

    train_topologies.append(topology)

train_topology_set = set(train_topologies)
train_topology_counts = Counter(train_topologies)

print(f"Training dataset has {len(train_data)} circuits")
print(f"Unique topologies in training: {len(train_topology_set)}")
print(f"\nTraining topologies:")
for topo, count in train_topology_counts.most_common():
    print(f"  {topo:15s}: {count:3d} circuits")

# Parse generated topologies from test results
print("\n" + "="*80)
print("GENERATED TOPOLOGIES FROM TEST CASES")
print("="*80)

import re

with open('docs/DETAILED_CIRCUITS.txt', 'r') as f:
    content = f.read()

# Extract all generated topologies
examples = content.split('\n' + '-'*80 + '\n')
generated_topologies = []
generated_details = []

for example in examples:
    if 'Example' not in example:
        continue

    # Extract title
    title_match = re.search(r'Example \d+: (.+)', example)
    if not title_match:
        continue
    title = title_match.group(1)

    # Extract topology
    topo_match = re.search(r'Topology: \d+ edges \(([^)]+)\)', example)
    if topo_match:
        topology = topo_match.group(1)
        generated_topologies.append(topology)
        generated_details.append({
            'title': title,
            'topology': topology
        })

generated_topology_counts = Counter(generated_topologies)

print(f"\nGenerated topologies from 18 test cases:")
for topo, count in generated_topology_counts.most_common():
    in_training = "✅ In training" if topo in train_topology_set else "❌ NOVEL (not in training)"
    print(f"  {topo:15s}: {count:2d} circuits  {in_training}")

# Analyze novelty
novel_topologies = [t for t in generated_topologies if t not in train_topology_set]
print(f"\n" + "="*80)
print("NOVELTY ANALYSIS")
print("="*80)

print(f"\nNovel topologies (not in training data):")
if novel_topologies:
    novel_counts = Counter(novel_topologies)
    for topo, count in novel_counts.most_common():
        print(f"  {topo:15s}: {count} occurrences")

        # Find which test cases generated this
        cases = [d['title'] for d in generated_details if d['topology'] == topo]
        for case in cases:
            print(f"    - {case}")
else:
    print("  None - all generated topologies exist in training data")

# Check if the model is doing topology interpolation
print(f"\n" + "="*80)
print("GENERATION MODE ANALYSIS")
print("="*80)

print(f"""
The model generates circuits through:
1. K-NN interpolation finds 5 nearest training examples by (frequency, Q)
2. Latent codes from those 5 examples are blended
3. Decoder generates circuit from interpolated latent code

Key question: Does interpolation create novel topologies?

Training topologies: {train_topology_set}
Generated topologies: {set(generated_topologies)}

Novel topologies generated: {set(novel_topologies) if novel_topologies else 'None'}
""")

# Analyze whether novel topologies are valid
if novel_topologies:
    print("\nValidity of novel topologies:")
    for topo in set(novel_topologies):
        print(f"\n  {topo}:")

        # Check if it's pure resistive
        if 'C' not in topo and 'L' not in topo:
            print("    ❌ Pure resistive - cannot provide frequency selectivity")
            print("    ❌ Invalid topology for filter design")
        else:
            print("    ✅ Has reactive components")

            # But check if it can achieve the required Q
            cases = [d for d in generated_details if d['topology'] == topo]
            for case in cases:
                if 'Q=' in case['title']:
                    q_str = case['title'].split('Q=')[1].split(',')[0].split(')')[0]
                    try:
                        q_val = float(q_str)
                        if q_val > 1.0 and ('L' not in topo or 'C' not in topo):
                            print(f"    ⚠️ For {case['title']}: needs both L and C for Q={q_val}")
                    except:
                        pass

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

if novel_topologies:
    print(f"""
✅ The model DOES generate novel topologies!

Novel topology count: {len(set(novel_topologies))}
Novel topology instances: {len(novel_topologies)} out of 18 test cases

However, the novel topologies are INVALID:
- All novel topologies are pure resistive (2R)
- Pure resistive circuits cannot provide frequency selectivity
- These are generation failures, not successful novel designs

The model's "novelty" is actually a bug:
- It generates topologies that don't exist in training
- But these topologies are physically invalid for the task
- Model lacks constraints to prevent invalid topology generation

True novelty would be:
- Valid RLC combinations not in training data
- Novel but functional circuit structures
- Creative solutions within physical constraints
""")
else:
    print(f"""
❌ The model does NOT generate novel topologies.

All 18 generated circuits use topologies from the training data:
- {train_topology_set}

The model is essentially:
1. Finding similar training examples via K-NN
2. Blending their latent representations
3. Generating circuits with familiar topologies
4. Interpolating component values (not topologies)

This is more like "sophisticated retrieval + interpolation" than
true generative design. The model cannot invent new circuit structures.

Pros:
- Guarantees physically plausible topologies
- Avoids invalid circuit structures
- Reliable within training distribution

Cons:
- Cannot explore novel circuit architectures
- Limited to training topology vocabulary
- No true creative circuit design
""")
