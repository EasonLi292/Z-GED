"""Analyze the actual edge count distribution in training vs validation."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from torch.utils.data import Subset
from collections import Counter

# Load dataset
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
split_data = torch.load('stratified_split.pt')

train_indices = split_data['train_indices']
val_indices = split_data['val_indices']

print("="*80)
print("DATASET DISTRIBUTION ANALYSIS")
print("="*80)

# Analyze training set
train_edge_counts = []
train_node_counts = []

for idx in train_indices:
    sample = dataset[idx]
    graph = sample['graph']

    # Count unique edges (undirected)
    num_directed_edges = graph.edge_attr.shape[0]
    num_undirected_edges = num_directed_edges // 2

    num_nodes = graph.x.shape[0]

    train_edge_counts.append(num_undirected_edges)
    train_node_counts.append(num_nodes)

# Analyze validation set
val_edge_counts = []
val_node_counts = []

for idx in val_indices:
    sample = dataset[idx]
    graph = sample['graph']

    num_directed_edges = graph.edge_attr.shape[0]
    num_undirected_edges = num_directed_edges // 2

    num_nodes = graph.x.shape[0]

    val_edge_counts.append(num_undirected_edges)
    val_node_counts.append(num_nodes)

print("\n" + "="*80)
print("TRAINING SET (96 circuits)")
print("="*80)

print(f"\nEdge Count Distribution:")
train_edge_dist = Counter(train_edge_counts)
for edge_count in sorted(train_edge_dist.keys()):
    count = train_edge_dist[edge_count]
    pct = 100 * count / len(train_edge_counts)
    bar = "█" * int(pct / 2)
    print(f"  {edge_count} edges: {count:2d} circuits ({pct:5.1f}%) {bar}")

print(f"\nStatistics:")
print(f"  Mean: {np.mean(train_edge_counts):.2f} edges")
print(f"  Std:  {np.std(train_edge_counts):.2f}")
print(f"  Min:  {min(train_edge_counts)} edges")
print(f"  Max:  {max(train_edge_counts)} edges")

print(f"\nNode Count Distribution:")
train_node_dist = Counter(train_node_counts)
for node_count in sorted(train_node_dist.keys()):
    count = train_node_dist[node_count]
    pct = 100 * count / len(train_node_counts)
    bar = "█" * int(pct / 2)
    print(f"  {node_count} nodes: {count:2d} circuits ({pct:5.1f}%) {bar}")

print("\n" + "="*80)
print("VALIDATION SET (24 circuits)")
print("="*80)

print(f"\nEdge Count Distribution:")
val_edge_dist = Counter(val_edge_counts)
for edge_count in sorted(val_edge_dist.keys()):
    count = val_edge_dist[edge_count]
    pct = 100 * count / len(val_edge_counts)
    bar = "█" * int(pct / 2)
    print(f"  {edge_count} edges: {count:2d} circuits ({pct:5.1f}%) {bar}")

print(f"\nStatistics:")
print(f"  Mean: {np.mean(val_edge_counts):.2f} edges")
print(f"  Std:  {np.std(val_edge_counts):.2f}")
print(f"  Min:  {min(val_edge_counts)} edges")
print(f"  Max:  {max(val_edge_counts)} edges")

print(f"\nNode Count Distribution:")
val_node_dist = Counter(val_node_counts)
for node_count in sorted(val_node_dist.keys()):
    count = val_node_dist[node_count]
    pct = 100 * count / len(val_node_counts)
    bar = "█" * int(pct / 2)
    print(f"  {node_count} nodes: {count:2d} circuits ({pct:5.1f}%) {bar}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\n{'Edge Count':<12} {'Train %':<12} {'Val %':<12} {'Delta'}")
print("-" * 60)

all_edge_counts = sorted(set(list(train_edge_dist.keys()) + list(val_edge_dist.keys())))
for edge_count in all_edge_counts:
    train_pct = 100 * train_edge_dist.get(edge_count, 0) / len(train_edge_counts)
    val_pct = 100 * val_edge_dist.get(edge_count, 0) / len(val_edge_counts)
    delta = val_pct - train_pct

    delta_str = f"{delta:+.1f}%"
    if abs(delta) > 5:
        delta_str += " ⚠️"

    print(f"{edge_count} edges    {train_pct:6.1f}%      {val_pct:6.1f}%      {delta_str}")

print("\n" + "="*80)
print("MODEL GENERATION STATISTICS")
print("="*80)

# What did the model actually generate?
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

def collate_circuit_batch(batch_list):
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)
    return {'graph': batched_graph, 'poles': poles, 'zeros': zeros}

device = 'cpu'

encoder = HierarchicalEncoder(
    node_feature_dim=4, edge_feature_dim=7, gnn_hidden_dim=64,
    gnn_num_layers=3, latent_dim=8, topo_latent_dim=2,
    values_latent_dim=2, pz_latent_dim=4, dropout=0.1
).to(device)

decoder = LatentGuidedGraphGPTDecoder(
    latent_dim=8, conditions_dim=2, hidden_dim=256,
    num_heads=8, num_node_layers=4, max_nodes=5,
    enforce_vin_connectivity=True
).to(device)

checkpoint = torch.load('checkpoints/phase3_joint_prediction/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

val_dataset = Subset(dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_circuit_batch)

gen_edge_counts = []

for batch in val_loader:
    graph = batch['graph'].to(device)
    poles_list = batch['poles']
    zeros_list = batch['zeros']

    with torch.no_grad():
        z, mu, logvar = encoder(
            graph.x, graph.edge_index, graph.edge_attr,
            graph.batch, poles_list, zeros_list
        )

        conditions = torch.randn(1, 2, device=device)
        circuit = decoder.generate(mu, conditions, verbose=False)

        edge_exist = circuit['edge_existence'][0]
        num_edges = (edge_exist > 0.5).sum().item() // 2
        gen_edge_counts.append(num_edges)

print(f"\nGenerated Edge Count Distribution:")
gen_edge_dist = Counter(gen_edge_counts)
for edge_count in sorted(gen_edge_dist.keys()):
    count = gen_edge_dist[edge_count]
    pct = 100 * count / len(gen_edge_counts)
    bar = "█" * int(pct / 2)
    print(f"  {edge_count} edges: {count:2d} circuits ({pct:5.1f}%) {bar}")

print(f"\nStatistics:")
print(f"  Mean: {np.mean(gen_edge_counts):.2f} edges")
print(f"  Target mean: {np.mean(val_edge_counts):.2f} edges")
print(f"  Difference: {np.mean(gen_edge_counts) - np.mean(val_edge_counts):.2f} edges")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Check if model generates what it was trained on
print("\nDoes the model generate what it was trained on?")
print("\nModel generates mostly 2-edge circuits")
print(f"  Training set has {train_edge_dist[2]} circuits with 2 edges ({100*train_edge_dist.get(2,0)/len(train_edge_counts):.1f}%)")
print(f"  Model generates {gen_edge_dist.get(2,0)} circuits with 2 edges ({100*gen_edge_dist.get(2,0)/len(gen_edge_counts):.1f}%)")

print("\nFor complex circuits (4+ edges):")
train_complex = sum([train_edge_dist.get(i, 0) for i in range(4, 10)])
val_complex = sum([val_edge_dist.get(i, 0) for i in range(4, 10)])
gen_complex = sum([gen_edge_dist.get(i, 0) for i in range(4, 10)])

print(f"  Training set: {train_complex}/{len(train_edge_counts)} = {100*train_complex/len(train_edge_counts):.1f}%")
print(f"  Validation set: {val_complex}/{len(val_edge_counts)} = {100*val_complex/len(val_edge_counts):.1f}%")
print(f"  Model generates: {gen_complex}/{len(gen_edge_counts)} = {100*gen_complex/len(gen_edge_counts):.1f}%")

if gen_complex < val_complex:
    print(f"\n  ❌ Model generates {val_complex - gen_complex} fewer complex circuits than expected!")
