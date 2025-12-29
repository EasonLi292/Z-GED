"""Comprehensive validation of Gumbel-Softmax model."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from ml.data.dataset import CircuitDataset
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from ml.models.gumbel_softmax_utils import masks_to_component_type

def collate_circuit_batch(batch_list):
    graphs = [item['graph'] for item in batch_list]
    poles = [item['poles'] for item in batch_list]
    zeros = [item['zeros'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)
    return {'graph': batched_graph, 'poles': poles, 'zeros': zeros}

print("="*70)
print("Comprehensive Gumbel-Softmax Validation")
print("="*70)

device = 'cpu'

# Load models
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

decoder = LatentGuidedGraphGPTDecoder(
    latent_dim=8,
    conditions_dim=2,
    hidden_dim=256,
    num_heads=8,
    num_node_layers=4,
    max_nodes=5,
    enforce_vin_connectivity=True
).to(device)

# Load best checkpoint
checkpoint = torch.load('checkpoints/gumbel_softmax/best.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

encoder.eval()
decoder.eval()

# Load stratified validation set
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
split_data = torch.load('stratified_split.pt')
val_indices = split_data['val_indices']
val_dataset = Subset(dataset, val_indices)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_circuit_batch
)

print(f"\n{'='*70}")
print("Validating on Full Validation Set")
print(f"{'='*70}\n")

comp_names = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']

# Confusion matrix
confusion_matrix = np.zeros((8, 8), dtype=int)
correct_count = 0
total_count = 0

# Store per-type accuracy
type_correct = {i: 0 for i in range(8)}
type_total = {i: 0 for i in range(8)}

for i, batch in enumerate(val_loader):
    graph = batch['graph'].to(device)
    poles_list = batch['poles']
    zeros_list = batch['zeros']

    # Get all ground truth component types from this circuit
    edge_attr = graph.edge_attr

    with torch.no_grad():
        z, mu, logvar = encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            poles_list,
            zeros_list
        )

        # Generate circuit
        conditions = torch.randn(1, 2, device=device)
        circuit = decoder.generate(mu, conditions, verbose=False)

        edge_vals = circuit['edge_values'][0]
        edge_exist = circuit['edge_existence'][0]

        # Check all edges in this circuit
        for edge_idx in range(edge_attr.shape[0]):
            true_masks = edge_attr[edge_idx, 3:6]
            true_type = masks_to_component_type(true_masks.unsqueeze(0))[0].item()

            # Find corresponding edge in generated circuit
            src = graph.edge_index[0, edge_idx].item()
            dst = graph.edge_index[1, edge_idx].item()

            pred_masks = edge_vals[src, dst, 3:6]
            pred_type = masks_to_component_type(pred_masks.unsqueeze(0))[0].item()

            # Update confusion matrix
            confusion_matrix[true_type, pred_type] += 1
            type_total[true_type] += 1
            total_count += 1

            if pred_type == true_type:
                correct_count += 1
                type_correct[true_type] += 1

# Print overall accuracy
print(f"Overall Component Type Accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.2f}%\n")

# Print per-type accuracy
print("Per-Type Accuracy:")
print("-" * 50)
for comp_type in sorted(type_total.keys()):
    if type_total[comp_type] > 0:
        acc = 100 * type_correct[comp_type] / type_total[comp_type]
        print(f"  {comp_names[comp_type]:8s}: {type_correct[comp_type]:3d}/{type_total[comp_type]:3d} = {acc:6.2f}%")

# Print confusion matrix
print(f"\n{'='*70}")
print("Confusion Matrix (rows=true, cols=predicted):")
print("="*70)

# Header
header = "True\\Pred  "
for name in comp_names:
    if confusion_matrix[:, comp_names.index(name)].sum() > 0:
        header += f"{name:>6s} "
print(header)
print("-" * 70)

# Rows
for i, name_i in enumerate(comp_names):
    if confusion_matrix[i, :].sum() > 0:
        row = f"{name_i:8s}  "
        for j, name_j in enumerate(comp_names):
            if confusion_matrix[:, j].sum() > 0:
                val = confusion_matrix[i, j]
                if val > 0:
                    if i == j:
                        row += f"{val:>6d} "  # Correct predictions
                    else:
                        row += f"({val:>4d}) "  # Wrong predictions
                else:
                    row += "     . "
        print(row)

print(f"\n{'='*70}")
print("✅ Validation Complete!")
print("="*70)
