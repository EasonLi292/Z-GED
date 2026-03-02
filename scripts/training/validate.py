"""Comprehensive validation of the latent-guided circuit generation model."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.data.dataset import CircuitDataset
from torch.utils.data import DataLoader, Subset
from ml.models.component_utils import masks_to_component_type
from ml.utils.runtime import load_encoder_decoder, make_collate_fn

print("="*70)
print("Circuit Generation Model Validation")
print("="*70)

device = 'cpu'

# Load models
encoder, decoder, checkpoint = load_encoder_decoder(
    checkpoint_path='checkpoints/production/best.pt',
    device=device,
    decoder_overrides={'max_nodes': 10},
)

print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

encoder.eval()
decoder.eval()

# Load stratified validation set
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')
split_data = torch.load('rlc_dataset/stratified_split.pt')
val_indices = split_data['val_indices']
val_dataset = Subset(dataset, val_indices)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=make_collate_fn(),
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

    # Get all ground truth component types from this circuit
    edge_attr = graph.edge_attr

    with torch.no_grad():
        _, mu, _ = encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch
        )

        # Generate from latent (no conditions needed)
        circuit = decoder.generate(mu, verbose=False)

        component_types = circuit['component_types'][0]
        edge_exist = circuit['edge_existence'][0]

        # Check all edges in this circuit
        for edge_idx in range(edge_attr.shape[0]):
            # Edge attr format: [log10(R), log10(C), log10(L)]
            # masks_to_component_type expects: [mask_C, mask_G, mask_L]
            is_R = (edge_attr[edge_idx, 0].abs() > 0.01).float()
            is_C = (edge_attr[edge_idx, 1].abs() > 0.01).float()
            is_L = (edge_attr[edge_idx, 2].abs() > 0.01).float()
            true_masks = torch.stack([is_C, is_R, is_L])  # Reorder to [C, G, L]
            true_type = masks_to_component_type(true_masks.unsqueeze(0))[0].item()

            # Find corresponding edge in generated circuit
            src = graph.edge_index[0, edge_idx].item()
            dst = graph.edge_index[1, edge_idx].item()

            # Check if edge exists in generated circuit
            if edge_exist[max(src, dst), min(src, dst)] > 0.5:
                pred_type = component_types[max(src, dst), min(src, dst)].item()
            else:
                # Edge doesn't exist → predicted as "None"
                pred_type = 0

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
        baseline_acc = {1: 82.35, 2: 62.50, 3: 33.33, 7: 100.0}.get(comp_type, 0)
        delta = acc - baseline_acc if baseline_acc > 0 else 0
        delta_str = f"({delta:+.1f}%)" if baseline_acc > 0 else ""
        print(f"  {comp_names[comp_type]:8s}: {type_correct[comp_type]:3d}/{type_total[comp_type]:3d} = {acc:6.2f}% {delta_str}")

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

# Summary comparison
print(f"\n{'='*70}")
print("Baseline vs Phase 3 Comparison:")
print("="*70)
print(f"{'Component':<10} {'Baseline':<12} {'Phase 3':<12} {'Improvement'}")
print("-" * 60)
baseline = {
    'R': (56, 68, 82.35),
    'C': (20, 32, 62.50),
    'L': (4, 12, 33.33),
    'RCL': (16, 16, 100.0)
}

for comp_name, (base_correct, base_total, base_acc) in baseline.items():
    comp_idx = comp_names.index(comp_name)
    if type_total[comp_idx] > 0:
        phase3_acc = 100 * type_correct[comp_idx] / type_total[comp_idx]
        delta = phase3_acc - base_acc
        print(f"{comp_name:<10} {base_acc:>6.1f}%       {phase3_acc:>6.1f}%       {delta:+.1f}%")

print(f"\nOverall: 75.0% → {100*correct_count/total_count:.2f}% ({100*correct_count/total_count - 75.0:+.1f}%)")
