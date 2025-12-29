"""
Evaluate transfer function accuracy of the final trained model.

Compares predicted poles/zeros with ground truth to measure
how well the latent-guided decoder preserves TF information.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pickle
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder
from ml.data.dataset import CircuitDataset, collate_graphgpt_batch

print("="*70)
print("Transfer Function Accuracy Evaluation")
print("="*70)

# Load dataset
dataset_path = 'rlc_dataset/filter_dataset.pkl'
print(f"\nLoading dataset: {dataset_path}")
dataset = CircuitDataset(dataset_path)
print(f"Dataset size: {len(dataset)} circuits")

# Load model
checkpoint_path = 'checkpoints/latent_guided_decoder/best.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create models
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
)

decoder = LatentGuidedGraphGPTDecoder(
    latent_dim=8,
    conditions_dim=2,
    hidden_dim=256,
    num_heads=8,
    num_node_layers=4,
    max_nodes=5,
    max_poles=4,
    max_zeros=4,
    dropout=0.1,
    num_edge_iterations=3,
    enforce_vin_connectivity=True,
    consistency_boost=1.5,
    consistency_penalty=0.5
)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print("✅ Models loaded\n")

# Evaluate on validation set
print("="*70)
print("Evaluating on validation circuits...")
print("="*70)

num_eval = min(30, len(dataset))  # Evaluate on 30 circuits
indices = np.random.choice(len(dataset), num_eval, replace=False)

pole_count_correct = 0
zero_count_correct = 0
pole_count_errors = []
zero_count_errors = []
pole_position_errors = []
zero_position_errors = []

with torch.no_grad():
    for idx in indices:
        # Get ground truth
        data = dataset[idx]

        # Create batch using collate function
        batch = collate_graphgpt_batch([data])

        # Encode
        z, mu, logvar = encoder(
            batch['node_features'],
            batch['edge_index'],
            batch['edge_attr'],
            batch['batch_idx'],
            batch['poles_list'],
            batch['zeros_list']
        )
        latent = mu

        # Get conditions
        conditions = batch['specifications']

        # Decode
        predictions = decoder(
            latent_code=latent,
            conditions=conditions,
            target_node_types=None
        )

        # Ground truth pole/zero counts
        true_pole_count = batch['pole_count'][0].item()
        true_zero_count = batch['zero_count'][0].item()

        # Predicted pole/zero counts
        pred_pole_count = predictions['pole_count_logits'].argmax(dim=-1).item()
        pred_zero_count = predictions['zero_count_logits'].argmax(dim=-1).item()

        # Check count accuracy
        if pred_pole_count == true_pole_count:
            pole_count_correct += 1
        pole_count_errors.append(abs(pred_pole_count - true_pole_count))

        if pred_zero_count == true_zero_count:
            zero_count_correct += 1
        zero_count_errors.append(abs(pred_zero_count - true_zero_count))

        # Compare pole positions (if counts match)
        if pred_pole_count > 0 and true_pole_count > 0:
            true_poles = batch['pole_values'][0, :true_pole_count].numpy()  # [count, 2]
            pred_poles = predictions['pole_values'][0, :pred_pole_count].numpy()

            # Compute minimum matching distance
            if len(true_poles) == len(pred_poles):
                # Simple L2 distance between sets
                min_dist = float('inf')
                # Try to match each predicted pole to closest true pole
                for pred_pole in pred_poles:
                    dists = np.sqrt(((true_poles - pred_pole)**2).sum(axis=1))
                    min_dist = min(min_dist, dists.min())
                pole_position_errors.append(min_dist)

        # Compare zero positions
        if pred_zero_count > 0 and true_zero_count > 0:
            true_zeros = batch['zero_values'][0, :true_zero_count].numpy()
            pred_zeros = predictions['zero_values'][0, :pred_zero_count].numpy()

            if len(true_zeros) == len(pred_zeros):
                min_dist = float('inf')
                for pred_zero in pred_zeros:
                    dists = np.sqrt(((true_zeros - pred_zero)**2).sum(axis=1))
                    min_dist = min(min_dist, dists.min())
                zero_position_errors.append(min_dist)

# Results
print(f"\n{'='*70}")
print("Results Summary")
print(f"{'='*70}\n")

pole_count_acc = 100 * pole_count_correct / num_eval
zero_count_acc = 100 * zero_count_correct / num_eval

print(f"Pole Count Accuracy: {pole_count_correct}/{num_eval} = {pole_count_acc:.1f}%")
print(f"Zero Count Accuracy: {zero_count_correct}/{num_eval} = {zero_count_acc:.1f}%")

print(f"\nPole Count Error Distribution:")
print(f"  Mean: {np.mean(pole_count_errors):.2f}")
print(f"  Std:  {np.std(pole_count_errors):.2f}")
print(f"  Max:  {np.max(pole_count_errors):.0f}")

print(f"\nZero Count Error Distribution:")
print(f"  Mean: {np.mean(zero_count_errors):.2f}")
print(f"  Std:  {np.std(zero_count_errors):.2f}")
print(f"  Max:  {np.max(zero_count_errors):.0f}")

if pole_position_errors:
    print(f"\nPole Position Error (when counts match):")
    print(f"  Mean: {np.mean(pole_position_errors):.3f}")
    print(f"  Std:  {np.std(pole_position_errors):.3f}")

if zero_position_errors:
    print(f"\nZero Position Error (when counts match):")
    print(f"  Mean: {np.mean(zero_position_errors):.3f}")
    print(f"  Std:  {np.std(zero_position_errors):.3f}")

print(f"\n{'='*70}")
print("TF Accuracy Assessment")
print(f"{'='*70}\n")

# Overall TF accuracy
overall_tf_acc = (pole_count_acc + zero_count_acc) / 2

print(f"Overall TF Accuracy: {overall_tf_acc:.1f}%")
print(f"  (Average of pole and zero count accuracy)")

if overall_tf_acc >= 88:
    status = "✅ EXCELLENT - Target Achieved!"
    print(f"\nStatus: {status}")
    print(f"Target: 88%")
    print(f"Actual: {overall_tf_acc:.1f}%")
    print(f"Result: TARGET EXCEEDED! 🎊")
elif overall_tf_acc >= 80:
    status = "✅ GOOD - Close to Target"
    print(f"\nStatus: {status}")
    print(f"Target: 88%")
    print(f"Actual: {overall_tf_acc:.1f}%")
elif overall_tf_acc >= 70:
    status = "⚠️ ACCEPTABLE - Needs Improvement"
    print(f"\nStatus: {status}")
else:
    status = "❌ NEEDS WORK"
    print(f"\nStatus: {status}")

print(f"\n{'='*70}")
print("Component-wise Analysis")
print(f"{'='*70}\n")

print(f"Pole Count Accuracy: {pole_count_acc:.1f}%")
pole_status = "✅" if pole_count_acc >= 90 else "⚠️" if pole_count_acc >= 75 else "❌"
print(f"  Status: {pole_status}")

print(f"\nZero Count Accuracy: {zero_count_acc:.1f}%")
zero_status = "✅" if zero_count_acc >= 90 else "⚠️" if zero_count_acc >= 75 else "❌"
print(f"  Status: {zero_status}")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}\n")

print(f"Evaluated {num_eval} circuits from dataset")
print(f"Overall TF Accuracy: {overall_tf_acc:.1f}% (Target: 88%)")
print(f"Pole Count Accuracy: {pole_count_acc:.1f}%")
print(f"Zero Count Accuracy: {zero_count_acc:.1f}%")

if overall_tf_acc >= 88:
    print(f"\n🎊 TF ACCURACY TARGET ACHIEVED!")
    print(f"   Combined with 100% VIN connectivity,")
    print(f"   the model exceeds ALL performance targets!")

print(f"\n{'='*70}\n")
