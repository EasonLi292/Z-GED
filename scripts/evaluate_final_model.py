"""
Comprehensive evaluation of the final trained latent-guided model.

Tests:
1. VIN/VOUT connectivity rate
2. Circuit generation quality
3. Edge distribution
4. Node type distribution
5. Consistency scores
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from ml.models.encoder import HierarchicalEncoder
from ml.models.graphgpt_decoder_latent_guided import LatentGuidedGraphGPTDecoder

print("="*70)
print("Final Model Evaluation - Latent-Guided GraphGPT")
print("="*70)

# Load final checkpoint
checkpoint_path = 'checkpoints/latent_guided_decoder/best.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Checkpoint info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best Val Loss: {checkpoint['val_loss']:.4f}")

# Create models
print("\nCreating models...")
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

# Test generation
num_tests = 50
print(f"{'='*70}")
print(f"Generating {num_tests} test circuits...")
print(f"{'='*70}\n")

# Metrics
vin_connected_count = 0
vout_connected_count = 0
both_connected_count = 0
edge_counts = []
node_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # GND, VIN, VOUT, INTERNAL, MASK
consistency_scores = []

with torch.no_grad():
    for i in range(num_tests):
        # Random latent and conditions
        latent = torch.randn(1, 8)
        conditions = torch.randn(1, 2)

        # Generate circuit
        circuit = decoder.generate(latent, conditions, verbose=False)

        # Check VIN (node 1) connectivity
        vin_id = 1
        vin_edges = circuit['edge_existence'][0, vin_id, :]
        vin_connected = vin_edges.sum() > 0

        # Check VOUT (node 2) connectivity
        vout_id = 2
        vout_edges = circuit['edge_existence'][0, vout_id, :]
        vout_connected = vout_edges.sum() > 0

        # Count edges (upper triangle only to avoid double counting)
        edge_matrix = circuit['edge_existence'][0]
        edges = torch.triu(edge_matrix, diagonal=1).sum().item()
        edge_counts.append(edges)

        # Node types
        node_types = circuit['node_types'][0].argmax(dim=-1)
        if node_types.dim() == 0:
            node_type_counts[node_types.item()] += 1
        else:
            for node_type in node_types:
                node_type_counts[node_type.item()] += 1

        # Consistency scores (if available)
        consistency = circuit.get('consistency_scores', None)
        if consistency is not None:
            avg_cons = torch.sigmoid(consistency[0]).mean().item()
            consistency_scores.append(avg_cons)

        # Track connectivity
        if vin_connected:
            vin_connected_count += 1
        if vout_connected:
            vout_connected_count += 1
        if vin_connected and vout_connected:
            both_connected_count += 1

        # Print every 10th circuit
        if (i + 1) % 10 == 0:
            status = "✅" if (vin_connected and vout_connected) else "⚠️"
            print(f"  Progress: {i+1}/{num_tests} circuits generated... {status}")

print(f"\n{'='*70}")
print("Connectivity Analysis")
print(f"{'='*70}\n")

vin_rate = 100 * vin_connected_count / num_tests
vout_rate = 100 * vout_connected_count / num_tests
both_rate = 100 * both_connected_count / num_tests

print(f"VIN Connectivity:   {vin_connected_count}/{num_tests} = {vin_rate:.1f}%")
print(f"VOUT Connectivity:  {vout_connected_count}/{num_tests} = {vout_rate:.1f}%")
print(f"Both Connected:     {both_connected_count}/{num_tests} = {both_rate:.1f}%")

# Status indicators
vin_status = "✅ EXCELLENT" if vin_rate >= 98 else "⚠️ GOOD" if vin_rate >= 90 else "❌ NEEDS WORK"
print(f"\n🎯 VIN Connectivity Status: {vin_status}")
print(f"   Target: 98%")
print(f"   Actual: {vin_rate:.1f}%")
if vin_rate >= 98:
    print(f"   Result: TARGET ACHIEVED! 🎊")

print(f"\n{'='*70}")
print("Edge Statistics")
print(f"{'='*70}\n")

avg_edges = np.mean(edge_counts)
std_edges = np.std(edge_counts)
min_edges = np.min(edge_counts)
max_edges = np.max(edge_counts)

print(f"Average Edges per Circuit: {avg_edges:.2f} ± {std_edges:.2f}")
print(f"Min Edges: {min_edges:.0f}")
print(f"Max Edges: {max_edges:.0f}")
print(f"Edge Distribution: {edge_counts[:10]}...")

print(f"\n{'='*70}")
print("Node Type Distribution")
print(f"{'='*70}\n")

node_type_names = {0: "GND", 1: "VIN", 2: "VOUT", 3: "INTERNAL", 4: "MASK"}
total_nodes = sum(node_type_counts.values())

for node_type, count in node_type_counts.items():
    pct = 100 * count / total_nodes if total_nodes > 0 else 0
    name = node_type_names[node_type]
    print(f"{name:10s}: {count:4d} ({pct:5.1f}%)")

print(f"\n{'='*70}")
print("Consistency Scores")
print(f"{'='*70}\n")

if consistency_scores:
    avg_cons = np.mean(consistency_scores)
    std_cons = np.std(consistency_scores)
    min_cons = np.min(consistency_scores)
    max_cons = np.max(consistency_scores)

    print(f"Average Consistency: {avg_cons:.3f} ± {std_cons:.3f}")
    print(f"Min Consistency: {min_cons:.3f}")
    print(f"Max Consistency: {max_cons:.3f}")

    cons_status = "✅ EXCELLENT" if avg_cons >= 0.7 else "⚠️ GOOD" if avg_cons >= 0.6 else "📈 IMPROVING"
    print(f"\n🎯 Consistency Status: {cons_status}")
    print(f"   Target: >0.7")
    print(f"   Actual: {avg_cons:.3f}")
else:
    print("No consistency scores available")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}\n")

print(f"✅ Circuits Generated: {num_tests}/{num_tests} (100%)")
print(f"✅ VIN Connectivity: {vin_rate:.1f}% (Target: 98%)")
print(f"✅ VOUT Connectivity: {vout_rate:.1f}%")
print(f"✅ Average Edges: {avg_edges:.1f}")

if vin_rate >= 98 and vout_rate >= 98:
    print(f"\n🎊 CONNECTIVITY TARGET ACHIEVED!")
    print(f"   The latent-guided approach has successfully solved")
    print(f"   the VIN/VOUT connectivity problem!")

print(f"\n{'='*70}")
print("Next Steps")
print(f"{'='*70}\n")

print("1. ✅ Connectivity evaluation complete")
print("2. 📋 Run TF accuracy evaluation (compare poles/zeros)")
print("3. 📋 Run component value analysis (R, L, C ranges)")
print("4. 📋 Generate sample circuits for visualization")
print("5. 📋 Create final comprehensive report")

print(f"\n{'='*70}\n")
