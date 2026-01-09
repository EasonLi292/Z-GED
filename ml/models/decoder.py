"""
Latent-Guided GraphGPT Decoder.

This is the integrated version that replaces standard GraphGPTDecoder
with latent-guided edge generation.

Key differences from original GraphGPTDecoder:
1. Uses LatentGuidedEdgeDecoder instead of AutoregressiveEdgeDecoder
2. Implements iterative edge generation with TF consistency checking
3. Smart VIN connectivity enforcement based on TF guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from ml.models.node_decoder import AutoregressiveNodeDecoder
from ml.models.decoder_components import (
    LatentDecomposer,
    LatentGuidedEdgeDecoder
)


class LatentGuidedGraphGPTDecoder(nn.Module):
    """
    GraphGPT decoder with latent-guided edge generation.

    Architecture:
    1. Context encoder: Project (latent + specs) to hidden_dim
    2. Autoregressive node generation
    3. LATENT-GUIDED edge generation:
       - Cross-attention to latent components (topology, values, TF)
       - TF consistency scoring (guides which edges help achieve target TF)
       - Iterative refinement
       - Smart VIN connectivity enforcement
    4. Transfer function evaluation via SPICE simulation (not predicted)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_node_layers: int = 4,
        max_nodes: int = 50,  # CHANGED: Safety limit for generation, not hard constraint
        max_training_nodes: int = 10,  # Max nodes expected during training (for efficient batching)
        edge_feature_dim: int = 7,
        dropout: float = 0.1,
        # Latent-guided parameters
        num_edge_iterations: int = 3,
        consistency_boost: float = 1.5,
        consistency_penalty: float = 0.5,
        enforce_vin_connectivity: bool = True
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.conditions_dim = conditions_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes  # Safety limit
        self.max_training_nodes = max_training_nodes  # Typical training max
        self.edge_feature_dim = edge_feature_dim

        # NEW: Latent-guided parameters
        self.num_edge_iterations = num_edge_iterations
        self.consistency_boost = consistency_boost
        self.consistency_penalty = consistency_penalty
        self.enforce_vin_connectivity = enforce_vin_connectivity

        # NEW: Topo latent dropout (Section 2 of Generation Plan)
        # During training, randomly drop z_topo to prevent over-reliance on topology latent
        self.topo_dropout_rate = 0.5  # 50% of batches drop topo latent

        # Node type constants
        self.NODE_TYPES = {
            'GND': 0,
            'VIN': 1,
            'VOUT': 2,
            'INTERNAL': 3,
            'MASK': 4
        }

        # Fixed generation order
        self.node_order = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'INTERNAL']

        # ==================================================================
        # Context Encoder (with Layer Normalization for stability)
        # ==================================================================

        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim + conditions_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # ADD: Layer normalization
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)   # ADD: Layer normalization
        )

        # ==================================================================
        # NEW: Latent Decomposer
        # ==================================================================

        self.latent_decomposer = LatentDecomposer(latent_dim=latent_dim)

        # ==================================================================
        # NEW: Direct Node Count Predictor
        # ==================================================================
        # This head directly predicts the number of nodes (3, 4, or 5)
        # from the topology latent. During generation, we use this prediction
        # to determine when to stop, avoiding the train/test mismatch issue.
        # Input: topology latent (2D) + conditions (2D) = 4D
        # Output: 3 classes (3-node, 4-node, 5-node)
        self.node_count_predictor = nn.Sequential(
            nn.Linear(2 + conditions_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 3)  # 3 classes: 3, 4, 5 nodes
        )

        # ==================================================================
        # Node Decoder (same as before)
        # ==================================================================

        self.node_decoder = AutoregressiveNodeDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_node_layers,
            num_node_types=5,
            max_position_embeddings=max_nodes,  # Can generate up to max_nodes
            dropout=dropout,
            use_stopping_criterion=True,  # Enable learned stopping (EOF)
            conditions_dim=conditions_dim,  # For behavior-centric stopping
            topo_latent_dim=2  # CRITICAL: For topology-based stopping decision!
        )

        # ==================================================================
        # NEW: Latent-Guided Edge Decoder
        # ==================================================================

        self.edge_decoder = LatentGuidedEdgeDecoder(
            hidden_dim=hidden_dim,
            topo_latent_dim=2,
            values_latent_dim=2,
            tf_latent_dim=4,
            conditions_dim=conditions_dim,  # NEW: Pass conditions_dim
            edge_feature_dim=edge_feature_dim,
            num_attention_heads=4,
            dropout=dropout
        )

        # ==================================================================
        # Pole/Zero prediction removed - using SPICE simulation instead
        # ==================================================================
        # TF latent and consistency scoring are kept for edge guidance,
        # but actual TF evaluation is done via SPICE simulation

    def forward(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        target_node_types: Optional[torch.Tensor] = None,
        target_edges: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing (for training).

        Args:
            latent_code: [batch, latent_dim]
            conditions: [batch, conditions_dim]
            target_node_types: [batch, num_nodes] - VARIABLE LENGTH per batch
            target_edges: [batch, num_nodes, num_nodes] - VARIABLE LENGTH

        Returns:
            Dictionary with predictions
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device

        # VARIABLE LENGTH: Get actual max nodes in this batch
        num_nodes = target_node_types.shape[1] if target_node_types is not None else self.max_training_nodes

        # Encode context
        context_input = torch.cat([latent_code, conditions], dim=-1)
        context = self.context_encoder(context_input)

        # Decompose latent
        latent_topo, latent_values, latent_tf = self.latent_decomposer(latent_code)

        # NEW: Topo latent dropout during training (Section 2 of Generation Plan)
        # Randomly zero-out z_topo to prevent decoder from over-relying on topology latent
        # This forces the model to learn stopping based on TF complexity, not memorized topology
        if self.training and torch.rand(1).item() < self.topo_dropout_rate:
            latent_topo = torch.zeros_like(latent_topo)

        # ==================================================================
        # NEW: Direct Node Count Prediction
        # ==================================================================
        # Predict number of nodes directly from topology latent + conditions
        node_count_input = torch.cat([latent_topo, conditions], dim=-1)
        node_count_logits = self.node_count_predictor(node_count_input)  # [batch, 3]

        # ==================================================================
        # 1. Node Generation (with teacher forcing) - VARIABLE LENGTH
        # ==================================================================

        node_embeddings = []
        node_logits_list = []
        stop_logits_list = []

        # CHANGED: Loop over actual number of nodes in batch, not fixed max_nodes
        for i in range(num_nodes):
            teacher_type = target_node_types[:, i] if target_node_types is not None else None

            node_logits, node_embed, stop_logit = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=teacher_type,
                conditions=conditions,
                latent_topo=latent_topo  # CRITICAL: Pass topology latent for stopping decision!
            )

            node_logits_list.append(node_logits)
            stop_logits_list.append(stop_logit)
            node_embeddings.append(node_embed)

        node_logits = torch.stack(node_logits_list, dim=1)  # [batch, num_nodes, num_types]
        stop_logits = torch.stack(stop_logits_list, dim=1) if stop_logits_list[0] is not None else None  # [batch, num_nodes]

        # ==================================================================
        # 2. LATENT-GUIDED Edge Generation with JOINT Edge-Component Prediction (Phase 3)
        # ==================================================================

        # VARIABLE LENGTH: Use actual num_nodes, not max_nodes
        edge_component_logits = torch.zeros(batch_size, num_nodes, num_nodes, 8, device=device)
        component_values = torch.zeros(batch_size, num_nodes, num_nodes, 3, device=device)
        is_parallel_logits = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        consistency_scores = torch.zeros(batch_size, num_nodes, num_nodes, device=device)

        for i in range(num_nodes):
            for j in range(i):  # Lower triangle
                # LATENT-GUIDED edge prediction with JOINT edge-component prediction
                (edge_comp_logits, comp_values, is_par_logit, consistency, _) = self.edge_decoder(
                    node_embeddings[i],
                    node_embeddings[j],
                    latent_topo,
                    latent_values,
                    latent_tf,
                    conditions  # NEW: Pass target specifications
                )

                edge_component_logits[:, i, j, :] = edge_comp_logits
                component_values[:, i, j, :] = comp_values
                is_parallel_logits[:, i, j] = is_par_logit
                consistency_scores[:, i, j] = consistency

                # Symmetric (undirected graph)
                edge_component_logits[:, j, i, :] = edge_comp_logits
                component_values[:, j, i, :] = comp_values
                is_parallel_logits[:, j, i] = is_par_logit
                consistency_scores[:, j, i] = consistency

        # ==================================================================
        # 3. Return Joint Edge-Component Predictions (Phase 3)
        # ==================================================================

        return {
            'node_types': node_logits,
            'stop_logits': stop_logits,  # Stopping criterion logits [batch, max_nodes]
            'node_count_logits': node_count_logits,  # NEW: Direct node count prediction [batch, 3]
            'edge_component_logits': edge_component_logits,  # PHASE 3: Joint prediction (class 0=no edge, 1-7=component)
            'component_values': component_values,             # Continuous values (used when edge exists)
            'is_parallel_logits': is_parallel_logits,         # Is parallel logit
            'consistency_scores': consistency_scores,
        }

    def generate(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        edge_threshold: float = 0.5,
        stop_threshold: float = 0.5,  # Threshold for stopping criterion
        min_internal_nodes: int = 0,  # NEW: Minimum internal nodes before stopping allowed
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuit with variable-length autoregressive generation.

        VARIABLE LENGTH: Generation continues until stopping criterion (EOF-like)
        or safety limit (max_nodes) is reached. No fixed topology constraint!

        Args:
            latent_code: [batch, latent_dim]
            conditions: [batch, conditions_dim]
            edge_threshold: Threshold for edge existence
            stop_threshold: Threshold for stopping generation (learned EOF)
            min_internal_nodes: Minimum internal nodes required before stop allowed (0=any, 1=at least one)
            verbose: Print generation details

        Returns:
            Generated circuit with VARIABLE number of nodes
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device

        if verbose and batch_size == 1:
            print(f"\n{'='*70}")
            print("Variable-Length Circuit Generation (EOF-style)")
            print(f"{'='*70}")

        # Encode context
        context_input = torch.cat([latent_code, conditions], dim=-1)
        context = self.context_encoder(context_input)

        # Decompose latent
        latent_topo, latent_values, latent_tf = self.latent_decomposer(latent_code)

        # ==================================================================
        # NEW: Direct Node Count Prediction (replaces learned EOF stopping)
        # ==================================================================
        # Predict node count directly from topology latent - avoids train/test mismatch!
        node_count_input = torch.cat([latent_topo, conditions], dim=-1)
        node_count_logits = self.node_count_predictor(node_count_input)  # [batch, 3]
        predicted_count_idx = torch.argmax(node_count_logits, dim=-1)  # [batch]
        predicted_node_count = predicted_count_idx + 3  # Class 0=3, 1=4, 2=5 nodes

        if verbose and batch_size == 1:
            print(f"\nLatent decomposition:")
            print(f"  Topology: {latent_topo[0].detach().cpu().numpy()}")
            print(f"  Values:   {latent_values[0].detach().cpu().numpy()}")
            print(f"  TF:       {latent_tf[0].detach().cpu().numpy()}")
            node_count_probs = F.softmax(node_count_logits, dim=-1)
            print(f"\nNode count prediction:")
            print(f"  Logits: {node_count_logits[0].detach().cpu().numpy()}")
            print(f"  Probs:  3-node={node_count_probs[0, 0].item():.3f}, 4-node={node_count_probs[0, 1].item():.3f}, 5-node={node_count_probs[0, 2].item():.3f}")
            print(f"  → Predicted: {predicted_node_count[0].item()} nodes")

        # ==================================================================
        # 1. Generate Nodes (FIXED LENGTH based on predicted count)
        # ==================================================================

        node_embeddings = []
        predicted_node_types = []
        target_nodes = predicted_node_count[0].item()  # Use predicted count (for batch size 1)

        if verbose and batch_size == 1:
            print(f"\nGenerating {target_nodes} nodes...")

        for i in range(target_nodes):
            node_logits, node_embed, stop_logit = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=None,
                conditions=conditions,
                latent_topo=latent_topo
            )

            # Generate node type
            if i < 3:
                # Required nodes: GND, VIN, VOUT
                node_type = torch.tensor(
                    [self.NODE_TYPES[self.node_order[i]]] * batch_size,
                    device=device
                )
            else:
                # Sample from distribution
                node_probs = F.softmax(node_logits, dim=-1)
                node_type = torch.argmax(node_probs, dim=-1)

            predicted_node_types.append(node_type)

            # Create embedding for next iteration
            node_embed = self.node_decoder.node_type_embedding(node_type)
            pos_embed = self.node_decoder.position_embedding(
                torch.tensor([i], device=device)
            ).expand(batch_size, -1)
            node_embed = node_embed + pos_embed

            node_embeddings.append(node_embed)

            if verbose and batch_size == 1 and i < 10:  # Only show first 10
                node_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']
                print(f"  Position {i}: {node_names[node_type[0].item()]}")

        # Stack generated nodes (variable length!)
        node_types = torch.stack(predicted_node_types, dim=1)  # [batch, actual_num_nodes]
        num_generated_nodes = node_types.shape[1]

        if verbose and batch_size == 1:
            print(f"\nGenerated nodes:")
            node_names = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'MASK']
            for i, nt in enumerate(node_types[0]):
                print(f"  Node {i}: {node_names[nt.item()]}")

        # ==================================================================
        # 2. ITERATIVE Latent-Guided Edge Generation (VARIABLE LENGTH)
        # ==================================================================

        # VARIABLE LENGTH: Use actual generated node count, not max_nodes
        edge_existence = torch.zeros(batch_size, num_generated_nodes, num_generated_nodes, device=device)
        edge_values = torch.zeros(batch_size, num_generated_nodes, num_generated_nodes, self.edge_feature_dim, device=device)

        for iteration in range(self.num_edge_iterations):
            if verbose and batch_size == 1:
                print(f"\n--- Edge Generation Iteration {iteration + 1}/{self.num_edge_iterations} ---")

            iteration_edges_added = 0
            iteration_edges_boosted = 0

            # CHANGED: Iterate over actual generated nodes only
            for i in range(num_generated_nodes):
                for j in range(i):
                    # No need to skip MASK nodes - we only have real nodes now!
                    # (stopped generation before adding MASK nodes)

                    # PHASE 3: JOINT edge-component prediction
                    (edge_component_logits, component_values_cont,
                     is_parallel_logit, consistency, _) = self.edge_decoder(
                        node_embeddings[i],
                        node_embeddings[j],
                        latent_topo,
                        latent_values,
                        latent_tf,
                        conditions  # NEW: Pass target specifications
                    )

                    # Edge existence probability: P(class > 0) = 1 - P(class 0)
                    edge_comp_probs = torch.softmax(edge_component_logits[0], dim=-1)  # [8]
                    base_prob = 1.0 - edge_comp_probs[0]  # Probability that edge exists

                    # CONSISTENCY-BASED BOOSTING (your key insight!)
                    if consistency[0] > 0.7:
                        adjusted_prob = base_prob * self.consistency_boost
                        if verbose and batch_size == 1:
                            iteration_edges_boosted += 1
                    elif consistency[0] < 0.3:
                        adjusted_prob = base_prob * self.consistency_penalty
                    else:
                        adjusted_prob = base_prob

                    # Sample edge: Check if edge should exist
                    # Also check that predicted component type is not "None" (class 0)
                    predicted_class = torch.argmax(edge_component_logits[0])

                    if adjusted_prob > edge_threshold and predicted_class > 0:
                        edge_existence[0, i, j] = 1.0
                        edge_existence[0, j, i] = 1.0

                        # Component type is directly from joint prediction (classes 1-7)
                        from ml.models.gumbel_softmax_utils import component_type_to_masks

                        component_type = predicted_class  # Already 1-7 (not 0)
                        masks = component_type_to_masks(component_type.unsqueeze(0), device=device)  # [1, 3]

                        # Combine into edge_values: [log(C), G, log(L_inv), mask_C, mask_G, mask_L, is_parallel]
                        is_parallel = (torch.sigmoid(is_parallel_logit[0]) > 0.5).float()  # Binary
                        edge_value = torch.cat([
                            component_values_cont[0],  # [3] - continuous values
                            masks[0],                  # [3] - binary masks from joint prediction
                            is_parallel.unsqueeze(0)   # [1] - is_parallel
                        ], dim=0)  # [7]

                        edge_values[0, i, j] = edge_value
                        edge_values[0, j, i] = edge_value

                        if verbose and batch_size == 1:
                            iteration_edges_added += 1

            if verbose and batch_size == 1:
                print(f"  Edges added: {iteration_edges_added}")
                print(f"  Edges boosted by TF consistency: {iteration_edges_boosted}")

            # ==================================================================
            # VIN CONNECTIVITY ENFORCEMENT (smart placement!)
            # ==================================================================

            if self.enforce_vin_connectivity and batch_size == 1:
                vin_id = 1  # VIN is always node 1
                vin_degree = edge_existence[0, vin_id, :].sum()

                if vin_degree < 0.5:
                    if verbose:
                        print(f"\n  ⚠️  VIN disconnected! Finding best target...")

                    # Find best target using TF consistency
                    best_target = None
                    best_consistency = -1.0

                    # CHANGED: Only search among actual generated nodes
                    for target_id in range(num_generated_nodes):
                        if target_id != vin_id:
                            (_, _, _, consistency, _) = self.edge_decoder(
                                node_embeddings[vin_id],
                                node_embeddings[target_id],
                                latent_topo,
                                latent_values,
                                latent_tf,
                                conditions  # NEW: Pass target specifications
                            )

                            if consistency[0] > best_consistency:
                                best_consistency = consistency[0].item()
                                best_target = target_id

                    # Force VIN → best_target
                    if best_target is not None:
                        (edge_comp_logits, comp_values, is_par_logit, _, _) = self.edge_decoder(
                            node_embeddings[vin_id],
                            node_embeddings[best_target],
                            latent_topo,
                            latent_values,
                            latent_tf,
                            conditions  # NEW: Pass target specifications
                        )

                        # Component type from joint prediction (classes 1-7)
                        from ml.models.gumbel_softmax_utils import component_type_to_masks
                        component_type = torch.argmax(edge_comp_logits[0], dim=-1)

                        # If predicted as "None" (class 0), force to R (class 1) as fallback
                        if component_type == 0:
                            component_type = torch.tensor(1, device=device)
                        masks = component_type_to_masks(component_type.unsqueeze(0), device=device)
                        is_parallel = (torch.sigmoid(is_par_logit[0]) > 0.5).float()

                        edge_val = torch.cat([
                            comp_values[0],            # [3] continuous
                            masks[0],                  # [3] masks
                            is_parallel.unsqueeze(0)   # [1] is_parallel
                        ], dim=0)  # [7]

                        edge_existence[0, vin_id, best_target] = 1.0
                        edge_existence[0, best_target, vin_id] = 1.0
                        edge_values[0, vin_id, best_target] = edge_val
                        edge_values[0, best_target, vin_id] = edge_val

                        if verbose:
                            target_name = ['GND', 'VIN', 'VOUT', 'INT', 'MASK'][node_types[0, best_target].item()]
                            print(f"  ✓ Forced VIN → {target_name} (consistency: {best_consistency:.3f})")

        # ==================================================================
        # 3. Done (TF evaluation done via SPICE, not prediction)
        # ==================================================================

        if verbose and batch_size == 1:
            print(f"\n{'='*70}")
            print(f"Generation Complete!")
            print(f"  Total edges: {int(edge_existence.sum() / 2)}")  # Divide by 2 for undirected
            print(f"  VIN connected: {'Yes' if edge_existence[0, 1, :].sum() > 0 else 'No'}")
            print(f"{'='*70}\n")

        # ==================================================================
        # Gumbel-Softmax Component Selection (already applied above)
        # ==================================================================
        # Component selection is done via hard argmax during generation.
        # Masks are already binary (0 or 1) from component_type_to_masks().
        # Edge values format: [log(C), G, log(L_inv), mask_C, mask_G, mask_L, is_parallel]

        return {
            'node_types': node_types,
            'edge_existence': edge_existence,
            'edge_values': edge_values,
        }


if __name__ == '__main__':
    """Test the latent-guided decoder."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    print("Testing Latent-Guided GraphGPT Decoder...")

    device = 'cpu'
    batch_size = 1

    decoder = LatentGuidedGraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=256,
        num_heads=8,
        num_node_layers=4,
        max_nodes=5,
        num_edge_iterations=2,  # Fewer for testing
        enforce_vin_connectivity=True
    ).to(device)

    # Test generation
    latent = torch.randn(batch_size, 8, device=device)
    conditions = torch.randn(batch_size, 2, device=device)

    print("\n1. Testing generation (verbose)...")
    circuit = decoder.generate(latent, conditions, verbose=True)

    print("✓ Circuit generated successfully")
    print(f"  Node types shape: {circuit['node_types'].shape}")
    print(f"  Edge existence shape: {circuit['edge_existence'].shape}")
    print(f"  VIN connected: {circuit['edge_existence'][0, 1, :].sum() > 0}")

    # Test forward pass (training mode)
    print("\n2. Testing forward pass (training mode)...")
    target_nodes = torch.randint(0, 5, (batch_size, 5), device=device)
    predictions = decoder(latent, conditions, target_node_types=target_nodes)

    print("✓ Forward pass successful")
    print(f"  Has consistency_scores: {'consistency_scores' in predictions}")
    print(f"  Consistency scores shape: {predictions['consistency_scores'].shape}")

    print("\n✅ All tests passed!")
    print("\nLatent-Guided GraphGPT Decoder is ready for training!")
