"""
Simplified Circuit Decoder for Topology-Only Generation.

Predicts circuit topology (nodes + edges with component types).
Does NOT predict component values - only topology matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ml.models.node_decoder import AutoregressiveNodeDecoder
from ml.models.decoder_components import LatentGuidedEdgeDecoder


class SimplifiedCircuitDecoder(nn.Module):
    """
    Minimal decoder for circuit topology generation.

    Predicts:
    - Node count (3, 4, or 5)
    - Node types (GND, VIN, VOUT, INTERNAL)
    - Edge-component (8-way: no edge, R, C, L, RC, RL, CL, RCL)

    Does NOT predict:
    - Component values (not needed for topology)
    - is_parallel (derived from component type)
    - Masks (derived from component type)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_node_layers: int = 4,
        max_nodes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.conditions_dim = conditions_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim + conditions_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Node count predictor (3, 4, or 5 nodes)
        self.node_count_predictor = nn.Sequential(
            nn.Linear(2 + conditions_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 3)
        )

        # Node decoder
        self.node_decoder = AutoregressiveNodeDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_node_layers,
            num_node_types=5,
            max_position_embeddings=max_nodes,
            dropout=dropout
        )

        # Edge decoder (topology only)
        self.edge_decoder = LatentGuidedEdgeDecoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            conditions_dim=conditions_dim,
            num_attention_heads=4,
            dropout=dropout
        )

    def forward(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        target_node_types: Optional[torch.Tensor] = None,
        target_edges: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training).

        Returns:
            node_types: [batch, num_nodes, 5] logits
            node_count_logits: [batch, 3] logits
            edge_component_logits: [batch, num_nodes, num_nodes, 8] logits
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device
        num_nodes = target_node_types.shape[1] if target_node_types is not None else self.max_nodes

        # Encode context
        context = self.context_encoder(torch.cat([latent_code, conditions], dim=-1))

        # Predict node count
        latent_topo = latent_code[:, :2]
        node_count_logits = self.node_count_predictor(torch.cat([latent_topo, conditions], dim=-1))

        # Generate nodes
        node_embeddings = []
        node_logits_list = []

        for i in range(num_nodes):
            teacher_type = target_node_types[:, i] if target_node_types is not None else None
            node_logits, node_embed = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=teacher_type
            )
            node_logits_list.append(node_logits)
            node_embeddings.append(node_embed)

        node_logits = torch.stack(node_logits_list, dim=1)

        # Generate edges (topology only)
        edge_component_logits = torch.zeros(batch_size, num_nodes, num_nodes, 8, device=device)

        for i in range(num_nodes):
            for j in range(i):
                logits = self.edge_decoder(
                    node_embeddings[i],
                    node_embeddings[j],
                    latent_code,
                    conditions
                )
                edge_component_logits[:, i, j, :] = logits
                edge_component_logits[:, j, i, :] = logits  # Symmetric

        return {
            'node_types': node_logits,
            'node_count_logits': node_count_logits,
            'edge_component_logits': edge_component_logits,
        }

    def generate(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        edge_threshold: float = 0.5,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuit topology from latent code.

        Returns:
            node_types: [batch, num_nodes] node type indices
            edge_existence: [batch, num_nodes, num_nodes] binary
            component_types: [batch, num_nodes, num_nodes] component type indices
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device

        # Encode context
        context = self.context_encoder(torch.cat([latent_code, conditions], dim=-1))

        # Predict node count
        latent_topo = latent_code[:, :2]
        node_count_logits = self.node_count_predictor(torch.cat([latent_topo, conditions], dim=-1))
        target_nodes = (torch.argmax(node_count_logits, dim=-1) + 3).item()

        if verbose:
            probs = F.softmax(node_count_logits, dim=-1)
            print(f"Node count: 3={probs[0,0]:.2f}, 4={probs[0,1]:.2f}, 5={probs[0,2]:.2f} → {target_nodes}")

        # Generate nodes
        node_embeddings = []
        predicted_node_types = []

        for i in range(target_nodes):
            node_logits, _ = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=None
            )

            # First 3 are fixed: GND, VIN, VOUT
            node_type = torch.tensor([i if i < 3 else torch.argmax(node_logits, dim=-1).item()], device=device)
            predicted_node_types.append(node_type)

            # Create embedding
            node_embed = self.node_decoder.node_type_embedding(node_type)
            pos_embed = self.node_decoder.position_embedding(torch.tensor([i], device=device))
            node_embeddings.append(node_embed + pos_embed)

        node_types = torch.stack(predicted_node_types, dim=1)

        if verbose:
            names = ['GND', 'VIN', 'VOUT', 'INT', 'MASK']
            print(f"Nodes: {[names[t.item()] for t in predicted_node_types]}")

        # Generate edges
        edge_existence = torch.zeros(batch_size, target_nodes, target_nodes, device=device)
        component_types = torch.zeros(batch_size, target_nodes, target_nodes, dtype=torch.long, device=device)

        for i in range(target_nodes):
            for j in range(i):
                logits = self.edge_decoder(
                    node_embeddings[i],
                    node_embeddings[j],
                    latent_code,
                    conditions
                )

                probs = F.softmax(logits[0], dim=-1)
                edge_prob = 1.0 - probs[0]
                predicted_class = torch.argmax(logits[0])

                if edge_prob > edge_threshold and predicted_class > 0:
                    edge_existence[0, i, j] = 1.0
                    edge_existence[0, j, i] = 1.0
                    component_types[0, i, j] = predicted_class
                    component_types[0, j, i] = predicted_class

        if verbose:
            num_edges = int(edge_existence.sum().item() // 2)
            print(f"Edges: {num_edges}, VIN connected: {edge_existence[0, 1].sum() > 0}")

        return {
            'node_types': node_types,
            'edge_existence': edge_existence,
            'component_types': component_types,
        }

    def generate_from_specification(
        self,
        frequency: float,
        q_factor: float,
        specs_db: Optional[torch.Tensor] = None,
        latents_db: Optional[torch.Tensor] = None,
        k: int = 5,
        num_samples: int = 1,
        edge_threshold: float = 0.5,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuit topology from frequency and Q specifications.

        Two modes:
        1. K-NN interpolation (recommended): If specs_db and latents_db provided,
           interpolates between k nearest training circuits by specification.
        2. Random sampling: If no database provided, samples from N(0,1) prior.

        Args:
            frequency: Cutoff frequency in Hz
            q_factor: Quality factor
            specs_db: [N, 2] tensor of training specifications [freq, Q] (for K-NN mode)
            latents_db: [N, 8] tensor of encoded latents from training data (for K-NN mode)
            k: Number of neighbors for K-NN interpolation (default: 5)
            num_samples: Number of samples to generate when using random mode (picks best)
            edge_threshold: Threshold for edge existence
            verbose: Print generation details

        Returns:
            node_types: [1, num_nodes] node type indices
            edge_existence: [1, num_nodes, num_nodes] binary
            component_types: [1, num_nodes, num_nodes] component type indices
        """
        import numpy as np

        device = next(self.parameters()).device

        # Normalize conditions (same as training)
        conditions = torch.tensor([[
            np.log10(max(frequency, 1.0)) / 4.0,
            np.log10(max(q_factor, 0.01)) / 2.0
        ]], dtype=torch.float32, device=device)

        if verbose:
            print(f"Generating circuit for f={frequency:.1f} Hz, Q={q_factor:.3f}")

        # Mode 1: K-NN interpolation (recommended)
        if specs_db is not None and latents_db is not None:
            latent = self._interpolate_latent(
                frequency, q_factor, specs_db, latents_db, k, verbose
            )
            latent = latent.unsqueeze(0).to(device)
            result = self.generate(latent, conditions, edge_threshold, verbose=verbose)
            result['latent'] = latent
            return result

        # Mode 2: Random sampling from N(0,1) prior
        if verbose:
            print(f"Using random sampling (N(0,1) prior)")

        best_result = None
        best_score = -1

        # Component names for checking reactive elements
        REACTIVE_TYPES = {2, 3, 4, 5, 6, 7}  # C, L, RC, RL, CL, RCL

        for i in range(num_samples):
            # Sample from prior N(0,1)
            latent = torch.randn(1, self.latent_dim, device=device)

            # Generate
            result = self.generate(latent, conditions, edge_threshold, verbose=False)

            # Score based on connectivity and reactive components
            edge_existence = result['edge_existence'][0]
            component_types = result['component_types'][0]
            num_nodes = result['node_types'].shape[1]

            # Check VIN (node 1) and VOUT (node 2) connectivity
            vin_connected = edge_existence[1].sum().item() > 0 if num_nodes > 1 else False
            vout_connected = edge_existence[2].sum().item() > 0 if num_nodes > 2 else False
            num_edges = int(edge_existence.sum().item() // 2)

            # Check for reactive components (C or L) - essential for a filter!
            has_reactive = False
            for ni in range(num_nodes):
                for nj in range(ni):
                    if edge_existence[ni, nj] > 0.5:
                        comp_type = component_types[ni, nj].item()
                        if comp_type in REACTIVE_TYPES:
                            has_reactive = True
                            break
                if has_reactive:
                    break

            # Score: connectivity + reactive bonus + edge count
            score = (2 if vin_connected else 0) + (1 if vout_connected else 0)
            score += (3 if has_reactive else 0)  # Big bonus for having C/L
            score += (0.1 * num_edges)

            if verbose and num_samples > 1:
                print(f"  Sample {i+1}: {num_nodes} nodes, {num_edges} edges, "
                      f"VIN={'Y' if vin_connected else 'N'}, VOUT={'Y' if vout_connected else 'N'}, "
                      f"reactive={'Y' if has_reactive else 'N'}, score={score:.1f}")

            if score > best_score:
                best_score = score
                best_result = result
                best_result['latent'] = latent

        if verbose:
            node_types = best_result['node_types'][0]
            edge_existence = best_result['edge_existence'][0]
            num_nodes = node_types.shape[0]
            num_edges = int(edge_existence.sum().item() // 2)
            names = ['GND', 'VIN', 'VOUT', 'INT', 'MASK']
            print(f"Best: {[names[t.item()] for t in node_types]}, {num_edges} edges")

        return best_result

    def _interpolate_latent(
        self,
        frequency: float,
        q_factor: float,
        specs_db: torch.Tensor,
        latents_db: torch.Tensor,
        k: int = 5,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Interpolate latent code from k nearest training circuits.

        Args:
            frequency: Target cutoff frequency
            q_factor: Target Q-factor
            specs_db: [N, 2] training specifications [freq, Q]
            latents_db: [N, 8] encoded latents from training
            k: Number of neighbors to interpolate
            verbose: Print interpolation details

        Returns:
            latent: [8] interpolated latent code
        """
        import numpy as np

        # Normalize target (log scale for frequency)
        target = torch.tensor([np.log10(max(frequency, 1.0)), q_factor])

        # Normalize database specs
        db_normalized = torch.stack([
            torch.log10(torch.clamp(specs_db[:, 0], min=1.0)),
            specs_db[:, 1]
        ], dim=1)

        # Compute distances (weighted: Q matters more)
        freq_weight = 1.0
        q_weight = 2.0
        distances = (
            freq_weight * (db_normalized[:, 0] - target[0])**2 +
            q_weight * (db_normalized[:, 1] - target[1])**2
        ).sqrt()

        # Find k nearest
        k = min(k, len(distances))
        nearest_indices = distances.argsort()[:k]
        nearest_distances = distances[nearest_indices]

        # Inverse distance weighting
        weights = 1.0 / (nearest_distances + 1e-6)
        weights = weights / weights.sum()

        # Weighted average of latents
        interpolated = (latents_db[nearest_indices] * weights.unsqueeze(1)).sum(dim=0)

        if verbose:
            print(f"K-NN interpolation (k={k}):")
            for i, idx in enumerate(nearest_indices[:3]):
                spec = specs_db[idx]
                print(f"  Neighbor {i+1}: f={spec[0]:.1f} Hz, Q={spec[1]:.3f}, "
                      f"dist={nearest_distances[i]:.3f}, weight={weights[i]:.3f}")

        return interpolated


# Alias for backward compatibility
LatentGuidedGraphGPTDecoder = SimplifiedCircuitDecoder


if __name__ == '__main__':
    print("Testing Simplified Decoder...")

    decoder = SimplifiedCircuitDecoder(latent_dim=8, conditions_dim=2, hidden_dim=256)
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Parameters: {params:,}")

    latent = torch.randn(1, 8)
    conditions = torch.randn(1, 2)

    print("\nGeneration:")
    circuit = decoder.generate(latent, conditions, verbose=True)

    print("\nForward pass:")
    target_nodes = torch.randint(0, 5, (1, 4))
    out = decoder(latent, conditions, target_node_types=target_nodes)
    print(f"  node_types: {out['node_types'].shape}")
    print(f"  edge_component_logits: {out['edge_component_logits'].shape}")

    print("\n✅ Test passed!")
