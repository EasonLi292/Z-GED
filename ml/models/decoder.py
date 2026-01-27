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
    - Node count (3 to max_nodes)
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
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_node_layers: int = 4,
        max_nodes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Context encoder (latent only, no conditions)
        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Node count predictor (3 to max_nodes) - uses topology dims of latent
        self.node_count_predictor = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, max_nodes - 2)
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
            num_attention_heads=4,
            dropout=dropout
        )

    def forward(
        self,
        latent_code: torch.Tensor,
        target_node_types: Optional[torch.Tensor] = None,
        target_edges: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training).

        Args:
            latent_code: [batch, latent_dim]
            target_node_types: [batch, num_nodes] for teacher forcing nodes
            target_edges: [batch, num_nodes, num_nodes] unified edge-component
                target (0=no edge, 1-7=component type) for teacher forcing edges.
                If None, uses predicted (argmax) edges as autoregressive input.

        Returns:
            node_types: [batch, num_nodes, 5] logits
            node_count_logits: [batch, max_nodes-2] logits
            edge_component_logits: [batch, num_nodes, num_nodes, 8] logits
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device
        num_nodes = target_node_types.shape[1] if target_node_types is not None else self.max_nodes

        # Encode context (latent only)
        context = self.context_encoder(latent_code)

        # Predict node count from topology dimensions of latent
        latent_topo = latent_code[:, :2]
        node_count_logits = self.node_count_predictor(latent_topo)

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

        # Generate edges autoregressively (teacher forcing if targets provided)
        edge_component_logits = torch.zeros(batch_size, num_nodes, num_nodes, 8, device=device)

        edge_hidden = self.edge_decoder.init_hidden(batch_size, device)
        prev_edge_token = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(num_nodes):
            for j in range(i):
                logits, edge_hidden = self.edge_decoder(
                    node_embeddings[i],
                    node_embeddings[j],
                    latent_code,
                    edge_hidden,
                    prev_edge_token
                )
                edge_component_logits[:, i, j, :] = logits
                edge_component_logits[:, j, i, :] = logits  # Symmetric

                # Teacher forcing: use ground truth; otherwise use predicted
                if target_edges is not None:
                    prev_edge_token = target_edges[:, i, j].long()
                else:
                    prev_edge_token = torch.argmax(logits, dim=-1)

        return {
            'node_types': node_logits,
            'node_count_logits': node_count_logits,
            'edge_component_logits': edge_component_logits,
        }

    def generate(
        self,
        latent_code: torch.Tensor,
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

        # Encode context (latent only)
        context = self.context_encoder(latent_code)

        # Predict node count from topology dimensions
        latent_topo = latent_code[:, :2]
        node_count_logits = self.node_count_predictor(latent_topo)
        target_nodes = min((torch.argmax(node_count_logits, dim=-1) + 3).item(), self.max_nodes)

        if verbose:
            probs = F.softmax(node_count_logits, dim=-1)
            parts = ", ".join(f"{i+3}={probs[0,i]:.2f}" for i in range(probs.shape[-1]))
            print(f"Node count: {parts} → {target_nodes}")

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

        # Generate edges autoregressively
        edge_existence = torch.zeros(batch_size, target_nodes, target_nodes, device=device)
        component_types = torch.zeros(batch_size, target_nodes, target_nodes, dtype=torch.long, device=device)

        edge_hidden = self.edge_decoder.init_hidden(batch_size, device)
        prev_edge_token = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(target_nodes):
            for j in range(i):
                logits, edge_hidden = self.edge_decoder(
                    node_embeddings[i],
                    node_embeddings[j],
                    latent_code,
                    edge_hidden,
                    prev_edge_token
                )

                probs = F.softmax(logits[0], dim=-1)
                edge_prob = 1.0 - probs[0]
                predicted_class = torch.argmax(logits[0])

                if edge_prob > edge_threshold and predicted_class > 0:
                    edge_existence[0, i, j] = 1.0
                    edge_existence[0, j, i] = 1.0
                    component_types[0, i, j] = predicted_class
                    component_types[0, j, i] = predicted_class

                # Feed back the predicted decision for next step
                prev_edge_token = torch.argmax(logits, dim=-1)

        if verbose:
            num_edges = int(edge_existence.sum().item() // 2)
            print(f"Edges: {num_edges}, VIN connected: {edge_existence[0, 1].sum() > 0}")

        return {
            'node_types': node_types,
            'edge_existence': edge_existence,
            'component_types': component_types,
        }

# Alias for backward compatibility
LatentGuidedGraphGPTDecoder = SimplifiedCircuitDecoder


if __name__ == '__main__':
    print("Testing Simplified Decoder...")

    decoder = SimplifiedCircuitDecoder(latent_dim=8, hidden_dim=256)
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Parameters: {params:,}")

    latent = torch.randn(1, 8)

    print("\nGeneration:")
    circuit = decoder.generate(latent, verbose=True)

    print("\nForward pass (with teacher forcing):")
    target_nodes = torch.randint(0, 5, (1, 4))
    target_edges = torch.randint(0, 8, (1, 4, 4))
    out = decoder(latent, target_node_types=target_nodes, target_edges=target_edges)
    print(f"  node_types: {out['node_types'].shape}")
    print(f"  edge_component_logits: {out['edge_component_logits'].shape}")

    print("\nForward pass (without teacher forcing):")
    out2 = decoder(latent, target_node_types=target_nodes)
    print(f"  node_types: {out2['node_types'].shape}")
    print(f"  edge_component_logits: {out2['edge_component_logits'].shape}")

    print("\n✅ Test passed!")
