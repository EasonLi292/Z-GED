"""
GraphGPT-style Autoregressive Decoder for RLC Circuit Generation

Generates circuits autoregressively:
1. Generate nodes one by one (GND -> VIN -> VOUT -> INTERNAL x2)
2. For each node, generate edges to all previously generated nodes
3. Generate poles/zeros using existing VariableLengthDecoder

This approach eliminates:
- Edge generation mode collapse (each edge is independent)
- Gradient explosion (standard transformer, no unbounded attention)
- Complex diffusion training (simple cross-entropy loss)

Reference: "GraphGPT: Graph Instruction Tuning for Large Language Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class AutoregressiveNodeDecoder(nn.Module):
    """
    Autoregressive decoder for generating nodes sequentially.

    For each node position i:
    - Attends to previous nodes [0, 1, ..., i-1]
    - Predicts node type (GND, VIN, VOUT, INTERNAL, MASK)
    - Predicts stopping criterion (EOF-like signal)

    IMPORTANT: No hard limit on node count! Generation continues until:
    1. Stopping criterion triggers (learned EOF)
    2. Safety limit reached (max_position_embeddings)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_node_types: int = 5,
        max_position_embeddings: int = 50,  # Safety limit, not architectural constraint
        dropout: float = 0.1,
        use_stopping_criterion: bool = True,
        conditions_dim: int = 2,  # Dimension of conditions (cutoff, Q)
        topo_latent_dim: int = 2  # NEW: Dimension of topology latent (encodes node count!)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.max_position_embeddings = max_position_embeddings
        self.use_stopping_criterion = use_stopping_criterion
        self.conditions_dim = conditions_dim
        self.topo_latent_dim = topo_latent_dim

        # Node type embedding (for teacher forcing and previous nodes)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)

        # Positional encoding - EXTENSIBLE to any position up to max_position_embeddings
        # This allows generation beyond training data (e.g., train on 3-5 nodes, generate 10+ nodes)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_types)
        )

        # Stopping criterion head with TOPOLOGY LATENT + CONDITIONS + POSITION
        # CRITICAL FIX: Use ONLY topology latent + conditions + position for stop decision
        # The topology latent encodes node count:
        #   3-node: topo_latent ≈ [-3.4, 2.3]
        #   4-node: topo_latent ≈ [-1.7, 0.0]
        #   5-node: topo_latent ≈ [0.5, -1.8]
        # DO NOT use transformer output - it causes train/test mismatch!
        if use_stopping_criterion:
            # Input: topology latent + conditions + position embedding
            # This forces the model to decide stop based on latent topology, not generated sequence
            stop_input_dim = topo_latent_dim + conditions_dim + 1  # +1 for position
            self.stop_head = nn.Sequential(
                nn.Linear(stop_input_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )

    def forward(
        self,
        context: torch.Tensor,
        position: int,
        previous_nodes: Optional[List[torch.Tensor]] = None,
        teacher_node_type: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,  # Conditions for stop head
        latent_topo: Optional[torch.Tensor] = None  # NEW: Topology latent for stop head
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate node at position i.

        Args:
            context: Context embedding [batch, hidden_dim] (latent + specs)
            position: Node position (0-4)
            previous_nodes: List of previous node embeddings
            teacher_node_type: Ground truth node type for teacher forcing [batch]
            conditions: Specifications [batch, conditions_dim] for stop decision
            latent_topo: Topology latent [batch, topo_latent_dim] - CRITICAL for stopping!

        Returns:
            node_logits: Node type logits [batch, num_node_types]
            node_embedding: Node embedding for future use [batch, hidden_dim]
            stop_logit: Stop generation logit [batch] (None if use_stopping_criterion=False)
        """
        batch_size = context.shape[0]
        device = context.device

        # Add positional encoding to context
        pos_embed = self.position_embedding(
            torch.tensor([position], device=device)
        ).expand(batch_size, -1)  # [batch, hidden_dim]

        query = context + pos_embed  # [batch, hidden_dim]
        query = query.unsqueeze(1)  # [batch, 1, hidden_dim]

        # If first node, no previous context to attend to
        if previous_nodes is None or len(previous_nodes) == 0:
            # Just use query directly
            output = query
        else:
            # Stack previous node embeddings as memory
            memory = torch.stack(previous_nodes, dim=1)  # [batch, num_prev, hidden_dim]

            # Transformer decoder: query attends to memory
            output = self.transformer(
                tgt=query,
                memory=memory
            )  # [batch, 1, hidden_dim]

        output = output.squeeze(1)  # [batch, hidden_dim]

        # Predict node type
        node_logits = self.output_head(output)  # [batch, num_node_types]

        # Predict stopping criterion using ONLY topology latent + conditions + position
        # CRITICAL FIX: Do NOT use transformer output - causes train/test mismatch!
        # The stop decision must be based solely on:
        #   1. Topology latent (encodes node count)
        #   2. Conditions (cutoff, Q)
        #   3. Current position (where we are in generation)
        stop_logit = None
        if self.use_stopping_criterion:
            # Build stop input: topology latent + conditions + position
            stop_parts = []

            # Add topology latent (or zeros if not provided)
            if latent_topo is not None:
                stop_parts.append(latent_topo)
            else:
                stop_parts.append(torch.zeros(batch_size, self.topo_latent_dim, device=device))

            # Add conditions (or zeros if not provided)
            if conditions is not None:
                stop_parts.append(conditions)
            else:
                stop_parts.append(torch.zeros(batch_size, self.conditions_dim, device=device))

            # Add position as a feature (normalized to [0, 1])
            position_feat = torch.full((batch_size, 1), position / self.max_position_embeddings, device=device)
            stop_parts.append(position_feat)

            stop_input = torch.cat(stop_parts, dim=-1)  # [batch, topo_latent_dim + conditions_dim + 1]
            stop_logit = self.stop_head(stop_input).squeeze(-1)  # [batch]

        # Create node embedding for next steps
        if teacher_node_type is not None:
            # Teacher forcing: use ground truth
            node_embedding = self.node_type_embedding(teacher_node_type)  # [batch, hidden_dim]
        else:
            # Sampling: use predicted type
            predicted_type = torch.argmax(node_logits, dim=-1)
            node_embedding = self.node_type_embedding(predicted_type)

        # Add positional encoding to node embedding
        node_embedding = node_embedding + pos_embed

        return node_logits, node_embedding, stop_logit


