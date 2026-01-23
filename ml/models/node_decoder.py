"""
Simplified Autoregressive Node Decoder for RLC Circuit Generation.

Generates nodes autoregressively using transformer decoder.
Node count is determined by direct prediction (not learned stopping).

Simplifications:
- Removed stopping criterion head (node count predicted directly in main decoder)
- Cleaner interface with fewer parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class AutoregressiveNodeDecoder(nn.Module):
    """
    Simplified autoregressive decoder for generating nodes sequentially.

    For each node position i:
    - Attends to previous nodes [0, 1, ..., i-1]
    - Predicts node type (GND, VIN, VOUT, INTERNAL, MASK)

    Node count is determined externally by the main decoder's node_count_predictor.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_node_types: int = 5,
        max_position_embeddings: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.max_position_embeddings = max_position_embeddings

        # Node type embedding (for teacher forcing and previous nodes)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)

        # Positional encoding
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

        # Output head for node type prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_types)
        )

    def forward(
        self,
        context: torch.Tensor,
        position: int,
        previous_nodes: Optional[List[torch.Tensor]] = None,
        teacher_node_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate node at position i.

        Args:
            context: Context embedding [batch, hidden_dim] (latent + specs)
            position: Node position (0, 1, 2, ...)
            previous_nodes: List of previous node embeddings
            teacher_node_type: Ground truth node type for teacher forcing [batch]

        Returns:
            node_logits: Node type logits [batch, num_node_types]
            node_embedding: Node embedding for future use [batch, hidden_dim]
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

        # Create node embedding for next steps
        if teacher_node_type is not None:
            # Teacher forcing: use ground truth
            node_embedding = self.node_type_embedding(teacher_node_type)
        else:
            # Sampling: use predicted type
            predicted_type = torch.argmax(node_logits, dim=-1)
            node_embedding = self.node_type_embedding(predicted_type)

        # Add positional encoding to node embedding
        node_embedding = node_embedding + pos_embed

        return node_logits, node_embedding


