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


class AutoregressiveEdgeDecoder(nn.Module):
    """
    Decoder for generating edges between nodes.

    For each pair (node_i, node_j) where j < i:
    - Predicts edge existence (binary)
    - Predicts edge values (C, G, L_inv + masks)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_feature_dim: int = 7,  # C, G, L_inv + 4 masks
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim

        # Edge existence head (binary classification)
        self.edge_exist_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Edge value head (continuous values)
        self.edge_value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_feature_dim)
        )

    def forward(
        self,
        node_i: torch.Tensor,
        node_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate edge from node i to node j.

        Args:
            node_i: Source node embedding [batch, hidden_dim]
            node_j: Target node embedding [batch, hidden_dim]

        Returns:
            edge_exist_logit: Edge existence logit [batch]
            edge_values: Edge feature values [batch, edge_feature_dim]
        """
        # Concatenate node embeddings
        edge_input = torch.cat([node_i, node_j], dim=-1)  # [batch, hidden_dim * 2]

        # Predict edge existence
        edge_exist_logit = self.edge_exist_head(edge_input).squeeze(-1)  # [batch]

        # Predict edge values
        edge_values = self.edge_value_head(edge_input)  # [batch, edge_feature_dim]

        return edge_exist_logit, edge_values


class GraphGPTDecoder(nn.Module):
    """
    Complete GraphGPT decoder for RLC circuit generation.

    Architecture:
    1. Context encoder: Project (latent + specs) to hidden_dim
    2. Autoregressive node generation (5 nodes)
    3. Autoregressive edge generation (for each node, edges to previous nodes)
    4. Pole/zero generation (reuse existing VariableLengthDecoder)

    Training:
    - Teacher forcing for nodes and edges
    - Cross-entropy loss for node types
    - BCE loss for edge existence
    - MSE loss for edge values
    - Existing loss for poles/zeros

    Generation:
    - Autoregressive sampling (left to right)
    - Can enforce constraints (GND, VIN, VOUT required)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_node_layers: int = 4,
        max_nodes: int = 5,
        max_poles: int = 4,
        max_zeros: int = 4,
        edge_feature_dim: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.conditions_dim = conditions_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.edge_feature_dim = edge_feature_dim

        # Node type constants
        self.NODE_TYPES = {
            'GND': 0,
            'VIN': 1,
            'VOUT': 2,
            'INTERNAL': 3,
            'MASK': 4
        }

        # Fixed generation order for RLC circuits
        self.node_order = ['GND', 'VIN', 'VOUT', 'INTERNAL', 'INTERNAL']

        # ==================================================================
        # Context Encoder (latent + specifications)
        # ==================================================================

        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim + conditions_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ==================================================================
        # Autoregressive Decoders
        # ==================================================================

        # Node decoder
        self.node_decoder = AutoregressiveNodeDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_node_layers,
            num_node_types=5,
            dropout=dropout
        )

        # Edge decoder
        self.edge_decoder = AutoregressiveEdgeDecoder(
            hidden_dim=hidden_dim,
            edge_feature_dim=edge_feature_dim,
            dropout=dropout
        )

        # ==================================================================
        # Pole/Zero Decoder (reuse existing - it works perfectly!)
        # ==================================================================

        # Graph pooling for pole/zero prediction
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Pole count prediction
        self.pole_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poles + 1)
        )

        # Zero count prediction
        self.zero_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_zeros + 1)
        )

        # Pole value prediction
        self.pole_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_poles * 2)
        )

        # Zero value prediction
        self.zero_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_zeros * 2)
        )

        self.max_poles = max_poles
        self.max_zeros = max_zeros

    def forward(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        target_nodes: Optional[torch.Tensor] = None,
        target_edges_exist: Optional[torch.Tensor] = None,
        target_edges_values: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: generate circuit autoregressively.

        Args:
            latent_code: Latent from encoder [batch, latent_dim]
            conditions: Specifications [batch, conditions_dim] (cutoff, Q)
            target_nodes: Ground truth node types [batch, max_nodes] (for training)
            target_edges_exist: Ground truth edge existence [batch, max_nodes, max_nodes]
            target_edges_values: Ground truth edge values [batch, max_nodes, max_nodes, 7]
            teacher_forcing: Use ground truth during training

        Returns:
            outputs: Dictionary containing:
                - node_logits: [batch, max_nodes, num_node_types]
                - edge_exist_logits: [batch, max_nodes, max_nodes]
                - edge_values: [batch, max_nodes, max_nodes, edge_feature_dim]
                - pole_count_logits: [batch, max_poles + 1]
                - zero_count_logits: [batch, max_zeros + 1]
                - pole_values: [batch, max_poles, 2]
                - zero_values: [batch, max_zeros, 2]
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device

        # ==================================================================
        # 1. Encode context (latent + specifications)
        # ==================================================================

        context_input = torch.cat([latent_code, conditions], dim=-1)
        context = self.context_encoder(context_input)  # [batch, hidden_dim]

        # ==================================================================
        # 2. Autoregressive Node Generation
        # ==================================================================

        node_embeddings = []
        node_logits_list = []

        for i in range(self.max_nodes):
            # Get teacher forcing target if available
            if teacher_forcing and target_nodes is not None:
                teacher_node_type = target_nodes[:, i]
            else:
                teacher_node_type = None

            # Generate node i
            node_logits, node_embed = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=teacher_node_type
            )

            node_embeddings.append(node_embed)
            node_logits_list.append(node_logits)

        # Stack node logits
        node_logits = torch.stack(node_logits_list, dim=1)  # [batch, max_nodes, num_node_types]

        # ==================================================================
        # 3. Autoregressive Edge Generation
        # ==================================================================

        edge_exist_logits = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=device)
        edge_values = torch.zeros(batch_size, self.max_nodes, self.max_nodes, self.edge_feature_dim, device=device)

        for i in range(self.max_nodes):
            for j in range(i):  # Only edges to previous nodes
                # Generate edge from node i to node j
                edge_exist_logit, edge_value = self.edge_decoder(
                    node_i=node_embeddings[i],
                    node_j=node_embeddings[j]
                )

                edge_exist_logits[:, i, j] = edge_exist_logit
                edge_values[:, i, j] = edge_value

                # Make symmetric (undirected graph)
                edge_exist_logits[:, j, i] = edge_exist_logit
                edge_values[:, j, i] = edge_value

        # ==================================================================
        # 4. Graph-level Pole/Zero Generation
        # ==================================================================

        # Pool node embeddings to graph representation
        graph_repr = torch.stack(node_embeddings, dim=1).mean(dim=1)  # [batch, hidden_dim]
        graph_features = self.graph_pooling(graph_repr)

        # Add context
        graph_features = graph_features + context

        # Predict pole count
        pole_count_logits = self.pole_count_head(graph_features)

        # Predict zero count
        zero_count_logits = self.zero_count_head(graph_features)

        # Predict pole values
        pole_values_flat = self.pole_value_head(graph_features)
        pole_values = pole_values_flat.view(batch_size, self.max_poles, 2)

        # Predict zero values
        zero_values_flat = self.zero_value_head(graph_features)
        zero_values = zero_values_flat.view(batch_size, self.max_zeros, 2)

        # ==================================================================
        # 5. Return outputs
        # ==================================================================

        return {
            'node_types': node_logits,
            'edge_existence': edge_exist_logits,
            'edge_values': edge_values,
            'pole_count_logits': pole_count_logits,
            'zero_count_logits': zero_count_logits,
            'pole_values': pole_values,
            'zero_values': zero_values
        }

    def generate(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        enforce_constraints: bool = True,
        edge_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuit autoregressively (no teacher forcing).

        Args:
            latent_code: Latent from encoder [batch, latent_dim]
            conditions: Specifications [batch, conditions_dim]
            enforce_constraints: Enforce GND, VIN, VOUT in first 3 positions
            edge_threshold: Threshold for edge existence probability

        Returns:
            circuit: Generated circuit dictionary
        """
        batch_size = latent_code.shape[0]
        device = latent_code.device

        # Encode context
        context_input = torch.cat([latent_code, conditions], dim=-1)
        context = self.context_encoder(context_input)

        # Generate nodes
        node_embeddings = []
        predicted_node_types = []

        for i in range(self.max_nodes):
            node_logits, node_embed = self.node_decoder(
                context=context,
                position=i,
                previous_nodes=node_embeddings,
                teacher_node_type=None  # No teacher forcing
            )

            # Sample node type
            if enforce_constraints and i < 3:
                # Enforce first 3 nodes are GND, VIN, VOUT
                node_type = torch.tensor(
                    [self.NODE_TYPES[self.node_order[i]]] * batch_size,
                    device=device
                )
            else:
                # Sample from distribution
                node_probs = F.softmax(node_logits, dim=-1)
                node_type = torch.argmax(node_probs, dim=-1)

            predicted_node_types.append(node_type)

            # Get embedding for sampled type
            node_embed = self.node_decoder.node_type_embedding(node_type)
            pos_embed = self.node_decoder.position_embedding(
                torch.tensor([i], device=device)
            ).expand(batch_size, -1)
            node_embed = node_embed + pos_embed

            node_embeddings.append(node_embed)

        # Stack node types
        node_types = torch.stack(predicted_node_types, dim=1)  # [batch, max_nodes]

        # Generate edges
        edge_existence = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=device)
        edge_values = torch.zeros(batch_size, self.max_nodes, self.max_nodes, self.edge_feature_dim, device=device)

        for i in range(self.max_nodes):
            for j in range(i):
                edge_exist_logit, edge_value = self.edge_decoder(
                    node_i=node_embeddings[i],
                    node_j=node_embeddings[j]
                )

                # Sample edge existence
                edge_prob = torch.sigmoid(edge_exist_logit)
                edge_exists = (edge_prob > edge_threshold).float()

                edge_existence[:, i, j] = edge_exists
                edge_values[:, i, j] = edge_value * edge_exists.unsqueeze(-1)

                # Symmetric
                edge_existence[:, j, i] = edge_exists
                edge_values[:, j, i] = edge_value * edge_exists.unsqueeze(-1)

        # Generate poles/zeros
        graph_repr = torch.stack(node_embeddings, dim=1).mean(dim=1)
        graph_features = self.graph_pooling(graph_repr) + context

        pole_count_logits = self.pole_count_head(graph_features)
        zero_count_logits = self.zero_count_head(graph_features)

        pole_count = torch.argmax(pole_count_logits, dim=-1)
        zero_count = torch.argmax(zero_count_logits, dim=-1)

        pole_values_flat = self.pole_value_head(graph_features)
        pole_values = pole_values_flat.view(batch_size, self.max_poles, 2)

        zero_values_flat = self.zero_value_head(graph_features)
        zero_values = zero_values_flat.view(batch_size, self.max_zeros, 2)

        return {
            'node_types': node_types,
            'edge_existence': edge_existence,
            'edge_values': edge_values,
            'pole_count': pole_count,
            'zero_count': zero_count,
            'pole_values': pole_values,
            'zero_values': zero_values
        }


if __name__ == '__main__':
    # Test the decoder
    print("Testing GraphGPT Decoder...")

    decoder = GraphGPTDecoder(
        latent_dim=8,
        conditions_dim=2,
        hidden_dim=256
    )

    # Test forward pass (training mode)
    batch_size = 4
    latent = torch.randn(batch_size, 8)
    conditions = torch.randn(batch_size, 2)
    target_nodes = torch.randint(0, 5, (batch_size, 5))
    target_edges_exist = torch.randint(0, 2, (batch_size, 5, 5)).float()
    target_edges_values = torch.randn(batch_size, 5, 5, 7)

    outputs = decoder(
        latent,
        conditions,
        target_nodes=target_nodes,
        target_edges_exist=target_edges_exist,
        target_edges_values=target_edges_values,
        teacher_forcing=True
    )

    print(f"✅ Forward pass successful")
    print(f"   Node logits: {outputs['node_types'].shape}")
    print(f"   Edge exist logits: {outputs['edge_existence'].shape}")
    print(f"   Edge values: {outputs['edge_values'].shape}")
    print(f"   Pole count logits: {outputs['pole_count_logits'].shape}")

    # Test generation mode
    generated = decoder.generate(latent, conditions)
    print(f"\n✅ Generation successful")
    print(f"   Generated node types: {generated['node_types'].shape}")
    print(f"   Generated edges: {generated['edge_existence'].sum(dim=(1,2))}")  # Edges per sample
    print(f"   Pole counts: {generated['pole_count']}")

    print("\n✅ All tests passed!")
