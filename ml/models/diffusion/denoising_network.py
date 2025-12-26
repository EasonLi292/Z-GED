"""
Denoising network for diffusion-based circuit graph generation.

Implements DiffusionGraphTransformer that predicts clean circuit graphs
from noisy inputs, conditioned on timestep, latent code, and specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .time_embedding import TimeEmbeddingMLP
from .graph_transformer import GraphTransformerStack, GraphPooling


class DiffusionGraphTransformer(nn.Module):
    """
    Main denoising network for diffusion-based circuit generation.

    Architecture:
        1. Time embedding (sinusoidal + MLP)
        2. Latent code projection
        3. Condition (specifications) embedding
        4. Node/edge input projections
        5. Graph Transformer stack (6 layers)
        6. Prediction heads:
           - Node types (5 classes: GND, VIN, VOUT, INTERNAL, MASK)
           - Edge existence (binary per pair)
           - Edge values (C, G, L_inv + masks)
           - Pole/zero counts
           - Pole/zero values

    Args:
        hidden_dim: Hidden dimension for transformer (default: 256)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        latent_dim: Dimension of latent code from encoder (default: 8)
        conditions_dim: Dimension of specifications (default: 2)
        max_nodes: Maximum number of nodes (default: 5)
        max_poles: Maximum number of poles (default: 4)
        max_zeros: Maximum number of zeros (default: 4)
        dropout: Dropout probability (default: 0.1)
        timesteps: Number of diffusion timesteps (default: 1000)

    Example:
        >>> model = DiffusionGraphTransformer(hidden_dim=256, num_layers=6)
        >>> # Noisy inputs at timestep t
        >>> noisy_nodes = torch.randn(4, 5, 5)  # [batch, max_nodes, node_dim]
        >>> noisy_edges = torch.randn(4, 5, 5, 7)  # [batch, max_nodes, max_nodes, edge_dim]
        >>> t = torch.randint(0, 1000, (4,))  # Timesteps
        >>> z = torch.randn(4, 8)  # Latent codes
        >>> c = torch.randn(4, 2)  # Conditions
        >>> outputs = model(noisy_nodes, noisy_edges, t, z, c)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        max_nodes: int = 5,
        max_poles: int = 4,
        max_zeros: int = 4,
        dropout: float = 0.1,
        timesteps: int = 1000
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.conditions_dim = conditions_dim
        self.max_nodes = max_nodes
        self.max_poles = max_poles
        self.max_zeros = max_zeros
        self.timesteps = timesteps

        # Node type constants
        self.num_node_types = 5  # GND, VIN, VOUT, INTERNAL, MASK

        # Edge feature dimension (C, G, L_inv + 4 masks)
        self.edge_feature_dim = 7

        # ==================================================================
        # Conditioning Networks
        # ==================================================================

        # Time embedding (sinusoidal + MLP)
        self.time_embed = TimeEmbeddingMLP(
            embedding_dim=128,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )

        # Latent code projection
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Condition (specifications) embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(conditions_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # ==================================================================
        # Input Projections
        # ==================================================================

        # Node feature projection (one-hot node type -> hidden_dim)
        self.node_input_proj = nn.Sequential(
            nn.Linear(self.num_node_types, hidden_dim),
            nn.SiLU()
        )

        # Edge feature projection
        # We'll create pairwise edge embeddings from edge features
        self.edge_input_proj = nn.Sequential(
            nn.Linear(self.edge_feature_dim, hidden_dim // 2),
            nn.SiLU()
        )

        # ==================================================================
        # Graph Transformer Stack
        # ==================================================================

        self.transformer = GraphTransformerStack(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=4 * hidden_dim,
            dropout=dropout,
            use_edge_features=True,
            edge_feature_dim=hidden_dim // 2  # From edge_input_proj
        )

        # ==================================================================
        # Prediction Heads
        # ==================================================================

        # Node type prediction head
        self.node_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_node_types)
        )

        # Edge existence prediction head
        # Takes concatenated features from two nodes
        self.edge_exist_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary classification
        )

        # Edge value prediction head (continuous: C, G, L_inv)
        self.edge_value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7)  # 3 values + 4 masks
        )

        # ==================================================================
        # Pole/Zero Prediction Heads
        # ==================================================================

        # Graph-level pooling for poles/zeros prediction
        self.graph_pooling = GraphPooling(pooling_type='attention', hidden_dim=hidden_dim)

        # Pole count prediction (0-4)
        self.pole_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poles + 1)  # 0, 1, 2, 3, 4
        )

        # Zero count prediction (0-4)
        self.zero_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_zeros + 1)
        )

        # Pole value prediction (real, imag for each pole)
        # We'll predict all max_poles poles and use count to mask
        self.pole_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_poles * 2)  # (real, imag) * max_poles
        )

        # Zero value prediction
        self.zero_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_zeros * 2)
        )

    def forward(
        self,
        noisy_nodes: torch.Tensor,
        noisy_edges: torch.Tensor,
        t: torch.Tensor,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict clean circuit from noisy inputs.

        Args:
            noisy_nodes: Noisy node features [batch_size, max_nodes, num_node_types]
                        (one-hot or soft categorical)
            noisy_edges: Noisy edge features [batch_size, max_nodes, max_nodes, edge_feature_dim]
            t: Timesteps [batch_size]
            latent_code: Latent code from encoder [batch_size, latent_dim]
            conditions: Specifications [batch_size, conditions_dim]
            node_mask: Optional mask for valid nodes [batch_size, max_nodes]

        Returns:
            outputs: Dictionary containing:
                - 'node_types': Logits [batch_size, max_nodes, num_node_types]
                - 'edge_existence': Logits [batch_size, max_nodes, max_nodes]
                - 'edge_values': Values [batch_size, max_nodes, max_nodes, 7]
                - 'pole_count_logits': Logits [batch_size, max_poles + 1]
                - 'zero_count_logits': Logits [batch_size, max_zeros + 1]
                - 'pole_values': Values [batch_size, max_poles, 2]
                - 'zero_values': Values [batch_size, max_zeros, 2]
        """
        batch_size = noisy_nodes.shape[0]
        device = noisy_nodes.device

        # ==================================================================
        # 1. Compute conditioning embeddings
        # ==================================================================

        # Time embedding [batch_size, hidden_dim]
        time_emb = self.time_embed(t)

        # Latent code projection [batch_size, hidden_dim]
        latent_emb = self.latent_proj(latent_code)

        # Condition embedding [batch_size, hidden_dim]
        cond_emb = self.condition_embed(conditions)

        # Combine conditioning (additive fusion)
        context = time_emb + latent_emb + cond_emb  # [batch_size, hidden_dim]

        # ==================================================================
        # 2. Project input node and edge features
        # ==================================================================

        # Project nodes to hidden dimension
        node_features = self.node_input_proj(noisy_nodes)  # [batch, max_nodes, hidden_dim]

        # Add context to each node
        node_features = node_features + context.unsqueeze(1)  # Broadcast across nodes

        # Project edge features
        edge_features = self.edge_input_proj(noisy_edges)  # [batch, max_nodes, max_nodes, hidden_dim//2]

        # ==================================================================
        # 3. Graph Transformer processing
        # ==================================================================

        # Create attention mask from node_mask if provided
        if node_mask is not None:
            # [batch, max_nodes] -> [batch, max_nodes, max_nodes]
            # attention_mask[i, j, k] = 1 if nodes j and k are both valid
            attention_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        else:
            attention_mask = None

        # Process through transformer
        node_features = self.transformer(
            node_features,
            edge_features=edge_features,
            attention_mask=attention_mask
        )  # [batch, max_nodes, hidden_dim]

        # ==================================================================
        # 4. Node-level predictions
        # ==================================================================

        # Predict node types
        node_type_logits = self.node_type_head(node_features)  # [batch, max_nodes, num_node_types]

        # ==================================================================
        # 5. Edge-level predictions
        # ==================================================================

        # Create pairwise node representations for edges
        # [batch, max_nodes, max_nodes, hidden_dim * 2]
        node_i = node_features.unsqueeze(2).expand(-1, -1, self.max_nodes, -1)
        node_j = node_features.unsqueeze(1).expand(-1, self.max_nodes, -1, -1)
        edge_node_pairs = torch.cat([node_i, node_j], dim=-1)  # [batch, max_nodes, max_nodes, 2*hidden_dim]

        # Predict edge existence
        edge_exist_logits = self.edge_exist_head(edge_node_pairs).squeeze(-1)  # [batch, max_nodes, max_nodes]

        # Predict edge values (continuous)
        edge_value_pred = self.edge_value_head(edge_node_pairs)  # [batch, max_nodes, max_nodes, 7]

        # ==================================================================
        # 6. Graph-level predictions (poles/zeros)
        # ==================================================================

        # Pool node features to graph-level representation
        graph_features = self.graph_pooling(node_features, node_mask)  # [batch, hidden_dim]

        # Add context to graph features
        graph_features = graph_features + context

        # Predict pole count
        pole_count_logits = self.pole_count_head(graph_features)  # [batch, max_poles + 1]

        # Predict zero count
        zero_count_logits = self.zero_count_head(graph_features)  # [batch, max_zeros + 1]

        # Predict pole values (all max_poles, will be masked by count)
        pole_values_flat = self.pole_value_head(graph_features)  # [batch, max_poles * 2]
        pole_values = pole_values_flat.view(batch_size, self.max_poles, 2)  # [batch, max_poles, 2]

        # Predict zero values
        zero_values_flat = self.zero_value_head(graph_features)  # [batch, max_zeros * 2]
        zero_values = zero_values_flat.view(batch_size, self.max_zeros, 2)  # [batch, max_zeros, 2]

        # ==================================================================
        # 7. Return outputs
        # ==================================================================

        outputs = {
            # Node predictions
            'node_types': node_type_logits,  # [batch, max_nodes, num_node_types]

            # Edge predictions
            'edge_existence': edge_exist_logits,  # [batch, max_nodes, max_nodes]
            'edge_values': edge_value_pred,  # [batch, max_nodes, max_nodes, 7]

            # Pole/zero count predictions
            'pole_count_logits': pole_count_logits,  # [batch, max_poles + 1]
            'zero_count_logits': zero_count_logits,  # [batch, max_zeros + 1]

            # Pole/zero value predictions
            'pole_values': pole_values,  # [batch, max_poles, 2]
            'zero_values': zero_values,  # [batch, max_zeros, 2]
        }

        return outputs

    def predict_noise(
        self,
        noisy_nodes: torch.Tensor,
        noisy_edges: torch.Tensor,
        noisy_poles: torch.Tensor,
        noisy_zeros: torch.Tensor,
        t: torch.Tensor,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict noise (epsilon) for continuous variables.

        This is used during training with the noise prediction objective.
        For discrete variables (node types, counts), we predict the clean value directly.

        Args:
            noisy_nodes: Noisy node types [batch, max_nodes, num_node_types]
            noisy_edges: Noisy edge values [batch, max_nodes, max_nodes, 7]
            noisy_poles: Noisy pole values [batch, max_poles, 2]
            noisy_zeros: Noisy zero values [batch, max_zeros, 2]
            t: Timesteps [batch]
            latent_code: Latent code [batch, latent_dim]
            conditions: Specifications [batch, conditions_dim]
            node_mask: Optional node mask [batch, max_nodes]

        Returns:
            predictions: Dictionary containing noise predictions
        """
        # Get predictions
        outputs = self.forward(
            noisy_nodes, noisy_edges, t, latent_code, conditions, node_mask
        )

        # For discrete variables (node types, counts), output is direct prediction
        # For continuous variables (edge values, poles, zeros), we need to convert
        # predictions to noise estimates

        # This will be used by the training loop to compute losses
        return outputs


class ConditionalDiffusionDecoder(nn.Module):
    """
    Complete diffusion decoder combining denoising network with sampling.

    This wraps DiffusionGraphTransformer and provides high-level interface
    for training and generation.

    Args:
        Same as DiffusionGraphTransformer

    Example:
        >>> decoder = ConditionalDiffusionDecoder(hidden_dim=256)
        >>> # Training
        >>> outputs = decoder.forward_training(batch, t, latent, conditions)
        >>> # Generation
        >>> circuit = decoder.generate(latent, conditions, num_steps=50)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        latent_dim: int = 8,
        conditions_dim: int = 2,
        max_nodes: int = 5,
        max_poles: int = 4,
        max_zeros: int = 4,
        dropout: float = 0.1,
        timesteps: int = 1000
    ):
        super().__init__()

        self.denoising_network = DiffusionGraphTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
            conditions_dim=conditions_dim,
            max_nodes=max_nodes,
            max_poles=max_poles,
            max_zeros=max_zeros,
            dropout=dropout,
            timesteps=timesteps
        )

        self.timesteps = timesteps
        self.max_nodes = max_nodes
        self.max_poles = max_poles
        self.max_zeros = max_zeros

    def forward(
        self,
        noisy_nodes: torch.Tensor,
        noisy_edges: torch.Tensor,
        t: torch.Tensor,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (delegates to denoising network)."""
        return self.denoising_network(
            noisy_nodes, noisy_edges, t, latent_code, conditions, node_mask
        )

    def generate(
        self,
        latent_code: torch.Tensor,
        conditions: torch.Tensor,
        num_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuit from latent code and conditions.

        This will be implemented in reverse_process.py to use DDPM/DDIM samplers.

        Args:
            latent_code: Latent code [batch, latent_dim]
            conditions: Specifications [batch, conditions_dim]
            num_steps: Number of denoising steps (50 for DDIM, 1000 for DDPM)
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance scale

        Returns:
            circuit: Generated circuit components
        """
        raise NotImplementedError("Generation will be implemented in reverse_process.py")
