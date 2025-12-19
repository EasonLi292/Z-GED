"""
Hybrid Decoder for GraphVAE.

Decodes latent vectors back to circuit graphs using a template-based approach:
    1. Topology: Classify filter type (6-way) → use template adjacency
    2. Component values: Predict edge features (impedance values)
    3. Poles/zeros: Optionally predict for validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Dict, Optional
import networkx as nx


# Template graph structures for each filter type
# Format: (num_nodes, edge_list)
CIRCUIT_TEMPLATES = {
    'low_pass': {
        'num_nodes': 3,
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],  # GND, VIN, VOUT
        'num_components': 2,  # R, C
        'node_types': [0, 1, 2]  # [GND, VIN, VOUT]
    },
    'high_pass': {
        'num_nodes': 3,
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],
        'num_components': 2,  # C, R
        'node_types': [0, 1, 2]
    },
    'band_pass': {
        'num_nodes': 4,
        'edges': [(0, 2), (1, 2), (2, 3), (3, 0), (2, 0), (2, 1), (0, 3)],
        'num_components': 3,  # R, L, C
        'node_types': [0, 1, 2, 3]  # [GND, VIN, VOUT, INTERNAL]
    },
    'band_stop': {
        'num_nodes': 5,
        'edges': [(0, 2), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (2, 0), (2, 1), (3, 2), (0, 4)],
        'num_components': 6,
        'node_types': [0, 1, 2, 3, 3]  # [GND, VIN, VOUT, INTERNAL, INTERNAL]
    },
    'rlc_series': {
        'num_nodes': 5,
        'edges': [(0, 2), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (2, 0), (2, 1), (3, 2), (0, 4)],
        'num_components': 4,  # R, L, C, R_series
        'node_types': [0, 1, 2, 3, 3]
    },
    'rlc_parallel': {
        'num_nodes': 4,
        'edges': [(0, 2), (1, 2), (2, 3), (3, 0), (2, 0), (2, 1), (0, 3)],
        'num_components': 4,  # R, L, C, R_parallel
        'node_types': [0, 1, 2, 3]
    }
}

# Filter type ordering (must match dataset)
FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']


class HybridDecoder(nn.Module):
    """
    Hybrid decoder using fixed topology templates and continuous value prediction.

    Architecture:
        Stage 1: Topology classification
            z_topo → MLP → logits [6] → softmax → filter_type
            → Retrieve template (nodes, edges)

        Stage 2: Component value prediction
            z_values → MLP → edge_features [num_edges × 3]
            → Apply to template edges

        Stage 3 (Optional): Poles/zeros prediction
            z_pz → MLP → poles, zeros (for validation)

    Args:
        latent_dim: Total latent dimension (default: 24)
        edge_feature_dim: Edge feature dimension (default: 3)
        hidden_dim: Hidden dimension for MLPs (default: 128)
        max_nodes: Maximum number of nodes in any template (default: 5)
        max_edges: Maximum number of edges in any template (default: 10)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        latent_dim: int = 24,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        max_nodes: int = 5,
        max_edges: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        assert latent_dim % 3 == 0, "Latent dim must be divisible by 3"

        self.latent_dim = latent_dim
        self.latent_dim_per_branch = latent_dim // 3
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        # Stage 1: Topology classification (from z_topo)
        self.topo_classifier = nn.Sequential(
            nn.Linear(self.latent_dim_per_branch, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(FILTER_TYPES))
        )

        # Stage 2: Component value prediction (from z_values)
        # Predict a fixed-size feature vector, then slice for actual edges
        self.value_decoder = nn.Sequential(
            nn.Linear(self.latent_dim_per_branch, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_edges * edge_feature_dim)
        )

        # Stage 3: Poles/zeros prediction (from z_pz) - optional for validation
        self.pz_decoder_enabled = True

        # Predict maximum 2 poles and 2 zeros (most complex case)
        self.pole_decoder = nn.Sequential(
            nn.Linear(self.latent_dim_per_branch, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * 2)  # 2 poles × [real, imag]
        )

        self.zero_decoder = nn.Sequential(
            nn.Linear(self.latent_dim_per_branch, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * 2)  # 2 zeros × [real, imag]
        )

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent vectors to circuit graphs.

        Args:
            z: Latent vector [B, latent_dim]
            temperature: Temperature for Gumbel-Softmax (default: 1.0)
            hard: Whether to use hard (one-hot) sampling (default: False)

        Returns:
            Dictionary with:
                - 'topo_logits': Topology logits [B, 6]
                - 'topo_probs': Topology probabilities [B, 6]
                - 'edge_features': Predicted edge features [B, max_edges, 3]
                - 'poles': Predicted poles [B, 2, 2]
                - 'zeros': Predicted zeros [B, 2, 2]
                - 'graphs': List of PyG Data objects (if hard=True)
        """
        batch_size = z.size(0)

        # Split latent vector
        z_topo = z[:, :self.latent_dim_per_branch]
        z_values = z[:, self.latent_dim_per_branch:2*self.latent_dim_per_branch]
        z_pz = z[:, 2*self.latent_dim_per_branch:]

        # Stage 1: Topology classification
        topo_logits = self.topo_classifier(z_topo)  # [B, 6]

        if self.training:
            # Use Gumbel-Softmax for differentiable sampling during training
            topo_probs = F.gumbel_softmax(topo_logits, tau=temperature, hard=hard, dim=-1)
        else:
            # Use standard softmax during inference
            topo_probs = F.softmax(topo_logits, dim=-1)

        # Stage 2: Component value prediction
        edge_features_flat = self.value_decoder(z_values)  # [B, max_edges × 3]
        edge_features = edge_features_flat.view(batch_size, self.max_edges, self.edge_feature_dim)

        # Clamp to reasonable ranges (after denormalization in loss function)
        # For now, keep raw predictions

        # Stage 3: Poles/zeros prediction
        poles_flat = self.pole_decoder(z_pz)  # [B, 4]
        zeros_flat = self.zero_decoder(z_pz)  # [B, 4]

        poles = poles_flat.view(batch_size, 2, 2)  # [B, 2 poles, [real, imag]]
        zeros = zeros_flat.view(batch_size, 2, 2)  # [B, 2 zeros, [real, imag]]

        output = {
            'topo_logits': topo_logits,
            'topo_probs': topo_probs,
            'edge_features': edge_features,
            'poles': poles,
            'zeros': zeros
        }

        # If hard sampling, construct actual graphs
        if hard or not self.training:
            graphs = self._construct_graphs(topo_probs, edge_features)
            output['graphs'] = graphs

        return output

    def _construct_graphs(
        self,
        topo_probs: torch.Tensor,
        edge_features: torch.Tensor
    ) -> List[Data]:
        """
        Construct PyG Data objects from topology and edge features.

        Args:
            topo_probs: Topology probabilities [B, 6]
            edge_features: Edge features [B, max_edges, 3]

        Returns:
            List of PyG Data objects
        """
        batch_size = topo_probs.size(0)
        graphs = []

        for i in range(batch_size):
            # Select filter type (argmax)
            filter_idx = topo_probs[i].argmax().item()
            filter_type = FILTER_TYPES[filter_idx]
            template = CIRCUIT_TEMPLATES[filter_type]

            # Get template structure
            num_nodes = template['num_nodes']
            edges = template['edges']
            node_types = template['node_types']

            # Create node features (one-hot encoding of node types)
            # Types: [GND=0, VIN=1, VOUT=2, INTERNAL=3]
            x = torch.zeros(num_nodes, 4, device=edge_features.device)
            for node_idx, node_type in enumerate(node_types):
                x[node_idx, node_type] = 1.0

            # Create edge index
            edge_index = torch.tensor(edges, dtype=torch.long, device=edge_features.device).t()

            # Get edge features (use first num_edges from predictions)
            num_edges = len(edges)
            edge_attr = edge_features[i, :num_edges, :]

            # Create PyG Data object
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                filter_type=filter_type
            )

            graphs.append(graph)

        return graphs

    def decode_to_circuits(
        self,
        z: torch.Tensor,
        denormalize_fn: Optional[callable] = None
    ) -> List[Dict]:
        """
        Decode latent vectors to full circuit dictionaries.

        Args:
            z: Latent vector [B, latent_dim]
            denormalize_fn: Optional function to denormalize edge features

        Returns:
            List of circuit dictionaries with graph structure and metadata
        """
        output = self.forward(z, hard=True)
        graphs = output['graphs']
        poles = output['poles']
        zeros = output['zeros']

        circuits = []
        for i, graph in enumerate(graphs):
            # Denormalize edge features if function provided
            edge_attr = graph.edge_attr
            if denormalize_fn is not None:
                edge_attr = denormalize_fn(edge_attr)

            circuit = {
                'filter_type': graph.filter_type,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'node_features': graph.x.cpu().numpy(),
                'edge_index': graph.edge_index.cpu().numpy(),
                'edge_features': edge_attr.cpu().numpy(),
                'predicted_poles': poles[i].cpu().numpy(),
                'predicted_zeros': zeros[i].cpu().numpy()
            }

            circuits.append(circuit)

        return circuits


class TemplateDecoder(nn.Module):
    """
    Simpler template-based decoder that directly maps latent codes to templates.

    This is an alternative to HybridDecoder that's more interpretable but less flexible.

    Args:
        latent_dim: Latent dimension
        num_templates: Number of filter type templates (default: 6)
    """

    def __init__(
        self,
        latent_dim: int = 24,
        num_templates: int = 6
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_templates = num_templates

        # Learn a prototype latent vector for each template
        self.template_prototypes = nn.Parameter(
            torch.randn(num_templates, latent_dim)
        )

        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10 * 3)  # max_edges × 3
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode by finding nearest template prototype.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Dictionary with decoded outputs
        """
        batch_size = z.size(0)

        # Compute distances to all prototypes
        # [B, num_templates]
        distances = torch.cdist(z, self.template_prototypes, p=2)

        # Softmax over distances (closer = higher prob)
        topo_logits = -distances  # Negate so closer = larger logit
        topo_probs = F.softmax(topo_logits, dim=-1)

        # Decode values
        edge_features_flat = self.value_decoder(z)
        edge_features = edge_features_flat.view(batch_size, 10, 3)

        return {
            'topo_logits': topo_logits,
            'topo_probs': topo_probs,
            'edge_features': edge_features
        }
