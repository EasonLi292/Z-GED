"""
Reconstruction loss for circuit graphs.

Measures how well the decoder reconstructs the original circuit:
    - Topology: Filter type classification accuracy
    - Node features: Cross-entropy (node types)
    - Edge features: MSE (impedance values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from ..models.decoder import CIRCUIT_TEMPLATES, FILTER_TYPES


class ReconstructionLoss(nn.Module):
    """
    Graph reconstruction loss combining topology, node, and edge reconstruction.

    Components:
        1. Topology loss: Cross-entropy for filter type classification
        2. Node loss: Cross-entropy for node type (GND, VIN, VOUT, INTERNAL)
        3. Edge loss: MSE for impedance features [log(C), log(G), log(L_inv)]

    Args:
        topo_weight: Weight for topology classification loss (default: 1.0)
        node_weight: Weight for node reconstruction loss (default: 0.5)
        edge_weight: Weight for edge feature reconstruction loss (default: 1.0)
    """

    def __init__(
        self,
        topo_weight: float = 1.0,
        node_weight: float = 0.5,
        edge_weight: float = 1.0
    ):
        super().__init__()

        self.topo_weight = topo_weight
        self.node_weight = node_weight
        self.edge_weight = edge_weight

        # Loss functions
        self.topo_criterion = nn.CrossEntropyLoss()
        self.node_criterion = nn.BCEWithLogitsLoss()
        self.edge_criterion = nn.MSELoss()

    def forward(
        self,
        decoder_output: Dict[str, torch.Tensor],
        target_topo: torch.Tensor,
        target_nodes: torch.Tensor = None,
        target_edges: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss.

        Args:
            decoder_output: Output from decoder with keys:
                - 'topo_logits': [B, 6] topology logits
                - 'topo_probs': [B, 6] topology probabilities
                - 'edge_features': [B, max_edges, 3] predicted edge features
            target_topo: Ground truth topology indices [B] or one-hot [B, 6]
            target_nodes: (Optional) Ground truth node features [N, 4]
            target_edges: Ground truth edge features [E, 3] or [B, max_edges, 3]

        Returns:
            loss: Total reconstruction loss (scalar)
            metrics: Dictionary of individual loss components
        """
        batch_size = decoder_output['topo_logits'].size(0)

        # 1. Topology loss
        topo_logits = decoder_output['topo_logits']  # [B, 6]

        if target_topo.dim() == 2:
            # One-hot encoding, convert to indices
            target_topo_idx = target_topo.argmax(dim=-1)
        else:
            target_topo_idx = target_topo

        loss_topo = self.topo_criterion(topo_logits, target_topo_idx)

        # Compute accuracy
        pred_topo = topo_logits.argmax(dim=-1)
        topo_accuracy = (pred_topo == target_topo_idx).float().mean()

        # 2. Node loss (optional, if templates differ from targets)
        if target_nodes is not None:
            # This is complex because templates have fixed node structures
            # For now, skip node loss as templates define correct nodes
            loss_node = torch.tensor(0.0, device=topo_logits.device)
        else:
            loss_node = torch.tensor(0.0, device=topo_logits.device)

        # 3. Edge feature loss
        if target_edges is not None:
            pred_edges = decoder_output['edge_features']  # [B, max_edges, 3]

            if target_edges.dim() == 2:
                # Need to convert to batched format
                # This is tricky - for now assume target_edges is already batched
                loss_edge = self.edge_criterion(pred_edges, target_edges)
            else:
                # Already batched [B, max_edges, 3]
                loss_edge = self.edge_criterion(pred_edges, target_edges)
        else:
            loss_edge = torch.tensor(0.0, device=topo_logits.device)

        # Total loss
        total_loss = (
            self.topo_weight * loss_topo +
            self.node_weight * loss_node +
            self.edge_weight * loss_edge
        )

        metrics = {
            'recon_total': total_loss.item(),
            'recon_topo': loss_topo.item(),
            'recon_node': loss_node.item(),
            'recon_edge': loss_edge.item(),
            'topo_accuracy': topo_accuracy.item()
        }

        return total_loss, metrics


class TemplateAwareReconstructionLoss(nn.Module):
    """
    Reconstruction loss that accounts for template-based decoder.

    Since the decoder uses fixed templates, we:
        1. Only penalize topology classification
        2. Penalize edge features within the predicted template structure

    This is more appropriate than trying to match exact graph structure.

    Args:
        topo_weight: Initial weight for topology loss (can be annealed via curriculum)
        edge_weight: Weight for edge feature loss
        use_curriculum: Whether to use curriculum learning for topology weight
        curriculum_warmup_epochs: Number of epochs to anneal topology weight
        curriculum_initial_multiplier: Initial multiplier for topology weight (e.g., 3-5)
    """

    def __init__(
        self,
        topo_weight: float = 1.0,
        edge_weight: float = 1.0,
        use_curriculum: bool = False,
        curriculum_warmup_epochs: int = 20,
        curriculum_initial_multiplier: float = 3.0
    ):
        super().__init__()

        self.base_topo_weight = topo_weight
        self.edge_weight = edge_weight

        # Curriculum learning
        self.use_curriculum = use_curriculum
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.curriculum_initial_multiplier = curriculum_initial_multiplier
        self.current_epoch = 0

        self.topo_criterion = nn.CrossEntropyLoss()
        self.edge_criterion = nn.MSELoss()

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def get_topology_weight(self) -> float:
        """Get current topology weight based on curriculum schedule."""
        if not self.use_curriculum:
            return self.base_topo_weight

        # Anneal from (base * multiplier) down to base over warmup_epochs
        if self.current_epoch >= self.curriculum_warmup_epochs:
            return self.base_topo_weight

        # Linear annealing
        progress = self.current_epoch / self.curriculum_warmup_epochs
        initial_weight = self.base_topo_weight * self.curriculum_initial_multiplier
        current_weight = initial_weight - progress * (initial_weight - self.base_topo_weight)

        return current_weight

    def _reorder_edges_to_template(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        filter_type_idx: int
    ) -> torch.Tensor:
        """
        Reorder edges to match the canonical template ordering.

        Args:
            edge_index: Edge index tensor [2, E_i]
            edge_attr: Edge features [E_i, feature_dim]
            filter_type_idx: Index of filter type (0-5)

        Returns:
            reordered_edge_attr: Edge features in canonical order [E_template, feature_dim]
                                 Pads with zeros if target has fewer edges than template
        """
        filter_type = FILTER_TYPES[filter_type_idx]
        template_edges = CIRCUIT_TEMPLATES[filter_type]['edges']

        # Create mapping from (source, target) to edge features
        edge_map = {}
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            edge_map[(src, tgt)] = edge_attr[i]

        # Reorder according to template
        reordered = []
        for template_edge in template_edges:
            if template_edge in edge_map:
                reordered.append(edge_map[template_edge])
            else:
                # Edge not in target graph, pad with zeros
                reordered.append(torch.zeros_like(edge_attr[0]))

        return torch.stack(reordered) if reordered else edge_attr

    def forward(
        self,
        decoder_output: Dict[str, torch.Tensor],
        target_filter_type: torch.Tensor,
        target_edge_attr: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_index: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute template-aware reconstruction loss with canonical edge ordering.

        Args:
            decoder_output: Decoder output
            target_filter_type: Ground truth filter type one-hot [B, 6]
            target_edge_attr: Ground truth edge features [E, edge_dim]
            edge_batch: Batch assignment for edges [E]
            edge_index: Edge connectivity [2, E] for canonical reordering

        Returns:
            loss: Total loss
            metrics: Loss components
        """
        batch_size = decoder_output['topo_logits'].size(0)

        # 1. Topology loss
        topo_logits = decoder_output['topo_logits']
        target_topo_idx = target_filter_type.argmax(dim=-1)
        loss_topo = self.topo_criterion(topo_logits, target_topo_idx)

        # Accuracy
        pred_topo = topo_logits.argmax(dim=-1)
        topo_accuracy = (pred_topo == target_topo_idx).float().mean()

        # 2. Edge feature loss with canonical ordering
        # For each graph in batch, reorder edges to match template
        pred_edges = decoder_output['edge_features']  # [B, max_edges, edge_dim]

        edge_losses = []

        for i in range(batch_size):
            # Get target edges for this graph
            edge_mask = edge_batch == i
            target_edges_i = target_edge_attr[edge_mask]  # [E_i, edge_dim]

            # Get edge index for this graph
            if edge_index is not None:
                edge_index_i = edge_index[:, edge_mask]  # [2, E_i]

                # Reorder target edges to match template canonical ordering
                target_edges_i_reordered = self._reorder_edges_to_template(
                    edge_index_i,
                    target_edges_i,
                    target_topo_idx[i].item()
                )
            else:
                # Fallback: use original ordering (no reordering)
                target_edges_i_reordered = target_edges_i

            # Get predicted edges (same number as reordered target)
            num_edges_i = target_edges_i_reordered.size(0)
            if num_edges_i > 0:
                pred_edges_i = pred_edges[i, :num_edges_i, :]  # [E_i, edge_dim]

                # MSE loss
                loss_i = F.mse_loss(pred_edges_i, target_edges_i_reordered)
                edge_losses.append(loss_i)

        loss_edge = torch.stack(edge_losses).mean() if edge_losses else torch.tensor(0.0, device=topo_logits.device)

        # Get current topology weight (may be scheduled via curriculum)
        current_topo_weight = self.get_topology_weight()

        # Total loss
        total_loss = (
            current_topo_weight * loss_topo +
            self.edge_weight * loss_edge
        )

        metrics = {
            'recon_total': total_loss.item(),
            'recon_topo': loss_topo.item(),
            'recon_edge': loss_edge.item(),
            'topo_accuracy': topo_accuracy.item(),
            'topo_weight_current': current_topo_weight
        }

        return total_loss, metrics


class EdgeFeatureReconstructionLoss(nn.Module):
    """
    Specialized loss for edge feature reconstruction.

    Handles the fact that edge features are normalized and log-scaled.
    Provides option to denormalize before computing loss.

    Args:
        use_mse: Use MSE (default) vs. L1
        denormalize: Whether to denormalize before loss computation
    """

    def __init__(
        self,
        use_mse: bool = True,
        denormalize: bool = False,
        impedance_mean: torch.Tensor = None,
        impedance_std: torch.Tensor = None
    ):
        super().__init__()

        self.use_mse = use_mse
        self.denormalize = denormalize

        if denormalize:
            assert impedance_mean is not None and impedance_std is not None
            self.register_buffer('impedance_mean', impedance_mean)
            self.register_buffer('impedance_std', impedance_std)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge feature reconstruction loss.

        Args:
            pred: Predicted edge features [*, 3]
            target: Target edge features [*, 3]

        Returns:
            loss: Scalar loss
        """
        if self.denormalize:
            # Denormalize both pred and target
            pred = pred * self.impedance_std + self.impedance_mean
            target = target * self.impedance_std + self.impedance_mean

        if self.use_mse:
            return F.mse_loss(pred, target)
        else:
            return F.l1_loss(pred, target)
