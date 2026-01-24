"""
Loss Function for Circuit Generation with Auxiliary Spec Prediction.

Losses:
1. Node type loss (cross-entropy)
2. Node count loss (cross-entropy for 3/4/5 nodes)
3. Edge-component loss (joint 8-way classification)
4. Connectivity loss (optional)
5. KL loss (VAE regularization)
6. Spec prediction loss (MSE on cutoff/Q from z[4:8]) - NEW

The spec prediction loss forces z[4:8] to encode meaningful transfer
function information, preventing posterior collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    from ml.losses.connectivity_loss import ConnectivityLoss
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ml.losses.connectivity_loss import ConnectivityLoss


class GumbelSoftmaxCircuitLoss(nn.Module):
    """
    Circuit generation loss with auxiliary spec prediction.

    Losses:
    1. Node type (cross-entropy)
    2. Node count (cross-entropy)
    3. Edge-component (8-way cross-entropy)
    4. Connectivity (optional)
    5. KL (VAE regularization)
    6. Spec prediction (MSE on cutoff/Q) - forces z[4:8] to encode transfer function
    """

    def __init__(
        self,
        node_type_weight: float = 1.0,
        node_count_weight: float = 5.0,
        edge_component_weight: float = 2.0,
        connectivity_weight: float = 5.0,
        kl_weight: float = 0.01,  # Low KL for topology separation in latent space
        spec_weight: float = 1.0,  # Weight for spec prediction loss
        use_connectivity_loss: bool = True,
        use_spec_loss: bool = True,
        pz_latent_start: int = 4,  # Start index of transfer function latent
        pz_latent_dim: int = 4     # Dimension of transfer function latent
    ):
        super().__init__()

        self.node_type_weight = node_type_weight
        self.node_count_weight = node_count_weight
        self.edge_component_weight = edge_component_weight
        self.connectivity_weight = connectivity_weight
        self.kl_weight = kl_weight
        self.spec_weight = spec_weight

        self.use_connectivity_loss = use_connectivity_loss
        self.use_spec_loss = use_spec_loss
        self.pz_latent_start = pz_latent_start
        self.pz_latent_dim = pz_latent_dim

        if use_connectivity_loss:
            self.connectivity_loss = ConnectivityLoss(
                vin_weight=10.0,
                vout_weight=5.0,
                graph_weight=3.0,
                isolated_weight=2.0
            )

        # Spec predictor: z[4:8] -> [log10(cutoff), Q]
        if use_spec_loss:
            from ml.models.spec_predictor import SpecPredictor
            self.spec_predictor = SpecPredictor(
                pz_latent_dim=pz_latent_dim,
                hidden_dim=32,
                num_layers=2
            )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        target_specs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute circuit generation loss with optional spec prediction.

        Args:
            predictions:
                - 'node_types': [batch, num_nodes, 5] logits
                - 'node_count_logits': [batch, 3]
                - 'edge_component_logits': [batch, num_nodes, num_nodes, 8]
            targets:
                - 'node_types': [batch, num_nodes]
                - 'edge_existence': [batch, num_nodes, num_nodes]
                - 'component_types': [batch, num_nodes, num_nodes]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
            latent: Sampled latent [batch, latent_dim] (used for spec prediction)
            target_specs: Target specifications [batch, 2] = [log10(cutoff), Q]
        """
        device = predictions['node_types'].device

        # 1. Node Type Loss
        node_logits = predictions['node_types']
        target_nodes = targets['node_types']

        node_logits_flat = node_logits.reshape(-1, node_logits.shape[-1])
        target_nodes_flat = target_nodes.reshape(-1)

        loss_node_type = F.cross_entropy(node_logits_flat, target_nodes_flat)

        with torch.no_grad():
            pred_nodes = torch.argmax(node_logits_flat, dim=-1)
            node_type_acc = (pred_nodes == target_nodes_flat).float().mean().item() * 100

        # 2. Node Count Loss
        if 'node_count_logits' in predictions:
            node_count_logits = predictions['node_count_logits']

            with torch.no_grad():
                target_node_counts = (target_nodes != 4).sum(dim=1)
                target_count_class = (target_node_counts - 3).clamp(0, 2).long()

            loss_node_count = F.cross_entropy(node_count_logits, target_count_class)

            with torch.no_grad():
                pred_count = torch.argmax(node_count_logits, dim=-1)
                node_count_acc = (pred_count == target_count_class).float().mean().item() * 100
        else:
            loss_node_count = torch.tensor(0.0, device=device)
            node_count_acc = 0.0

        # 3. Edge-Component Loss (8-way)
        edge_component_logits = predictions['edge_component_logits']
        target_edges = targets['edge_existence']
        target_component_types = targets['component_types']

        # Unified target: 0=no edge, 1-7=edge with component type
        target_edge_component = torch.where(
            target_edges > 0.5,
            target_component_types,
            torch.zeros_like(target_component_types)
        ).long()

        triu_mask = torch.triu(torch.ones_like(target_edges), diagonal=1)

        edge_comp_logits_flat = edge_component_logits.reshape(-1, 8)
        target_edge_comp_flat = target_edge_component.reshape(-1)
        triu_mask_flat = triu_mask.reshape(-1)

        if triu_mask_flat.sum() > 0:
            loss_per_edge = F.cross_entropy(edge_comp_logits_flat, target_edge_comp_flat, reduction='none')
            loss_edge_component = (loss_per_edge * triu_mask_flat).sum() / (triu_mask_flat.sum() + 1e-6)
        else:
            loss_edge_component = torch.tensor(0.0, device=device)

        with torch.no_grad():
            pred_edge_comp = torch.argmax(edge_comp_logits_flat, dim=-1)

            pred_has_edge = (pred_edge_comp > 0).float()
            target_has_edge = (target_edge_comp_flat > 0).float()
            edge_acc = ((pred_has_edge == target_has_edge).float() * triu_mask_flat).sum()
            edge_acc = (edge_acc / (triu_mask_flat.sum() + 1e-6)).item() * 100

            edge_mask_flat = (target_edge_comp_flat > 0).float()
            if edge_mask_flat.sum() > 0:
                comp_type_acc = ((pred_edge_comp == target_edge_comp_flat).float() * edge_mask_flat).sum()
                comp_type_acc = (comp_type_acc / (edge_mask_flat.sum() + 1e-6)).item() * 100
            else:
                comp_type_acc = 0.0

        # 4. Connectivity Loss
        if self.use_connectivity_loss:
            edge_probs = F.softmax(edge_component_logits, dim=-1)
            edge_exist_probs = torch.clamp(1.0 - edge_probs[..., 0], 1e-6, 1.0 - 1e-6)
            edge_logits = torch.log(edge_exist_probs / (1.0 - edge_exist_probs))

            loss_connectivity, metrics_connectivity = self.connectivity_loss(
                edge_logits, predictions['node_types']
            )
        else:
            loss_connectivity = torch.tensor(0.0, device=device)
            metrics_connectivity = {}

        # 5. KL Loss
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # 6. Spec Prediction Loss (forces z[4:8] to encode transfer function)
        if self.use_spec_loss and latent is not None and target_specs is not None:
            # Extract transfer function latent z[4:8]
            z_pz = latent[:, self.pz_latent_start:self.pz_latent_start + self.pz_latent_dim]

            # Predict specs from z[4:8]
            pred_specs = self.spec_predictor(z_pz)  # [batch, 2]

            # Normalize target specs for loss computation
            # target_specs[:, 0] = cutoff (Hz), convert to log10
            # target_specs[:, 1] = Q (already linear)
            target_log_cutoff = torch.log10(target_specs[:, 0].clamp(min=1.0))
            target_q = target_specs[:, 1]
            target_specs_normalized = torch.stack([target_log_cutoff, target_q], dim=1)

            # MSE loss
            loss_spec = F.mse_loss(pred_specs, target_specs_normalized)

            # Compute accuracy metrics
            with torch.no_grad():
                # Cutoff error (in decades)
                cutoff_error = (pred_specs[:, 0] - target_specs_normalized[:, 0]).abs().mean().item()
                # Q error (absolute)
                q_error = (pred_specs[:, 1] - target_specs_normalized[:, 1]).abs().mean().item()
        else:
            loss_spec = torch.tensor(0.0, device=device)
            cutoff_error = 0.0
            q_error = 0.0

        # Total
        total_loss = (
            self.node_type_weight * loss_node_type +
            self.node_count_weight * loss_node_count +
            self.edge_component_weight * loss_edge_component +
            self.connectivity_weight * loss_connectivity +
            self.kl_weight * kl_loss +
            self.spec_weight * loss_spec
        )

        metrics = {
            'loss_node_type': loss_node_type.item(),
            'loss_node_count': loss_node_count.item(),
            'loss_edge_component': loss_edge_component.item(),
            'loss_connectivity': loss_connectivity.item() if self.use_connectivity_loss else 0.0,
            'loss_kl': kl_loss.item(),
            'loss_spec': loss_spec.item() if self.use_spec_loss else 0.0,
            'node_type_acc': node_type_acc,
            'node_count_acc': node_count_acc,
            'edge_exist_acc': edge_acc,
            'component_type_acc': comp_type_acc,
            'cutoff_error_decades': cutoff_error,
            'q_error': q_error,
        }

        if self.use_connectivity_loss:
            metrics.update(metrics_connectivity)

        return total_loss, metrics


if __name__ == '__main__':
    print("Testing Circuit Loss with Spec Prediction...")

    batch_size = 2
    max_nodes = 5
    latent_dim = 8

    predictions = {
        'node_types': torch.randn(batch_size, max_nodes, 5),
        'node_count_logits': torch.randn(batch_size, 3),
        'edge_component_logits': torch.randn(batch_size, max_nodes, max_nodes, 8),
    }

    targets = {
        'node_types': torch.randint(0, 5, (batch_size, max_nodes)),
        'edge_existence': torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        'component_types': torch.randint(0, 8, (batch_size, max_nodes, max_nodes)),
    }

    # Test without spec loss
    print("\n1. Without spec loss:")
    loss_fn = GumbelSoftmaxCircuitLoss(use_connectivity_loss=False, use_spec_loss=False)
    total_loss, metrics = loss_fn(predictions, targets)

    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test with spec loss
    print("\n2. With spec loss:")
    loss_fn_spec = GumbelSoftmaxCircuitLoss(
        use_connectivity_loss=False,
        use_spec_loss=True,
        spec_weight=1.0
    )

    latent = torch.randn(batch_size, latent_dim)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    target_specs = torch.tensor([
        [10000.0, 0.707],  # 10kHz, Q=0.707
        [1000.0, 2.0],    # 1kHz, Q=2.0
    ])

    total_loss, metrics = loss_fn_spec(
        predictions, targets,
        mu=mu, logvar=logvar,
        latent=latent,
        target_specs=target_specs
    )

    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ All tests passed!")
