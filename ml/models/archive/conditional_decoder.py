"""
Conditional Variable-Length Decoder for GraphVAE.

Extends VariableLengthDecoder to accept conditions (specifications) as input.
Conditions modulate the decoding process to generate circuits matching specs.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .variable_decoder import VariableLengthDecoder


class ConditionalVariableLengthDecoder(VariableLengthDecoder):
    """
    Conditional decoder that generates circuits matching specified conditions.

    Architecture:
        1. Project conditions to each latent branch
        2. Add condition projections to latent codes
        3. Decode as usual with condition-augmented latents

    This allows the decoder to generate circuits that match the specified
    cutoff frequency, Q factor, etc.

    Args:
        conditions_dim: Dimension of condition vector (default: 2)
        All other args same as VariableLengthDecoder
    """

    def __init__(
        self,
        latent_dim: int = 24,
        hidden_dim: int = 128,
        topo_latent_dim: int = 8,
        values_latent_dim: int = 8,
        pz_latent_dim: int = 8,
        max_nodes: int = 5,
        max_edges: int = 10,
        max_poles: int = 4,
        max_zeros: int = 4,
        num_filter_types: int = 6,
        # Conditional parameters
        conditions_dim: int = 2
    ):
        # Initialize base decoder
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            topo_latent_dim=topo_latent_dim,
            values_latent_dim=values_latent_dim,
            pz_latent_dim=pz_latent_dim,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_poles=max_poles,
            max_zeros=max_zeros
        )

        self.conditions_dim = conditions_dim

        # Condition projection networks (project conditions to latent space)
        # These learn how to map specifications to latent representations

        # Project to topology latent
        self.topo_condition_proj = nn.Sequential(
            nn.Linear(conditions_dim, topo_latent_dim),
            nn.Tanh()  # Bounded output
        )

        # Project to values latent
        self.values_condition_proj = nn.Sequential(
            nn.Linear(conditions_dim, values_latent_dim),
            nn.Tanh()
        )

        # Project to poles/zeros latent
        self.pz_condition_proj = nn.Sequential(
            nn.Linear(conditions_dim, pz_latent_dim),
            nn.Tanh()
        )

    def forward(
        self,
        z: torch.Tensor,
        hard: bool = False,
        gt_filter_type: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional conditions.

        Args:
            z: Latent code [B, latent_dim]
            hard: Whether to use hard (argmax) or soft (gumbel-softmax) sampling
            gt_filter_type: Ground truth filter type for teacher forcing (optional)
            conditions: Condition tensor [B, conditions_dim] (optional)
                       If None, no conditioning is applied

        Returns:
            Dictionary with decoder outputs (same as base decoder)
        """
        batch_size = z.size(0)

        # Split latent into branches
        z_topo = z[:, :self.topo_latent_dim]
        z_values = z[:, self.topo_latent_dim:self.topo_latent_dim + self.values_latent_dim]
        z_pz = z[:, -self.pz_latent_dim:]

        # Apply condition modulation if provided
        if conditions is not None:
            # Project conditions to each latent branch
            topo_cond = self.topo_condition_proj(conditions)      # [B, topo_latent_dim]
            values_cond = self.values_condition_proj(conditions)  # [B, values_latent_dim]
            pz_cond = self.pz_condition_proj(conditions)          # [B, pz_latent_dim]

            # Add conditions to latent codes
            # This modulates the latent representation based on desired specs
            z_topo = z_topo + topo_cond
            z_values = z_values + values_cond
            z_pz = z_pz + pz_cond

        # Decode topology (filter type)
        topo_logits = self.topo_head(z_topo)  # [B, num_filter_types]

        if gt_filter_type is not None:
            # Teacher forcing: use ground truth topology
            topo_probs = gt_filter_type
        else:
            if hard:
                # Hard assignment during generation
                topo_idx = torch.argmax(topo_logits, dim=-1)
                topo_probs = torch.zeros_like(topo_logits)
                topo_probs.scatter_(1, topo_idx.unsqueeze(1), 1.0)
            else:
                # Soft assignment during training (Gumbel-Softmax)
                topo_probs = torch.nn.functional.gumbel_softmax(
                    topo_logits, tau=1.0, hard=False
                )

        # Decode component values
        values_output = self.value_decoder(z_values)  # [B, max_edges * num_component_types]

        # Reshape to [B, max_edges, num_component_types]
        num_component_types = 7  # R, C, L, has_R, has_C, has_L, is_parallel
        values_decoded = values_output.view(batch_size, self.max_edges, num_component_types)

        # Extract component values and existence masks
        R_pred = values_decoded[:, :, 0]  # [B, max_edges]
        C_pred = values_decoded[:, :, 1]
        L_pred = values_decoded[:, :, 2]

        component_exists = torch.sigmoid(values_decoded[:, :, 3:6])  # [B, max_edges, 3]
        is_parallel = torch.sigmoid(values_decoded[:, :, 6:7])  # [B, max_edges, 1]

        # Decode pole/zero counts
        pole_count_logits = self.pole_count_head(z_pz)  # [B, max_poles+1]
        zero_count_logits = self.zero_count_head(z_pz)  # [B, max_zeros+1]

        if hard:
            num_poles = pole_count_logits.argmax(dim=-1)  # [B]
            num_zeros = zero_count_logits.argmax(dim=-1)  # [B]
        else:
            # Soft counts for training (expectation)
            pole_probs = torch.softmax(pole_count_logits, dim=-1)
            zero_probs = torch.softmax(zero_count_logits, dim=-1)
            num_poles = (pole_probs * torch.arange(self.max_poles + 1, device=z.device)).sum(dim=-1)
            num_zeros = (zero_probs * torch.arange(self.max_zeros + 1, device=z.device)).sum(dim=-1)

        # Decode pole/zero values
        poles_all = self.pole_decoder(z_pz).view(batch_size, self.max_poles, 2)  # [B, max_poles, 2]
        zeros_all = self.zero_decoder(z_pz).view(batch_size, self.max_zeros, 2)  # [B, max_zeros, 2]

        return {
            # Topology
            'topo_logits': topo_logits,
            'topo_probs': topo_probs,

            # Component values
            'R': R_pred,
            'C': C_pred,
            'L': L_pred,
            'component_exists': component_exists,
            'is_parallel': is_parallel,

            # Pole/zero counts
            'pole_count_logits': pole_count_logits,
            'zero_count_logits': zero_count_logits,
            'num_poles': num_poles,
            'num_zeros': num_zeros,

            # Pole/zero values
            'poles_all': poles_all,
            'zeros_all': zeros_all
        }

    def generate(
        self,
        num_samples: int,
        conditions: torch.Tensor,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate circuits from conditions (without reference latent code).

        Args:
            num_samples: Number of circuits to generate
            conditions: Condition tensor [num_samples, conditions_dim]
                       or [1, conditions_dim] (broadcast to all samples)
            device: Device to generate on

        Returns:
            Dictionary of generated circuits
        """
        # Sample random latent codes from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Broadcast conditions if needed
        if conditions.size(0) == 1 and num_samples > 1:
            conditions = conditions.expand(num_samples, -1)

        # Generate with hard sampling
        outputs = self.forward(z, hard=True, conditions=conditions)

        return outputs
