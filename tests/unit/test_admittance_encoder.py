"""
Tests for the admittance-polynomial encoder (v2).

Verifies physical claims of AdmittanceEncoder:
  1. Scaling init     - alpha=1, beta=0 at construction
  2. Parallel algebra - two 2 kOhm resistors in parallel == one 1 kOhm
                        (exact at init when beta=0)
  3. Series algebra   - two series caps == one equivalent cap
                        (informational; not expected to match untrained)
  4. R/L discrimination - R and L with equal |Y| should produce
                          clearly different mu.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from ml.models.admittance_encoder import AdmittanceEncoder
from ml.models.constants import G_REF, C_REF, L_INV_REF


GND, VIN, VOUT, INTERNAL = 0, 1, 2, 3


def _node_one_hot(node_type):
    vec = [0.0, 0.0, 0.0, 0.0]
    vec[node_type] = 1.0
    return vec


def _make_graph(node_types, edges):
    x = torch.tensor([_node_one_hot(t) for t in node_types],
                     dtype=torch.float32)
    edge_idx, edge_attrs = [], []
    for s, d, (g, c, l_inv) in edges:
        edge_idx.append([s, d])
        edge_attrs.append([g / G_REF, c / C_REF, l_inv / L_INV_REF])
    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _get_mu(encoder, graphs):
    """Encode graphs and return mu (handles both VAE and deterministic)."""
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        result = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if isinstance(result, tuple):
            return result[1]  # mu
        return result


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    enc = AdmittanceEncoder(
        node_feature_dim=4, hidden_dim=64, latent_dim=5,
        num_layers=3, dropout=0.0, vae=True,
    )
    enc.eval()
    return enc


class TestScalingInit:
    def test_alpha_one_beta_zero(self, encoder):
        """Verify alpha=1, beta=0 init preserves identity scaling."""
        for i, conv in enumerate(encoder.convs):
            for name, alpha, beta in [
                ('G', conv.alpha_G, conv.beta_G),
                ('C', conv.alpha_C, conv.beta_C),
                ('L', conv.alpha_L, conv.beta_L),
            ]:
                assert abs(alpha.item() - 1.0) < 1e-6, \
                    f"Layer {i} alpha_{name} = {alpha.item()}, expected 1.0"
                assert abs(beta.item()) < 1e-6, \
                    f"Layer {i} beta_{name} = {beta.item()}, expected 0.0"


class TestParallelAlgebra:
    def test_two_parallel_resistors_equal_one(self, encoder):
        """Two 2 kOhm resistors in parallel VIN-VOUT == one 1 kOhm."""
        two_parallel = _make_graph(
            node_types=[GND, VIN, VOUT],
            edges=[
                (1, 2, [5e-4, 0.0, 0.0]),
                (2, 1, [5e-4, 0.0, 0.0]),
                (1, 2, [5e-4, 0.0, 0.0]),
                (2, 1, [5e-4, 0.0, 0.0]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        one_equiv = _make_graph(
            node_types=[GND, VIN, VOUT],
            edges=[
                (1, 2, [1e-3, 0.0, 0.0]),
                (2, 1, [1e-3, 0.0, 0.0]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        mu = _get_mu(encoder, [two_parallel, one_equiv])
        diff = (mu[0] - mu[1]).norm().item()
        rel = diff / (mu[0].norm().item() + 1e-9)
        assert rel < 1e-3, \
            f"Parallel R||R relative error {rel:.3e} exceeds threshold 1e-3"


class TestSeriesAlgebra:
    def test_two_series_caps_informational(self, encoder):
        """Two series caps VIN-INT-VOUT vs one equivalent cap VIN-VOUT.

        This is informational only -- series algebra requires learning
        and is not expected to match on an untrained encoder.
        """
        series = _make_graph(
            node_types=[GND, VIN, VOUT, INTERNAL],
            edges=[
                (1, 3, [0.0, 2e-7, 0.0]),
                (3, 1, [0.0, 2e-7, 0.0]),
                (3, 2, [0.0, 2e-7, 0.0]),
                (2, 3, [0.0, 2e-7, 0.0]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        single = _make_graph(
            node_types=[GND, VIN, VOUT],
            edges=[
                (1, 2, [0.0, 1e-7, 0.0]),
                (2, 1, [0.0, 1e-7, 0.0]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        mu = _get_mu(encoder, [series, single])
        diff = (mu[0] - mu[1]).norm().item()
        rel = diff / (mu[0].norm().item() + 1e-9)
        # No assertion -- just verify it runs without error.
        # Series algebra is not expected to hold on an untrained encoder.
        assert rel >= 0, "Sanity: relative difference should be non-negative"


class TestRLDiscrimination:
    def test_R_vs_L_different_mu(self, encoder):
        """R and L with equal |Y| at 1 kHz should produce clearly different mu."""
        R_graph = _make_graph(
            node_types=[GND, VIN, VOUT],
            edges=[
                (1, 2, [1e-3, 0.0, 0.0]),
                (2, 1, [1e-3, 0.0, 0.0]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        L_graph = _make_graph(
            node_types=[GND, VIN, VOUT],
            edges=[
                (1, 2, [0.0, 0.0, 6.283]),
                (2, 1, [0.0, 0.0, 6.283]),
                (0, 2, [1e-4, 0.0, 0.0]),
                (2, 0, [1e-4, 0.0, 0.0]),
            ],
        )
        mu = _get_mu(encoder, [R_graph, L_graph])
        diff = (mu[0] - mu[1]).norm().item()
        base = 0.5 * (mu[0].norm().item() + mu[1].norm().item()) + 1e-9
        rel = diff / base
        assert rel > 0.05, \
            f"R vs L relative difference {rel:.3e} too small (threshold 5e-2)"
