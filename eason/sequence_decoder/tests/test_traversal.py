"""Tests for Eulerian circuit generation and augmentation."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
from ml.data.bipartite_graph import BipartiteCircuitGraph, from_spice_netlist
from ml.data.traversal import (
    hierholzer,
    augment_traversals,
    verify_traversal,
    walk_to_circuit,
)


def _rc_lowpass():
    """R between VIN-VOUT, C between VOUT-VSS."""
    return BipartiteCircuitGraph(
        net_nodes=['VSS', 'VIN', 'VOUT'],
        comp_nodes=['R1', 'C1'],
        comp_terminals={'R1': ('VIN', 'VOUT'), 'C1': ('VOUT', 'VSS')},
        comp_types={'R1': 'R', 'C1': 'C'},
    )


def _rlc_bandpass():
    """R: VIN→INT1, L: INT1→VOUT, C: VOUT→VSS."""
    return BipartiteCircuitGraph(
        net_nodes=['VSS', 'VIN', 'VOUT', 'INTERNAL_1'],
        comp_nodes=['R1', 'L1', 'C1'],
        comp_terminals={
            'R1': ('VIN', 'INTERNAL_1'),
            'L1': ('INTERNAL_1', 'VOUT'),
            'C1': ('VOUT', 'VSS'),
        },
        comp_types={'R1': 'R', 'L1': 'L', 'C1': 'C'},
    )


class TestHierholzer:
    """Test Eulerian circuit finding."""

    def test_rc_lowpass_walk_valid(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        assert verify_traversal(g, walk, start='VSS')

    def test_walk_starts_and_ends_at_vss(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        assert walk[0] == 'VSS'
        assert walk[-1] == 'VSS'

    def test_walk_length_correct(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        assert len(walk) == 4 * 2 + 1  # 4K+1, K=2

    def test_walk_alternates_net_component(self):
        """Walk should alternate between net and component nodes."""
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        nets = set(g.net_nodes)
        comps = set(g.comp_nodes)
        for i, node in enumerate(walk):
            if i % 2 == 0:
                assert node in nets, f"Position {i} ({node}) should be a net"
            else:
                assert node in comps, f"Position {i} ({node}) should be a component"

    def test_rlc_bandpass_walk_valid(self):
        g = _rlc_bandpass()
        walk = hierholzer(g, start='VSS')
        assert verify_traversal(g, walk, start='VSS')
        assert len(walk) == 4 * 3 + 1  # K=3

    def test_randomized_walk_still_valid(self):
        import random
        g = _rlc_bandpass()
        rng = random.Random(42)
        walk = hierholzer(g, start='VSS', rng=rng)
        assert verify_traversal(g, walk, start='VSS')

    def test_invalid_start_raises(self):
        g = _rc_lowpass()
        with pytest.raises(ValueError, match="Start node"):
            hierholzer(g, start='NONEXISTENT')

    def test_from_spice_netlist_roundtrip(self):
        netlist = """
        R1 vin vout 1000
        C1 vout gnd 1e-9
        L1 vin gnd 0.01
        """
        g = from_spice_netlist(netlist)
        walk = hierholzer(g, start='VSS')
        assert verify_traversal(g, walk, start='VSS')
        assert len(walk) == 4 * 3 + 1


class TestAugmentation:
    """Test traversal augmentation."""

    def test_produces_multiple_distinct_walks(self):
        """Use a graph with parallel paths for more augmentation diversity."""
        # Two resistors in parallel between VIN-VOUT, plus C to VSS
        # This gives more branching at VIN and VOUT (degree 2 each from R1, R2)
        g = BipartiteCircuitGraph(
            net_nodes=['VSS', 'VIN', 'VOUT'],
            comp_nodes=['R1', 'R2', 'C1'],
            comp_terminals={
                'R1': ('VIN', 'VOUT'),
                'R2': ('VIN', 'VOUT'),
                'C1': ('VOUT', 'VSS'),
            },
            comp_types={'R1': 'R', 'R2': 'R', 'C1': 'C'},
        )
        walks = augment_traversals(g, n=5, seed=42)
        assert len(walks) >= 2  # parallel paths enable distinct walks
        # All distinct
        unique = set(tuple(w) for w in walks)
        assert len(unique) == len(walks)

    def test_all_augmented_walks_valid(self):
        g = _rlc_bandpass()
        walks = augment_traversals(g, n=10, seed=42)
        for walk in walks:
            assert verify_traversal(g, walk, start='VSS')

    def test_canonical_always_first(self):
        g = _rc_lowpass()
        walks = augment_traversals(g, n=5, seed=42)
        canonical = hierholzer(g, start='VSS', rng=None)
        assert walks[0] == canonical

    def test_single_traversal_for_minimal_circuit(self):
        """A 2-component circuit may have very few distinct Euler circuits."""
        g = _rc_lowpass()
        walks = augment_traversals(g, n=10, seed=42)
        # Should get at least 1
        assert len(walks) >= 1
        for walk in walks:
            assert verify_traversal(g, walk, start='VSS')


class TestVerifyTraversal:
    """Test traversal verification."""

    def test_empty_walk_invalid(self):
        g = _rc_lowpass()
        assert not verify_traversal(g, [], start='VSS')

    def test_wrong_start_invalid(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        walk[0] = 'VIN'  # corrupt start
        assert not verify_traversal(g, walk, start='VSS')

    def test_wrong_length_invalid(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        assert not verify_traversal(g, walk + ['VSS'], start='VSS')

    def test_valid_walk_passes(self):
        g = _rlc_bandpass()
        walk = hierholzer(g, start='VSS')
        assert verify_traversal(g, walk, start='VSS')


class TestWalkToCircuit:
    """Test parsing walks back to circuit connections."""

    def test_rc_lowpass_roundtrip(self):
        g = _rc_lowpass()
        walk = hierholzer(g, start='VSS')
        net_set = set(g.net_nodes)
        result = walk_to_circuit(walk, net_set)

        assert 'R1' in result
        assert 'C1' in result
        # R1 connects VIN and VOUT
        _, net_a, net_b = result['R1']
        assert set([net_a, net_b]) == {'VIN', 'VOUT'}
        # C1 connects VOUT and VSS
        _, net_a, net_b = result['C1']
        assert set([net_a, net_b]) == {'VOUT', 'VSS'}

    def test_component_types_preserved(self):
        g = _rlc_bandpass()
        walk = hierholzer(g, start='VSS')
        net_set = set(g.net_nodes)
        result = walk_to_circuit(walk, net_set)
        assert result['R1'][0] == 'R'
        assert result['L1'][0] == 'L'
        assert result['C1'][0] == 'C'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
