"""Tests for bipartite graph conversion."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
from ml.data.bipartite_graph import (
    BipartiteCircuitGraph,
    from_pickle_circuit,
    from_spice_netlist,
)


class TestFromSpiceNetlist:
    """Test SPICE netlist parsing."""

    def test_simple_rc_lowpass(self):
        netlist = """
        R1 vin vout 1000
        C1 vout 0 1e-9
        """
        g = from_spice_netlist(netlist)
        assert g.net_nodes == ['VSS', 'VIN', 'VOUT']
        assert g.comp_nodes == ['R1', 'C1']
        assert g.comp_terminals['R1'] == ('VIN', 'VOUT')
        assert g.comp_terminals['C1'] == ('VOUT', 'VSS')
        assert g.comp_types == {'R1': 'R', 'C1': 'C'}

    def test_rlc_series(self):
        netlist = """
        R1 vin n1 500
        L1 n1 n2 0.001
        C1 n2 vout 1e-8
        R2 vout gnd 10000
        """
        g = from_spice_netlist(netlist)
        assert len(g.net_nodes) == 5  # VSS, VIN, VOUT, INTERNAL_1, INTERNAL_2
        assert len(g.comp_nodes) == 4
        assert all(g.comp_types[c] in ('R', 'C', 'L') for c in g.comp_nodes)

    def test_component_degree_always_2(self):
        netlist = """
        R1 vin vout 1000
        C1 vout gnd 1e-9
        L1 vin gnd 0.01
        """
        g = from_spice_netlist(netlist)
        g.verify_degrees()
        for comp in g.comp_nodes:
            net_a, net_b = g.comp_terminals[comp]
            assert net_a in g.net_nodes
            assert net_b in g.net_nodes

    def test_connected(self):
        netlist = """
        R1 vin vout 1000
        C1 vout gnd 1e-9
        """
        g = from_spice_netlist(netlist)
        assert g.verify_connected()

    def test_net_canonicalization(self):
        netlist = """
        R1 VIN VOUT 100
        C1 VOUT VSS 1e-9
        """
        g = from_spice_netlist(netlist)
        assert 'VSS' in g.net_nodes
        assert 'VIN' in g.net_nodes
        assert 'VOUT' in g.net_nodes

    def test_internal_net_naming(self):
        netlist = """
        R1 vin node_a 100
        R2 node_a node_b 200
        R3 node_b vout 300
        C1 vout gnd 1e-9
        """
        g = from_spice_netlist(netlist)
        assert 'INTERNAL_1' in g.net_nodes
        assert 'INTERNAL_2' in g.net_nodes

    def test_skips_comments_and_directives(self):
        netlist = """
        * This is a comment
        .subckt test vin vout
        R1 vin vout 1000
        .ends
        """
        g = from_spice_netlist(netlist)
        assert len(g.comp_nodes) == 1

    def test_directed_arcs_count(self):
        """Each component produces 4 directed arcs (2 per terminal × 2 directions)."""
        netlist = """
        R1 vin vout 100
        C1 vout gnd 1e-9
        """
        g = from_spice_netlist(netlist)
        arcs = g.directed_arcs()
        assert len(arcs) == 4 * len(g.comp_nodes)  # 4 * 2 = 8


class TestFromPickleCircuit:
    """Test pickle dataset conversion."""

    def _make_circuit(self, edges):
        """Helper: build a minimal circuit dict from edge specs.

        edges: list of (src_id, tgt_id, C, G, L_inv)
        """
        # Collect nodes
        node_ids = set()
        for src, tgt, *_ in edges:
            node_ids.add(src)
            node_ids.add(tgt)

        # Assign node features (0=GND, 1=VIN, 2=VOUT, 3+=INTERNAL)
        nodes = []
        for nid in sorted(node_ids):
            if nid == 0:
                feat = [1, 0, 0, 0]
            elif nid == 1:
                feat = [0, 1, 0, 0]
            elif nid == 2:
                feat = [0, 0, 1, 0]
            else:
                feat = [0, 0, 0, 1]
            nodes.append({'id': nid, 'features': feat})

        # Build adjacency (both directions)
        adjacency = {nid: [] for nid in sorted(node_ids)}
        for src, tgt, C, G, L_inv in edges:
            adjacency[src].append({'id': tgt, 'impedance_den': [C, G, L_inv]})
            adjacency[tgt].append({'id': src, 'impedance_den': [C, G, L_inv]})

        return {
            'graph_adj': {
                'nodes': nodes,
                'adjacency': [adjacency[nid] for nid in sorted(node_ids)],
            }
        }

    def test_simple_rc(self):
        # R between node1(VIN) and node2(VOUT), C between node2(VOUT) and node0(GND)
        circuit = self._make_circuit([
            (1, 2, 0, 0.001, 0),     # R: G=0.001 → R=1000Ω
            (2, 0, 1e-9, 0, 0),       # C: C=1e-9F
        ])
        g = from_pickle_circuit(circuit)
        assert len(g.comp_nodes) == 2
        assert g.comp_types == {'R1': 'R', 'C1': 'C'}
        assert g.verify_connected()

    def test_multi_component_edge_compound(self):
        """An edge with both R and C should produce one RC compound node."""
        circuit = self._make_circuit([
            (1, 2, 1e-9, 0.001, 0),  # RC parallel: both C and G nonzero
        ])
        g = from_pickle_circuit(circuit)
        assert len(g.comp_nodes) == 1
        assert g.comp_nodes[0] == 'RC1'
        assert g.comp_types['RC1'] == 'RC'

    def test_rlc_edge(self):
        """An edge with R, C, and L should produce one RCL compound node."""
        circuit = self._make_circuit([
            (1, 2, 1e-9, 0.001, 100),  # RCL: all three nonzero
        ])
        g = from_pickle_circuit(circuit)
        assert len(g.comp_nodes) == 1
        assert g.comp_nodes[0] == 'RCL1'
        assert g.comp_types['RCL1'] == 'RCL'


class TestBipartiteGraphMethods:
    """Test BipartiteCircuitGraph methods."""

    def test_adjacency_lists(self):
        g = BipartiteCircuitGraph(
            net_nodes=['VSS', 'VIN', 'VOUT'],
            comp_nodes=['R1', 'C1'],
            comp_terminals={'R1': ('VIN', 'VOUT'), 'C1': ('VOUT', 'VSS')},
            comp_types={'R1': 'R', 'C1': 'C'},
        )
        adj = g.adjacency_lists()
        assert 'R1' in adj['VIN']
        assert 'R1' in adj['VOUT']
        assert 'C1' in adj['VOUT']
        assert 'C1' in adj['VSS']
        assert len(adj['R1']) == 2  # VIN and VOUT

    def test_to_adjacency(self):
        g = BipartiteCircuitGraph(
            net_nodes=['VSS', 'VIN', 'VOUT'],
            comp_nodes=['R1'],
            comp_terminals={'R1': ('VIN', 'VOUT')},
            comp_types={'R1': 'R'},
        )
        adj = g.to_adjacency()
        assert len(adj['VIN']) == 1
        assert adj['VIN'][0]['net'] == 'VOUT'
        assert adj['VIN'][0]['type'] == 'R'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
