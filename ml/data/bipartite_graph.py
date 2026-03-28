"""
Bipartite circuit graph: nets as one partition, components as the other.

Converts from:
  - Existing pickle dataset (impedance_den edge format)
  - SPICE-like RLC netlists

Each 2-terminal R/L/C component node connects to exactly two net nodes.
The symmetric directed expansion (each undirected edge → two directed arcs)
guarantees Eulerian circuits exist for any connected circuit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# Net node roles
NET_ROLES = {'VSS', 'VIN', 'VOUT', 'VDD'}


@dataclass
class BipartiteCircuitGraph:
    """
    Bipartite graph where net nodes and component nodes form two partitions.

    Attributes:
        net_nodes: ordered list of net names (e.g. ['VSS', 'VIN', 'VOUT', 'INTERNAL_1'])
        comp_nodes: ordered list of component names (e.g. ['R1', 'C1', 'L1'])
        comp_terminals: dict mapping component name → (net_a, net_b)
        comp_types: dict mapping component name → 'R' | 'C' | 'L'
    """
    net_nodes: List[str] = field(default_factory=list)
    comp_nodes: List[str] = field(default_factory=list)
    comp_terminals: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    comp_types: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Directed arc helpers                                                #
    # ------------------------------------------------------------------ #

    def directed_arcs(self) -> List[Tuple[str, str]]:
        """
        Return all directed arcs in the symmetric directed bipartite graph.

        Each undirected incidence {net, comp} yields two arcs:
            (net → comp) and (comp → net).

        Returns:
            List of (source, target) pairs.
        """
        arcs: List[Tuple[str, str]] = []
        for comp, (net_a, net_b) in self.comp_terminals.items():
            arcs.append((net_a, comp))
            arcs.append((comp, net_a))
            arcs.append((net_b, comp))
            arcs.append((comp, net_b))
        return arcs

    def adjacency_lists(self) -> Dict[str, List[str]]:
        """
        Build adjacency lists for the symmetric directed graph.

        Returns:
            Dict mapping node → list of outgoing neighbors (with duplicates
            for multi-edges).
        """
        adj: Dict[str, List[str]] = {n: [] for n in self.net_nodes + self.comp_nodes}
        for comp, (net_a, net_b) in self.comp_terminals.items():
            adj[net_a].append(comp)
            adj[comp].append(net_a)
            adj[net_b].append(comp)
            adj[comp].append(net_b)
        return adj

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def verify_degrees(self) -> None:
        """Assert every component node has exactly degree 2 (undirected)."""
        for comp in self.comp_nodes:
            net_a, net_b = self.comp_terminals[comp]
            assert net_a in self.net_nodes, f"{comp} terminal {net_a} not in net_nodes"
            assert net_b in self.net_nodes, f"{comp} terminal {net_b} not in net_nodes"
            # Degree 2 is guaranteed by construction (exactly 2 terminals).

    def verify_connected(self) -> bool:
        """Check the undirected bipartite graph is connected via BFS."""
        if not self.net_nodes and not self.comp_nodes:
            return True
        all_nodes = set(self.net_nodes + self.comp_nodes)
        adj = self.adjacency_lists()
        visited: Set[str] = set()
        queue = [self.net_nodes[0]]
        visited.add(queue[0])
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited == all_nodes

    # ------------------------------------------------------------------ #
    #  Back-conversion to adjacency                                        #
    # ------------------------------------------------------------------ #

    def to_adjacency(self) -> Dict[str, List[Dict]]:
        """
        Convert back to an adjacency-list representation (net-centric).

        Returns:
            Dict mapping net_name → list of {net, comp_type} dicts
            representing edges to other nets via components.
        """
        adj: Dict[str, List[Dict]] = {n: [] for n in self.net_nodes}
        for comp, (net_a, net_b) in self.comp_terminals.items():
            ctype = self.comp_types[comp]
            adj[net_a].append({'net': net_b, 'comp': comp, 'type': ctype})
            adj[net_b].append({'net': net_a, 'comp': comp, 'type': ctype})
        return adj


# ====================================================================== #
#  Constructors                                                            #
# ====================================================================== #

# Node type index → canonical net name
_NODE_TYPE_MAP = {0: 'VSS', 1: 'VIN', 2: 'VOUT', 3: 'INTERNAL'}


def from_pickle_circuit(circuit: dict) -> BipartiteCircuitGraph:
    """
    Convert a circuit from the existing pickle dataset format.

    The pickle stores:
        graph_adj.nodes[i].features  — 4D one-hot [GND, VIN, VOUT, INTERNAL]
        graph_adj.adjacency[src]     — list of {id, impedance_den: [C, G, L_inv]}

    Multi-component edges (e.g. RC parallel) are decomposed into individual
    R / C / L component nodes.

    Args:
        circuit: One element from the pickle dataset list.

    Returns:
        BipartiteCircuitGraph
    """
    graph_adj = circuit['graph_adj']
    nodes = graph_adj['nodes']
    adjacency = graph_adj['adjacency']

    # --- Build net node list with canonical names ---
    internal_count = 0
    node_id_to_name: Dict[int, str] = {}
    net_nodes: List[str] = []

    for node in nodes:
        nid = node['id']
        ntype = node['features'].index(1) if 1 in node['features'] else 3
        if ntype == 3:  # INTERNAL
            internal_count += 1
            name = f'INTERNAL_{internal_count}'
        else:
            name = _NODE_TYPE_MAP[ntype]
        node_id_to_name[nid] = name
        net_nodes.append(name)

    # --- Extract unique undirected edges as compound component nodes ---
    # Each edge becomes ONE component node whose type reflects all present
    # elements (e.g. 'RCL' if R, C, and L are all present on that edge).
    comp_nodes: List[str] = []
    comp_terminals: Dict[str, Tuple[str, str]] = {}
    comp_types: Dict[str, str] = {}

    # Per-type counters for naming: R1, C1, RC1, RCL1, etc.
    type_counts: Dict[str, int] = {}
    seen_edges: Set[Tuple[int, int]] = set()

    for node_idx, neighbors in enumerate(adjacency):
        src_id = nodes[node_idx]['id']
        for neighbor in neighbors:
            tgt_id = neighbor['id']
            edge_key = (min(src_id, tgt_id), max(src_id, tgt_id))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            C_val, G_val, L_inv = neighbor['impedance_den']
            net_a = node_id_to_name[edge_key[0]]
            net_b = node_id_to_name[edge_key[1]]

            # Build compound type string (always R before C before L)
            ctype = ''
            if G_val > 1e-12:
                ctype += 'R'
            if C_val > 1e-12:
                ctype += 'C'
            if L_inv > 1e-12:
                ctype += 'L'

            if not ctype:
                continue  # no components on this edge

            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            name = f'{ctype}{type_counts[ctype]}'
            comp_nodes.append(name)
            comp_terminals[name] = (net_a, net_b)
            comp_types[name] = ctype

    graph = BipartiteCircuitGraph(
        net_nodes=net_nodes,
        comp_nodes=comp_nodes,
        comp_terminals=comp_terminals,
        comp_types=comp_types,
    )
    graph.verify_degrees()
    return graph


def from_spice_netlist(netlist: str) -> BipartiteCircuitGraph:
    """
    Parse a SPICE-like RLC netlist into a bipartite circuit graph.

    Expected format (one component per line):
        R1 net1 net2 1000
        C1 net2 0 1e-9
        L1 net1 net3 1e-3

    Net names are canonicalized:
        '0' or 'gnd' or 'vss' → 'VSS'
        'vin'                  → 'VIN'
        'vout'                 → 'VOUT'
        'vdd'                  → 'VDD'
        anything else          → 'INTERNAL_k'

    Args:
        netlist: Multi-line string of SPICE component statements.

    Returns:
        BipartiteCircuitGraph
    """
    # Canonical net name mapping
    special_nets = {
        '0': 'VSS', 'gnd': 'VSS', 'vss': 'VSS',
        'vin': 'VIN', 'vout': 'VOUT', 'vdd': 'VDD',
    }

    internal_map: Dict[str, str] = {}
    internal_count = 0

    def canonicalize_net(raw: str) -> str:
        nonlocal internal_count
        lower = raw.lower().strip()
        if lower in special_nets:
            return special_nets[lower]
        if raw in internal_map:
            return internal_map[raw]
        internal_count += 1
        name = f'INTERNAL_{internal_count}'
        internal_map[raw] = name
        return name

    net_set: Set[str] = set()
    comp_nodes: List[str] = []
    comp_terminals: Dict[str, Tuple[str, str]] = {}
    comp_types: Dict[str, str] = {}

    comp_pattern = re.compile(
        r'^([RCL]\w*)\s+(\S+)\s+(\S+)\s+(\S+)', re.IGNORECASE
    )

    for line in netlist.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue

        m = comp_pattern.match(line)
        if not m:
            continue

        comp_name = m.group(1).upper()
        net_a = canonicalize_net(m.group(2))
        net_b = canonicalize_net(m.group(3))
        # value = m.group(4)  # not used for topology

        # Determine component type from first character
        ctype = comp_name[0]  # 'R', 'C', or 'L'
        assert ctype in ('R', 'C', 'L'), f"Unknown component type: {comp_name}"

        net_set.add(net_a)
        net_set.add(net_b)
        comp_nodes.append(comp_name)
        comp_terminals[comp_name] = (net_a, net_b)
        comp_types[comp_name] = ctype

    # Order net nodes: VSS first, then VIN, VOUT, VDD, then INTERNAL_* sorted
    priority = {'VSS': 0, 'VIN': 1, 'VOUT': 2, 'VDD': 3}
    net_nodes = sorted(net_set, key=lambda n: (priority.get(n, 10), n))

    graph = BipartiteCircuitGraph(
        net_nodes=net_nodes,
        comp_nodes=comp_nodes,
        comp_terminals=comp_terminals,
        comp_types=comp_types,
    )
    graph.verify_degrees()
    return graph
