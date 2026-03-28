"""
Eulerian circuit generation for symmetric directed bipartite graphs.

Uses Hierholzer's algorithm (randomizable) to find directed Eulerian circuits
that start and end at VSS. Supports augmentation by producing multiple
distinct valid circuits via randomized arc ordering.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ml.data.bipartite_graph import BipartiteCircuitGraph


def hierholzer(
    graph: BipartiteCircuitGraph,
    start: str = 'VSS',
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Find a directed Eulerian circuit using Hierholzer's algorithm.

    The graph is the symmetric directed bipartite graph where each undirected
    incidence {net, comp} is expanded into arcs (net→comp) and (comp→net).

    Args:
        graph: BipartiteCircuitGraph (must be connected).
        start: Starting node (default 'VSS').
        rng: Optional Random instance for shuffling arc order. If None,
             a deterministic (sorted) order is used.

    Returns:
        List of node names forming a closed walk that visits every
        directed arc exactly once. Length = 4K + 1 for K components.

    Raises:
        ValueError: If start node not in graph or graph is disconnected.
    """
    if start not in graph.net_nodes:
        raise ValueError(f"Start node '{start}' not in net_nodes")

    # Build mutable adjacency lists (directed arcs)
    adj: Dict[str, List[str]] = defaultdict(list)
    for comp, (net_a, net_b) in graph.comp_terminals.items():
        adj[net_a].append(comp)
        adj[comp].append(net_a)
        adj[net_b].append(comp)
        adj[comp].append(net_b)

    # Optionally shuffle each adjacency list for augmentation
    if rng is not None:
        for node in adj:
            rng.shuffle(adj[node])
    else:
        for node in adj:
            adj[node].sort()

    # Hierholzer's: walk until stuck, then splice in sub-walks
    circuit: List[str] = []
    stack: List[str] = [start]

    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop()
            stack.append(u)
        else:
            circuit.append(stack.pop())

    circuit.reverse()

    # Verify length: 4K + 1
    expected_len = 4 * len(graph.comp_nodes) + 1
    if len(circuit) != expected_len:
        raise RuntimeError(
            f"Euler circuit length {len(circuit)} != expected {expected_len}. "
            f"Graph may be disconnected."
        )

    return circuit


def enumerate_euler_circuits(
    graph: BipartiteCircuitGraph,
    start: str = 'VSS',
    max_circuits: int = 200,
) -> List[List[str]]:
    """
    Enumerate all distinct Eulerian circuits via backtracking.

    Exhaustively explores all possible arc orderings to find every
    distinct Euler circuit, up to max_circuits. This is the AnalogGenie-style
    augmentation approach.

    Args:
        graph: BipartiteCircuitGraph.
        start: Starting node (default 'VSS').
        max_circuits: Maximum number of circuits to find.

    Returns:
        List of distinct Eulerian circuit walks.
    """
    if start not in graph.net_nodes:
        raise ValueError(f"Start node '{start}' not in net_nodes")

    # Build adjacency lists (directed arcs)
    adj: Dict[str, List[str]] = defaultdict(list)
    for comp, (net_a, net_b) in graph.comp_terminals.items():
        adj[net_a].append(comp)
        adj[comp].append(net_a)
        adj[net_b].append(comp)
        adj[comp].append(net_b)

    # Sort for deterministic enumeration
    for node in adj:
        adj[node].sort()

    total_arcs = sum(len(neighbors) for neighbors in adj.values())
    circuits: List[List[str]] = []

    def backtrack(current: str, path: List[str], used_count: int):
        if len(circuits) >= max_circuits:
            return

        if used_count == total_arcs:
            if current == start:
                circuits.append(path[:])
            return

        neighbors = adj[current]
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            # Remove this arc
            adj[current] = neighbors[:i] + neighbors[i + 1:]
            path.append(neighbor)

            backtrack(neighbor, path, used_count + 1)

            # Restore
            path.pop()
            adj[current] = neighbors

            if len(circuits) >= max_circuits:
                return

    backtrack(start, [start], 0)
    return circuits


def augment_traversals(
    graph: BipartiteCircuitGraph,
    n: int = 10,
    start: str = 'VSS',
    seed: Optional[int] = None,
) -> List[List[str]]:
    """
    Generate up to n distinct Eulerian circuits by randomizing arc order.

    Args:
        graph: BipartiteCircuitGraph.
        n: Maximum number of distinct traversals to produce.
        start: Starting node (default 'VSS').
        seed: Random seed for reproducibility.

    Returns:
        List of distinct traversal sequences (deduplicated).
    """
    rng = random.Random(seed)
    seen: Set[Tuple[str, ...]] = set()
    results: List[List[str]] = []

    # Always include the canonical (sorted) traversal first
    canonical = hierholzer(graph, start=start, rng=None)
    seen.add(tuple(canonical))
    results.append(canonical)

    # Generate randomized traversals
    max_attempts = n * 5
    for _ in range(max_attempts):
        if len(results) >= n:
            break
        walk = hierholzer(graph, start=start, rng=rng)
        key = tuple(walk)
        if key not in seen:
            seen.add(key)
            results.append(walk)

    return results


def verify_traversal(
    graph: BipartiteCircuitGraph,
    walk: List[str],
    start: str = 'VSS',
) -> bool:
    """
    Verify a walk is a valid directed Eulerian circuit.

    Checks:
        1. Starts and ends at start node
        2. Every directed arc is covered exactly once
        3. Length is correct (4K + 1)

    Args:
        graph: The bipartite graph.
        walk: Sequence of node names.
        start: Expected start/end node.

    Returns:
        True if valid, False otherwise.
    """
    if not walk:
        return False
    if walk[0] != start or walk[-1] != start:
        return False

    expected_len = 4 * len(graph.comp_nodes) + 1
    if len(walk) != expected_len:
        return False

    # Collect all arcs used in the walk
    used_arcs: List[Tuple[str, str]] = []
    for i in range(len(walk) - 1):
        used_arcs.append((walk[i], walk[i + 1]))

    # Build expected arc multiset
    expected_arcs: List[Tuple[str, str]] = graph.directed_arcs()

    # Compare as sorted lists (multiset equality)
    return sorted(used_arcs) == sorted(expected_arcs)


def walk_to_circuit(
    walk: List[str],
    net_nodes: Set[str],
) -> Dict[str, Tuple[str, str, str]]:
    """
    Parse a walk back into component connections.

    For each component node in the walk, determine which two net nodes
    it connects to.

    Args:
        walk: Eulerian circuit sequence.
        net_nodes: Set of net node names (to distinguish from components).

    Returns:
        Dict mapping component_name → (comp_type, net_a, net_b).
        comp_type is derived from the first character ('R', 'C', or 'L').
    """
    # For each component, collect its neighboring nets in the walk
    comp_neighbors: Dict[str, Set[str]] = defaultdict(set)

    for i, node in enumerate(walk):
        if node not in net_nodes:
            # This is a component node; its walk neighbors are net nodes
            if i > 0 and walk[i - 1] in net_nodes:
                comp_neighbors[node].add(walk[i - 1])
            if i < len(walk) - 1 and walk[i + 1] in net_nodes:
                comp_neighbors[node].add(walk[i + 1])

    result: Dict[str, Tuple[str, str, str]] = {}
    for comp, nets in comp_neighbors.items():
        net_list = sorted(nets)
        if len(net_list) != 2:
            raise ValueError(
                f"Component {comp} has {len(net_list)} terminal nets "
                f"(expected 2): {net_list}"
            )
        ctype = comp[0]  # R, C, or L
        result[comp] = (ctype, net_list[0], net_list[1])

    return result
