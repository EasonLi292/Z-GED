#!/usr/bin/env python3
"""
Graph Edit Distance (GED) for Circuit Graphs

Implements GED computation for circuit graphs with impedance-based edges.
Supports similarity search, clustering, and nearest neighbor retrieval.

Key features:
- Anchored nodes (VIN, VOUT, GND must match)
- Impedance distance metric for edge replacement
- Configurable cost parameters
- Fast approximate GED using NetworkX
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple replacement for tqdm
    def tqdm(iterable, desc=None):
        return iterable


def load_graph_from_dataset(graph_adj_dict: Dict) -> nx.Graph:
    """
    Convert dataset graph representation to NetworkX Graph.

    Args:
        graph_adj_dict: Graph adjacency dict from dataset with structure:
            {
                'nodes': [{'id': int, 'features': [4D one-hot]}, ...],
                'adjacency': [[{'id': int, 'impedance_num': [...],
                               'impedance_den': [...]}, ...], ...]
            }

    Returns:
        NetworkX Graph with node features and edge impedances
    """
    G = nx.Graph()

    # Add nodes with features
    for node_data in graph_adj_dict['nodes']:
        G.add_node(node_data['id'], features=tuple(node_data['features']))

    # Add edges with impedances (avoid duplicates in undirected graph)
    added_edges = set()
    for source_id, neighbors in enumerate(graph_adj_dict['adjacency']):
        for edge in neighbors:
            target_id = edge['id']
            edge_key = tuple(sorted([source_id, target_id]))

            if edge_key not in added_edges:
                G.add_edge(
                    source_id, target_id,
                    impedance_num=tuple(edge['impedance_num']),
                    impedance_den=tuple(edge['impedance_den'])
                )
                added_edges.add(edge_key)

    return G


class CircuitGED:
    """
    Graph Edit Distance calculator for circuit graphs.

    Computes GED between circuits using impedance-aware cost functions
    and enforces matching of special nodes (GND, VIN, VOUT).

    Attributes:
        w_C: Weight for capacitance differences
        w_G: Weight for conductance differences
        w_L_inv: Weight for inverse inductance differences
        node_cost: Cost for node insertion/deletion
        simple_edge_cost: Cost for simple component edges (R, L, or C)
        complex_edge_cost: Cost for parallel combination edges
    """

    def __init__(self,
                 w_C: float = 1e12,
                 w_G: float = 1e2,
                 w_L_inv: float = 1e-3,
                 node_cost: float = 1.0,
                 simple_edge_cost: float = 0.5,
                 complex_edge_cost: float = 1.0):
        """
        Initialize GED calculator with cost parameters.

        Args:
            w_C: Capacitance normalization weight (default: 1e12)
            w_G: Conductance normalization weight (default: 1e2)
            w_L_inv: Inverse inductance normalization weight (default: 1e-3)
            node_cost: Cost for node insertion/deletion (default: 1.0)
            simple_edge_cost: Cost for simple component (default: 0.5)
            complex_edge_cost: Cost for parallel combination (default: 1.0)
        """
        self.w_C = w_C
        self.w_G = w_G
        self.w_L_inv = w_L_inv
        self.node_cost = node_cost
        self.simple_edge_cost = simple_edge_cost
        self.complex_edge_cost = complex_edge_cost

    def impedance_distance(self,
                          imp1: Tuple[Tuple[float, ...], Tuple[float, ...]],
                          imp2: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> float:
        """
        Calculate distance between two impedance polynomials.

        Z(s) = s / (C·s² + G·s + L_inv)

        Args:
            imp1, imp2: Tuples of (numerator, denominator) coefficients
                numerator: (1.0, 0.0) for all circuits
                denominator: (C, G, L_inv)

        Returns:
            Normalized Euclidean distance in coefficient space
        """
        num1, den1 = imp1
        num2, den2 = imp2

        # Validate numerators (should both be (1.0, 0.0) = s)
        if num1 != num2:
            return float('inf')

        # Extract denominator coefficients
        C1, G1, L_inv1 = den1[0], den1[1], den1[2]
        C2, G2, L_inv2 = den2[0], den2[1], den2[2]

        # Normalized squared differences
        d_C = self.w_C * (C1 - C2)**2
        d_G = self.w_G * (G1 - G2)**2
        d_L = self.w_L_inv * (L_inv1 - L_inv2)**2

        # Euclidean distance
        distance = np.sqrt(d_C + d_G + d_L)

        # Scale to be comparable with node costs (~1 for typical difference)
        return distance / 10.0

    def _node_match(self, node1_attrs: Dict[str, Any],
                   node2_attrs: Dict[str, Any]) -> bool:
        """
        Check if two nodes can be matched (enforces anchoring).

        Special nodes (GND, VIN, VOUT) must match their types exactly.
        Internal nodes can only match other internal nodes.

        Args:
            node1_attrs, node2_attrs: Node attribute dicts with 'features'

        Returns:
            True if nodes can be matched, False otherwise
        """
        features1 = node1_attrs['features']
        features2 = node2_attrs['features']

        # Check if either is a special node (GND, VIN, VOUT)
        is_special1 = features1[:3] != (0, 0, 0)
        is_special2 = features2[:3] != (0, 0, 0)

        if is_special1 or is_special2:
            # Special nodes must match type exactly
            return features1 == features2

        # Both are internal nodes - can match each other
        return features1[3] == features2[3]

    def _node_subst_cost(self, node1_attrs: Dict[str, Any],
                        node2_attrs: Dict[str, Any]) -> float:
        """
        Cost of substituting one node with another.

        Args:
            node1_attrs, node2_attrs: Node attribute dicts

        Returns:
            0.0 if same type, inf if incompatible
        """
        if node1_attrs['features'] == node2_attrs['features']:
            return 0.0
        else:
            # Cannot substitute different node types
            return float('inf')

    def _node_ins_cost(self, node_attrs: Dict[str, Any]) -> float:
        """
        Cost of inserting a node.

        Only internal nodes can be inserted.

        Args:
            node_attrs: Node attribute dict

        Returns:
            node_cost for internal nodes, inf for special nodes
        """
        features = node_attrs['features']

        # Only allow insertion of internal nodes
        if features[3] == 1:  # INTERNAL node
            return self.node_cost
        else:
            # Cannot insert special nodes (GND, VIN, VOUT)
            return float('inf')

    def _node_del_cost(self, node_attrs: Dict[str, Any]) -> float:
        """
        Cost of deleting a node.

        Only internal nodes can be deleted.

        Args:
            node_attrs: Node attribute dict

        Returns:
            node_cost for internal nodes, inf for special nodes
        """
        features = node_attrs['features']

        # Only allow deletion of internal nodes
        if features[3] == 1:  # INTERNAL node
            return self.node_cost
        else:
            # Cannot delete special nodes (GND, VIN, VOUT)
            return float('inf')

    def _edge_subst_cost(self, edge1_attrs: Dict[str, Any],
                        edge2_attrs: Dict[str, Any]) -> float:
        """
        Cost of replacing one edge with another.

        Uses impedance distance metric.

        Args:
            edge1_attrs, edge2_attrs: Edge attribute dicts with impedance data

        Returns:
            Impedance distance between edges
        """
        imp1 = (edge1_attrs['impedance_num'], edge1_attrs['impedance_den'])
        imp2 = (edge2_attrs['impedance_num'], edge2_attrs['impedance_den'])

        return self.impedance_distance(imp1, imp2)

    def _edge_ins_cost(self, edge_attrs: Dict[str, Any]) -> float:
        """
        Cost of inserting an edge.

        Simple components (R, L, or C alone) have lower cost than
        parallel combinations.

        Args:
            edge_attrs: Edge attribute dict with impedance data

        Returns:
            simple_edge_cost or complex_edge_cost
        """
        return self._edge_component_cost(edge_attrs)

    def _edge_del_cost(self, edge_attrs: Dict[str, Any]) -> float:
        """
        Cost of deleting an edge.

        Symmetric with insertion cost.

        Args:
            edge_attrs: Edge attribute dict with impedance data

        Returns:
            simple_edge_cost or complex_edge_cost
        """
        return self._edge_component_cost(edge_attrs)

    def _edge_component_cost(self, edge_attrs: Dict[str, Any]) -> float:
        """
        Determine edge cost based on component complexity.

        Args:
            edge_attrs: Edge attribute dict with impedance data

        Returns:
            simple_edge_cost if single component, complex_edge_cost if parallel
        """
        num = edge_attrs['impedance_num']
        den = edge_attrs['impedance_den']

        # Extract coefficients
        C = den[0]
        G = den[1]
        L_inv = den[2]

        # Count non-zero components
        num_components = sum([C > 0, G > 0, L_inv > 0])

        if num_components == 1:
            # Single component (R, L, or C alone)
            return self.simple_edge_cost
        else:
            # Parallel combination
            return self.complex_edge_cost

    def compute_ged(self,
                    graph1: nx.Graph,
                    graph2: nx.Graph,
                    timeout: float = 60.0) -> float:
        """
        Compute approximate GED between two circuit graphs.

        Uses NetworkX's optimize_graph_edit_distance with custom
        cost functions and anchored node matching.

        Args:
            graph1, graph2: NetworkX graphs from dataset
            timeout: Maximum computation time in seconds

        Returns:
            Approximate graph edit distance (float)
        """
        # Use generator version for progressive refinement
        ged_generator = nx.optimize_graph_edit_distance(
            graph1, graph2,
            node_match=self._node_match,
            edge_match=None,  # Use cost functions instead
            node_subst_cost=self._node_subst_cost,
            node_del_cost=self._node_del_cost,
            node_ins_cost=self._node_ins_cost,
            edge_subst_cost=self._edge_subst_cost,
            edge_del_cost=self._edge_del_cost,
            edge_ins_cost=self._edge_ins_cost,
            upper_bound=None
        )

        # Get best approximation within timeout
        start = time.time()
        best_ged = float('inf')

        try:
            for ged in ged_generator:
                best_ged = min(best_ged, ged)
                if time.time() - start > timeout:
                    break
        except nx.NetworkXError as e:
            # Handle cases where graphs cannot be matched
            print(f"Warning: GED computation failed: {e}")
            return float('inf')

        return best_ged

    def compute_ged_matrix(self,
                          graphs: List[nx.Graph],
                          show_progress: bool = True) -> np.ndarray:
        """
        Compute pairwise GED matrix for a list of graphs.

        Args:
            graphs: List of NetworkX graphs
            show_progress: Show progress bar

        Returns:
            N×N symmetric matrix of GED values
        """
        n = len(graphs)
        matrix = np.zeros((n, n))

        # Compute upper triangle (GED is symmetric)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

        if show_progress:
            pairs = tqdm(pairs, desc="Computing GED matrix")

        for i, j in pairs:
            ged = self.compute_ged(graphs[i], graphs[j])
            matrix[i, j] = ged
            matrix[j, i] = ged  # Symmetric

        return matrix

    def find_nearest_neighbors(self,
                              query_graph: nx.Graph,
                              graph_database: List[nx.Graph],
                              k: int = 5) -> List[Tuple[int, float]]:
        """
        Find k-nearest graphs by GED.

        Args:
            query_graph: Query circuit graph
            graph_database: Database of circuit graphs
            k: Number of neighbors to return

        Returns:
            List of (index, ged_distance) tuples, sorted by distance
        """
        distances = []

        for idx, graph in enumerate(tqdm(graph_database,
                                         desc="Computing distances")):
            ged = self.compute_ged(query_graph, graph)
            distances.append((idx, ged))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]


def validate_ged_properties(ged_calc: CircuitGED,
                            G1: nx.Graph,
                            G2: nx.Graph,
                            G3: Optional[nx.Graph] = None,
                            tol: float = 1e-3) -> Dict[str, bool]:
    """
    Validate mathematical properties of GED.

    Args:
        ged_calc: CircuitGED instance
        G1, G2, G3: Test graphs
        tol: Tolerance for numerical comparisons

    Returns:
        Dict of property names and validation results
    """
    results = {}

    # 1. Non-negativity
    ged = ged_calc.compute_ged(G1, G2, timeout=10)
    results['non_negative'] = ged >= 0

    # 2. Identity: GED(G, G) = 0
    ged_self = ged_calc.compute_ged(G1, G1, timeout=10)
    results['identity'] = abs(ged_self) < tol

    # 3. Symmetry: GED(G1, G2) = GED(G2, G1)
    ged_12 = ged_calc.compute_ged(G1, G2, timeout=10)
    ged_21 = ged_calc.compute_ged(G2, G1, timeout=10)
    results['symmetry'] = abs(ged_12 - ged_21) < tol

    # 4. Triangle inequality (if G3 provided)
    # Note: May not hold exactly due to approximate algorithm
    if G3 is not None:
        ged_13 = ged_calc.compute_ged(G1, G3, timeout=10)
        ged_23 = ged_calc.compute_ged(G2, G3, timeout=10)
        results['triangle_inequality'] = ged_13 <= ged_12 + ged_23 + tol

    return results


if __name__ == '__main__':
    print("Graph Edit Distance module for circuit graphs")
    print("=" * 60)
    print("Import this module to use CircuitGED class")
    print()
    print("Example usage:")
    print("  from graph_edit_distance import CircuitGED, load_graph_from_dataset")
    print("  ged_calc = CircuitGED()")
    print("  ged = ged_calc.compute_ged(graph1, graph2)")
