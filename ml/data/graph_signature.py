"""
Walk validation, parsing, and topology canonicalisation.

The decoder produces token sequences over the CircuitVocabulary alphabet
(VSS, VIN, VOUT, INTERNAL_*, R1, C2, L3, ...). To compare generated walks
to known templates and to each other, we need:

  • `well_formed`            : structural check (alternating net/comp tokens,
                               start/end at VSS, every component visited
                               exactly twice as Eulerian convention demands).
  • `walk_to_graph`          : reconstruct the underlying BipartiteCircuitGraph.
  • `walk_topology_signature`: a hashable canonical fingerprint that is
                               invariant under (a) the choice of Eulerian
                               traversal and (b) relabeling of INTERNAL nets.
  • `is_electrically_valid`  : VIN, VOUT, GND each connect to ≥1 component
                               and every component touches ≥2 distinct nets.
  • `dedupe_by_isomorphism`  : collapse a list of walks to one per signature.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple

from ml.data.bipartite_graph import BipartiteCircuitGraph


# Net tokens recognised in walks. Anything else with this prefix
# is also a net (INTERNAL_*).
_FIXED_NETS = {'VSS', 'VIN', 'VOUT', 'VDD'}


def _is_net_token(tok: str) -> bool:
    return tok in _FIXED_NETS or tok.startswith('INTERNAL_')


def _comp_type(tok: str) -> str:
    """Strip trailing digits — 'R12' → 'R', 'RC3' → 'RC'."""
    i = 0
    while i < len(tok) and tok[i].isalpha():
        i += 1
    return tok[:i]


def well_formed(walk: Iterable[str]) -> bool:
    """Check that `walk` is a valid Eulerian circuit token sequence.

    Required structure:
      • length is odd and ≥ 3 (one component → 5 tokens VSS comp X comp VSS)
      • positions 0,2,4,... are net tokens
      • positions 1,3,5,... are component tokens (and not nets)
      • starts and ends at VSS
      • every component appears exactly twice (Eulerian arc convention:
        each undirected component edge is traversed once in each direction)
    """
    walk = list(walk)
    if len(walk) < 3 or len(walk) % 2 == 0:
        return False
    if walk[0] != 'VSS' or walk[-1] != 'VSS':
        return False
    for i in range(0, len(walk), 2):
        if not _is_net_token(walk[i]):
            return False
    for i in range(1, len(walk), 2):
        if _is_net_token(walk[i]):
            return False
    comp_counts = Counter(walk[1::2])
    if any(c != 2 for c in comp_counts.values()):
        return False
    return True


def walk_to_graph(walk: Iterable[str]) -> Optional[BipartiteCircuitGraph]:
    """Parse a well-formed walk back into a BipartiteCircuitGraph.

    Returns None if the walk is malformed or a component is found
    spanning more or fewer than two distinct nets.
    """
    walk = list(walk)
    if not well_formed(walk):
        return None

    comp_neighbors: dict = defaultdict(set)
    for i in range(1, len(walk), 2):
        comp = walk[i]
        left = walk[i - 1]
        right = walk[i + 1]
        comp_neighbors[comp].add(left)
        comp_neighbors[comp].add(right)

    g = BipartiteCircuitGraph()
    nets_seen: set = set()
    for comp, nets in comp_neighbors.items():
        if len(nets) != 2:
            return None
        a, b = sorted(nets)
        g.comp_terminals[comp] = (a, b)
        g.comp_types[comp] = _comp_type(comp)
        nets_seen.update(nets)
    g.comp_nodes = sorted(comp_neighbors.keys())
    # Stable order: fixed nets first (VSS, VIN, VOUT, VDD), then INTERNAL_*
    fixed = [n for n in ('VSS', 'VIN', 'VOUT', 'VDD') if n in nets_seen]
    internal = sorted(n for n in nets_seen if n.startswith('INTERNAL_'))
    g.net_nodes = fixed + internal
    return g


def is_electrically_valid(walk: Iterable[str]) -> bool:
    """Sanity electrical checks beyond pure structural well-formedness.

      • VIN, VOUT, VSS each connect to ≥1 component
      • no component is a self-loop (its two terminals must differ)
      • every INTERNAL net touches ≥2 components (otherwise it's a
        dangling stub with nowhere for current to flow)
    """
    g = walk_to_graph(walk)
    if g is None:
        return False
    incident: dict = defaultdict(int)
    for comp, (a, b) in g.comp_terminals.items():
        if a == b:
            return False
        incident[a] += 1
        incident[b] += 1
    for required in ('VSS', 'VIN', 'VOUT'):
        if incident.get(required, 0) < 1:
            return False
    for net in g.net_nodes:
        if net.startswith('INTERNAL_') and incident[net] < 2:
            return False
    return True


def walk_topology_signature(walk: Iterable[str]) -> Optional[frozenset]:
    """Return a hashable structural fingerprint of the underlying circuit.

    Two walks share a signature iff they describe the same circuit up to:
      • re-traversal (different Eulerian circuit on the same graph)
      • relabeling of INTERNAL_* nets (internal naming is arbitrary)
      • renaming of component instances WITHIN a type (e.g. R1↔R2)

    Returns None if the walk is malformed.

    Implementation:
      1. Parse to BipartiteCircuitGraph.
      2. Canonicalise INTERNAL net labels by sorting them by their
         (multiset of incident component types). For ties, fall back
         to a positional index.
      3. Multiset of (comp_type, frozenset({net_a, net_b})) is the sig.

    The internal-net canonicalisation is a refinement-style hash, not a
    full graph-isomorphism canonical form, but it correctly distinguishes
    the topologies in our 10-template space and most plausible novel
    variants. False merges (different graphs hashing the same) are
    possible only when two graphs have identical refinement classes —
    rare for circuits with ≤6 components.
    """
    g = walk_to_graph(walk)
    if g is None:
        return None

    # Step 1: gather incident component types per net.
    incident_types: dict = defaultdict(list)
    for comp, (a, b) in g.comp_terminals.items():
        ct = g.comp_types[comp]
        incident_types[a].append(ct)
        incident_types[b].append(ct)

    # Step 2: canonical label for each INTERNAL net based on its
    # incident type multiset, broken by lexicographic order.
    internals = [n for n in g.net_nodes if n.startswith('INTERNAL_')]
    keyed = [
        (tuple(sorted(incident_types[n])), n) for n in internals
    ]
    keyed.sort()
    relabel: dict = {}
    for new_idx, (_, old) in enumerate(keyed):
        relabel[old] = f'_INT{new_idx}'
    # Fixed nets keep their names.
    for n in g.net_nodes:
        if n not in relabel:
            relabel[n] = n

    # Step 3: emit multiset of edges with canonical net labels.
    edges = Counter()
    for comp, (a, b) in g.comp_terminals.items():
        ct = g.comp_types[comp]
        pair = frozenset({relabel[a], relabel[b]})
        edges[(ct, pair)] += 1

    return frozenset(edges.items())


def dedupe_by_isomorphism(walks: Iterable[Iterable[str]]) -> List[Tuple[frozenset, Tuple[str, ...]]]:
    """Collapse a list of walks to one representative per topology signature.

    Returns a list of (signature, walk) pairs in first-seen order.
    Malformed walks are dropped.
    """
    seen: dict = {}
    for w in walks:
        w = tuple(w)
        sig = walk_topology_signature(w)
        if sig is None or sig in seen:
            continue
        seen[sig] = w
    return [(sig, walk) for sig, walk in seen.items()]
