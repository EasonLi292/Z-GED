"""Circuit formatting and validation helpers for generated walks."""

from collections import defaultdict, deque
from typing import Dict, List, Optional

import torch

from ml.models.vocabulary import CircuitVocabulary

COMPONENT_NAMES = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']
BASE_NODE_NAMES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INT', 4: 'INT'}


def walk_to_string(walk_tokens: List[str], vocab: CircuitVocabulary) -> str:
    """Convert walk token strings to a compact edge-list description.

    Example output: "VSS--R--VIN, VIN--C--VOUT"
    """
    comp_nets: Dict[str, set] = defaultdict(set)
    for i, tok in enumerate(walk_tokens):
        if vocab.token_type(tok) == 'component':
            if i > 0 and vocab.token_type(walk_tokens[i - 1]) == 'net':
                comp_nets[tok].add(walk_tokens[i - 1])
            if i < len(walk_tokens) - 1 and vocab.token_type(walk_tokens[i + 1]) == 'net':
                comp_nets[tok].add(walk_tokens[i + 1])

    if not comp_nets:
        return '(no components)'

    parts = []
    for comp in sorted(comp_nets.keys()):
        nets = sorted(comp_nets[comp])
        ctype = vocab.component_type(comp)
        if len(nets) == 2:
            parts.append(f"{nets[0]}--{ctype}--{nets[1]}")
        elif len(nets) == 1:
            parts.append(f"{nets[0]}--{ctype}--{nets[0]}")
        else:
            parts.append(f"{ctype}({','.join(nets)})")
    return ', '.join(parts)


def is_valid_walk(walk_tokens: List[str], vocab: Optional[CircuitVocabulary] = None) -> bool:
    """Check physical validity of a generated circuit walk.

    Requirements:
    1. Starts and ends at VSS
    2. Contains at least one component
    3. Contains VIN and VOUT nodes
    4. VIN is connected to both VOUT and VSS through components
    """
    if not walk_tokens:
        return False
    if walk_tokens[0] != 'VSS' or walk_tokens[-1] != 'VSS':
        return False

    if vocab is None:
        return len(walk_tokens) >= 3

    has_comp = any(vocab.token_type(t) == 'component' for t in walk_tokens)
    if not has_comp:
        return False

    nets = set(t for t in walk_tokens if vocab.token_type(t) == 'net')
    if 'VIN' not in nets or 'VOUT' not in nets:
        return False

    # Build net adjacency (two nets are adjacent if they share a component)
    comp_nets: Dict[str, set] = defaultdict(set)
    for i, tok in enumerate(walk_tokens):
        if vocab.token_type(tok) == 'component':
            if i > 0 and vocab.token_type(walk_tokens[i - 1]) == 'net':
                comp_nets[tok].add(walk_tokens[i - 1])
            if i < len(walk_tokens) - 1 and vocab.token_type(walk_tokens[i + 1]) == 'net':
                comp_nets[tok].add(walk_tokens[i + 1])

    net_adj: Dict[str, set] = defaultdict(set)
    for comp, cnets in comp_nets.items():
        cnets_list = list(cnets)
        for a in cnets_list:
            for b in cnets_list:
                if a != b:
                    net_adj[a].add(b)

    # BFS from VIN — must reach both VOUT and VSS
    visited = set()
    queue = deque(['VIN'])
    visited.add('VIN')
    while queue:
        n = queue.popleft()
        for neighbor in net_adj[n]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return 'VOUT' in visited and 'VSS' in visited


def generate_walk(
    decoder,
    latent: torch.Tensor,
    vocab: CircuitVocabulary,
    max_length: int = 32,
    greedy: bool = True,
) -> List[str]:
    """Convenience: latent vector -> decoded walk token strings.

    Args:
        decoder: SequenceDecoder instance (eval mode).
        latent: [latent_dim] or [1, latent_dim] tensor.
        vocab: CircuitVocabulary for decoding.
        max_length: Maximum walk tokens to generate.
        greedy: Greedy (argmax) vs sampling.

    Returns:
        List of token strings (e.g. ['VSS', 'R1', 'VIN', ...]).
    """
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)
    latent = latent.float()
    with torch.no_grad():
        generated = decoder.generate(
            latent, max_length=max_length, greedy=greedy, eos_id=vocab.eos_id,
        )
    return vocab.decode(generated[0])
