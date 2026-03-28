"""
Token vocabulary for circuit sequence representation.

Fixed, deterministic vocabulary covering:
    - Special tokens: PAD, EOS
    - Net tokens: VSS, VIN, VOUT, VDD, INTERNAL_1..N
    - Component tokens: R1..M, C1..M, L1..M, RC1..M, RL1..M, CL1..M, RCL1..M
"""

from __future__ import annotations

from typing import Dict, List, Optional


# All 7 component type prefixes, matching the adjacency decoder's classification
COMPONENT_PREFIXES = ('R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL')


class CircuitVocabulary:
    """
    Deterministic vocabulary for bipartite circuit graph walks.

    Token layout:
        0       PAD
        1       EOS
        2       VSS
        3       VIN
        4       VOUT
        5       VDD
        6..5+N  INTERNAL_1 .. INTERNAL_N
        then    R1..R_M, C1..C_M, L1..L_M, RC1..RC_M, RL1..RL_M, CL1..CL_M, RCL1..RCL_M

    Args:
        max_internal: Maximum number of internal net tokens (default 10).
        max_components: Maximum count per component type (default 10).
    """

    PAD_TOKEN = 'PAD'
    EOS_TOKEN = 'EOS'

    # Fixed net ordering
    FIXED_NETS = ['VSS', 'VIN', 'VOUT', 'VDD']

    def __init__(
        self,
        max_internal: int = 10,
        max_components: int = 10,
    ):
        self.max_internal = max_internal
        self.max_components = max_components

        tokens: List[str] = []

        # Special tokens
        tokens.append(self.PAD_TOKEN)
        tokens.append(self.EOS_TOKEN)

        # Fixed nets
        tokens.extend(self.FIXED_NETS)

        # Internal nets
        for i in range(1, max_internal + 1):
            tokens.append(f'INTERNAL_{i}')

        # Components: all 7 types
        for prefix in COMPONENT_PREFIXES:
            for i in range(1, max_components + 1):
                tokens.append(f'{prefix}{i}')

        self.tokens = tokens
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.itos: Dict[int, str] = {i: t for i, t in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return self.stoi['EOS']

    @property
    def vss_id(self) -> int:
        return self.stoi['VSS']

    def encode(self, sequence: List[str]) -> List[int]:
        """Convert a list of token strings to integer IDs."""
        return [self.stoi[t] for t in sequence]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert integer IDs back to token strings."""
        return [self.itos[i] for i in ids]

    def token_type(self, token: str) -> str:
        """
        Classify a token as 'pad', 'eos', 'net', or 'component'.
        """
        if token == self.PAD_TOKEN:
            return 'pad'
        if token == self.EOS_TOKEN:
            return 'eos'
        if token in self.FIXED_NETS or token.startswith('INTERNAL_'):
            return 'net'
        return 'component'

    def component_type(self, token: str) -> Optional[str]:
        """
        Get the component type prefix ('R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL')
        for a component token. Returns None for non-component tokens.
        """
        if self.token_type(token) != 'component':
            return None
        # Strip trailing digits to get the prefix
        i = 0
        while i < len(token) and token[i].isalpha():
            i += 1
        return token[:i]

    def is_net(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a net node."""
        return self.token_type(self.itos[token_id]) == 'net'

    def is_component(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a component node."""
        return self.token_type(self.itos[token_id]) == 'component'

    def __repr__(self) -> str:
        return (
            f"CircuitVocabulary(size={self.vocab_size}, "
            f"internal={self.max_internal}, components={self.max_components})"
        )
