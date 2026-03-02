"""Circuit formatting and validation helpers for generated graphs."""

from typing import Dict

import torch

COMPONENT_NAMES = ['None', 'R', 'C', 'L', 'RC', 'RL', 'CL', 'RCL']
BASE_NODE_NAMES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INT', 4: 'INT'}


def _first_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize generated outputs that may include a batch dimension."""
    return tensor[0] if tensor.dim() >= 3 else tensor


def circuit_to_string(circuit: Dict[str, torch.Tensor]) -> str:
    """Convert decoder output to a compact edge-list string."""
    edge_exist = _first_batch(circuit['edge_existence'])
    comp_types = _first_batch(circuit['component_types'])
    node_types = circuit['node_types'][0] if circuit['node_types'].dim() > 1 else circuit['node_types']

    num_nodes = node_types.shape[0]

    node_names = []
    int_counter = 1
    for idx in range(num_nodes):
        node_type = int(node_types[idx].item())
        if node_type >= 3:
            node_names.append(f'INT{int_counter}')
            int_counter += 1
        else:
            node_names.append(BASE_NODE_NAMES[node_type])

    edges = []
    for ni in range(num_nodes):
        for nj in range(ni):
            if edge_exist[ni, nj] > 0.5:
                comp = COMPONENT_NAMES[int(comp_types[ni, nj].item())]
                edges.append(f"{node_names[nj]}--{comp}--{node_names[ni]}")

    return ', '.join(edges) if edges else '(no edges)'


def is_valid_circuit(circuit: Dict[str, torch.Tensor]) -> bool:
    """Check basic validity: VIN and VOUT are both connected."""
    edge_exist = _first_batch(circuit['edge_existence'])
    vin_connected = bool((edge_exist[1, :] > 0.5).any() or (edge_exist[:, 1] > 0.5).any())
    vout_connected = bool((edge_exist[2, :] > 0.5).any() or (edge_exist[:, 2] > 0.5).any())
    return vin_connected and vout_connected
