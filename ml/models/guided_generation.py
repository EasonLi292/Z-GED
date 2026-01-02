"""
Guided Circuit Generation with Connectivity Guarantees.

This module provides post-processing to ensure generated circuits
have proper connectivity, particularly VIN connections.

This is Option 1C from ARCHITECTURE_INVESTIGATION.md - an immediate
fix that requires no retraining.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def ensure_vin_connectivity(
    circuit: Dict[str, torch.Tensor],
    decoder,
    latent: torch.Tensor,
    conditions: torch.Tensor,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Post-process generated circuit to ensure VIN is connected.

    If VIN has no edges, adds VIN→VOUT edge with inferred component values.

    Args:
        circuit: Generated circuit from decoder.generate()
        decoder: GraphGPT decoder (for component value inference)
        latent: Latent code [batch, latent_dim]
        conditions: Specifications [batch, conditions_dim]
        verbose: Print debug information

    Returns:
        Modified circuit with VIN connectivity guaranteed
    """
    node_types = circuit['node_types']  # [batch, max_nodes]
    edge_existence = circuit['edge_existence']  # [batch, max_nodes, max_nodes]
    edge_values = circuit['edge_values']  # [batch, max_nodes, max_nodes, edge_dim]

    batch_size = node_types.shape[0]
    device = node_types.device

    fixed_count = 0

    for batch_idx in range(batch_size):
        # Find VIN node (type = 1)
        vin_mask = (node_types[batch_idx] == 1)
        if vin_mask.sum() == 0:
            continue  # No VIN in this circuit (shouldn't happen)

        vin_id = vin_mask.nonzero()[0].item()

        # Check VIN connectivity
        vin_degree = edge_existence[batch_idx, vin_id, :].sum() + \
                    edge_existence[batch_idx, :, vin_id].sum()

        if vin_degree < 0.5:  # VIN is disconnected
            if verbose:
                print(f"  Batch {batch_idx}: VIN disconnected, adding edge...")

            # Find VOUT node (type = 2)
            vout_mask = (node_types[batch_idx] == 2)
            if vout_mask.sum() == 0:
                # Fallback: connect to GND (type = 0)
                target_id = (node_types[batch_idx] == 0).nonzero()[0].item()
            else:
                target_id = vout_mask.nonzero()[0].item()

            # Add edge VIN ↔ target
            edge_existence[batch_idx, vin_id, target_id] = 1.0
            edge_existence[batch_idx, target_id, vin_id] = 1.0

            # Infer component values for this edge
            # Use decoder's edge_decoder to predict reasonable values
            with torch.no_grad():
                # Get node embeddings
                vin_type = node_types[batch_idx, vin_id]
                target_type = node_types[batch_idx, target_id]

                node_i_embed = decoder.node_decoder.node_type_embedding(vin_type.unsqueeze(0))
                node_j_embed = decoder.node_decoder.node_type_embedding(target_type.unsqueeze(0))

                # Add positional encoding
                pos_i = decoder.node_decoder.position_embedding(
                    torch.tensor([vin_id], device=device)
                )
                pos_j = decoder.node_decoder.position_embedding(
                    torch.tensor([target_id], device=device)
                )

                node_i_embed = node_i_embed + pos_i
                node_j_embed = node_j_embed + pos_j

                # Predict edge values
                _, edge_value = decoder.edge_decoder(node_i_embed, node_j_embed)

                # Set edge values (symmetric)
                edge_values[batch_idx, vin_id, target_id] = edge_value.squeeze(0)
                edge_values[batch_idx, target_id, vin_id] = edge_value.squeeze(0)

            fixed_count += 1

    if verbose and fixed_count > 0:
        print(f"Fixed {fixed_count}/{batch_size} circuits with disconnected VIN")

    # Update circuit dict
    circuit['edge_existence'] = edge_existence
    circuit['edge_values'] = edge_values

    return circuit


def ensure_graph_connectivity(
    circuit: Dict[str, torch.Tensor],
    decoder,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Ensure entire graph is connected (single component).

    Uses BFS to find connected components and connects them if needed.

    Args:
        circuit: Generated circuit
        decoder: GraphGPT decoder (for value inference)
        verbose: Print debug information

    Returns:
        Modified circuit with full connectivity
    """
    node_types = circuit['node_types']
    edge_existence = circuit['edge_existence']
    edge_values = circuit['edge_values']

    batch_size = node_types.shape[0]
    max_nodes = node_types.shape[1]
    device = node_types.device

    for batch_idx in range(batch_size):
        # Get non-MASK nodes
        valid_mask = (node_types[batch_idx] < 4)  # MASK = 4
        valid_nodes = valid_mask.nonzero().squeeze(-1).cpu().numpy()

        if len(valid_nodes) <= 1:
            continue

        # Build adjacency list for BFS
        adj = edge_existence[batch_idx].cpu().numpy()

        # Find connected components using BFS
        visited = set()
        components = []

        for start_node in valid_nodes:
            if start_node in visited:
                continue

            # BFS from start_node
            component = []
            queue = [start_node]
            visited.add(start_node)

            while queue:
                node = queue.pop(0)
                component.append(node)

                # Check neighbors
                for neighbor in range(max_nodes):
                    if neighbor not in visited and adj[node, neighbor] > 0.5:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        # If multiple components, connect them
        if len(components) > 1:
            if verbose:
                print(f"  Batch {batch_idx}: {len(components)} components, connecting...")

            # Connect each component to the first one
            for i in range(1, len(components)):
                # Pick arbitrary nodes to connect
                node_a = components[0][0]  # From first component
                node_b = components[i][0]  # From i-th component

                # Add edge
                edge_existence[batch_idx, node_a, node_b] = 1.0
                edge_existence[batch_idx, node_b, node_a] = 1.0

                # Infer edge values
                with torch.no_grad():
                    type_a = node_types[batch_idx, node_a]
                    type_b = node_types[batch_idx, node_b]

                    embed_a = decoder.node_decoder.node_type_embedding(type_a.unsqueeze(0))
                    embed_b = decoder.node_decoder.node_type_embedding(type_b.unsqueeze(0))

                    pos_a = decoder.node_decoder.position_embedding(
                        torch.tensor([node_a], device=device)
                    )
                    pos_b = decoder.node_decoder.position_embedding(
                        torch.tensor([node_b], device=device)
                    )

                    embed_a = embed_a + pos_a
                    embed_b = embed_b + pos_b

                    _, edge_value = decoder.edge_decoder(embed_a, embed_b)

                    edge_values[batch_idx, node_a, node_b] = edge_value.squeeze(0)
                    edge_values[batch_idx, node_b, node_a] = edge_value.squeeze(0)

    circuit['edge_existence'] = edge_existence
    circuit['edge_values'] = edge_values

    return circuit


def guided_generate(
    decoder,
    latent_code: torch.Tensor,
    conditions: torch.Tensor,
    ensure_vin: bool = True,
    ensure_connected: bool = True,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate circuit with structural guarantees.

    This is a drop-in replacement for decoder.generate() that ensures:
    1. VIN is connected (if ensure_vin=True)
    2. Graph is fully connected (if ensure_connected=True)

    Args:
        decoder: GraphGPT decoder
        latent_code: Latent from encoder [batch, latent_dim]
        conditions: Specifications [batch, conditions_dim]
        ensure_vin: Guarantee VIN connectivity
        ensure_connected: Guarantee graph connectivity
        verbose: Print debug information

    Returns:
        Generated circuit with structural guarantees
    """
    # Standard generation
    circuit = decoder.generate(latent_code, conditions)

    # Post-process for connectivity
    if ensure_vin:
        circuit = ensure_vin_connectivity(
            circuit, decoder, latent_code, conditions, verbose
        )

    if ensure_connected:
        circuit = ensure_graph_connectivity(circuit, decoder, verbose)

    return circuit


if __name__ == '__main__':
    """Test guided generation."""
    print("Testing guided generation...")

    # This would require loading actual models
    # For now, just show the API

    print("""
Example usage:

    from ml.models.node_decoder import GraphGPTDecoder
    from ml.models.guided_generation import guided_generate

    # Load decoder
    decoder = GraphGPTDecoder(...)
    decoder.load_state_dict(checkpoint)

    # Generate with guarantees
    circuit = guided_generate(
        decoder,
        latent_code=latent,
        conditions=specs,
        ensure_vin=True,       # Guarantee VIN connected
        ensure_connected=True, # Guarantee full connectivity
        verbose=True           # Show fixes
    )

    # Use circuit normally
    # Now VIN connectivity is guaranteed!
    """)

    print("✅ Module loaded successfully")
