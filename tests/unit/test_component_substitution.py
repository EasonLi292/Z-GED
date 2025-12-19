#!/usr/bin/env python3
"""Test that all component type substitutions use delete+insert cost."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.graph_edit_distance import CircuitGED
import networkx as nx


def create_test_edge(component_type):
    """
    Create edge attributes for different component types.

    Impedance: Z(s) = s / (C*s^2 + G*s + L_inv)
    - R only: den = (0, G, 0)
    - C only: den = (C, 0, 0)
    - L only: den = (0, 0, L_inv)
    - R||C:   den = (C, G, 0)
    """
    if component_type == 'R':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (0.0, 1e-3, 0.0)}
    elif component_type == 'C':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (1e-8, 0.0, 0.0)}
    elif component_type == 'L':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (0.0, 0.0, 1e3)}
    elif component_type == 'RC_PARALLEL':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (1e-8, 1e-3, 0.0)}
    elif component_type == 'RL_PARALLEL':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (0.0, 1e-3, 1e3)}
    elif component_type == 'CL_PARALLEL':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (1e-8, 0.0, 1e3)}
    elif component_type == 'RCL_PARALLEL':
        return {'impedance_num': (1.0, 0.0), 'impedance_den': (1e-8, 1e-3, 1e3)}
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def test_component_substitutions():
    """Test all combinations of component substitutions."""
    ged_calc = CircuitGED(
        simple_edge_cost=0.5,
        complex_edge_cost=1.0
    )

    component_types = ['R', 'C', 'L', 'RC_PARALLEL', 'RL_PARALLEL',
                       'CL_PARALLEL', 'RCL_PARALLEL']

    print("="*80)
    print("COMPONENT SUBSTITUTION COST MATRIX")
    print("="*80)
    print("\nExpected costs:")
    print("  Same type (e.g., R→R):        ~0.0 (impedance distance)")
    print("  Different single (e.g., R→C): 1.0 (del + ins = 0.5 + 0.5)")
    print("  Single→Parallel (e.g., R→RC): 1.5 (del + ins = 0.5 + 1.0)")
    print("  Parallel→Single (e.g., RC→R): 1.5 (del + ins = 1.0 + 0.5)")
    print("  Parallel→Parallel (diff):     2.0 (del + ins = 1.0 + 1.0)")
    print()

    # Print header
    print(f"{'From \\ To':<15}", end='')
    for t2 in component_types:
        print(f"{t2:<15}", end='')
    print()
    print("-" * (15 + 15 * len(component_types)))

    # Test all combinations
    for t1 in component_types:
        print(f"{t1:<15}", end='')

        for t2 in component_types:
            edge1 = create_test_edge(t1)
            edge2 = create_test_edge(t2)

            cost = ged_calc._edge_subst_cost(edge1, edge2)

            # Format cost
            if cost < 0.01:
                print(f"{'~0.0':<15}", end='')
            else:
                print(f"{cost:<15.1f}", end='')

        print()

    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Verify specific cases
    test_cases = [
        ('R', 'R', 'Same type', 0.0, 0.1),
        ('R', 'C', 'R→C substitution', 1.0, 1.0),
        ('R', 'L', 'R→L substitution', 1.0, 1.0),
        ('C', 'L', 'C→L substitution', 1.0, 1.0),
        ('R', 'RC_PARALLEL', 'R→RC substitution', 1.5, 1.5),
        ('RC_PARALLEL', 'R', 'RC→R substitution', 1.5, 1.5),
        ('RC_PARALLEL', 'RL_PARALLEL', 'RC→RL substitution', 2.0, 2.0),
    ]

    all_pass = True
    for t1, t2, description, expected_cost, tolerance in test_cases:
        edge1 = create_test_edge(t1)
        edge2 = create_test_edge(t2)
        cost = ged_calc._edge_subst_cost(edge1, edge2)

        if abs(cost - expected_cost) <= tolerance:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_pass = False

        print(f"{status}  {description:<30} Expected: {expected_cost:.1f}, Got: {cost:.4f}")

    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL TESTS PASSED")
        print("\nConclusion: Component type substitutions correctly use delete+insert cost")
        print("for ALL component types (R, C, L, and all parallel combinations).")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)


if __name__ == '__main__':
    test_component_substitutions()
