#!/usr/bin/env python3
"""
Demonstration of Specification-Based Circuit Generation

Shows how to generate circuits from user specifications for all 6 filter types.
"""

import sys
sys.path.insert(0, 'tools')

from circuit_generator import FilterGenerator, extract_poles_zeros_gain_analytical, create_compact_graph_representation
import numpy as np


def demo_specification_to_circuit():
    """
    Demonstrate the complete workflow: Specification → Circuit → Verification
    """
    print("="*80)
    print("SPECIFICATION-DRIVEN CIRCUIT GENERATION DEMO")
    print("="*80)
    print("\nWorkflow: User Specs → Pole-Zero → Component Values → Circuit Graph\n")

    gen = FilterGenerator()

    # Example 1: User wants a low-pass filter at 1kHz
    print("\n" + "="*80)
    print("Example 1: Low-Pass Filter at 1kHz")
    print("="*80)
    print("\nUser specification: 'I need a low-pass filter with cutoff at 1kHz'")

    fc = gen.from_low_pass_spec(fc=1000)
    print(f"\n✅ Generated low-pass filter at {fc:.2f} Hz")
    print(f"\nComponents:")
    for comp in gen.components:
        if comp['type'] == 'R':
            print(f"  {comp['name']}: {comp['value']:.2f} Ω")
        elif comp['type'] == 'C':
            print(f"  {comp['name']}: {comp['value']*1e9:.2f} nF")

    # Verify with pole extraction
    poles, zeros, gain = extract_poles_zeros_gain_analytical(gen.filter_type, gen.components)
    print(f"\nVerification:")
    print(f"  Pole: {poles[0]:.2f}")
    print(f"  Expected: {-2*np.pi*1000:.2f}")
    print(f"  Match: {'✅ Perfect!' if abs(poles[0] - (-2*np.pi*1000))/abs(2*np.pi*1000) < 0.01 else '❌ Error'}")

    # Example 2: User wants a band-pass filter
    print("\n" + "="*80)
    print("Example 2: Band-Pass Filter at 10kHz with Q=5")
    print("="*80)
    print("\nUser specification: 'I need a band-pass filter centered at 10kHz")
    print("                     with bandwidth of 2kHz (Q=5)'")

    f0 = gen.from_band_pass_spec(f0=10000, Q=5.0)
    print(f"\n✅ Generated band-pass filter at {f0:.2f} Hz, BW={f0/5:.2f} Hz")
    print(f"\nComponents:")
    for comp in gen.components:
        if comp['type'] == 'R':
            print(f"  {comp['name']}: {comp['value']:.2f} Ω")
        elif comp['type'] == 'L':
            print(f"  {comp['name']}: {comp['value']*1e3:.4f} mH")
        elif comp['type'] == 'C':
            print(f"  {comp['name']}: {comp['value']*1e9:.2f} nF")

    poles, zeros, gain = extract_poles_zeros_gain_analytical(gen.filter_type, gen.components)
    actual_Q = abs(poles[0]) / (-2*poles[0].real)
    print(f"\nVerification:")
    print(f"  Center frequency: {abs(poles[0])/(2*np.pi):.2f} Hz")
    print(f"  Q factor: {actual_Q:.2f}")
    print(f"  Match: {'✅ Perfect!' if abs(actual_Q - 5.0)/5.0 < 0.05 else '❌ Error'}")

    # Example 3: All filter types at once
    print("\n" + "="*80)
    print("Example 3: Generate All Filter Types")
    print("="*80)
    print("\nGenerating one of each filter type...\n")

    specs = [
        ("Low-Pass",     lambda: gen.from_low_pass_spec(fc=1000),              "1kHz cutoff"),
        ("High-Pass",    lambda: gen.from_high_pass_spec(fc=10000),            "10kHz cutoff"),
        ("Band-Pass",    lambda: gen.from_band_pass_spec(f0=10000, Q=5.0),     "10kHz, Q=5"),
        ("Band-Stop",    lambda: gen.from_band_stop_spec(f0=10000, Q=10.0),    "10kHz notch, Q=10"),
        ("RLC Series",   lambda: gen.from_rlc_series_spec(f0=20000, Q=5.0),    "20kHz resonance, Q=5"),
        ("RLC Parallel", lambda: gen.from_rlc_parallel_spec(f0=20000, Q=5.0),  "20kHz resonance, Q=5"),
    ]

    for name, gen_func, spec in specs:
        freq = gen_func()
        num_components = len(gen.components)
        print(f"{name:<15}: {spec:<30} → {num_components} components")

    # Example 4: Export to graph for ML
    print("\n" + "="*80)
    print("Example 4: Export to Graph Representation for ML")
    print("="*80)

    gen.from_band_pass_spec(f0=10000, Q=5.0)

    # Create compact graph representation for ML
    graph = create_compact_graph_representation(gen.graph, gen.filter_type)

    print(f"\nGenerated band-pass filter at 10kHz")
    print(f"Graph structure:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Node features: 4D one-hot [GND, VIN, VOUT, INTERNAL]")
    print(f"  Edge features: Impedance encoding [C, G, L_inv]")

    for node in graph.nodes(data=True):
        print(f"  Node {node[0]}: {node[1]['features']}")

    print("\n✅ Ready for GED-based ML or circuit generation pipeline!")

    # Example 5: Frequency limits
    print("\n" + "="*80)
    print("Example 5: Automatic Component Adjustment")
    print("="*80)
    print("\nThe system automatically adjusts components to stay within practical ranges\n")

    test_frequencies = [10, 100, 1000, 10000, 100000, 1000000]

    for fc in test_frequencies:
        try:
            actual_fc = gen.from_low_pass_spec(fc=fc)
            R = gen.components[0]['value']
            C = gen.components[1]['value']
            print(f"fc={fc:>7} Hz: ✅ R={R:>8.1f}Ω, C={C*1e9:>6.2f}nF")
        except ValueError as e:
            print(f"fc={fc:>7} Hz: ❌ {str(e)[:50]}...")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The specification-based generation system allows users to:

1. Specify filter type and characteristics (frequency, Q factor)
2. Automatically calculate component values (R, L, C)
3. Generate circuit graphs for ML pipelines
4. Verify correctness with pole-zero extraction

Key features:
✅ 6 filter types supported
✅ Analytical inverse formulas (exact, not optimization)
✅ Component validation and auto-adjustment
✅ Round-trip verification (< 1% error)
✅ Integration with existing GED and ML pipeline

Next steps:
- Use generated circuits for ML training
- Build specification → circuit generation pipeline
- Integrate with GED for circuit similarity search
""")


if __name__ == '__main__':
    demo_specification_to_circuit()
