#!/usr/bin/env python3
"""
Comprehensive test suite for specification-based circuit generation.

Tests all 6 filter types with round-trip validation.
"""

import sys
sys.path.insert(0, 'tools')

from circuit_generator import FilterGenerator, extract_poles_zeros_gain_analytical
import numpy as np


def check_filter_spec(filter_type, gen_method, spec, tolerance_freq=0.01, tolerance_Q=0.05):
    """
    Generic test for any filter specification method.

    Args:
        filter_type: Filter type string
        gen_method: Generator method to call
        spec: Dictionary of specifications
        tolerance_freq: Frequency error tolerance (default 1%)
        tolerance_Q: Q factor error tolerance (default 5%)

    Returns:
        True if passed, False if failed
    """
    print(f"\n{'='*70}")
    print(f"TEST: {filter_type.upper().replace('_', '-')} Filter")
    print(f"{'='*70}")

    # Generate circuit
    if 'fc' in spec:
        actual_freq = gen_method(fc=spec['fc'])
        expected_freq = spec['fc']
        freq_key = 'fc'
    elif 'f0' in spec:
        Q = spec.get('Q', 5.0)
        actual_freq = gen_method(f0=spec['f0'], Q=Q)
        expected_freq = spec['f0']
        freq_key = 'f0'
    else:
        print("❌ FAIL: Invalid spec")
        return False

    gen = gen_method.__self__  # Get the generator instance

    print(f"\nSpecification:")
    print(f"  {freq_key} = {expected_freq} Hz")
    if 'Q' in spec:
        print(f"  Q  = {spec['Q']}")

    print(f"\nGenerated components:")
    for comp in gen.components:
        value = comp['value']
        if comp['type'] == 'R':
            if value >= 1000:
                print(f"  {comp['name']:<12}: {value/1000:.2f} kΩ")
            else:
                print(f"  {comp['name']:<12}: {value:.2f} Ω")
        elif comp['type'] == 'L':
            print(f"  {comp['name']:<12}: {value*1e3:.4f} mH")
        elif comp['type'] == 'C':
            if value >= 1e-6:
                print(f"  {comp['name']:<12}: {value*1e6:.2f} μF")
            else:
                print(f"  {comp['name']:<12}: {value*1e9:.2f} nF")

    # Extract poles, zeros, gain
    poles, zeros, gain = extract_poles_zeros_gain_analytical(
        gen.filter_type, gen.components
    )

    print(f"\nPole-zero analysis:")
    print(f"  Poles: {len(poles)}")
    for i, p in enumerate(poles):
        if abs(p.imag) < 1e-6:
            print(f"    p{i+1}: {p.real:.2f}")
        else:
            print(f"    p{i+1}: {p.real:.2f} ± j{abs(p.imag):.2f}")

    print(f"  Zeros: {len(zeros)}")
    for i, z in enumerate(zeros):
        if abs(z) < 1e-6:
            print(f"    z{i+1}: 0")
        elif abs(z.imag) < 1e-6:
            print(f"    z{i+1}: {z.real:.2f}")
        else:
            print(f"    z{i+1}: {z.real:.2f} ± j{abs(z.imag):.2f}")

    print(f"  Gain:  {gain:.4e}")

    # Validation
    print(f"\nValidation:")

    if filter_type in ['low_pass', 'high_pass']:
        # Check pole location
        expected_pole = -2 * np.pi * expected_freq
        actual_pole = poles[0].real
        error_freq = abs(actual_pole - expected_pole) / abs(expected_pole)

        print(f"  Expected pole: {expected_pole:.2f}")
        print(f"  Actual pole:   {actual_pole:.2f}")
        print(f"  Error:         {error_freq*100:.4f}%")

        passed = error_freq < tolerance_freq

    else:
        # Check resonant frequency
        expected_ω0 = 2 * np.pi * expected_freq
        actual_ω0 = abs(poles[0])
        error_freq = abs(actual_ω0 - expected_ω0) / expected_ω0

        print(f"  Expected ω0: {expected_ω0:.2f}")
        print(f"  Actual ω0:   {actual_ω0:.2f}")
        print(f"  Freq error:  {error_freq*100:.4f}%")

        passed = error_freq < tolerance_freq

        # Check Q factor if specified
        if 'Q' in spec:
            ζ = -poles[0].real / actual_ω0
            actual_Q = 1 / (2 * ζ)
            error_Q = abs(actual_Q - spec['Q']) / spec['Q']

            print(f"  Expected Q:  {spec['Q']:.2f}")
            print(f"  Actual Q:    {actual_Q:.2f}")
            print(f"  Q error:     {error_Q*100:.4f}%")

            passed = passed and (error_Q < tolerance_Q)

    if passed:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL")

    return passed


def run_all_tests():
    """Run comprehensive tests for all 6 filter types."""
    print("="*70)
    print("COMPREHENSIVE SPECIFICATION-BASED GENERATION TEST SUITE")
    print("="*70)

    gen = FilterGenerator()
    results = []

    # Test 1: Low-pass at 1kHz
    results.append(check_filter_spec(
        'low_pass',
        gen.from_low_pass_spec,
        {'fc': 1000}
    ))

    # Test 2: High-pass at 10kHz
    results.append(check_filter_spec(
        'high_pass',
        gen.from_high_pass_spec,
        {'fc': 10000}
    ))

    # Test 3: Band-pass at 10kHz, Q=5
    results.append(check_filter_spec(
        'band_pass',
        gen.from_band_pass_spec,
        {'f0': 10000, 'Q': 5.0}
    ))

    # Test 4: Band-stop at 10kHz, Q=10
    results.append(check_filter_spec(
        'band_stop',
        gen.from_band_stop_spec,
        {'f0': 10000, 'Q': 10.0}
    ))

    # Test 5: RLC series at 20kHz, Q=5
    results.append(check_filter_spec(
        'rlc_series',
        gen.from_rlc_series_spec,
        {'f0': 20000, 'Q': 5.0}
    ))

    # Test 6: RLC parallel at 20kHz, Q=5
    results.append(check_filter_spec(
        'rlc_parallel',
        gen.from_rlc_parallel_spec,
        {'f0': 20000, 'Q': 5.0}
    ))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")

    filter_names = [
        'Low-pass', 'High-pass', 'Band-pass',
        'Band-stop', 'RLC Series', 'RLC Parallel'
    ]

    for name, passed in zip(filter_names, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<15}: {status}")

    print()
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
