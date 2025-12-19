#!/usr/bin/env python3
"""Test basic specification-based generation (low-pass and high-pass)."""

import sys
sys.path.insert(0, 'tools')

from circuit_generator import FilterGenerator, extract_poles_zeros_gain_analytical
import numpy as np


def test_low_pass_spec():
    """Test low-pass filter generation from specification."""
    print("="*70)
    print("TEST: Low-Pass Filter from Specification")
    print("="*70)

    gen = FilterGenerator()

    # Test 1kHz cutoff
    fc_desired = 1000
    fc_actual = gen.from_low_pass_spec(fc=fc_desired)

    print(f"\nDesired cutoff: {fc_desired} Hz")
    print(f"Actual cutoff:  {fc_actual:.2f} Hz")
    print(f"\nGenerated components:")
    for comp in gen.components:
        if comp['type'] == 'R':
            print(f"  {comp['name']}: {comp['value']:.2f} Ω")
        elif comp['type'] == 'C':
            print(f"  {comp['name']}: {comp['value']*1e9:.2f} nF")

    # Validate with pole extraction
    poles, zeros, gain = extract_poles_zeros_gain_analytical(
        gen.filter_type, gen.components
    )

    print(f"\nPole-zero analysis:")
    print(f"  Poles: {poles}")
    print(f"  Zeros: {zeros}")
    print(f"  Gain:  {gain:.4f}")

    # Check pole location
    expected_pole = -2 * np.pi * fc_desired
    actual_pole = poles[0].real
    error = abs(actual_pole - expected_pole) / abs(expected_pole)

    print(f"\nValidation:")
    print(f"  Expected pole: {expected_pole:.2f}")
    print(f"  Actual pole:   {actual_pole:.2f}")
    print(f"  Error:         {error*100:.4f}%")

    if error < 0.01:
        print("  ✅ PASS: Pole within 1% tolerance")
        return True
    else:
        print("  ❌ FAIL: Pole error > 1%")
        return False


def test_high_pass_spec():
    """Test high-pass filter generation from specification."""
    print("\n" + "="*70)
    print("TEST: High-Pass Filter from Specification")
    print("="*70)

    gen = FilterGenerator()

    # Test 10kHz cutoff
    fc_desired = 10000
    fc_actual = gen.from_high_pass_spec(fc=fc_desired)

    print(f"\nDesired cutoff: {fc_desired} Hz")
    print(f"Actual cutoff:  {fc_actual:.2f} Hz")
    print(f"\nGenerated components:")
    for comp in gen.components:
        if comp['type'] == 'R':
            print(f"  {comp['name']}: {comp['value']:.2f} Ω")
        elif comp['type'] == 'C':
            print(f"  {comp['name']}: {comp['value']*1e9:.2f} nF")

    # Validate with pole extraction
    poles, zeros, gain = extract_poles_zeros_gain_analytical(
        gen.filter_type, gen.components
    )

    print(f"\nPole-zero analysis:")
    print(f"  Poles: {poles}")
    print(f"  Zeros: {zeros}")
    print(f"  Gain:  {gain:.4f}")

    # Check pole location
    expected_pole = -2 * np.pi * fc_desired
    actual_pole = poles[0].real
    error = abs(actual_pole - expected_pole) / abs(expected_pole)

    print(f"\nValidation:")
    print(f"  Expected pole: {expected_pole:.2f}")
    print(f"  Actual pole:   {actual_pole:.2f}")
    print(f"  Error:         {error*100:.4f}%")

    if error < 0.01:
        print("  ✅ PASS: Pole within 1% tolerance")
        return True
    else:
        print("  ❌ FAIL: Pole error > 1%")
        return False


def test_frequency_limits():
    """Test edge cases for frequency limits."""
    print("\n" + "="*70)
    print("TEST: Frequency Limit Validation")
    print("="*70)

    gen = FilterGenerator()

    # Test very low frequency
    print("\n1. Very low frequency (10 Hz):")
    try:
        fc = gen.from_low_pass_spec(fc=10)
        print(f"   ✅ Generated at {fc:.2f} Hz")
        print(f"   Components: R={gen.components[0]['value']:.0f}Ω, "
              f"C={gen.components[1]['value']*1e6:.2f}μF")
    except ValueError as e:
        print(f"   ❌ Error: {e}")

    # Test very high frequency
    print("\n2. Very high frequency (1 MHz):")
    try:
        fc = gen.from_high_pass_spec(fc=1e6)
        print(f"   ✅ Generated at {fc:.2e} Hz")
        print(f"   Components: R={gen.components[1]['value']:.0f}Ω, "
              f"C={gen.components[0]['value']*1e9:.2f}nF")
    except ValueError as e:
        print(f"   ❌ Error: {e}")

    # Test unrealizable frequency (too high)
    print("\n3. Unrealizable frequency (100 MHz):")
    try:
        fc = gen.from_low_pass_spec(fc=100e6)
        print(f"   ⚠️  Unexpectedly succeeded at {fc:.2e} Hz")
    except ValueError as e:
        print(f"   ✅ Correctly rejected: {str(e)[:60]}...")


if __name__ == '__main__':
    results = []

    results.append(test_low_pass_spec())
    results.append(test_high_pass_spec())
    test_frequency_limits()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
