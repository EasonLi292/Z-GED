#!/usr/bin/env python3
"""Test band-pass specification-based generation."""

import sys
sys.path.insert(0, 'tools')

from circuit_generator import FilterGenerator, extract_poles_zeros_gain_analytical
import numpy as np


def test_band_pass_spec():
    """Test band-pass filter generation from specification."""
    print("="*70)
    print("TEST: Band-Pass Filter from Specification")
    print("="*70)

    gen = FilterGenerator()

    # Test 10kHz center frequency, Q=5
    f0_desired = 10000
    Q_desired = 5.0
    f0_actual = gen.from_band_pass_spec(f0=f0_desired, Q=Q_desired)

    print(f"\nDesired center frequency: {f0_desired} Hz")
    print(f"Desired Q factor:         {Q_desired}")
    print(f"Actual center frequency:  {f0_actual:.2f} Hz")
    print(f"Bandwidth (BW = f0/Q):    {f0_actual/Q_desired:.2f} Hz")

    print(f"\nGenerated components:")
    for comp in gen.components:
        if comp['type'] == 'R':
            print(f"  {comp['name']}: {comp['value']:.2f} Ω")
        elif comp['type'] == 'L':
            print(f"  {comp['name']}: {comp['value']*1e3:.4f} mH")
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

    # Check resonant frequency
    expected_ω0 = 2 * np.pi * f0_desired
    actual_ω0 = abs(poles[0])
    error_freq = abs(actual_ω0 - expected_ω0) / expected_ω0

    # Check Q factor
    ζ = -poles[0].real / actual_ω0
    actual_Q = 1 / (2 * ζ)
    error_Q = abs(actual_Q - Q_desired) / Q_desired

    print(f"\nValidation:")
    print(f"  Expected ω0: {expected_ω0:.2f}")
    print(f"  Actual ω0:   {actual_ω0:.2f}")
    print(f"  Freq error:  {error_freq*100:.4f}%")
    print(f"  Expected Q:  {Q_desired:.2f}")
    print(f"  Actual Q:    {actual_Q:.2f}")
    print(f"  Q error:     {error_Q*100:.4f}%")

    if error_freq < 0.01 and error_Q < 0.05:
        print("  ✅ PASS: Frequency within 1%, Q within 5%")
        return True
    else:
        print(f"  ❌ FAIL: Error too large")
        return False


def test_different_Q_factors():
    """Test band-pass with various Q factors."""
    print("\n" + "="*70)
    print("TEST: Band-Pass with Different Q Factors")
    print("="*70)

    gen = FilterGenerator()
    f0 = 10000  # 10kHz

    Q_values = [1.0, 5.0, 10.0, 20.0]

    for Q in Q_values:
        print(f"\nQ = {Q}:")
        try:
            f0_actual = gen.from_band_pass_spec(f0=f0, Q=Q)

            # Extract Q from poles
            poles, _, _ = extract_poles_zeros_gain_analytical(
                gen.filter_type, gen.components
            )
            ω0 = abs(poles[0])
            ζ = -poles[0].real / ω0
            actual_Q = 1 / (2 * ζ)

            print(f"  Bandwidth:  {f0_actual/Q:.2f} Hz")
            print(f"  Actual Q:   {actual_Q:.2f}")
            print(f"  R={gen.components[0]['value']:.2f}Ω, "
                  f"L={gen.components[1]['value']*1e3:.4f}mH, "
                  f"C={gen.components[2]['value']*1e9:.2f}nF")
        except ValueError as e:
            print(f"  ❌ Error: {e}")


def test_frequency_range():
    """Test band-pass at different frequencies."""
    print("\n" + "="*70)
    print("TEST: Band-Pass Frequency Range")
    print("="*70)

    gen = FilterGenerator()
    Q = 5.0

    frequencies = [1000, 5000, 10000, 50000, 100000]

    for f0 in frequencies:
        print(f"\nf0 = {f0} Hz:")
        try:
            f0_actual = gen.from_band_pass_spec(f0=f0, Q=Q)
            print(f"  ✅ Generated at {f0_actual:.2f} Hz")
            print(f"  L={gen.components[1]['value']*1e3:.4f}mH, "
                  f"C={gen.components[2]['value']*1e9:.2f}nF")
        except ValueError as e:
            print(f"  ❌ Error: {str(e)[:60]}...")


if __name__ == '__main__':
    results = []

    results.append(test_band_pass_spec())
    test_different_Q_factors()
    test_frequency_range()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
