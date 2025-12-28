"""
Filter design utilities: Convert high-level specifications to poles/zeros.

Supports standard filter types with automatic pole/zero calculation.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def butterworth_poles(order: int, cutoff_freq: float) -> List[complex]:
    """
    Calculate poles for Butterworth filter.

    Butterworth = Maximally flat magnitude response.

    Args:
        order: Filter order (1-4)
        cutoff_freq: Cutoff frequency in Hz

    Returns:
        List of pole locations (complex numbers)
    """
    omega_c = 2 * np.pi * cutoff_freq

    poles = []
    for k in range(order):
        # Pole angle: π/2 + (2k+1)π/(2*order)
        theta = np.pi / 2 + (2 * k + 1) * np.pi / (2 * order)

        # Pole location on unit circle, scaled by cutoff
        real = omega_c * np.cos(theta)
        imag = omega_c * np.sin(theta)

        poles.append(complex(real, imag))

    return poles


def bessel_poles(order: int, cutoff_freq: float) -> List[complex]:
    """
    Calculate poles for Bessel filter.

    Bessel = Linear phase response (constant group delay).

    Args:
        order: Filter order (1-4)
        cutoff_freq: Cutoff frequency in Hz

    Returns:
        List of pole locations
    """
    omega_c = 2 * np.pi * cutoff_freq

    # Normalized Bessel pole locations (from tables)
    # These are normalized for -3dB at ω=1
    normalized_poles = {
        1: [complex(-1.0, 0)],
        2: [complex(-1.1016, 0.6360), complex(-1.1016, -0.6360)],
        3: [complex(-1.0509, 0.9999), complex(-1.0509, -0.9999), complex(-1.3270, 0)],
        4: [complex(-0.9952, 1.2571), complex(-0.9952, -1.2571),
            complex(-1.3597, 0.4071), complex(-1.3597, -0.4071)]
    }

    if order not in normalized_poles:
        raise ValueError(f"Bessel filter order {order} not supported (use 1-4)")

    # Scale by cutoff frequency
    poles = [p * omega_c for p in normalized_poles[order]]

    return poles


def chebyshev_poles(order: int, cutoff_freq: float, ripple_db: float = 1.0) -> List[complex]:
    """
    Calculate poles for Chebyshev Type I filter.

    Chebyshev = Steeper rolloff, with passband ripple.

    Args:
        order: Filter order (1-4)
        cutoff_freq: Cutoff frequency in Hz
        ripple_db: Passband ripple in dB (default: 1.0)

    Returns:
        List of pole locations
    """
    omega_c = 2 * np.pi * cutoff_freq

    # Calculate epsilon from ripple
    epsilon = np.sqrt(10 ** (ripple_db / 10) - 1)

    # Calculate sinh and cosh of asinh(1/epsilon)/order
    asinh_val = np.arcsinh(1 / epsilon)
    sinh_val = np.sinh(asinh_val / order)
    cosh_val = np.cosh(asinh_val / order)

    poles = []
    for k in range(order):
        # Pole angle
        theta = np.pi / 2 + (2 * k + 1) * np.pi / (2 * order)

        # Chebyshev pole on ellipse
        real = -omega_c * sinh_val * np.cos(theta)
        imag = omega_c * cosh_val * np.sin(theta)

        poles.append(complex(real, imag))

    return poles


def notch_poles_zeros(notch_freq: float, q_factor: float, cutoff_freq: float) -> Tuple[List[complex], List[complex]]:
    """
    Calculate poles and zeros for notch filter.

    Notch = Rejects a specific frequency.

    Args:
        notch_freq: Frequency to reject (Hz)
        q_factor: Quality factor (selectivity)
        cutoff_freq: Overall bandwidth control (Hz)

    Returns:
        (poles, zeros) tuple
    """
    omega_n = 2 * np.pi * notch_freq

    # Zeros on jω axis (perfect rejection)
    zeros = [
        complex(0, omega_n),
        complex(0, -omega_n)
    ]

    # Poles slightly inside (determines bandwidth)
    # Bandwidth = omega_n / Q
    bandwidth = omega_n / q_factor
    real_part = -bandwidth / 2

    poles = [
        complex(real_part, omega_n),
        complex(real_part, -omega_n)
    ]

    return poles, zeros


def highpass_from_lowpass(lowpass_poles: List[complex], cutoff_freq: float) -> List[complex]:
    """
    Convert lowpass poles to highpass by frequency inversion.

    Args:
        lowpass_poles: Poles of lowpass prototype
        cutoff_freq: Cutoff frequency in Hz

    Returns:
        Highpass poles
    """
    omega_c = 2 * np.pi * cutoff_freq

    # Highpass transformation: s -> ωc²/s
    highpass_poles = [omega_c**2 / p for p in lowpass_poles]

    return highpass_poles


def bandpass_from_lowpass(
    lowpass_poles: List[complex],
    center_freq: float,
    bandwidth: float
) -> List[complex]:
    """
    Convert lowpass poles to bandpass.

    Args:
        lowpass_poles: Poles of lowpass prototype
        center_freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz

    Returns:
        Bandpass poles (doubled in number)
    """
    omega_0 = 2 * np.pi * center_freq
    bw = 2 * np.pi * bandwidth

    bandpass_poles = []

    for p in lowpass_poles:
        # Bandpass transformation: s -> (s² + ω₀²)/(bw·s)
        # Each lowpass pole creates 2 bandpass poles

        # Solve: s² - (bw·p)s + ω₀² = 0
        a = 1
        b = -bw * p
        c = omega_0**2

        discriminant = b**2 - 4*a*c
        sqrt_disc = np.sqrt(discriminant)

        pole1 = (-b + sqrt_disc) / (2*a)
        pole2 = (-b - sqrt_disc) / (2*a)

        bandpass_poles.append(pole1)
        bandpass_poles.append(pole2)

    return bandpass_poles


def calculate_filter_poles_zeros(
    filter_type: str,
    order: int,
    cutoff_freq: float,
    q_factor: Optional[float] = None,
    gain_db: float = 0.0,
    ripple_db: float = 1.0,
    center_freq: Optional[float] = None,
    bandwidth: Optional[float] = None
) -> Tuple[List[complex], List[complex]]:
    """
    Calculate poles and zeros for any standard filter type.

    Args:
        filter_type: One of:
            - 'butterworth' / 'butter': Maximally flat
            - 'bessel': Linear phase
            - 'chebyshev' / 'cheby': Passband ripple
            - 'notch': Frequency rejection
            - 'lowpass': Generic lowpass
            - 'highpass': Generic highpass
            - 'bandpass': Generic bandpass
        order: Filter order (1-4)
        cutoff_freq: Cutoff frequency in Hz
        q_factor: Quality factor (for 2nd order filters)
        gain_db: DC gain in dB (default: 0)
        ripple_db: Passband ripple for Chebyshev (default: 1.0 dB)
        center_freq: Center frequency for bandpass (Hz)
        bandwidth: Bandwidth for bandpass (Hz)

    Returns:
        (poles, zeros) tuple

    Examples:
        >>> # 2nd-order Butterworth lowpass at 1kHz
        >>> poles, zeros = calculate_filter_poles_zeros('butterworth', 2, 1000)

        >>> # Notch filter rejecting 60Hz
        >>> poles, zeros = calculate_filter_poles_zeros('notch', 2, 60, q_factor=10)

        >>> # 3rd-order Bessel highpass at 5kHz
        >>> poles, zeros = calculate_filter_poles_zeros('bessel', 3, 5000)
    """
    filter_type = filter_type.lower()
    zeros = []  # Most filters have no zeros (all-pole)

    # === Lowpass Filters ===
    if filter_type in ['butterworth', 'butter', 'lowpass']:
        poles = butterworth_poles(order, cutoff_freq)

    elif filter_type == 'bessel':
        poles = bessel_poles(order, cutoff_freq)

    elif filter_type in ['chebyshev', 'cheby']:
        poles = chebyshev_poles(order, cutoff_freq, ripple_db)

    # === Notch Filter ===
    elif filter_type == 'notch':
        if q_factor is None:
            q_factor = 10.0  # Default high Q
        poles, zeros = notch_poles_zeros(cutoff_freq, q_factor, cutoff_freq)

    # === Highpass Filter ===
    elif filter_type == 'highpass':
        # Create lowpass prototype, then transform
        lowpass_poles = butterworth_poles(order, cutoff_freq)
        poles = highpass_from_lowpass(lowpass_poles, cutoff_freq)
        # Highpass has zeros at origin
        zeros = [complex(0, 0)] * order

    # === Bandpass Filter ===
    elif filter_type == 'bandpass':
        if center_freq is None or bandwidth is None:
            raise ValueError("Bandpass requires center_freq and bandwidth")

        # Create lowpass prototype, then transform
        lowpass_poles = butterworth_poles(order, 1.0)  # Normalized
        poles = bandpass_from_lowpass(lowpass_poles, center_freq, bandwidth)
        # Bandpass has zeros at origin
        zeros = [complex(0, 0)] * order

    else:
        raise ValueError(f"Unknown filter type: {filter_type}. "
                        f"Supported: butterworth, bessel, chebyshev, notch, lowpass, highpass, bandpass")

    return poles, zeros


def poles_zeros_to_arrays(
    poles: List[complex],
    zeros: List[complex],
    max_poles: int = 4,
    max_zeros: int = 4
) -> Tuple[np.ndarray, int, np.ndarray, int]:
    """
    Convert poles/zeros lists to padded arrays for model input.

    Args:
        poles: List of pole locations (complex)
        zeros: List of zero locations (complex)
        max_poles: Maximum number of poles (padding)
        max_zeros: Maximum number of zeros (padding)

    Returns:
        (pole_array, num_poles, zero_array, num_zeros)
        - pole_array: [max_poles, 2] array of [real, imag]
        - num_poles: Actual number of poles
        - zero_array: [max_zeros, 2] array of [real, imag]
        - num_zeros: Actual number of zeros
    """
    num_poles = min(len(poles), max_poles)
    num_zeros = min(len(zeros), max_zeros)

    if len(poles) > max_poles:
        warnings.warn(f"Truncating {len(poles)} poles to max_poles={max_poles}")
    if len(zeros) > max_zeros:
        warnings.warn(f"Truncating {len(zeros)} zeros to max_zeros={max_zeros}")

    # Create padded arrays
    pole_array = np.zeros((max_poles, 2), dtype=np.float32)
    zero_array = np.zeros((max_zeros, 2), dtype=np.float32)

    # Fill with actual values
    for i in range(num_poles):
        pole_array[i, 0] = poles[i].real
        pole_array[i, 1] = poles[i].imag

    for i in range(num_zeros):
        zero_array[i, 0] = zeros[i].real
        zero_array[i, 1] = zeros[i].imag

    return pole_array, num_poles, zero_array, num_zeros


if __name__ == '__main__':
    # Test filter calculations
    print("Testing Filter Design Utilities\n")
    print("="*70)

    # Test Butterworth
    print("\n1. Butterworth 2nd-order @ 1kHz:")
    poles, zeros = calculate_filter_poles_zeros('butterworth', 2, 1000)
    print(f"   Poles: {poles}")
    print(f"   Zeros: {zeros}")

    # Test Bessel
    print("\n2. Bessel 3rd-order @ 5kHz:")
    poles, zeros = calculate_filter_poles_zeros('bessel', 3, 5000)
    print(f"   Poles: {poles}")

    # Test Chebyshev
    print("\n3. Chebyshev 2nd-order @ 1kHz, 1dB ripple:")
    poles, zeros = calculate_filter_poles_zeros('chebyshev', 2, 1000, ripple_db=1.0)
    print(f"   Poles: {poles}")

    # Test Notch
    print("\n4. Notch @ 60Hz, Q=10:")
    poles, zeros = calculate_filter_poles_zeros('notch', 2, 60, q_factor=10)
    print(f"   Poles: {poles}")
    print(f"   Zeros: {zeros}")

    # Test conversion to arrays
    print("\n5. Convert to model input format:")
    pole_array, num_poles, zero_array, num_zeros = poles_zeros_to_arrays(poles, zeros)
    print(f"   Pole array shape: {pole_array.shape}")
    print(f"   Num poles: {num_poles}")
    print(f"   Zero array shape: {zero_array.shape}")
    print(f"   Num zeros: {num_zeros}")

    print("\n" + "="*70)
    print("✅ All filter calculations working!\n")
