"""
Circuit Analysis Utilities.

Compute transfer functions from circuit graphs using nodal analysis.
"""

import numpy as np
from typing import List, Tuple, Dict
import warnings

def compute_transfer_function_simple(
    circuit_graph: Dict,
    freq_range: np.ndarray = None
) -> Tuple[List[complex], List[complex], np.ndarray, np.ndarray]:
    """
    Compute transfer function from circuit graph using nodal analysis.

    This is a simplified analysis for passive RLC circuits with:
    - One voltage source (VIN)
    - One output node (VOUT)
    - Ground reference (GND)

    Args:
        circuit_graph: Dict with 'nodes' and 'adjacency'
        freq_range: Optional frequency range for frequency response

    Returns:
        (poles, zeros, frequencies, H_magnitude)
        - poles: List of pole locations (complex)
        - zeros: List of zero locations (complex)
        - frequencies: Frequency array (if freq_range provided)
        - H_magnitude: Magnitude response (if freq_range provided)
    """
    nodes = circuit_graph['nodes']
    adjacency = circuit_graph['adjacency']

    # Find special nodes
    gnd_id = None
    vin_id = None
    vout_id = None

    for node in nodes:
        if node['type'] == 'GND':
            gnd_id = node['id']
        elif node['type'] == 'VIN':
            vin_id = node['id']
        elif node['type'] == 'VOUT':
            vout_id = node['id']

    if gnd_id is None or vin_id is None or vout_id is None:
        warnings.warn("Circuit missing required nodes (GND, VIN, VOUT)")
        return [], [], np.array([]), np.array([])

    # Build admittance matrix in Laplace domain
    # Y(s) = G + sC + 1/(sL)
    # For each edge: Y = C*s + G + L_inv*s^-1

    num_nodes = len(nodes)

    # We'll build the system in frequency domain for now
    # Then estimate poles/zeros from the frequency response

    if freq_range is None:
        freq_range = np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz

    omega = 2 * np.pi * freq_range
    s = 1j * omega

    H = np.zeros(len(s), dtype=complex)

    for idx, s_val in enumerate(s):
        # Build admittance matrix for this frequency
        Y = np.zeros((num_nodes, num_nodes), dtype=complex)
        I = np.zeros(num_nodes, dtype=complex)

        # Add admittances from edges
        for src_id, neighbors in enumerate(adjacency):
            for edge in neighbors:
                dst_id = edge['id']
                C, G, L_inv = edge['impedance_den']

                # Admittance: Y(s) = G + sC + (1/L)/(s) = G + sC + L_inv/s
                Y_edge = G + s_val * C
                if L_inv > 1e-15:
                    Y_edge += L_inv / s_val

                # Add to matrix (KCL)
                Y[src_id, src_id] += Y_edge
                Y[dst_id, dst_id] += Y_edge
                Y[src_id, dst_id] -= Y_edge
                Y[dst_id, src_id] -= Y_edge

        # Apply voltage source at VIN
        # Set V[vin_id] = 1V
        # This is done by modifying the matrix

        # Remove GND row/col (reference)
        nodes_to_keep = [i for i in range(num_nodes) if i != gnd_id]
        Y_reduced = Y[np.ix_(nodes_to_keep, nodes_to_keep)]
        I_reduced = I[nodes_to_keep]

        # Find vin and vout indices in reduced system
        vin_idx_reduced = nodes_to_keep.index(vin_id)
        vout_idx_reduced = nodes_to_keep.index(vout_id)

        # Set VIN = 1V by modifying the equation
        # Replace VIN row with: V[vin] = 1
        Y_reduced[vin_idx_reduced, :] = 0
        Y_reduced[vin_idx_reduced, vin_idx_reduced] = 1
        I_reduced[vin_idx_reduced] = 1

        # Solve for node voltages
        try:
            V = np.linalg.solve(Y_reduced, I_reduced)
            H[idx] = V[vout_idx_reduced]  # Transfer function = Vout/Vin = Vout (since Vin=1)
        except np.linalg.LinAlgError:
            H[idx] = 0

    # Extract magnitude and phase
    H_mag = np.abs(H)
    H_phase = np.angle(H)

    # Estimate poles from frequency response using advanced method
    poles = estimate_poles_from_response(freq_range, H, max_poles=4)

    # Zeros are harder to estimate - look for magnitude nulls
    zeros = estimate_zeros_from_response(freq_range, H, max_zeros=4)

    return poles, zeros, freq_range, H_mag


def estimate_poles_from_response(
    frequencies: np.ndarray,
    H: np.ndarray,
    max_poles: int = 4
) -> List[complex]:
    """
    Estimate pole locations from frequency response.

    This is an approximate method using peak detection.
    For accurate results, use symbolic circuit analysis.

    Args:
        frequencies: Frequency array (Hz)
        H: Complex frequency response
        max_poles: Maximum number of poles to find

    Returns:
        List of estimated pole locations (complex)
    """
    from scipy import signal

    H_mag = np.abs(H)
    H_phase = np.angle(H)
    omega = 2 * np.pi * frequencies

    # Find peaks in magnitude (resonances indicate poles)
    peaks, _ = signal.find_peaks(H_mag, prominence=0.1*np.max(H_mag))

    poles = []

    for peak in peaks[:max_poles]:
        omega_peak = omega[peak]

        # Estimate Q from peak sharpness
        # For 2nd-order system: peak_mag ≈ Q at resonance
        Q = H_mag[peak] / (H_mag[0] + 1e-10)

        # Estimate damping
        zeta = 1 / (2 * Q) if Q > 0 else 0.5

        # Pole location: s = -zeta*omega_n ± j*omega_n*sqrt(1-zeta^2)
        omega_n = omega_peak
        real_part = -zeta * omega_n
        imag_part = omega_n * np.sqrt(1 - zeta**2) if zeta < 1 else 0

        if imag_part > 0:
            poles.append(complex(real_part, imag_part))
            poles.append(complex(real_part, -imag_part))
        else:
            poles.append(complex(real_part, 0))

    return poles[:max_poles]


def estimate_zeros_from_response(
    frequencies: np.ndarray,
    H: np.ndarray,
    max_zeros: int = 4
) -> List[complex]:
    """
    Estimate zero locations from frequency response.

    Zeros appear as magnitude nulls in the frequency response.

    Args:
        frequencies: Frequency array (Hz)
        H: Complex frequency response
        max_zeros: Maximum number of zeros to find

    Returns:
        List of estimated zero locations (complex)
    """
    from scipy import signal

    H_mag = np.abs(H)
    omega = 2 * np.pi * frequencies

    # Find valleys (nulls) in magnitude response
    # Invert magnitude to find peaks (which are nulls in original)
    if np.max(H_mag) > 1e-10:
        H_inv = 1.0 / (H_mag + 1e-10)
    else:
        return []  # No clear response, can't estimate zeros

    # Find peaks in inverted response (= nulls in original)
    peaks, properties = signal.find_peaks(H_inv, prominence=0.1*np.max(H_inv))

    zeros = []

    for peak in peaks[:max_zeros]:
        omega_zero = omega[peak]

        # For passive circuits, zeros are typically on imaginary axis or negative real axis
        # Simple approximation: zeros at ±jω where magnitude is minimum
        zeros.append(complex(0, omega_zero))
        zeros.append(complex(0, -omega_zero))

    return zeros[:max_zeros]


def compare_transfer_functions(
    target_poles: List[complex],
    target_zeros: List[complex],
    actual_poles: List[complex],
    actual_zeros: List[complex]
) -> Dict:
    """
    Compare target and actual transfer functions.

    Uses Hungarian algorithm to match poles/zeros and compute errors.

    Returns:
        Dictionary with comparison metrics
    """
    from scipy.optimize import linear_sum_assignment

    results = {
        'pole_count_match': len(actual_poles) == len(target_poles),
        'zero_count_match': len(actual_zeros) == len(target_zeros),
        'pole_errors': [],
        'zero_errors': [],
        'avg_pole_error': float('inf'),
        'avg_zero_error': float('inf')
    }

    # Compare poles
    if len(target_poles) > 0 and len(actual_poles) > 0:
        # Build cost matrix
        n_target = len(target_poles)
        n_actual = len(actual_poles)
        max_len = max(n_target, n_actual)

        cost_matrix = np.zeros((max_len, max_len))
        for i in range(min(n_target, max_len)):
            for j in range(min(n_actual, max_len)):
                if i < n_target and j < n_actual:
                    cost_matrix[i, j] = abs(target_poles[i] - actual_poles[j])
                else:
                    cost_matrix[i, j] = 1e10

        # Find optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute errors for matched pairs
        for i, j in zip(row_ind, col_ind):
            if i < n_target and j < n_actual:
                target = target_poles[i]
                actual = actual_poles[j]

                mag_target = abs(target)
                mag_actual = abs(actual)
                mag_error = abs(mag_target - mag_actual) / (mag_target + 1e-10)

                phase_target = np.angle(target)
                phase_actual = np.angle(actual)
                phase_error = abs(phase_target - phase_actual)

                results['pole_errors'].append({
                    'target': target,
                    'actual': actual,
                    'mag_error': mag_error,
                    'phase_error': phase_error
                })

        if results['pole_errors']:
            results['avg_pole_error'] = np.mean([e['mag_error'] for e in results['pole_errors']])

    # Compare zeros (similar logic)
    if len(target_zeros) > 0 and len(actual_zeros) > 0:
        n_target = len(target_zeros)
        n_actual = len(actual_zeros)
        max_len = max(n_target, n_actual)

        cost_matrix = np.zeros((max_len, max_len))
        for i in range(min(n_target, max_len)):
            for j in range(min(n_actual, max_len)):
                if i < n_target and j < n_actual:
                    cost_matrix[i, j] = abs(target_zeros[i] - actual_zeros[j])
                else:
                    cost_matrix[i, j] = 1e10

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if i < n_target and j < n_actual:
                target = target_zeros[i]
                actual = actual_zeros[j]

                mag_error = abs(abs(target) - abs(actual)) / (abs(target) + 1e-10)

                results['zero_errors'].append({
                    'target': target,
                    'actual': actual,
                    'mag_error': mag_error
                })

        if results['zero_errors']:
            results['avg_zero_error'] = np.mean([e['mag_error'] for e in results['zero_errors']])

    return results
