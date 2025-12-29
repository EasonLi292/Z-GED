"""
SPICE-based circuit simulation and transfer function extraction.

Converts generated circuits to SPICE netlists, runs AC analysis,
and extracts transfer function characteristics.
"""

import numpy as np
import torch
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class CircuitSimulator:
    """Simulate circuits using SPICE and extract transfer functions."""

    def __init__(self,
                 simulator='ngspice',
                 freq_points=100,
                 freq_start=1.0,
                 freq_stop=1e6):
        """
        Args:
            simulator: SPICE simulator to use ('ngspice', 'ltspice')
            freq_points: Number of frequency points for AC analysis
            freq_start: Starting frequency (Hz)
            freq_stop: Stopping frequency (Hz)
        """
        self.simulator = simulator
        self.freq_points = freq_points
        self.freq_start = freq_start
        self.freq_stop = freq_stop

    def circuit_to_netlist(self,
                          node_types: torch.Tensor,
                          edge_existence: torch.Tensor,
                          edge_values: torch.Tensor,
                          ac_amplitude: float = 1.0) -> str:
        """
        Convert circuit graph to SPICE netlist.

        Args:
            node_types: [num_nodes, 5] one-hot node types (GND/VIN/VOUT/INTERNAL/MASK)
            edge_existence: [num_nodes, num_nodes] binary adjacency matrix
            edge_values: [num_nodes, num_nodes, 7] edge features
                         [log(C), log(G), log(L_inv), mask_C, mask_G, mask_L, exists]
            ac_amplitude: AC source amplitude (V)

        Returns:
            SPICE netlist string
        """
        # Decode node types
        node_type_ids = node_types.argmax(dim=-1).cpu().numpy()  # [num_nodes]
        num_nodes = len(node_type_ids)

        # Node type mapping
        NODE_TYPES = {0: 'GND', 1: 'VIN', 2: 'VOUT', 3: 'INTERNAL', 4: 'MASK'}

        # Find special nodes
        gnd_node = None
        vin_node = None
        vout_node = None

        for i, node_type_id in enumerate(node_type_ids):
            node_type = NODE_TYPES[node_type_id]
            if node_type == 'GND':
                gnd_node = i
            elif node_type == 'VIN':
                vin_node = i
            elif node_type == 'VOUT':
                vout_node = i

        if gnd_node is None or vin_node is None or vout_node is None:
            raise ValueError("Circuit must have GND, VIN, and VOUT nodes")

        # Build netlist
        netlist_lines = []
        netlist_lines.append("* Auto-generated circuit netlist")
        netlist_lines.append("")

        # AC voltage source at VIN
        # In SPICE, node 0 is always ground
        # Map our GND node to SPICE node 0
        netlist_lines.append(f"VIN n{vin_node} 0 DC 0 AC {ac_amplitude}")
        netlist_lines.append("")

        # Component counter
        component_counters = {'C': 0, 'R': 0, 'L': 0}

        # Add components for each edge
        edge_existence_np = edge_existence.cpu().numpy()
        edge_values_np = edge_values.cpu().numpy()

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Upper triangle only
                if edge_existence_np[i, j] < 0.5:
                    continue  # No edge

                # Skip edges involving masked nodes
                if node_type_ids[i] == 4 or node_type_ids[j] == 4:
                    continue

                # Get edge values
                log_C = edge_values_np[i, j, 0]
                log_G = edge_values_np[i, j, 1]
                log_L_inv = edge_values_np[i, j, 2]
                mask_C = edge_values_np[i, j, 3]
                mask_G = edge_values_np[i, j, 4]
                mask_L = edge_values_np[i, j, 5]

                # Convert log values to actual values
                C_value = np.exp(log_C)  # Farads
                G_value = np.exp(log_G)  # Siemens
                L_value = 1.0 / np.exp(log_L_inv)  # Henrys

                # Convert G to R
                R_value = 1.0 / (G_value + 1e-12)  # Ohms

                # Map nodes (use GND=0, others as-is)
                node_i = 0 if i == gnd_node else f"n{i}"
                node_j = 0 if j == gnd_node else f"n{j}"

                # Add components based on masks
                if mask_C > 0.5:
                    component_counters['C'] += 1
                    # Capacitors in Farads
                    netlist_lines.append(
                        f"C{component_counters['C']} {node_i} {node_j} {C_value:.12e}"
                    )

                if mask_G > 0.5:
                    component_counters['R'] += 1
                    # Resistors in Ohms
                    netlist_lines.append(
                        f"R{component_counters['R']} {node_i} {node_j} {R_value:.12e}"
                    )

                if mask_L > 0.5:
                    component_counters['L'] += 1
                    # Inductors in Henrys
                    netlist_lines.append(
                        f"L{component_counters['L']} {node_i} {node_j} {L_value:.12e}"
                    )

        netlist_lines.append("")

        # AC analysis command
        netlist_lines.append(
            f".ac dec {self.freq_points} {self.freq_start} {self.freq_stop}"
        )

        # Print transfer function (VOUT/VIN)
        vout_node_spice = 0 if vout_node == gnd_node else f"n{vout_node}"
        netlist_lines.append(f".print ac v({vout_node_spice})")

        # Control commands
        netlist_lines.append(".control")
        netlist_lines.append("run")
        netlist_lines.append(f"set hcopydevtype=ascii")
        netlist_lines.append("print frequency v(n2)")
        netlist_lines.append(".endc")
        netlist_lines.append("")
        netlist_lines.append(".end")

        return "\n".join(netlist_lines)

    def run_ac_analysis(self, netlist: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run AC analysis using ngspice.

        Args:
            netlist: SPICE netlist string

        Returns:
            frequencies: [num_points] frequency array (Hz)
            response: [num_points] complex transfer function H(jω)
        """
        # Create temporary netlist file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_path = f.name

        try:
            # Run ngspice in batch mode
            cmd = [
                'ngspice',
                '-b',  # Batch mode
                netlist_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise RuntimeError(f"ngspice failed: {result.stderr}")

            # Parse stdout output
            frequencies, response = self._parse_ac_output_from_text(result.stdout)

            return frequencies, response

        finally:
            # Cleanup
            if os.path.exists(netlist_path):
                os.remove(netlist_path)

    def _parse_ac_output_from_text(self, output_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse ngspice AC analysis output from stdout text.

        Args:
            output_text: ngspice stdout text

        Returns:
            frequencies: [num_points] frequency array (Hz)
            response: [num_points] complex transfer function H(jω)
        """
        frequencies = []
        real_parts = []
        imag_parts = []

        lines = output_text.split('\n')

        in_data_section = False
        for line in lines:
            line = line.strip()

            # Detect start of data section
            # Look for: "Index   frequency       v(n2)"
            if 'Index' in line and 'frequency' in line:
                in_data_section = True
                continue

            # Skip separator lines
            if '---' in line:
                continue

            # Detect end of data section (ngspice done message)
            if 'ngspice' in line and 'done' in line:
                break

            if not in_data_section:
                continue

            # Skip empty lines
            if not line:
                continue

            # Parse data line
            # Format: "0\t1.000000e+00\t9.999605e-01,\t-6.28294e-03"
            # Index, frequency, real, imaginary (comma-separated)
            parts = line.split()

            if len(parts) >= 3:
                try:
                    # Index (not used)
                    idx = int(parts[0])

                    # Frequency
                    freq = float(parts[1])

                    # Complex value: "real,\timag" or "real,imag"
                    # Remove comma from real part
                    real_str = parts[2].rstrip(',')
                    real_val = float(real_str)

                    # Imaginary part (if present)
                    if len(parts) >= 4:
                        imag_val = float(parts[3])
                    else:
                        imag_val = 0.0

                    frequencies.append(freq)
                    real_parts.append(real_val)
                    imag_parts.append(imag_val)

                except (ValueError, IndexError):
                    # Not a data line, skip
                    continue

        if not frequencies:
            raise ValueError("No AC analysis data found in output")

        frequencies = np.array(frequencies)
        real_parts = np.array(real_parts)
        imag_parts = np.array(imag_parts)

        # Construct complex response
        response = real_parts + 1j * imag_parts

        return frequencies, response

    def extract_poles_zeros(self,
                           frequencies: np.ndarray,
                           response: np.ndarray,
                           max_poles: int = 4,
                           max_zeros: int = 4) -> Dict[str, np.ndarray]:
        """
        Extract poles and zeros from frequency response.

        Uses simple peak detection and curve fitting.

        Args:
            frequencies: [num_points] frequency array (Hz)
            response: [num_points] complex transfer function
            max_poles: Maximum number of poles to extract
            max_zeros: Maximum number of zeros to extract

        Returns:
            dict with:
                'pole_count': int
                'zero_count': int
                'poles': [max_poles, 2] array of [real, imag] in rad/s
                'zeros': [max_zeros, 2] array of [real, imag] in rad/s
        """
        from scipy.signal import find_peaks

        magnitude = np.abs(response)
        phase = np.angle(response)
        log_mag = 20 * np.log10(magnitude + 1e-12)  # dB

        # Convert frequency to angular frequency
        omega = 2 * np.pi * frequencies

        # Find poles (peaks in magnitude response or phase drops)
        # Poles cause magnitude to increase (resonances)
        pole_indices, _ = find_peaks(log_mag, prominence=3.0)

        # Find zeros (valleys in magnitude response)
        # Zeros cause magnitude to decrease (anti-resonances)
        zero_indices, _ = find_peaks(-log_mag, prominence=3.0)

        # Limit to max counts
        pole_indices = pole_indices[:max_poles]
        zero_indices = zero_indices[:max_zeros]

        pole_count = len(pole_indices)
        zero_count = len(zero_indices)

        # Extract pole/zero frequencies
        poles = np.zeros((max_poles, 2))  # [real, imag]
        zeros = np.zeros((max_zeros, 2))

        for i, idx in enumerate(pole_indices):
            omega_pole = omega[idx]
            # Assume poles are on negative real axis or complex conjugate pairs
            # Simple approximation: real = -omega/Q, imag = omega
            # For now, use negative real axis (overdamped)
            poles[i, 0] = -omega_pole * 0.1  # Negative real part
            poles[i, 1] = omega_pole  # Imaginary part

        for i, idx in enumerate(zero_indices):
            omega_zero = omega[idx]
            zeros[i, 0] = -omega_zero * 0.1
            zeros[i, 1] = omega_zero

        return {
            'pole_count': pole_count,
            'zero_count': zero_count,
            'poles': poles,
            'zeros': zeros,
            'pole_frequencies': omega[pole_indices] if pole_count > 0 else np.array([]),
            'zero_frequencies': omega[zero_indices] if zero_count > 0 else np.array([]),
        }

    def compare_transfer_functions(self,
                                  freq1: np.ndarray,
                                  response1: np.ndarray,
                                  freq2: np.ndarray,
                                  response2: np.ndarray) -> Dict[str, float]:
        """
        Compare two transfer functions.

        Args:
            freq1, response1: First TF
            freq2, response2: Second TF

        Returns:
            dict with comparison metrics:
                'magnitude_mse': Mean squared error in dB
                'magnitude_correlation': Correlation coefficient
                'phase_mse': Mean squared error in phase (radians)
        """
        # Interpolate to common frequency grid
        from scipy.interpolate import interp1d

        # Use freq1 as reference
        mag1 = 20 * np.log10(np.abs(response1) + 1e-12)
        phase1 = np.angle(response1)

        # Interpolate response2 to freq1
        interp_mag2 = interp1d(
            np.log10(freq2),
            20 * np.log10(np.abs(response2) + 1e-12),
            kind='linear',
            fill_value='extrapolate'
        )
        interp_phase2 = interp1d(
            np.log10(freq2),
            np.unwrap(np.angle(response2)),
            kind='linear',
            fill_value='extrapolate'
        )

        mag2 = interp_mag2(np.log10(freq1))
        phase2 = interp_phase2(np.log10(freq1))

        # Compute metrics
        magnitude_mse = np.mean((mag1 - mag2) ** 2)
        magnitude_correlation = np.corrcoef(mag1, mag2)[0, 1]

        phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))  # Wrap to [-π, π]
        phase_mse = np.mean(phase_diff ** 2)

        return {
            'magnitude_mse': magnitude_mse,
            'magnitude_correlation': magnitude_correlation,
            'phase_mse': phase_mse,
        }


def compare_pole_zero_counts(pred_count: int,
                             target_count: int,
                             pred_positions: np.ndarray,
                             target_positions: np.ndarray,
                             tolerance: float = 0.1) -> Dict[str, float]:
    """
    Compare pole or zero counts and positions.

    Args:
        pred_count: Predicted count
        target_count: Target count
        pred_positions: [max_count, 2] predicted [real, imag]
        target_positions: [max_count, 2] target [real, imag]
        tolerance: Position matching tolerance (relative)

    Returns:
        dict with:
            'count_match': 1.0 if counts match, 0.0 otherwise
            'position_error': Mean position error (if counts match)
            'match_rate': Fraction of positions matched within tolerance
    """
    count_match = 1.0 if pred_count == target_count else 0.0

    if pred_count == 0 and target_count == 0:
        return {
            'count_match': 1.0,
            'position_error': 0.0,
            'match_rate': 1.0,
        }

    if pred_count == 0 or target_count == 0:
        return {
            'count_match': 0.0,
            'position_error': float('inf'),
            'match_rate': 0.0,
        }

    # Extract valid positions
    pred_pos = pred_positions[:pred_count]
    target_pos = target_positions[:target_count]

    # Compute pairwise distances
    min_count = min(pred_count, target_count)

    if pred_count != target_count:
        position_error = float('inf')
        match_rate = 0.0
    else:
        # Compute minimum matching distance
        from scipy.spatial.distance import cdist

        distances = cdist(pred_pos, target_pos, metric='euclidean')

        # Greedy matching
        matched = 0
        total_error = 0.0

        for _ in range(min_count):
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            min_dist = distances[i, j]

            # Check if within tolerance
            target_magnitude = np.linalg.norm(target_pos[j])
            if min_dist / (target_magnitude + 1e-6) < tolerance:
                matched += 1

            total_error += min_dist

            # Remove matched pair
            distances[i, :] = float('inf')
            distances[:, j] = float('inf')

        position_error = total_error / min_count
        match_rate = matched / min_count

    return {
        'count_match': count_match,
        'position_error': position_error,
        'match_rate': match_rate,
    }
