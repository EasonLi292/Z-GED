"""
Utility functions for extracting and computing conditions from circuits.

Conditions are circuit specifications that we want to control:
- Cutoff frequency (Hz)
- Q factor (quality factor)
- DC gain (dB)
- Filter order (number of poles)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class ConditionExtractor:
    """
    Extracts conditions (specifications) from circuits for conditional VAE.

    Supports:
    - Cutoff frequency extraction from poles/zeros
    - Q factor computation from pole locations
    - DC/HF gain estimation
    - Filter order (number of poles)
    """

    def __init__(self,
                 normalize: bool = True,
                 condition_stats: Optional[Dict] = None):
        """
        Args:
            normalize: Whether to normalize conditions
            condition_stats: Pre-computed statistics for normalization
                            (mean/std for each condition)
        """
        self.normalize = normalize
        self.condition_stats = condition_stats

    def compute_cutoff_frequency(self,
                                 poles: np.ndarray,
                                 zeros: np.ndarray,
                                 filter_type: str) -> float:
        """
        Compute cutoff frequency from poles/zeros.

        Args:
            poles: Complex poles [N, 2] as [real, imag]
            zeros: Complex zeros [M, 2] as [real, imag]
            filter_type: 'low_pass', 'high_pass', 'band_pass', etc.

        Returns:
            Cutoff frequency in Hz
        """
        # Convert to complex numbers
        if len(poles) > 0:
            poles_complex = poles[:, 0] + 1j * poles[:, 1]
        else:
            poles_complex = np.array([])

        if len(zeros) > 0:
            zeros_complex = zeros[:, 0] + 1j * zeros[:, 1]
        else:
            zeros_complex = np.array([])

        # Extract cutoff based on filter type
        if filter_type == 'low_pass':
            # Cutoff = magnitude of dominant pole
            if len(poles_complex) > 0:
                # Use the pole closest to origin (lowest frequency)
                cutoff = np.min(np.abs(poles_complex))
            else:
                cutoff = 1.0  # Default

        elif filter_type == 'high_pass':
            # Cutoff = magnitude of dominant zero
            if len(zeros_complex) > 0:
                cutoff = np.min(np.abs(zeros_complex))
            else:
                cutoff = 1.0  # Default

        elif filter_type in ['band_pass', 'band_stop']:
            # Cutoff = geometric mean of pole frequencies
            if len(poles_complex) >= 2:
                pole_freqs = np.abs(poles_complex)
                cutoff = np.sqrt(pole_freqs[0] * pole_freqs[1])
            elif len(poles_complex) == 1:
                cutoff = np.abs(poles_complex[0])
            else:
                cutoff = 1.0

        elif filter_type in ['rlc_series', 'rlc_parallel']:
            # Resonant frequency = geometric mean of poles
            if len(poles_complex) >= 2:
                pole_freqs = np.abs(poles_complex)
                cutoff = np.sqrt(np.prod(pole_freqs))
            elif len(poles_complex) == 1:
                cutoff = np.abs(poles_complex[0])
            else:
                cutoff = 1.0
        else:
            cutoff = 1.0  # Unknown type

        # Convert from rad/s to Hz (poles/zeros are typically in rad/s)
        cutoff_hz = cutoff / (2 * np.pi)

        return float(cutoff_hz)

    def compute_q_factor(self,
                        poles: np.ndarray,
                        filter_type: str) -> float:
        """
        Compute Q factor from pole locations.

        Q = 1 / (2 * damping_ratio)
        For complex conjugate poles: Q = |pole| / (2 * |real_part|)

        Args:
            poles: Complex poles [N, 2] as [real, imag]
            filter_type: Filter type

        Returns:
            Q factor (dimensionless)
        """
        if len(poles) == 0:
            return 0.5  # Default for no poles

        # Convert to complex
        poles_complex = poles[:, 0] + 1j * poles[:, 1]

        # For 1st order filters (1 real pole)
        if len(poles_complex) == 1:
            if np.abs(poles_complex[0].imag) < 1e-6:
                # Real pole → Q ≈ 0.5 (no resonance)
                return 0.5
            else:
                # Complex pole (rare for 1st order, but handle it)
                pole = poles_complex[0]
                q = np.abs(pole) / (2 * np.abs(pole.real))
                return float(q)

        # For 2nd order filters (complex conjugate pair)
        elif len(poles_complex) >= 2:
            # Find the dominant complex conjugate pair
            # (closest to imaginary axis = highest Q)
            max_q = 0.5

            for pole in poles_complex:
                if np.abs(pole.imag) > 1e-6:  # Complex pole
                    q = np.abs(pole) / (2 * np.abs(pole.real))
                    max_q = max(max_q, q)

            return float(max_q)

        return 0.5  # Default

    def compute_dc_gain(self,
                       gain: float,
                       zeros: np.ndarray,
                       poles: np.ndarray,
                       filter_type: str) -> float:
        """
        Estimate DC gain (gain at f=0).

        Args:
            gain: Scale factor from circuit
            zeros: Complex zeros
            poles: Complex poles
            filter_type: Filter type

        Returns:
            DC gain in dB
        """
        # For low-pass: DC gain ≈ gain
        # For high-pass: DC gain ≈ 0 (or -inf dB)
        # For band-pass/band-stop: depends on center frequency

        if filter_type == 'low_pass':
            dc_gain_linear = abs(gain)
        elif filter_type == 'high_pass':
            dc_gain_linear = 1e-6  # Very small (high-pass blocks DC)
        elif filter_type in ['band_pass', 'rlc_series']:
            dc_gain_linear = 1e-6  # Band-pass blocks DC
        elif filter_type in ['band_stop', 'rlc_parallel']:
            dc_gain_linear = abs(gain)  # Band-stop passes DC
        else:
            dc_gain_linear = abs(gain)

        # Convert to dB
        dc_gain_db = 20 * np.log10(dc_gain_linear + 1e-10)

        return float(dc_gain_db)

    def extract_conditions(self,
                          poles: np.ndarray,
                          zeros: np.ndarray,
                          gain: float,
                          filter_type: str) -> Dict[str, float]:
        """
        Extract all conditions from a circuit.

        Args:
            poles: Poles [N, 2]
            zeros: Zeros [M, 2]
            gain: Gain value
            filter_type: Filter type string

        Returns:
            Dictionary of conditions
        """
        conditions = {
            'cutoff_frequency': self.compute_cutoff_frequency(poles, zeros, filter_type),
            'q_factor': self.compute_q_factor(poles, filter_type),
            'dc_gain': self.compute_dc_gain(gain, zeros, poles, filter_type),
            'num_poles': len(poles),
            'num_zeros': len(zeros)
        }

        return conditions

    def normalize_conditions(self,
                            conditions: Dict[str, float]) -> torch.Tensor:
        """
        Normalize conditions to zero mean, unit variance.

        Args:
            conditions: Dictionary of condition values

        Returns:
            Normalized condition tensor [condition_dim]
        """
        if not self.normalize or self.condition_stats is None:
            # Just stack as tensor without normalization
            condition_values = [
                conditions['cutoff_frequency'],
                conditions['q_factor']
            ]
            return torch.tensor(condition_values, dtype=torch.float32)

        # Normalize each condition
        normalized = []

        # Cutoff frequency (log-scale)
        cutoff = conditions['cutoff_frequency']
        log_cutoff = np.log10(cutoff + 1e-10)
        norm_cutoff = (log_cutoff - self.condition_stats['cutoff_mean']) / self.condition_stats['cutoff_std']
        normalized.append(norm_cutoff)

        # Q factor (log-scale)
        q = conditions['q_factor']
        log_q = np.log10(q + 1e-10)
        norm_q = (log_q - self.condition_stats['q_mean']) / self.condition_stats['q_std']
        normalized.append(norm_q)

        return torch.tensor(normalized, dtype=torch.float32)

    def denormalize_conditions(self,
                               normalized_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Convert normalized condition tensor back to original values.

        Args:
            normalized_tensor: Normalized conditions [2]

        Returns:
            Dictionary of denormalized conditions
        """
        if not self.normalize or self.condition_stats is None:
            cutoff, q = normalized_tensor.tolist()
            return {
                'cutoff_frequency': cutoff,
                'q_factor': q
            }

        norm_cutoff, norm_q = normalized_tensor.tolist()

        # Denormalize cutoff (undo log-scale)
        log_cutoff = norm_cutoff * self.condition_stats['cutoff_std'] + self.condition_stats['cutoff_mean']
        cutoff = 10 ** log_cutoff

        # Denormalize Q factor (undo log-scale)
        log_q = norm_q * self.condition_stats['q_std'] + self.condition_stats['q_mean']
        q = 10 ** log_q

        return {
            'cutoff_frequency': cutoff,
            'q_factor': q
        }


def compute_condition_statistics(dataset) -> Dict[str, float]:
    """
    Compute mean and std for normalization across entire dataset.

    Args:
        dataset: CircuitDataset instance

    Returns:
        Dictionary with statistics
    """
    extractor = ConditionExtractor(normalize=False)

    all_cutoffs = []
    all_qs = []

    for i in range(len(dataset)):
        circuit = dataset.circuits[i]

        poles = circuit['label']['poles']
        zeros = circuit['label']['zeros']
        gain = circuit['label']['gain']
        filter_type = circuit['filter_type']

        # Convert poles/zeros to array format
        if len(poles) > 0:
            poles_array = np.array([[p.real, p.imag] for p in poles])
        else:
            poles_array = np.zeros((0, 2))

        if len(zeros) > 0:
            zeros_array = np.array([[z.real, z.imag] for z in zeros])
        else:
            zeros_array = np.zeros((0, 2))

        # Extract conditions
        conditions = extractor.extract_conditions(
            poles_array, zeros_array, gain, filter_type
        )

        all_cutoffs.append(conditions['cutoff_frequency'])
        all_qs.append(conditions['q_factor'])

    # Compute statistics in log-space
    log_cutoffs = np.log10(np.array(all_cutoffs) + 1e-10)
    log_qs = np.log10(np.array(all_qs) + 1e-10)

    stats = {
        'cutoff_mean': float(log_cutoffs.mean()),
        'cutoff_std': float(log_cutoffs.std()),
        'q_mean': float(log_qs.mean()),
        'q_std': float(log_qs.std()),

        # Also store raw statistics for reference
        'cutoff_min': float(np.min(all_cutoffs)),
        'cutoff_max': float(np.max(all_cutoffs)),
        'q_min': float(np.min(all_qs)),
        'q_max': float(np.max(all_qs))
    }

    print(f"\n📊 Condition Statistics:")
    print(f"   Cutoff frequency: {stats['cutoff_min']:.2f} - {stats['cutoff_max']:.2f} Hz")
    print(f"   Q factor:         {stats['q_min']:.3f} - {stats['q_max']:.3f}")
    print(f"   Log cutoff:       mean={stats['cutoff_mean']:.3f}, std={stats['cutoff_std']:.3f}")
    print(f"   Log Q:            mean={stats['q_mean']:.3f}, std={stats['q_std']:.3f}")

    return stats
