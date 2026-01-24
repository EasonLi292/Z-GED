import os
import random
import numpy as np
import networkx as nx
import pickle
import uuid
import warnings

# PySpice Imports
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_kOhm, u_Ohm, u_H, u_F, u_uH, u_mH, u_uF, u_nF, u_pF, u_MHz, u_Hz, u_kHz

# Scipy for fitting Transfer Functions
from scipy import signal
from scipy.optimize import least_squares

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATASET_DIR = "rlc_dataset"
NUM_SAMPLES_PER_FILTER = 60  # Generate 60 variations of each filter type (3x original)
FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

class FilterGenerator:
    def __init__(self):
        self.components = []
        self.graph = nx.MultiGraph()
        self.filter_type = None

    def generate_low_pass_filter(self):
        """RC Low-pass filter: Vin --R-- Vout --C-- GND"""
        self.filter_type = 'low_pass'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(2, 5)  # 100 Ohm to 100 kOhm
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF

        # Cutoff frequency: fc = 1 / (2*pi*R*C)
        fc = 1 / (2 * np.pi * R * C)

        self.components = [
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 1, 'node2': 2},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return fc

    def generate_high_pass_filter(self):
        """RC High-pass filter: Vin --C-- Vout --R-- GND"""
        self.filter_type = 'high_pass'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(2, 5)  # 100 Ohm to 100 kOhm
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF

        # Cutoff frequency: fc = 1 / (2*pi*R*C)
        fc = 1 / (2 * np.pi * R * C)

        self.components = [
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 1, 'node2': 2},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return fc

    def generate_band_pass_filter(self):
        """RLC Band-pass filter (series RLC, measure across R)"""
        self.filter_type = 'band_pass'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(2, 4)  # 100 Ohm to 10 kOhm
        L = 10 ** random.uniform(-4, -2)  # 0.1mH to 10mH
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF

        # Center frequency: f0 = 1 / (2*pi*sqrt(L*C))
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))

        # Series RLC: Vin --L--C-- Vout --R-- GND, measure across R
        # At DC: C blocks → I=0 → V_R=0
        # At resonance: Z_LC=0 (short) → I_max → V_R max (band-pass peak)
        # At HF: L blocks → I→0 → V_R→0
        self.components = [
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 1, 'node2': 3},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 3, 'node2': 2},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return f0

    def generate_band_stop_filter(self):
        """RLC Band-stop (notch) filter (series LC in shunt path)"""
        self.filter_type = 'band_stop'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(2, 4)  # 100 Ohm to 10 kOhm
        L = 10 ** random.uniform(-4, -2)  # 0.1mH to 10mH
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF
        R_load = 10 ** random.uniform(3, 5)  # 1k to 100k load

        # Center frequency: f0 = 1 / (2*pi*sqrt(L*C))
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))

        # Vin --R_series-- Node3[series LC to GND] --R_load-- Vout --R_out-- GND
        # At resonance: series LC shorts to ground → maximum attenuation (notch)
        # Away from resonance: series LC has high impedance → signal passes through
        self.components = [
            {'name': 'R_series', 'type': 'R', 'value': R, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 4},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 4, 'node2': 0},
            {'name': 'R_load', 'type': 'R', 'value': R_load, 'node1': 3, 'node2': 2},
            {'name': 'R_out', 'type': 'R', 'value': R_load, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return f0

    def generate_rlc_series_resonant(self):
        """Series RLC resonant circuit"""
        self.filter_type = 'rlc_series'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(1, 3)  # 10 Ohm to 1 kOhm
        L = 10 ** random.uniform(-4, -2)  # 0.1mH to 10mH
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF

        # Resonant frequency: f0 = 1 / (2*pi*sqrt(L*C))
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))

        # Simple series: Vin --R--L--C-- GND, measure across everything (node 2 as output)
        self.components = [
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 4},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 4, 'node2': 2},
            {'name': 'R_load', 'type': 'R', 'value': R * 10, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return f0

    def generate_rlc_parallel_resonant(self):
        """Parallel RLC resonant circuit"""
        self.filter_type = 'rlc_parallel'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(3, 5)  # 1k to 100k (high for parallel)
        L = 10 ** random.uniform(-4, -2)  # 0.1mH to 10mH
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF
        R_source = 10 ** random.uniform(2, 3)  # 100-1k source resistance

        # Resonant frequency: f0 = 1 / (2*pi*sqrt(L*C))
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))

        # Vin --R_source-- (parallel RLC) -- GND, measure at node 2
        self.components = [
            {'name': 'R_source', 'type': 'R', 'value': R_source, 'node1': 1, 'node2': 2},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 2, 'node2': 0},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 2, 'node2': 0},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return f0

    def _build_graph(self):
        """Build NetworkX graph from components"""
        self.graph.clear()
        nodes = set()
        for comp in self.components:
            nodes.add(comp['node1'])
            nodes.add(comp['node2'])

        self.graph.add_nodes_from(nodes)
        for comp in self.components:
            self.graph.add_edge(comp['node1'], comp['node2'],
                              key=comp['name'], **comp)

    def to_pyspice(self):
        """Converts internal structure to PySpice Circuit."""
        circuit = Circuit(f'{self.filter_type}_{uuid.uuid4().hex[:8]}')

        # Voltage Source (AC 1V) at Node 1, referenced to GND (0)
        circuit.SinusoidalVoltageSource('in', 1, 0, amplitude=1.0, ac_magnitude=1.0)

        for comp in self.components:
            n1 = comp['node1']
            n2 = comp['node2']
            val = comp['value']

            if comp['type'] == 'R':
                circuit.Resistor(comp['name'], n1, n2, val@u_Ohm)
            elif comp['type'] == 'L':
                # Values are stored in Henry; use base unit to avoid double-scaling
                circuit.Inductor(comp['name'], n1, n2, val@u_H)
            elif comp['type'] == 'C':
                # Values are stored in Farad; use base unit to avoid double-scaling
                circuit.Capacitor(comp['name'], n1, n2, val@u_F)

        return circuit

    # ========================================================================
    # SPECIFICATION-BASED GENERATION METHODS
    # ========================================================================

    def _validate_component(self, comp_type, value):
        """
        Validate component is within practical range.

        Args:
            comp_type: 'R', 'L', or 'C'
            value: Component value in base SI units

        Returns:
            True if valid

        Raises:
            ValueError: If value is out of practical range
        """
        RANGES = {
            'R': (10, 100e3),      # 10Ω to 100kΩ
            'L': (100e-6, 10e-3),  # 100μH to 10mH
            'C': (1e-9, 1e-6)      # 1nF to 1μF
        }

        if comp_type not in RANGES:
            raise ValueError(f"Unknown component type: {comp_type}")

        min_val, max_val = RANGES[comp_type]
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"{comp_type}={value:.2e} out of practical range "
                f"[{min_val:.2e}, {max_val:.2e}]"
            )
        return True

    def _auto_adjust_RC(self, fc):
        """
        Find valid R,C combination for given cutoff frequency.

        Tries standard capacitor values to find a valid resistor value.

        Args:
            fc: Cutoff frequency (Hz)

        Returns:
            (R, C): Resistor and capacitor values

        Raises:
            ValueError: If frequency cannot be realized with practical components
        """
        ω_c = 2 * np.pi * fc

        # Try standard capacitor values
        C_candidates = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9, 1e-6]

        for C in C_candidates:
            R = 1 / (ω_c * C)
            if 10 <= R <= 100e3:
                return R, C

        # If still out of range, frequency may be unrealizable
        min_fc = 1 / (2 * np.pi * 100e3 * 1e-6)  # R_max * C_max
        max_fc = 1 / (2 * np.pi * 10 * 1e-9)     # R_min * C_min

        raise ValueError(
            f"Cannot realize fc={fc:.2e}Hz with practical components. "
            f"Valid range: {min_fc:.2f}Hz to {max_fc:.2e}Hz"
        )

    def from_low_pass_spec(self, fc, C=100e-9):
        """
        Generate low-pass RC filter from cutoff frequency specification.

        Circuit topology: Vin --R-- Vout --C-- GND

        Args:
            fc: Cutoff frequency (Hz)
            C: Capacitance (F), default 100nF

        Returns:
            actual_fc: Achieved cutoff frequency

        Raises:
            ValueError: If specification cannot be realized
        """
        self.filter_type = 'low_pass'
        self.components = []
        self.graph.clear()

        ω_c = 2 * np.pi * fc

        # Calculate R from fc and C
        R = 1 / (ω_c * C)

        # Validate and auto-adjust if needed
        try:
            self._validate_component('C', C)
            self._validate_component('R', R)
        except ValueError:
            # Try to auto-adjust
            R, C = self._auto_adjust_RC(fc)

        # Build component list
        self.components = [
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 1, 'node2': 2},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_fc = 1 / (2 * np.pi * R * C)
        return actual_fc

    def from_high_pass_spec(self, fc, C=100e-9):
        """
        Generate high-pass RC filter from cutoff frequency specification.

        Circuit topology: Vin --C-- Vout --R-- GND

        Args:
            fc: Cutoff frequency (Hz)
            C: Capacitance (F), default 100nF

        Returns:
            actual_fc: Achieved cutoff frequency

        Raises:
            ValueError: If specification cannot be realized
        """
        self.filter_type = 'high_pass'
        self.components = []
        self.graph.clear()

        ω_c = 2 * np.pi * fc

        # Calculate R from fc and C (same as low-pass)
        R = 1 / (ω_c * C)

        # Validate and auto-adjust if needed
        try:
            self._validate_component('C', C)
            self._validate_component('R', R)
        except ValueError:
            # Try to auto-adjust
            R, C = self._auto_adjust_RC(fc)

        # Build component list (different order than low-pass)
        self.components = [
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 1, 'node2': 2},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_fc = 1 / (2 * np.pi * R * C)
        return actual_fc

    def from_band_pass_spec(self, f0, Q=5.0, C=100e-9):
        """
        Generate band-pass series RLC filter from specification.

        Circuit topology: Vin --L--C-- Vout --R-- GND, measure across R

        Args:
            f0: Center frequency (Hz)
            Q: Quality factor (default 5.0), controls bandwidth
            C: Capacitance (F), default 100nF

        Returns:
            actual_f0: Achieved center frequency

        Raises:
            ValueError: If specification cannot be realized

        Notes:
            - Bandwidth BW = f0 / Q
            - Damping ratio ζ = 1 / (2Q)
            - Underdamped (oscillatory) requires Q > 0.5
        """
        self.filter_type = 'band_pass'
        self.components = []
        self.graph.clear()

        if Q < 0.5:
            raise ValueError(
                f"Q={Q} too low for band-pass filter. "
                "Minimum Q=0.5 for underdamped response."
            )

        if Q > 50:
            print(f"Warning: Q={Q} is very high, may be difficult to realize")

        ω_0 = 2 * np.pi * f0
        ζ = 1 / (2 * Q)

        # Calculate L from resonance condition: ω_0 = 1/√(LC)
        L = 1 / (ω_0**2 * C)

        # Calculate R from damping: ζ = R/(2) * √(C/L)
        # Rearranging: R = 2*ζ*√(L/C) = 2*ζ*ω_0*L
        R = 2 * ζ * ω_0 * L

        # Validate components
        try:
            self._validate_component('R', R)
            self._validate_component('L', L)
            self._validate_component('C', C)
        except ValueError as e:
            # Try to adjust C to get valid component values
            found_valid = False
            C_candidates = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9, 1e-6]

            for C_try in C_candidates:
                L_try = 1 / (ω_0**2 * C_try)
                R_try = 2 * ζ * ω_0 * L_try

                try:
                    self._validate_component('R', R_try)
                    self._validate_component('L', L_try)
                    self._validate_component('C', C_try)
                    R, L, C = R_try, L_try, C_try
                    found_valid = True
                    break
                except ValueError:
                    continue

            if not found_valid:
                raise ValueError(
                    f"Cannot realize band-pass at f0={f0}Hz, Q={Q} "
                    f"with practical components. Original error: {e}"
                )

        # Build component list (series RLC, measure across R)
        self.components = [
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 1, 'node2': 3},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 3, 'node2': 2},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        return actual_f0

    def from_band_stop_spec(self, f0, Q=10.0, C=100e-9):
        """
        Generate band-stop (notch) series LC shunt filter from specification.

        Circuit topology: Vin --R_series-- Node3[series LC to GND]
                          --R_load-- Vout --R_out-- GND

        Args:
            f0: Notch frequency (Hz)
            Q: Quality factor (default 10.0), controls notch sharpness
            C: Capacitance (F), default 100nF

        Returns:
            actual_f0: Achieved notch frequency

        Raises:
            ValueError: If specification cannot be realized

        Notes:
            - Higher Q = sharper notch (more selective rejection)
            - Typical Q for notch filters: 5-20
            - Uses series LC in shunt path for proper notch behavior
        """
        self.filter_type = 'band_stop'
        self.components = []
        self.graph.clear()

        if Q < 0.5:
            raise ValueError(f"Q={Q} too low for band-stop filter. Minimum Q=0.5")

        if Q > 50:
            print(f"Warning: Q={Q} is very high for notch filter")

        ω_0 = 2 * np.pi * f0

        # Calculate L from resonance
        L = 1 / (ω_0**2 * C)

        # For series LC shunt notch filter:
        # The characteristic impedance is Z0 = sqrt(L/C)
        # Q factor for the notch: Q = Z0 / R_series
        # Therefore: R_series = Z0 / Q = sqrt(L/C) / Q
        #
        # R_load and R_out provide the output voltage divider
        # They should be high enough not to load the notch significantly

        Z0 = np.sqrt(L / C)  # Characteristic impedance
        R_series = Z0 / Q    # Calculate R_series from Q

        # R_load should be much larger than R_series to avoid loading
        R_load = max(10 * R_series, 1000)  # At least 10x R_series or 1kΩ
        R_out = R_load

        # Validate and adjust components
        found_valid = False
        C_candidates = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9, 1e-6]

        for C_try in C_candidates:
            L_try = 1 / (ω_0**2 * C_try)
            Z0_try = np.sqrt(L_try / C_try)
            R_series_try = Z0_try / Q
            R_load_try = max(10 * R_series_try, 1000)

            try:
                self._validate_component('L', L_try)
                self._validate_component('C', C_try)
                self._validate_component('R', R_series_try)
                self._validate_component('R', R_load_try)
                L, C, R_series, R_load, R_out = L_try, C_try, R_series_try, R_load_try, R_load_try
                found_valid = True
                break
            except ValueError:
                continue

        if not found_valid:
            raise ValueError(
                f"Cannot realize band-stop at f0={f0}Hz, Q={Q} "
                f"with practical components."
            )

        # Build component list (series LC in shunt path)
        self.components = [
            {'name': 'R_series', 'type': 'R', 'value': R_series, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 4},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 4, 'node2': 0},
            {'name': 'R_load', 'type': 'R', 'value': R_load, 'node1': 3, 'node2': 2},
            {'name': 'R_out', 'type': 'R', 'value': R_out, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        return actual_f0

    def from_rlc_series_spec(self, f0, Q=5.0, C=100e-9):
        """
        Generate RLC series resonant circuit from specification.

        Circuit topology: Vin --R--L--C-- Vout --R_load-- GND

        Args:
            f0: Resonant frequency (Hz)
            Q: Quality factor (default 5.0)
            C: Capacitance (F), default 100nF

        Returns:
            actual_f0: Achieved resonant frequency

        Raises:
            ValueError: If specification cannot be realized
        """
        self.filter_type = 'rlc_series'
        self.components = []
        self.graph.clear()

        if Q < 0.5:
            raise ValueError(f"Q={Q} too low. Minimum Q=0.5")

        ω_0 = 2 * np.pi * f0
        ζ = 1 / (2 * Q)

        # Calculate L from resonance
        L = 1 / (ω_0**2 * C)

        # Total resistance from damping (R + R_load)
        R_total = 2 * ζ * ω_0 * L

        # Split between R and R_load (R_load = 10*R typical)
        R = R_total / 11
        R_load = 10 * R

        # Validate components
        try:
            self._validate_component('R', R)
            self._validate_component('L', L)
            self._validate_component('C', C)
            self._validate_component('R', R_load)
        except ValueError as e:
            # Try different C values
            found_valid = False
            C_candidates = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9, 1e-6]

            for C_try in C_candidates:
                L_try = 1 / (ω_0**2 * C_try)
                R_total_try = 2 * ζ * ω_0 * L_try
                R_try = R_total_try / 11
                R_load_try = 10 * R_try

                try:
                    self._validate_component('R', R_try)
                    self._validate_component('L', L_try)
                    self._validate_component('C', C_try)
                    self._validate_component('R', R_load_try)
                    R, L, C, R_load = R_try, L_try, C_try, R_load_try
                    found_valid = True
                    break
                except ValueError:
                    continue

            if not found_valid:
                raise ValueError(
                    f"Cannot realize RLC series at f0={f0}Hz, Q={Q} "
                    f"with practical components. Original error: {e}"
                )

        # Build component list
        self.components = [
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 4},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 4, 'node2': 2},
            {'name': 'R_load', 'type': 'R', 'value': R_load, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        return actual_f0

    def from_rlc_parallel_spec(self, f0, Q=5.0, C=100e-9):
        """
        Generate RLC parallel resonant circuit from specification.

        Circuit topology: Vin --R_source-- Node2[L||C||R to GND]

        Args:
            f0: Resonant frequency (Hz)
            Q: Quality factor (default 5.0)
            C: Capacitance (F), default 100nF

        Returns:
            actual_f0: Achieved resonant frequency

        Raises:
            ValueError: If specification cannot be realized
        """
        self.filter_type = 'rlc_parallel'
        self.components = []
        self.graph.clear()

        if Q < 0.5:
            raise ValueError(f"Q={Q} too low. Minimum Q=0.5")

        ω_0 = 2 * np.pi * f0

        # Calculate L from resonance
        L = 1 / (ω_0**2 * C)

        # For parallel RLC, Q factor depends on parallel combination of resistances
        # Q = R_eff * √(C/L) where R_eff = (R * R_source) / (R + R_source)
        #
        # Choose R_source = R/10 (typical ratio), then:
        # R_eff = (R * R/10) / (R + R/10) = R²/10 / (11R/10) = R/11
        # Q = (R/11) * √(C/L)
        # Therefore: R = 11 * Q * √(L/C)

        R = 11 * Q * np.sqrt(L / C)

        # Source resistance (R_source = R/10)
        R_source = R / 10

        # Validate components
        try:
            self._validate_component('R', R)
            self._validate_component('L', L)
            self._validate_component('C', C)
            self._validate_component('R', R_source)
        except ValueError as e:
            # Try different C values
            found_valid = False
            C_candidates = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9, 1e-6]

            for C_try in C_candidates:
                L_try = 1 / (ω_0**2 * C_try)
                R_try = Q * np.sqrt(L_try / C_try)
                R_source_try = R_try / 10

                try:
                    self._validate_component('R', R_try)
                    self._validate_component('L', L_try)
                    self._validate_component('C', C_try)
                    self._validate_component('R', R_source_try)
                    R, L, C, R_source = R_try, L_try, C_try, R_source_try
                    found_valid = True
                    break
                except ValueError:
                    continue

            if not found_valid:
                raise ValueError(
                    f"Cannot realize RLC parallel at f0={f0}Hz, Q={Q} "
                    f"with practical components. Original error: {e}"
                )

        # Build component list
        self.components = [
            {'name': 'R_source', 'type': 'R', 'value': R_source, 'node1': 1, 'node2': 2},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 2, 'node2': 0},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 2, 'node2': 0},
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()

        # Return actual achieved frequency
        actual_f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        return actual_f0


def extract_poles_zeros_gain_analytical(filter_type, components):
    """
    Analytically calculates poles, zeros, and gain based on circuit topology.
    Uses proper circuit analysis with impedance calculations and voltage dividers.
    """
    # Extract component values
    comp_dict = {}
    for comp in components:
        comp_dict[comp['name']] = comp['value']

    poles = []
    zeros = []
    gain = 1.0

    if filter_type == 'low_pass':
        # RC Low-pass: Vin(1) --R1-- Vout(2) --C1-- GND(0)
        # Z_C = 1/(sC), H(s) = Z_C / (R + Z_C) = 1 / (1 + sRC)
        R = comp_dict.get('R1', 0)
        C = comp_dict.get('C1', 0)
        if R > 0 and C > 0:
            pole = -1.0 / (R * C)
            poles = [pole]
            zeros = []
            # Gain: H(s) = K / (s - p), H(0) = K/(-p) = 1, so K = -p
            gain = -pole

    elif filter_type == 'high_pass':
        # RC High-pass: Vin(1) --C1-- Vout(2) --R1-- GND(0)
        # Z_C = 1/(sC), H(s) = R / (R + Z_C) = sRC / (1 + sRC)
        # H(s) = s / (s - p) where p = -1/(RC)
        R = comp_dict.get('R1', 0)
        C = comp_dict.get('C1', 0)
        if R > 0 and C > 0:
            pole = -1.0 / (R * C)
            poles = [pole]
            zeros = [0.0]  # Zero at DC
            # Gain: H(s) = K·s / (s - p), as s→∞: H(∞) = K = 1
            gain = 1.0  # HF gain is 1 (already correct)

    elif filter_type == 'band_pass':
        # Series RLC: Vin(1) --L1-- (3) --C1-- Vout(2) --R1-- GND(0)
        # Measuring voltage across R (node 2)
        # I = Vin / (sL + 1/(sC) + R)
        # V_R = I × R = Vin × R / (sL + 1/(sC) + R)
        # H(s) = R / (sL + 1/(sC) + R) = sRC / (s²LC + sRC + 1)
        R = comp_dict.get('R1', 0)
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)

        if R > 0 and L > 0 and C > 0:
            # Numerator: sRC → zero at s=0
            zeros = [0.0]

            # Denominator: s²LC + sRC + 1
            # Standard form: s² + (R/L)s + 1/(LC)
            omega_0 = 1.0 / np.sqrt(L * C)
            zeta = R / (2.0) * np.sqrt(C / L)

            if zeta < 1:  # Underdamped
                real_part = -R / (2 * L)
                imag_part = omega_0 * np.sqrt(1 - zeta**2)
                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:  # Overdamped
                sqrt_term = np.sqrt((R / (2 * L))**2 - 1 / (L * C))
                poles = [-R / (2 * L) + sqrt_term, -R / (2 * L) - sqrt_term]

            # Gain: H(s) = K·s / ((s-p1)(s-p2))
            # From transfer function: H(s) = (RC)·s / (LC·s² + RC·s + 1)
            # The coefficient of s in numerator is RC
            # K = RC × |p1·p2| where p1·p2 = 1/(LC)
            gain = R * C * np.abs(poles[0] * poles[1])

    elif filter_type == 'band_stop':
        # Band-stop (notch) filter
        # Vin(1) --R_series-- Node3[series LC to GND] --R_load-- Vout(2) --R_out-- GND
        # Series LC in shunt: L(3→4), C(4→0)
        #
        # At resonance: series LC shorts to ground → maximum attenuation (notch)
        # Away from resonance: series LC has high impedance → signal passes through
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)
        R_series = comp_dict.get('R_series', 0)
        R_load = comp_dict.get('R_load', 0)
        R_out = comp_dict.get('R_out', 0)

        if L > 0 and C > 0 and R_out > 0:
            omega_0 = 1.0 / np.sqrt(L * C)

            # Zeros: at resonance frequency on imaginary axis
            # The series LC impedance becomes zero (short to ground)
            zeros = [complex(0, omega_0), complex(0, -omega_0)]

            # For series LC shunt notch filter:
            # Q = Z0 / R_series where Z0 = sqrt(L/C) is characteristic impedance
            Z0 = np.sqrt(L / C)
            if R_series > 0:
                Q = Z0 / R_series
            else:
                Q = 10.0  # Default

            # Clamp Q to reasonable range
            Q = max(0.5, min(Q, 100.0))

            # Poles: damped oscillation
            # For a second-order system: poles at -ω₀/(2Q) ± jω₀√(1 - 1/(4Q²))
            zeta = 1.0 / (2 * Q)  # Damping ratio
            if zeta < 1:  # Underdamped
                real_part = -omega_0 * zeta
                imag_part = omega_0 * np.sqrt(1 - zeta**2)
                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:  # Overdamped
                poles = [
                    -omega_0 * (zeta + np.sqrt(zeta**2 - 1)),
                    -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
                ]

            # Passband gain (away from notch): voltage divider from Vin to Vout
            R_right = (R_load + R_out) if R_load > 0 else R_out
            Z_parallel_passband = R_right
            V3_gain = Z_parallel_passband / (R_series + Z_parallel_passband) if (R_series + Z_parallel_passband) > 0 else 0.5
            Vout_gain = R_out / R_right if R_right > 0 else 1.0
            H_passband = V3_gain * Vout_gain

            # For the pole-zero form: K = H_passband × |p1·p2| / ω₀²
            gain = H_passband * np.abs(poles[0] * poles[1]) / (omega_0**2)

    elif filter_type == 'rlc_series':
        # Vin(1) --R1-- (3) --L1-- (4) --C1-- Vout(2) --R_load-- GND(0)
        # Measuring voltage across R_load only (node 2 to GND)
        # NOT measuring across C1+R_load, just R_load!
        R = comp_dict.get('R1', 0)
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)
        R_load = comp_dict.get('R_load', 0)

        if R > 0 and L > 0 and C > 0 and R_load > 0:
            R_total = R + R_load

            # Correct transfer function: V_out = I × R_load
            # I = Vin / Z_total = Vin / (R_total + sL + 1/(sC))
            # H(s) = R_load / (R_total + sL + 1/(sC))
            #      = sC·R_load / (s²LC + sC·R_total + 1)
            #
            # Numerator: sC·R_load → zero at s = 0
            # Denominator: s²LC + sC·R_total + 1 → poles from characteristic equation

            zeros = [0.0]  # Zero at origin (capacitor blocks DC)

            # Poles from denominator: s² + (R_total/L)s + 1/(LC) = 0
            omega_0 = 1.0 / np.sqrt(L * C)
            zeta = R_total / (2.0) * np.sqrt(C / L)

            if zeta < 1:
                real_part = -R_total / (2 * L)
                imag_part = omega_0 * np.sqrt(1 - zeta**2)
                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:
                sqrt_term = np.sqrt((R_total / (2 * L))**2 - 1 / (L * C))
                poles = [-R_total / (2 * L) + sqrt_term, -R_total / (2 * L) - sqrt_term]

            # Gain: H(s) = K·s / ((s-p1)(s-p2))
            # From the transfer function: H(s) = (C·R_load)·s / (LC·s² + C·R_total·s + 1)
            # The coefficient of s in numerator is C·R_load
            # When factored: H(s) = K·s / ((s-p1)(s-p2)) = K·s / (s² - (p1+p2)s + p1·p2)
            # Comparing: K / (LC) = C·R_load
            # Therefore: K = LC · C · R_load = C²·L·R_load
            # But simpler: K = C·R_load · (p1·p2) where p1·p2 = 1/(LC)
            gain = C * R_load * np.abs(poles[0] * poles[1])

    elif filter_type == 'rlc_parallel':
        # Vin(1) --R_source-- Vout(2: parallel RLC)-- GND(0)
        R_source = comp_dict.get('R_source', 0)
        R = comp_dict.get('R1', 0)
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)

        if L > 0 and C > 0 and R_source > 0:
            omega_0 = 1.0 / np.sqrt(L * C)

            # Quality factor with both resistances
            if R > 0:
                R_eff = (R * R_source) / (R + R_source)
                Q = R_eff * np.sqrt(C / L)
                damping = 1.0 / Q

                real_part = -omega_0 * damping / 2.0
                imag_part = omega_0 * np.sqrt(1 - (damping/2.0)**2) if damping < 2.0 else omega_0 * 0.5

                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:
                poles = [complex(-omega_0*0.01, omega_0), complex(-omega_0*0.01, -omega_0)]

            # Zero at origin (inductor shorts at DC)
            zeros = [0.0]

            # Gain: H(s) = K·s / ((s-p1)(s-p2))
            # At high frequency, voltage divider dominates: H(∞) = R/(R+R_source)
            # As s→∞: H(s) ≈ K·s/s² = K/s → 0, but before that it peaks
            # Better: match the transfer function coefficient
            # H(s) = K·s/((s-p1)(s-p2)) = K·s/(s² - (p1+p2)s + p1·p2)
            # From circuit analysis, K should normalize to voltage divider at resonance
            gain = R / (R + R_source) if R > 0 else 0.5

    # Convert complex numbers to proper format
    poles = [complex(p) if not isinstance(p, complex) else p for p in poles]
    zeros = [complex(z) if not isinstance(z, complex) else z for z in zeros]

    return poles, zeros, gain


def create_compact_graph_representation(nx_multigraph, filter_type):
    """
    Converts the component multigraph into a simple graph where
    parallel components are merged into a single impedance function.
    """
    G = nx.Graph()
    G.add_nodes_from(nx_multigraph.nodes())

    # Add Node Features
    # 0: GND, 1: IN, 2: OUT, >2: Internal
    for n in G.nodes():
        if n == 0:
            ntype = [1, 0, 0, 0]  # GND
        elif n == 1:
            ntype = [0, 1, 0, 0]  # IN
        elif n == 2:
            ntype = [0, 0, 1, 0]  # OUT
        else:
            ntype = [0, 0, 0, 1]  # Internal

        G.nodes[n]['features'] = ntype

    # Add filter type as graph attribute
    G.graph['filter_type'] = filter_type

    # Iterate over unique pairs of connected nodes
    for u, v in set(nx_multigraph.edges()):
        if u == v:
            continue

        # Get all edges (components) between u and v
        edges = nx_multigraph.get_edge_data(u, v)

        # Admittance Parameters
        total_conductance = 0.0  # 1/R
        total_capacitance = 0.0  # C
        total_inverse_inductance = 0.0  # 1/L

        for key, attr in edges.items():
            val = attr['value']
            ctype = attr['type']

            if ctype == 'R':
                total_conductance += 1.0 / val
            elif ctype == 'C':
                total_capacitance += val
            elif ctype == 'L':
                total_inverse_inductance += 1.0 / val

        # Net admittance: Y(s) = G + C*s + (1/L_eq)/s where (1/L_eq) is total_inverse_inductance.
        # Net impedance:  Z(s) = s / (C*s^2 + G*s + total_inverse_inductance)
        # Store polynomial coefficients (highest degree first) for numerator/denominator in s-domain.
        if total_conductance == 0.0 and total_capacitance == 0.0 and total_inverse_inductance == 0.0:
            # Safety: should not happen, but keep a well-defined edge
            imp_num = [1.0]
            imp_den = [1.0]
        else:
            imp_num = [1.0, 0.0]  # s
            imp_den = [total_capacitance, total_conductance, total_inverse_inductance]  # C*s^2 + G*s + L_inv

        G.add_edge(u, v, impedance_num=imp_num, impedance_den=imp_den)

    return G

def main():
    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)

    dataset = []

    print("Generating filter circuits...")
    print(f"Filter types: {FILTER_TYPES}")
    print(f"Samples per type: {NUM_SAMPLES_PER_FILTER}")
    print(f"Total circuits: {len(FILTER_TYPES) * NUM_SAMPLES_PER_FILTER}\n")

    success_count = 0

    for filter_type in FILTER_TYPES:
        print(f"\nGenerating {filter_type} filters...")

        for i in range(NUM_SAMPLES_PER_FILTER):
            gen = FilterGenerator()

            # Generate appropriate filter type
            if filter_type == 'low_pass':
                char_freq = gen.generate_low_pass_filter()
            elif filter_type == 'high_pass':
                char_freq = gen.generate_high_pass_filter()
            elif filter_type == 'band_pass':
                char_freq = gen.generate_band_pass_filter()
            elif filter_type == 'band_stop':
                char_freq = gen.generate_band_stop_filter()
            elif filter_type == 'rlc_series':
                char_freq = gen.generate_rlc_series_resonant()
            elif filter_type == 'rlc_parallel':
                char_freq = gen.generate_rlc_parallel_resonant()

            # Extract Poles, Zeros, Gain using analytical method (no SPICE needed)
            poles, zeros, gain = extract_poles_zeros_gain_analytical(filter_type, gen.components)

            # Create ML-ready Graph Representation
            ml_graph = create_compact_graph_representation(gen.graph, filter_type)

            # Generate synthetic frequency response from poles/zeros (for compatibility)
            freqs = np.logspace(1, 8, 100)  # 10 Hz to 100 MHz
            s = 1j * 2 * np.pi * freqs

            # Compute H(s) from poles/zeros/gain
            H = np.ones_like(s, dtype=complex) * gain
            for z in zeros:
                H *= (s - z)
            for p in poles:
                H /= (s - p)

            # Pack data
            data_point = {
                'id': uuid.uuid4().hex,
                'filter_type': filter_type,
                'characteristic_frequency': char_freq,
                'components': gen.components,
                'graph_adj': nx.adjacency_data(ml_graph),
                'frequency_response': {
                    'freqs': freqs,
                    'H_magnitude': np.abs(H),
                    'H_phase': np.angle(H),
                    'H_complex': H
                },
                'label': {
                    'poles': poles if poles is not None else [],
                    'zeros': zeros if zeros is not None else [],
                    'gain': gain if gain is not None else 0.0
                }
            }

            dataset.append(data_point)
            success_count += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  ✓ {filter_type} #{i+1}/{NUM_SAMPLES_PER_FILTER} | fc={char_freq:.2f} Hz | Poles: {len(poles)}")

    # Save dataset
    output_file = os.path.join(DATASET_DIR, "filter_dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'='*60}")
    print(f"Dataset saved to {output_file}")
    print(f"Total circuits generated: {success_count}/{len(FILTER_TYPES) * NUM_SAMPLES_PER_FILTER}")
    print(f"{'='*60}\n")

    # Show sample
    if dataset:
        sample = dataset[0]
        print("--- Sample Data Point ---")
        print(f"ID: {sample['id']}")
        print(f"Filter Type: {sample['filter_type']}")
        print(f"Characteristic Frequency: {sample['characteristic_frequency']:.2f} Hz")
        print(f"Label Gain: {sample['label']['gain']:.4e}")
        print(f"Poles: {len(sample['label']['poles'])}")
        print(f"Zeros: {len(sample['label']['zeros'])}")
        print(f"Graph Nodes: {len(sample['graph_adj']['nodes'])}")
        print(f"Graph Edges: {len(sample['graph_adj']['adjacency'])}")
        print(f"Components: {len(sample['components'])}")
        print(f"Frequency points: {len(sample['frequency_response']['freqs'])}")

if __name__ == "__main__":
    main()
