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
NUM_SAMPLES_PER_FILTER = 20  # Generate 20 variations of each filter type
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
        """RLC Band-pass filter (series RLC)"""
        self.filter_type = 'band_pass'
        self.components = []
        self.graph.clear()

        # Randomize component values
        R = 10 ** random.uniform(2, 4)  # 100 Ohm to 10 kOhm
        L = 10 ** random.uniform(-4, -2)  # 0.1mH to 10mH
        C = 10 ** random.uniform(-9, -6)  # 1nF to 1uF

        # Center frequency: f0 = 1 / (2*pi*sqrt(L*C))
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))

        # Series RLC: Vin --R--L--C-- GND, Vout across C
        self.components = [
            {'name': 'R1', 'type': 'R', 'value': R, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 2},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 2, 'node2': 0}
        ]

        self._build_graph()
        return f0

    def generate_band_stop_filter(self):
        """RLC Band-stop (notch) filter (parallel RLC)"""
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

        # Vin --R_series-- (parallel RLC to GND) -- Vout -- R_load -- GND
        self.components = [
            {'name': 'R_series', 'type': 'R', 'value': R, 'node1': 1, 'node2': 3},
            {'name': 'L1', 'type': 'L', 'value': L, 'node1': 3, 'node2': 0},
            {'name': 'C1', 'type': 'C', 'value': C, 'node1': 3, 'node2': 0},
            {'name': 'R_parallel', 'type': 'R', 'value': R * 2, 'node1': 3, 'node2': 0},
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
        # Series RLC: Vin(1) --R1-- (3) --L1-- Vout(2) --C1-- GND(0)
        # Measuring voltage at node 2 (between L and C)
        # This measures voltage across C only
        # I = Vin / (R + sL + 1/(sC))
        # V_C = I / (sC) = Vin / (sC(R + sL + 1/(sC)))
        # H(s) = 1 / (s²LC + sRC + 1)
        R = comp_dict.get('R1', 0)
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)

        if R > 0 and L > 0 and C > 0:
            # Numerator: 1 (constant, no zeros)
            zeros = []

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

            # Gain: H(s) = K / ((s-p1)(s-p2))
            # At s=0: H(0) = K/(p1*p2) = 1, so K = p1*p2
            # Note: p1*p2 = omega_0^2 = 1/(LC)
            gain = poles[0] * poles[1]
            gain = np.abs(gain)  # Make it real and positive

    elif filter_type == 'band_stop':
        # Band-stop (notch) filter
        # Vin(1) --R_series-- Node3[L||C||R_parallel to GND] --R_load-- Vout(2) --R_out-- GND
        #
        # The parallel LC creates high impedance at resonance, passing signal through
        # At DC and HF, the LC shorts to ground, attenuating the signal
        L = comp_dict.get('L1', 0)
        C = comp_dict.get('C1', 0)
        R_series = comp_dict.get('R_series', 0)
        R_parallel = comp_dict.get('R_parallel', 0)
        R_load = comp_dict.get('R_load', 0)
        R_out = comp_dict.get('R_out', 0)

        if L > 0 and C > 0 and R_out > 0:
            omega_0 = 1.0 / np.sqrt(L * C)

            # Zeros: at resonance frequency on imaginary axis
            # The parallel LC impedance becomes very high
            zeros = [complex(0, omega_0), complex(0, -omega_0)]

            # Poles: The full network is 4th order, but we approximate with dominant 2nd order poles
            # The effective damping comes from the ratio of series to parallel resistance
            # and the LC network. Better approximation accounts for loading.
            if R_parallel > 0:
                # Effective resistance seen by LC tank includes loading from R_series and R_right
                R_right = R_load + R_out if R_load > 0 else R_out
                R_load_eff = (R_series * R_right) / (R_series + R_right) if (R_series + R_right) > 0 else R_series

                # Total parallel resistance is R_parallel || R_load_eff
                R_eff = (R_parallel * R_load_eff) / (R_parallel + R_load_eff) if (R_parallel + R_load_eff) > 0 else R_parallel

                # Quality factor with effective resistance
                Q = R_eff * np.sqrt(C / L)
                damping = 1.0 / Q

                # For high-Q notch filters, poles are very close to imaginary axis
                # Use more accurate pole placement
                real_part = -omega_0 / (2 * Q)  # More accurate than -omega_0*damping/2
                imag_part = omega_0 * np.sqrt(1 - 1/(4*Q**2)) if Q > 0.5 else omega_0 * 0.5

                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:
                # Without R_parallel, poles are on imaginary axis (undamped)
                poles = [complex(-omega_0*0.01, omega_0), complex(-omega_0*0.01, -omega_0)]

            # Gain: The transfer function in pole-zero form is:
            # H(s) = K*(s² + ω₀²) / ((s-p1)(s-p2))
            #
            # Near resonance (s ≈ jω), both numerator and denominator ≈ 0
            # Away from resonance, the gain approaches a constant
            #
            # At very low or very high frequencies, the LC shorts out, so H→0
            # This means we need K to be small enough that H→0 as s→0 or s→∞
            #
            # The passband gain (at resonance vicinity) depends on voltage division
            # At resonance specifically, H = 0 due to zeros
            # Just below/above resonance, we get the peak passband response

            R_right = (R_load + R_out) if R_load > 0 else R_out
            if R_parallel > 0 and R_right > 0:
                # At resonance, parallel combination of R_parallel and R_right
                Z_par_res = (R_parallel * R_right) / (R_parallel + R_right)
                V3_gain = Z_par_res / (R_series + Z_par_res) if R_series + Z_par_res > 0 else 0
                Vout_gain = R_out / R_right if R_right > 0 else 1.0
                H_passband = V3_gain * Vout_gain
            else:
                H_passband = 0.5

            # The gain K must be chosen to match the passband level
            # For the pole-zero form: H(s) = K(s² + ω₀²) / ((s-p1)(s-p2))
            # The gain is: K = H_passband × |p1·p2| / ω₀²
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

            # Convert to PySpice
            circuit = gen.to_pyspice()

            # Simulate (AC Analysis)
            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            try:
                # Sweep from 10Hz to 100MHz
                analysis = simulator.ac(start_frequency=10@u_Hz, stop_frequency=100@u_MHz,
                                      number_of_points=100, variation='dec')
            except Exception as e:
                print(f"  Sim fail for {filter_type} #{i+1}: {e}")
                continue

            # Extract Transfer Function H(s) = Vout / Vin
            freqs = analysis.frequency.as_ndarray()
            vin = analysis['1'].as_ndarray()
            vout = analysis['2'].as_ndarray()

            # Calculate complex transfer function
            H = vout / vin

            # Extract Poles, Zeros, Gain using analytical method
            poles, zeros, gain = extract_poles_zeros_gain_analytical(filter_type, gen.components)

            # Create ML-ready Graph Representation
            ml_graph = create_compact_graph_representation(gen.graph, filter_type)

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
