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

def extract_poles_zeros_gain(freqs, H_complex):
    """
    Fits a rational transfer function H(s) = K * (s-z1)... / (s-p1)...
    to the frequency response data.
    """
    w = 2 * np.pi * freqs
    s = 1j * w

    # Try orders 1 through 6
    best_error = float('inf')
    best_sys = None

    for order in range(1, 7):
        try:
            b, a = signal.invfreqs(H_complex, w, nb=order, na=order)

            # Check reconstruction error
            H_fit = signal.freqs(b, a, w)[1]
            error = np.mean(np.abs(H_complex - H_fit)**2)

            if error < best_error:
                best_error = error
                best_sys = (b, a)
        except:
            continue

    if best_sys is None:
        return None, None, None

    b, a = best_sys

    # Extract ZPK
    z, p, k = signal.tf2zpk(b, a)

    return list(p), list(z), k

def create_compact_graph_representation(nx_multigraph, filter_type):
    """
    Converts the component multigraph into a simple graph where
    parallel components are merged into a single edge feature vector.
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

        G.add_edge(u, v, features=[total_conductance, total_capacitance, total_inverse_inductance])

    return G

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    dataset = []

    print(f"Generating filter circuits...")
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

            # Extract Poles, Zeros, Gain (optional, may fail for some circuits)
            poles, zeros, gain = extract_poles_zeros_gain(freqs, H)

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
            pole_info = f"Poles: {len(poles)}" if poles is not None else "No poles extracted"
            print(f"  ✓ {filter_type} #{i+1}/{NUM_SAMPLES_PER_FILTER} | fc={char_freq:.2f} Hz | {pole_info}")

    # Save to disk
    filepath = os.path.join(DATASET_DIR, "filter_dataset.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'='*60}")
    print(f"Dataset saved to {filepath}")
    print(f"Total circuits generated: {success_count}/{len(FILTER_TYPES) * NUM_SAMPLES_PER_FILTER}")
    print(f"{'='*60}")

    # Print sample statistics
    if dataset:
        print("\n--- Sample Data Point ---")
        first = dataset[0]
        print(f"ID: {first['id']}")
        print(f"Filter Type: {first['filter_type']}")
        print(f"Characteristic Frequency: {first['characteristic_frequency']:.2f} Hz")
        print(f"Label Gain: {first['label']['gain']:.4e}")
        print(f"Poles: {len(first['label']['poles'])}")
        print(f"Zeros: {len(first['label']['zeros'])}")
        print(f"Graph Nodes: {len(first['graph_adj']['nodes'])}")
        print(f"Graph Edges: {len(first['graph_adj']['adjacency'])}")
        print(f"Components: {len(first['components'])}")
        print(f"Frequency points: {len(first['frequency_response']['freqs'])}")

if __name__ == "__main__":
    main()
