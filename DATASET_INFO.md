# RLC Filter Circuit Dataset

## Overview
This dataset contains 120 simulated RLC filter circuits with their frequency response data and graph representations for impedance-based analysis.

**Generated:** December 2, 2025
**Dataset Size:** 3.3 MB
**Format:** Python pickle file (`.pkl`)

## Dataset Composition

### Filter Types (20 samples each)
1. **Low-Pass Filters (RC)**
   - Topology: Vin --R-- Vout --C-- GND
   - Cutoff frequency range: 10.18 Hz - 182.5 kHz

2. **High-Pass Filters (RC)**
   - Topology: Vin --C-- Vout --R-- GND
   - Cutoff frequency range: 3.24 Hz - 190.4 kHz

3. **Band-Pass Filters (RLC Series)**
   - Topology: Vin --R--L--C-- GND (output across C)
   - Center frequency range: 2.5 kHz - 297 kHz

4. **Band-Stop/Notch Filters (RLC Parallel)**
   - Topology: Parallel RLC tank with series/load resistors
   - Notch frequency range: 2.7 kHz - 245 kHz

5. **RLC Series Resonant**
   - Topology: Vin --R--L--C-- Load --GND
   - Resonant frequency range: 2.7 kHz - 370 kHz

6. **RLC Parallel Resonant**
   - Topology: Vin --Rsource-- (parallel RLC) --GND
   - Resonant frequency range: 2.7 kHz - 154 kHz

## Component Value Ranges

### Resistors
- Range: 100 Ω to 100 kΩ
- Distribution: Log-uniform

### Inductors
- Range: 0.1 mH to 10 mH (1×10⁻⁴ to 1×10⁻² H)
- Distribution: Log-uniform
- **Note:** Values stored in Henry (base SI unit)

### Capacitors
- Range: 1 nF to 1 μF (1×10⁻⁹ to 1×10⁻⁶ F)
- Distribution: Log-uniform
- **Note:** Values stored in Farad (base SI unit)

## Simulation Details

### AC Analysis
- **Frequency sweep:** 10 Hz to 100 MHz
- **Number of points:** 701 (logarithmically spaced)
- **Variation:** Decade sweep
- **Simulator:** ngspice 45
- **Temperature:** 25°C

### Node Convention
- **Node 0:** Ground (GND)
- **Node 1:** Input (Vin)
- **Node 2:** Output (Vout)
- **Node 3+:** Internal nodes

## Data Structure

Each circuit data point contains:

```python
{
    'id': str,                          # Unique UUID
    'filter_type': str,                 # Filter classification
    'characteristic_frequency': float,   # Cutoff/resonant freq (Hz)
    'components': [                     # List of components
        {
            'name': str,                # Component identifier
            'type': str,                # 'R', 'L', or 'C'
            'value': float,             # Value in base SI units
            'node1': int,               # Connection node 1
            'node2': int                # Connection node 2
        },
        ...
    ],
    'graph_adj': {                      # NetworkX graph data
        'nodes': [...],                 # Node list with features
        'adjacency': [...]              # Edge list with features
    },
    'frequency_response': {
        'freqs': ndarray,               # Frequency points (701)
        'H_magnitude': ndarray,         # |H(jω)|
        'H_phase': ndarray,             # ∠H(jω) in radians
        'H_complex': ndarray            # Complex H(jω)
    },
    'label': {
        'poles': list,                  # Transfer function poles
        'zeros': list,                  # Transfer function zeros
        'gain': float                   # DC gain
    }
}
```

## Graph Representation

### Node Features (One-hot encoding)
Each node has a 4-element feature vector:
- `[1,0,0,0]` - Ground (GND)
- `[0,1,0,0]` - Input (IN)
- `[0,0,1,0]` - Output (OUT)
- `[0,0,0,1]` - Internal node

### Edge Features (Impedance polynomial)
Each edge stores the net impedance of parallel components as polynomials in `s`:
- `impedance_num`: `[1, 0]`  (numerator coefficients for \(s\))
- `impedance_den`: `[C, G, L_inv]` giving \(Z(s) = s / (C s^2 + G s + L_{inv})\)
  - `C` = total capacitance in Farads
  - `G` = total conductance (1/R) in Siemens
  - `L_inv` = inverse total inductance (1/Henry)

## Usage Example

```python
import pickle
import numpy as np

# Load dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Access first circuit
circuit = dataset[0]
print(f"Filter type: {circuit['filter_type']}")
print(f"Cutoff frequency: {circuit['characteristic_frequency']:.2f} Hz")

# Plot frequency response
import matplotlib.pyplot as plt
freqs = circuit['frequency_response']['freqs']
mag = circuit['frequency_response']['H_magnitude']
plt.loglog(freqs, mag)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
```

## Notes

- All component values are stored in **base SI units** (Ohm, Henry, Farad)
- Frequency responses include full 701-point sweeps from 10 Hz to 100 MHz
- Graph representation uses admittance-based edge features for impedance analysis
- Pole/zero extraction may fail for some circuits (check if list is empty)
- PySpice warnings about spinit file and ngspice version can be safely ignored

## Files

- `rlc_dataset/filter_dataset.pkl` - Main dataset (3.3 MB)
- `tools/circuit_generator.py` - Generator script
- `tools/verify_dataset.py` - Dataset verification script

## Citation

If you use this dataset, please cite:
```
Z-GED: Impedance-based Graph Circuit Dataset
Generated using PySpice and ngspice
December 2025
```
