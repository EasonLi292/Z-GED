# Z-GED
Impedance-based Graph of Circuits Dataset

## Overview
This repository contains a dataset of 120 RLC filter circuits with their frequency response data and graph representations for machine learning applications focused on impedance analysis.

## Quick Start

### Prerequisites
- Python 3.12+
- ngspice (installed via Homebrew on macOS)

### Installation
```bash
# Install ngspice
brew install ngspice libngspice

# Install Python dependencies
pip3 install PySpice numpy scipy networkx
```

### Running the Generator

**Option 1: Using the wrapper script (recommended)**
```bash
./run_generator.sh
```

**Option 2: Manual execution**
```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
python3 "Generated Circuits/circuit_generator.py"
```

**Option 3: Add to your shell profile**

Add this to your `~/.zshrc` or `~/.bash_profile`:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

Then simply run:
```bash
python3 "Generated Circuits/circuit_generator.py"
```

## Dataset Information

- **Total Circuits:** 120
- **Filter Types:** 6 (low-pass, high-pass, band-pass, band-stop, RLC series, RLC parallel)
- **Samples per Type:** 20
- **Dataset Size:** 3.3 MB
- **Format:** Python pickle (`.pkl`)

See [DATASET_INFO.md](DATASET_INFO.md) for detailed documentation.

## Files

- `rlc_dataset/filter_dataset.pkl` - Main dataset file
- `Generated Circuits/circuit_generator.py` - Dataset generator script
- `verify_dataset.py` - Dataset verification script
- `run_generator.sh` - Convenient wrapper script
- `DATASET_INFO.md` - Complete dataset documentation

## Usage Example

```python
import pickle
import matplotlib.pyplot as plt

# Load dataset
with open('rlc_dataset/filter_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Examine first circuit
circuit = dataset[0]
print(f"Filter type: {circuit['filter_type']}")
print(f"Frequency: {circuit['characteristic_frequency']:.2f} Hz")

# Plot frequency response
freqs = circuit['frequency_response']['freqs']
mag = circuit['frequency_response']['H_magnitude']
plt.loglog(freqs, mag)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude |H(jω)|')
plt.grid(True)
plt.show()
```

## Troubleshooting

### "libngspice.dylib not found" Error

This error occurs because PySpice can't find the ngspice library. Solutions:

1. **Use the wrapper script:** `./run_generator.sh`
2. **Set environment variable manually:**
   ```bash
   export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
   ```
3. **Make it permanent:** Add the export to your shell profile

### "Unsupported Ngspice version" Warning

This warning can be safely ignored. PySpice works correctly with ngspice 45.

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this dataset, please cite:
```
Z-GED: Impedance-based Graph Circuit Dataset
Generated using PySpice and ngspice
December 2025
```
