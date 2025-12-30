# Component Value Denormalization Fix

## Root Cause Analysis

After implementing CVAE conditioning and achieving 100% training accuracy, the model still generates circuits with incorrect specifications (1668% average error). The issue is **missing denormalization** in the SPICE simulator.

### The Problem

**During Training:**
1. Dataset normalizes component values:
   ```python
   # Clip to practical ranges
   C_practical = np.clip(C, 1e-12, 1e-6)  # 1pF to 1μF
   R_practical = np.clip(R, 10, 100e3)     # 10Ω to 100kΩ
   L_practical = np.clip(L, 1e-9, 10e-3)   # 1nH to 10mH

   # Log transform
   log_C = np.log(C_practical)

   # Z-score normalization
   log_C_normalized = (log_C - C_mean) / C_std
   ```

2. Model learns to predict **normalized** values: `[log_C_norm, G_norm, log_L_inv_norm]`

3. Training uses these normalized values with teacher forcing → 100% accuracy ✓

**During Generation:**
1. Model outputs **normalized** values (z-score)
2. SPICE simulator directly applies:
   ```python
   C_value = np.exp(log_C_normalized)  # ❌ WRONG!
   ```
3. Should denormalize first:
   ```python
   log_C_raw = log_C_normalized * C_std + C_mean
   C_value = np.exp(log_C_raw)  # ✓ CORRECT
   ```

### Evidence

**Test results show:**
- Generated values: C = 10-45 Farads (should be pF-μF)
- Generated values: R = 0.5-100 Ω (should be 10Ω-100kΩ)
- Negative resistances (from negative normalized G values)
- Most circuits measure at 1 Hz regardless of target

**Values ARE changing with target specs**, proving the decoder is working. But the absolute scale is wrong due to missing denormalization.

## The Fix

### Step 1: Pass normalization stats to simulator

```python
# In test_unseen_specs.py
dataset = CircuitDataset('rlc_dataset/filter_dataset.pkl')

# Get normalization stats from dataset
impedance_mean = dataset.impedance_mean.numpy()  # [C_mean, G_mean, L_inv_mean]
impedance_std = dataset.impedance_std.numpy()    # [C_std, G_std, L_inv_std]

# Create simulator with normalization stats
simulator = CircuitSimulator(
    simulator='ngspice',
    freq_points=200,
    freq_start=1.0,
    freq_stop=1e6,
    impedance_mean=impedance_mean,  # NEW
    impedance_std=impedance_std      # NEW
)
```

### Step 2: Update CircuitSimulator to denormalize

```python
# In ml/utils/spice_simulator.py

class CircuitSimulator:
    def __init__(self, ..., impedance_mean=None, impedance_std=None):
        self.impedance_mean = impedance_mean
        self.impedance_std = impedance_std

    def circuit_to_netlist(self, ...):
        # Get edge values (normalized)
        log_C_norm = edge_values_np[i, j, 0]
        log_G_norm = edge_values_np[i, j, 1]
        log_L_inv_norm = edge_values_np[i, j, 2]

        # Denormalize if stats provided
        if self.impedance_mean is not None:
            log_C = log_C_norm * self.impedance_std[0] + self.impedance_mean[0]
            log_G = log_G_norm * self.impedance_std[1] + self.impedance_mean[1]
            log_L_inv = log_L_inv_norm * self.impedance_std[2] + self.impedance_mean[2]
        else:
            # Assume already denormalized (for backward compatibility)
            log_C = log_C_norm
            log_G = log_G_norm
            log_L_inv = log_L_inv_norm

        # Convert to actual values
        C_value = np.exp(log_C)  # Now correct!
        G_value = np.exp(log_G)
        L_value = 1.0 / np.exp(log_L_inv)
```

### Step 3: Update all scripts that use CircuitSimulator

- `scripts/test_unseen_specs.py` ✓
- `scripts/generate_from_specs.py` (if it uses SPICE)
- Any other scripts that create CircuitSimulator

## Expected Impact

After this fix:
- Component values will be in correct ranges (pF-μF, 10Ω-100kΩ, nH-mH)
- No more negative resistances
- Cutoff frequencies should match targets within 10-20% (limited by training data coverage)
- Q-factors should match targets within 10-20%

## Implementation Priority

**CRITICAL** - This is a blocking bug that prevents the model from generating usable circuits.

## Status

- [ ] Update CircuitSimulator class
- [ ] Update test_unseen_specs.py
- [ ] Update generate_from_specs.py
- [ ] Run tests to validate fix
- [ ] Compare before/after accuracy
