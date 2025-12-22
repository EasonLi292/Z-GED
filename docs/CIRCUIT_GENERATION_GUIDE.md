# Circuit Generation Guide

## Overview

The trained GraphVAE model can generate novel circuits through multiple strategies. This guide explains how to use the circuit generation capabilities to create new filter circuits.

## Quick Start

```bash
# Generate 5 random circuits
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode prior \
    --num-samples 5 \
    --save-json

# Generate 5 low-pass filters
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode conditional \
    --filter-type low_pass \
    --num-samples 5 \
    --save-json

# Interpolate between two circuits
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode interpolate \
    --circuit1-idx 0 \
    --circuit2-idx 20 \
    --interp-steps 5 \
    --save-json

# Modify a specific circuit property
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode modify \
    --circuit-idx 5 \
    --modify-branch values \
    --modify-amount 0.5 \
    --save-json
```

## Generation Modes

### 1. Prior Sampling (Unconditional)

Sample circuits randomly from the latent prior distribution N(0, I).

**When to use**: Explore the full diversity of circuits the model can generate.

```bash
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode prior \
    --num-samples 10 \
    --temperature 1.0
```

**Parameters**:
- `--num-samples`: Number of circuits to generate (default: 5)
- `--temperature`: Sampling temperature
  - `< 1.0`: More conservative, stay closer to training distribution (e.g., 0.5)
  - `= 1.0`: Standard sampling (default)
  - `> 1.0`: More exploratory, generate more diverse circuits (e.g., 2.0)

**Example Output**:
```
Circuit 1: high_pass (80.82% confidence) - 3 nodes, 4 edges
Circuit 2: low_pass (99.37% confidence) - 3 nodes, 4 edges
Circuit 3: band_stop (99.62% confidence) - 5 nodes, 9 edges
```

### 2. Conditional Generation

Generate circuits of a specific filter type with 100% accuracy using teacher forcing.

**When to use**: Need circuits with specific characteristics (filter type).

```bash
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode conditional \
    --filter-type band_pass \
    --num-samples 5 \
    --temperature 0.8
```

**Filter Types**:
- `low_pass`: Low-pass filters (3 nodes, 4 edges)
- `high_pass`: High-pass filters (3 nodes, 4 edges)
- `band_pass`: Band-pass filters (4 nodes, 7 edges)
- `band_stop`: Band-stop/notch filters (5 nodes, 9 edges)
- `rlc_series`: RLC series resonant circuits (3 nodes, 3 edges)
- `rlc_parallel`: RLC parallel resonant circuits (3 nodes, 3 edges)

**Example Output**:
```
Circuit 1: band_pass (100.00% confidence) - 4 nodes, 7 edges
Circuit 2: band_pass (100.00% confidence) - 4 nodes, 7 edges
Circuit 3: band_pass (100.00% confidence) - 4 nodes, 7 edges
```

### 3. Latent Space Interpolation

Smoothly transition between two circuits in latent space.

**When to use**:
- Explore the latent space structure
- Create smooth transitions between circuit designs
- Understand what's "in between" two different circuits

```bash
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode interpolate \
    --circuit1-idx 0 \
    --circuit2-idx 20 \
    --interp-steps 7 \
    --interp-type linear
```

**Parameters**:
- `--circuit1-idx`, `--circuit2-idx`: Indices of circuits from the dataset
- `--interp-steps`: Number of intermediate circuits (including endpoints)
- `--interp-type`: Interpolation method
  - `linear`: Linear interpolation z = (1-α)z₁ + αz₂
  - `spherical`: Spherical linear interpolation (slerp)
  - `branch`: Interpolate each latent branch independently

**Example Output** (low-pass → high-pass):
```
Circuit 1: low_pass (99.89% confidence)  - α=0.00
Circuit 2: low_pass (99.30% confidence)  - α=0.25
Circuit 3: low_pass (67.66% confidence)  - α=0.50 ← Transition zone
Circuit 4: high_pass (98.83% confidence) - α=0.75
Circuit 5: high_pass (99.91% confidence) - α=1.00
```

### 4. Branch Modification

Modify a specific aspect of a circuit by adjusting one latent branch.

**When to use**:
- Change topology while keeping component values (modify `topo` branch)
- Change component values while keeping topology (modify `values` branch)
- Change transfer function characteristics (modify `pz` branch)

```bash
python scripts/generate.py \
    --checkpoint checkpoints/20251220_225827/best.pt \
    --mode modify \
    --circuit-idx 10 \
    --modify-branch values \
    --modify-amount 1.0 \
    --modify-op add
```

**Parameters**:
- `--circuit-idx`: Index of circuit to modify
- `--modify-branch`: Which latent branch to modify
  - `topo`: Topology branch (2D) - affects filter type
  - `values`: Component values branch (2D) - affects R, L, C values
  - `pz`: Poles/zeros branch (4D) - affects transfer function
- `--modify-op`: How to modify
  - `add`: Add constant to branch
  - `multiply`: Scale branch by factor
  - `replace`: Set to specific value
- `--modify-amount`: Modification amount

**Example**: Varying component values
```bash
# Original circuit
--modify-amount 0.0  → Original component values

# Increase component values
--modify-amount 0.5  → Larger R, L, C values
--modify-amount 1.0  → Even larger values

# Decrease component values
--modify-amount -0.5 → Smaller R, L, C values
```

## Output Format

### Console Output

```
======================================================================
GENERATED 3 CIRCUITS
======================================================================

Circuit 1:
  Filter Type: low_pass (99.89% confidence)
  Topology: 3 nodes, 4 edges
  Components: 4 edges with features
  Transfer Function: 2 poles, 2 zeros
```

### JSON Output

With `--save-json`, results are saved to `generated_circuits/`:

```json
{
  "mode": "conditional",
  "num_circuits": 3,
  "model_config": {
    "latent_dim": 8,
    "branch_dims": [2, 2, 4]
  },
  "circuits": [
    {
      "filter_type": "low_pass",
      "filter_confidence": 0.9989,
      "num_nodes": 3,
      "num_edges": 4,
      "edge_features": [
        [4.17e-12, 1.14e-09, 2.57e-12, 0.19, 0.0, 0.01, 0.20],
        // [C, G, L_inv, has_C, has_R, has_L, is_parallel]
        ...
      ],
      "predicted_poles": [[-428160.0, 0.0], ...],
      "predicted_zeros": [[0.0, 0.0], ...],
      "node_types": [0, 1, 2]  // [GND, VIN, VOUT]
    }
  ]
}
```

**Edge Features Format**:
- `[0:3]`: Impedance values `[C, G, L_inv]` in denormalized units
  - C: Capacitance (Farads)
  - G: Conductance = 1/R (Siemens)
  - L_inv: 1/Inductance (1/Henries)
- `[3:7]`: Binary masks `[has_C, has_R, has_L, is_parallel]`
  - Indicates which components are present on each edge

## Advanced Usage

### Temperature Tuning

Temperature controls the diversity/quality tradeoff:

```bash
# Conservative (high quality, low diversity)
--temperature 0.5

# Balanced (default)
--temperature 1.0

# Exploratory (lower quality, high diversity)
--temperature 1.5
```

### Batch Generation

Generate many circuits at once:

```bash
# Generate 100 band-pass filters
python scripts/generate.py \
    --mode conditional \
    --filter-type band_pass \
    --num-samples 100 \
    --save-json
```

### Custom Interpolation

Interpolate with different methods to explore latent space:

```bash
# Linear interpolation (default)
--interp-type linear

# Spherical interpolation (better for normalized spaces)
--interp-type spherical

# Branch-wise interpolation (independent control)
--interp-type branch
```

## Model Performance

The 8D model (2D+2D+4D) used for generation has:
- **Val loss**: 0.3115 (10.2% better than baseline)
- **Topology accuracy**: 100% (with teacher forcing)
- **Latent space**: Highly organized by filter type
- **Training**: Converged at epoch 127

## Validation

Generated circuits are automatically:
1. ✅ **Topology-valid**: Follow known circuit templates
2. ✅ **Component-valid**: Positive R, L, C values
3. ✅ **Transfer function**: Include predicted poles/zeros

For physical validation:
- Extract component values from edge_features
- Verify impedance ranges are realistic
- Simulate transfer function if needed

## Next Steps

After generating circuits:
1. **Validate**: Check that component values are physically reasonable
2. **Simulate**: Use SPICE or symbolic tools to verify transfer function
3. **Optimize**: Fine-tune component values for specific frequency requirements
4. **Fabricate**: Use generated designs as starting points for PCB design

## Troubleshooting

**Q: Generated circuits have unrealistic component values**
- Try lower temperature: `--temperature 0.7`
- Use conditional generation with specific filter type

**Q: Want more diversity in conditional generation**
- Increase temperature: `--temperature 1.2`
- Generate more samples to get variety

**Q: Interpolation produces invalid intermediate circuits**
- Use `--interp-type branch` for smoother transitions
- Increase `--interp-steps` for finer granularity

**Q: How to interpret edge_features?**
- First 3 values are [C, G, L_inv] in denormalized units
- R = 1/G, L = 1/L_inv
- Last 4 values are binary masks indicating component types
