# Scripts Organization

This directory contains organized scripts for training, testing, generation, and analysis.

## Directory Structure

```
scripts/
├── training/       # Model training and validation
├── testing/        # Test suites
├── generation/     # Circuit generation
└── analysis/       # Analysis and debugging
```

## Training

**Main Training:**
- **train.py** - Main training script for the circuit generation model
  - Trains encoder + decoder end-to-end
  - Uses configuration from `configs/`
  - Saves checkpoints to `checkpoints/`

**Data Preparation:**
- **create_stratified_split.py** - Create stratified train/val/test splits
  - Ensures balanced distribution of filter types
  - Preserves specification diversity

**Validation:**
- **validate.py** - Comprehensive validation script
  - Tests reconstruction quality
  - Evaluates latent space properties

## Testing

**Main Test Suite:**
- **test_comprehensive_specs.py** - Full test suite (USE THIS)
  - Tests various cutoff frequency and Q-factor combinations
  - Validates generated circuits with SPICE simulation
  - Comprehensive output with topology analysis

**Specialized Tests:**
- **test_single_spec.py** - Test single specification in detail
  - Useful for debugging specific cases

## Generation

**Main Generation:**
- **generate_from_specs.py** - Generate circuits from user specifications
  - Input: cutoff frequency and Q-factor
  - Output: Circuit with component values and netlist
  - Uses K-NN interpolation in latent space

**Interpolation:**
- **interpolate_filter_types.py** - Interpolate between filter types in latent space
  - Smooth transitions between topologies
  - Useful for exploring latent space structure

## Usage Examples

**Training:**
```bash
python scripts/training/train.py
```

**Testing:**
```bash
python scripts/testing/test_comprehensive_specs.py
```

**Generation:**
```bash
python scripts/generation/generate_from_specs.py --cutoff 10000 --q-factor 0.707
```
