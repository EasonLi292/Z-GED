# Scripts Organization

This directory contains organized scripts for training, testing, generation, and analysis.

## Directory Structure

```
scripts/
├── training/       # Model training and validation
├── testing/        # Test suites
├── generation/     # Circuit generation
├── analysis/       # Analysis and debugging
├── utils/          # Utilities
└── archive/        # Archived/experimental scripts
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

- **test_hybrid_specs.py** - Test hybrid/cross-type specifications
  - Forces interpolation across filter types
  - Tests generalization capability

## Generation

**Main Generation:**
- **generate_from_specs.py** - Generate circuits from user specifications
  - Input: cutoff frequency and Q-factor
  - Output: Circuit with component values and netlist
  - Uses K-NN interpolation in latent space

**Utilities:**
- **extract_circuit_diagrams.py** - Extract detailed topology for documentation
  - Outputs circuit diagrams
  - Useful for creating visualizations

## Analysis

**Error Analysis:**
- **analyze_error_sources.py** - Investigate specification error sources
  - Decoder reconstruction error
  - K-NN interpolation error
  - Condition signal strength
  - Topology diversity

**Model Analysis:**
- **analyze_overfitting.py** - Analyze potential overfitting
  - Model complexity vs dataset size
  - Training/validation loss analysis

- **analyze_filter_types.py** - Analyze filter type distribution
  - Training data balance
  - Generated topology patterns

**Debugging:**
- **debug_circuit_netlist.py** - Debug SPICE netlist generation
  - Outputs actual SPICE netlist
  - Useful for troubleshooting simulation issues

## Utils

**Dataset Tools:**
- **inspect_dataset.py** - Inspect dataset contents
  - Shows circuit statistics
  - Validates data integrity

- **generate_doubled_dataset.py** - Generate augmented dataset
  - Creates larger dataset through augmentation
  - Useful for experiments

**System Tools:**
- **check_gpu.py** - Check GPU availability and CUDA setup
  - Validates PyTorch GPU access
  - Tests MPS (Apple Silicon) if available

- **precompute_ged.py** - Precompute graph edit distances
  - For GED-based metrics
  - Speeds up evaluation

## Archive

Older/experimental scripts preserved for reference:
- `test_unseen_specs.py` - Redundant with test_comprehensive_specs
- `test_topology_viability.py` - One-off topology analysis
- `evaluate_tf.py` - Transfer function evaluation
- `generate_targeted_tf.py` - Targeted generation (experimental)

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

**Analysis:**
```bash
python scripts/analysis/analyze_error_sources.py
```
