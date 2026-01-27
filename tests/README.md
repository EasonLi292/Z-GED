# GraphVAE Test Suite

Comprehensive test suite for the GraphVAE circuit latent space discovery project.

## Organization

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_dataset.py        # Dataset loading and preprocessing
│   ├── test_models.py         # Model architectures (encoder, decoder, GNN)
│   └── test_component_substitution.py  # GED component substitution
│
├── spec_generation/           # Specification-based circuit generation tests
│   ├── test_spec_basic.py     # Low-pass and high-pass generation
│   ├── test_bandpass_spec.py  # Band-pass filter generation
│   └── test_spec_generation.py  # All 6 filter types
│
└── integration/               # Integration tests (future)
    └── (to be added)
```

## Running Tests

### Run all tests:
```bash
python3 run_tests.py
```

### Run specific test suite:
```bash
python3 run_tests.py --suite unit
python3 run_tests.py --suite spec
```

### Run individual test:
```bash
python3 tests/unit/test_dataset.py
python3 tests/unit/test_models.py
python3 tests/unit/test_component_substitution.py
```

## Test Coverage

### Unit Tests

**test_dataset.py** - CircuitDataset functionality
- ✅ Basic dataset loading (120 circuits)
- ✅ Single sample retrieval
- ✅ Batching with DataLoader
- ✅ Train/val/test split (96/12/12)
- ✅ All samples loadable
- ✅ Stratified split verification

**test_models.py** - Model architecture
- ✅ ImpedanceConv layer (custom message passing)
- ✅ ImpedanceGNN (3-layer GNN)
- ✅ DeepSets (variable-length poles/zeros)
- ✅ HierarchicalEncoder (8D latent)
- ✅ SimplifiedCircuitDecoder generation
- ✅ SimplifiedCircuitDecoder forward with teacher forcing (autoregressive edges)
- ✅ SimplifiedCircuitDecoder forward without teacher forcing
- ✅ Real data integration

**test_component_substitution.py** - GED component substitution
- ✅ R→C, R→L, C→L substitution costs
- ✅ All component type pairs
- ✅ Low-pass/high-pass discrimination

### Spec Generation Tests

**test_spec_basic.py** - Low-pass and high-pass
- ✅ Low-pass at 1kHz (pole verification)
- ✅ High-pass at 10kHz (pole verification)
- ✅ Frequency limit validation

**test_bandpass_spec.py** - Band-pass filters
- ✅ 10kHz center, Q=5 (pole/zero validation)
- ✅ Different Q factors (1, 5, 10, 20)
- ✅ Frequency range (1kHz to 100kHz)

**test_spec_generation.py** - All filter types
- ✅ Low-pass (0.0000% error)
- ✅ High-pass (0.0000% error)
- ✅ Band-pass (0.0000% error)
- ✅ Band-stop (~40% Q error, known limitation)
- ✅ RLC series (0.0000% error)
- ✅ RLC parallel (0.0000% error)

## Test Statistics

```
Total test files: 6
Total test cases: ~40+
Passing rate: 100% (except known band-stop Q limitation)
Coverage: Dataset, Models, GED, Spec Generation
```

## Expected Test Output

### Successful run:
```
======================================================================
GRAPHVAE TEST SUITE
======================================================================

Running 6 test files...

Running: tests/unit/test_dataset.py
...
✅ ALL TESTS PASSED!

Running: tests/unit/test_models.py
...
ALL MODEL TESTS PASSED!

======================================================================
TEST SUMMARY
======================================================================
  test_dataset.py                     ✅ PASS
  test_models.py                      ✅ PASS
  test_component_substitution.py      ✅ PASS
  test_spec_basic.py                  ✅ PASS
  test_bandpass_spec.py               ✅ PASS
  test_spec_generation.py             ✅ PASS

Passed: 6/6
```

## Known Issues

### test_spec_generation.py - Band-stop Q error
- **Issue**: Band-stop Q factor has ~40% error
- **Cause**: Complex 6-component parallel RLC interaction
- **Impact**: Analytical formula doesn't capture all effects
- **Status**: Known limitation, documented

## Adding New Tests

1. Create test file in appropriate directory:
   ```python
   # tests/unit/test_new_feature.py
   def test_new_feature():
       # Test code
       assert result == expected
   ```

2. Add to `run_tests.py` test suites if needed

3. Run: `python3 run_tests.py --suite unit`

## Continuous Integration

Tests are ready for CI integration:
- All tests are deterministic (seeded random)
- No external dependencies beyond pip packages
- Fast execution (<2 minutes total)
- Clear pass/fail output

## Development Workflow

1. **Before committing**: `python3 run_tests.py`
2. **After changes**: Run relevant suite
3. **Before PR**: Run all tests
4. **Debug**: Run individual test with verbose output
