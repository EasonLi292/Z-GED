# Test Suite

## Layout

```text
tests/
├── unit/
├── spec_generation/
├── integration/
└── run_tests.py
```

## Run Tests

All:

```bash
.venv/bin/python tests/run_tests.py --suite all
```

Specific suites:

```bash
.venv/bin/python tests/run_tests.py --suite unit
.venv/bin/python tests/run_tests.py --suite spec
```

Verbose output:

```bash
.venv/bin/python tests/run_tests.py --suite all --verbose
```

## Unit Coverage

- `unit/test_dataset.py`
- `unit/test_models.py`
- `unit/test_component_substitution.py`

## Spec Generation Coverage

- `spec_generation/test_spec_basic.py`
- `spec_generation/test_bandpass_spec.py`
- `spec_generation/test_spec_generation.py`

## Notes

- These tests depend on local dataset/checkpoint artifacts in this repo.
- Use `scripts/setup_venv.sh dev` for a consistent test environment.
