# Usage Guide

This guide documents the current workflows in this repository.

## 1) Environment Setup

### Runtime environment

```bash
./scripts/setup_venv.sh runtime
source .venv/bin/activate
```

### Development environment

```bash
./scripts/setup_venv.sh dev
source .venv/bin/activate
```

### Quick health check

```bash
make doctor
```

## 2) Pole/Zero-Driven Generation (Primary)

Use the decoder-only generation path with dominant pole/zero inputs.

```bash
.venv/bin/python scripts/generation/generate_from_specs.py \
  --pole-real -6283 \
  --pole-imag 0 \
  --zero-real 0 \
  --zero-imag 0 \
  --num-samples 5
```

Arguments:
- `--pole-real` / `--pole-imag`: dominant pole parts
- `--zero-real` / `--zero-imag`: dominant zero parts (optional)
- `--num-samples`: number of generated topologies
- `--checkpoint`: defaults to `checkpoints/production/best.pt`
- `--device`: `cpu` or `cuda`

## 3) Training

```bash
.venv/bin/python scripts/training/train.py
```

Training data and split:
- `rlc_dataset/filter_dataset.pkl`
- `rlc_dataset/stratified_split.pt`

Output checkpoint directory:
- `checkpoints/production/`

## 4) Validation and Evaluation

### Component/topology validation

```bash
.venv/bin/python scripts/training/validate.py
```

### Pole/zero latent quality

```bash
.venv/bin/python scripts/eval/eval_pz.py
```

## 5) Latent Exploration Utilities

### Interpolate between filter centroids

```bash
.venv/bin/python scripts/generation/interpolate_filter_types.py \
  --from low_pass \
  --to high_pass \
  --steps 7
```

### Regenerate report outputs

```bash
.venv/bin/python scripts/generation/regenerate_all_results.py
```

## 6) Tests

Run suite:

```bash
.venv/bin/python tests/run_tests.py --suite all
```

Run subsets:

```bash
.venv/bin/python tests/run_tests.py --suite unit
.venv/bin/python tests/run_tests.py --suite spec
```

## 7) Python API (Minimal)

```python
import torch
from ml.utils.runtime import load_decoder
from ml.utils.circuit_ops import circuit_to_string, is_valid_circuit

decoder, _ = load_decoder('checkpoints/production/best.pt', device='cpu')
z = torch.randn(1, 8)

with torch.no_grad():
    circuit = decoder.generate(z)

print(circuit_to_string(circuit))
print('valid:', is_valid_circuit(circuit))
```

## 8) Common Make Targets

```bash
make setup        # create runtime venv
make setup-dev    # create dev venv
make doctor       # import check
make train
make eval-pz
make generate-pz
make test
```

## 9) Compatibility Notes

- `scripts/generation/generate_from_specs.py` is pole/zero-based.
- Several exploratory scripts in `scripts/testing/` still evaluate cutoff/Q interpolation behavior built from encoded dataset latents.
- If you regenerate the dataset with `tools/circuit_generator.py`, regenerate split files before training.
