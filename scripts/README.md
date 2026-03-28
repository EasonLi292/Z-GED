# Scripts Overview

This directory contains executable workflows grouped by purpose.

## Structure

```text
scripts/
├── training/
├── generation/
├── eval/
└── testing/
```

## training/

- `train.py`
  - Main end-to-end training entry point.
  - Uses `rlc_dataset/filter_dataset.pkl` + `rlc_dataset/stratified_split.pt`.
  - Saves best checkpoint to `checkpoints/production/best.pt`.
- `validate.py`
  - Encodes validation set, generates walks, reports topology match rate and valid walk rate.
- `create_stratified_split.py`
  - Rebuilds train/val split file from dataset filter-type distribution.

## generation/

- `generate_from_specs.py`
  - **Primary generation script**.
  - Pole/zero-driven decoder-only generation.
  - Inputs: `--pole-real/--pole-imag/--zero-real/--zero-imag`.
- `interpolate_filter_types.py`
  - Builds per-filter latent centroids and interpolates between types.
- `regenerate_all_results.py`
  - Recomputes the major results sections used in repo reports.

## eval/

- `eval_pz.py`
  - Evaluates pole/zero latent prediction quality (`mu[:, 4:]` vs `pz_target`).

## testing/

Exploratory analysis scripts for specification interpolation behavior.

- `test_single_spec.py`
  - Single cutoff/Q interpolation case deep dive.
- `test_comprehensive_specs.py`
  - Batch cutoff/Q interpolation sweep and summary.

## Typical Commands

```bash
.venv/bin/python scripts/training/train.py
.venv/bin/python scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 3
.venv/bin/python scripts/eval/eval_pz.py
```
