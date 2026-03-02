# Development Workflow

## Bootstrap

```bash
./scripts/setup_venv.sh dev
source .venv/bin/activate
make doctor
```

## Common Commands

```bash
make train
make eval-pz
make generate-pz
make test
```

## Rebuild Data Split

If dataset contents change:

```bash
.venv/bin/python scripts/training/create_stratified_split.py
```

## Notes

- Model and collate defaults are centralized in `ml/utils/runtime.py`.
- Reusable circuit formatting/validity helpers are in `ml/utils/circuit_ops.py`.
- Keep generation script interfaces aligned with docs (`README.md`, `USAGE.md`).
