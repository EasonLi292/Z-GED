# Configuration Notes

This folder contains YAML experiment configurations from earlier training workflows.

## Current State

- The active production training entry point is `scripts/training/train.py`.
- Runtime model defaults are centralized in `ml/utils/runtime.py`:
  - `DEFAULT_ENCODER_CONFIG`
  - `DEFAULT_DECODER_CONFIG`

## Files in This Directory

- `production.yaml`
- `optimized_8d.yaml`
- `test.yaml`

These files are kept for reference/experiments and are not the primary source of truth for the current default training path.

## Current Production Defaults

- Latent dim: 8 (`2 + 2 + 4` split)
- Encoder hidden dim: 64
- Decoder hidden dim: 256
- Decoder max nodes: 10
- Loss weights: node=1.0, count=5.0, edge_component=2.0, connectivity=5.0, KL=0.01, pz=5.0
- Optimizer: Adam, lr=1e-4
- Epochs: 100
- Batch size: 16
