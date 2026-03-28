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
- Decoder: sequence decoder (GPT-style), d_model=256, 4 heads, 4 layers, max_seq_len=33
- Vocabulary: 86 tokens (nets + components)
- Loss: CE (next-token) + 0.01 * KL (with warmup)
- Optimizer: AdamW, lr=3e-4
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Epochs: 100
- Batch size: 32
