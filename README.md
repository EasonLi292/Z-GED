# Z-GED: Graph-VAE Circuit Topology Generation

Z-GED generates RLC circuit topologies from an 8D latent space.

Current generation entry points:
- **Pole/zero-driven generation** (decoder-only): `scripts/generation/generate_from_specs.py`
- **Latent interpolation/exploration**: `scripts/generation/interpolate_filter_types.py`
- **Result regeneration report**: `scripts/generation/regenerate_all_results.py`

## Current Repository Snapshot

- Dataset file: `rlc_dataset/filter_dataset.pkl`
- Included dataset size: **1920 circuits**
- Filter types: `low_pass`, `high_pass`, `band_pass`, `band_stop`, `rlc_series`, `rlc_parallel`, `lc_lowpass`, `cl_highpass`
- Stratified split file: `rlc_dataset/stratified_split.pt` (1536 train / 384 val)
- Production checkpoint: `checkpoints/production/best.pt` (epoch 21, val_loss 0.023)

## Step-by-Step: Run the Project

This sequence assumes you already have this repository checked out locally.

### 1) Create environment

```bash
./scripts/setup_venv.sh runtime
source .venv/bin/activate
```

Or with dev tools:

```bash
./scripts/setup_venv.sh dev
source .venv/bin/activate
```

### 2) Verify dependencies

```bash
make doctor
```

### 3) Verify required artifacts

```bash
ls checkpoints/production/best.pt
ls rlc_dataset/filter_dataset.pkl
ls rlc_dataset/stratified_split.pt
```

### 4) Run generation (primary workflow)

```bash
.venv/bin/python scripts/generation/generate_from_specs.py \
  --pole-real -6283 \
  --pole-imag 0 \
  --num-samples 5
```

### 5) Run tests

```bash
make test
```

### 6) Train

```bash
.venv/bin/python scripts/training/train.py
```

### 7) Validate/evaluate

```bash
.venv/bin/python scripts/training/validate.py
.venv/bin/python scripts/eval/eval_pz.py
```

### 8) Optional: latent exploration

```bash
.venv/bin/python scripts/generation/interpolate_filter_types.py --from low_pass --to high_pass --steps 7
.venv/bin/python scripts/generation/regenerate_all_results.py
```

## One-Command Shortcuts

```bash
make setup       # runtime venv
make setup-dev   # dev venv
make doctor      # dependency health check
make generate-pz # sample pole/zero generation
make train
make eval-pz
make test
```

## Core Architecture

- Encoder: `ml/models/encoder.py`
  - 3-layer impedance-aware GNN
  - Hierarchical latent split: `[z_topology(2) | z_values(2) | z_pz(4)]`
- Decoder: `ml/models/decoder.py`
  - GPT-style autoregressive transformer (sequence decoder)
  - Generates circuits as Eulerian walk token sequences over a bipartite graph
  - 86-token vocabulary (nets + components), latent prefix conditioning
- Loss: CE (next-token prediction) + KL divergence

## Project Structure

```text
Z-GED/
├── ml/
│   ├── data/            # Dataset loading, bipartite graph, traversal
│   ├── models/          # Encoder, sequence decoder, vocabulary
│   └── utils/           # Runtime, circuit ops, evaluation helpers
├── scripts/
│   ├── training/        # Training and validation scripts
│   ├── generation/      # Generation and latent exploration
│   ├── eval/            # Evaluation scripts
│   └── testing/         # Spec-focused exploratory tests
├── tests/               # Unit/spec test suite
├── tools/               # Dataset and GED utilities
├── checkpoints/         # Trained checkpoints
└── rlc_dataset/         # Dataset artifacts
```

## Documentation

- `ARCHITECTURE.md` - model and training design
- `USAGE.md` - command reference and workflows
- `GENERATION_RESULTS.md` - generated results and analysis
- `docs/pole_zero_prediction.md` - pole/zero representation details

## Notes

- The decoder generates Eulerian walk token sequences (not adjacency matrices). See `ARCHITECTURE.md` for details.
- The current `generate_from_specs.py` interface is pole/zero based (`--pole-real`, `--pole-imag`, `--zero-real`, `--zero-imag`).

## License

MIT (see `LICENSE`)
