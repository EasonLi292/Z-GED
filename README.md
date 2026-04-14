# Z-GED: Generate RLC Circuit Topologies

Z-GED is a generative model that produces RLC filter circuit topologies from electrical specifications. Give it the pole/zero behavior you want, and it outputs valid circuit structures.

## Quick Start

```bash
# 1. Set up environment
./scripts/setup_venv.sh runtime
source .venv/bin/activate

# 2. Generate a circuit (requires checkpoints/production/best.pt)
.venv/bin/python scripts/generation/generate_from_specs.py \
  --pole-real -6283 --pole-imag 0 --num-samples 5
```

This generates 5 RC low-pass topologies with a dominant pole at -6283 rad/s (~1 kHz cutoff).

## Generating Circuits

### By Pole/Zero Specification (Primary Interface)

Specify the dominant pole and/or zero of the transfer function you want. The model generates circuit topologies that match that behavior.

```bash
.venv/bin/python scripts/generation/generate_from_specs.py \
  --pole-real <real> --pole-imag <imag> \
  --zero-real <real> --zero-imag <imag> \
  --num-samples <N>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pole-real` | -1000 | Real part of the dominant pole (rad/s) |
| `--pole-imag` | 0 | Imaginary part of the dominant pole (rad/s) |
| `--zero-real` | 0 | Real part of the dominant zero (0 = no zero) |
| `--zero-imag` | 0 | Imaginary part of the dominant zero (0 = no zero) |
| `--num-samples` | 5 | Number of circuit topologies to generate |
| `--checkpoint` | `checkpoints/production/best.pt` | Model checkpoint |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed` | 42 | Random seed for reproducibility |

### Choosing Parameters for Common Filter Types

The pole/zero values map directly to the filter behavior you want:

| Desired Behavior | Pole (real + imag) | Zero | Example Command |
|------------------|-------------------|------|-----------------|
| **RC low-pass** (~1 kHz) | -6283 + 0j | none | `--pole-real -6283` |
| **RC low-pass** (~10 kHz) | -62832 + 0j | none | `--pole-real -62832` |
| **RC low-pass** (~100 kHz) | -628318 + 0j | none | `--pole-real -628318` |
| **RC high-pass** (~1 kHz) | none | -6283 + 0j | `--zero-real -6283` |
| **Resonant / band-pass** | -3142 + 49348j | none | `--pole-real -3142 --pole-imag 49348` |
| **Band-stop (notch)** | -3142 + 49348j | 0 + 49348j | `--pole-real -3142 --pole-imag 49348 --zero-imag 49348` |

**Frequency rule of thumb:** pole magnitude in rad/s = 2 * pi * cutoff_Hz. For a 1 kHz cutoff: 2 * pi * 1000 = 6283 rad/s.

**What you get back:** Each sample is a circuit topology string like `VOUT--C--VSS, VIN--R--VOUT` (an RC low-pass) or `INTERNAL_1--C--VOUT, INTERNAL_1--L--VIN, VOUT--R--VSS` (a band-pass).

### By Filter Type Interpolation

Morph smoothly between two filter types in the latent space:

```bash
.venv/bin/python scripts/generation/interpolate_filter_types.py \
  --from low_pass --to high_pass --steps 7
```

Available filter types: `low_pass`, `high_pass`, `band_pass`, `band_stop`, `rlc_series`, `rlc_parallel`, `lc_lowpass`, `cl_highpass`

This requires the encoder and dataset (not just the checkpoint).

### Python API

```python
import torch
from ml.utils.runtime import load_decoder
from ml.utils.circuit_ops import walk_to_string, is_valid_walk, generate_walk

# Load the decoder (no encoder or dataset needed)
decoder, vocab, _ = load_decoder('checkpoints/production/best.pt', device='cpu')

# Build a latent vector: z[0:4] = random topology, z[4:8] = pole/zero encoding
z = torch.randn(1, 8)
# Or construct z[4:8] from specific pole/zero values (see generate_from_specs.py)

walk = generate_walk(decoder, z, vocab)
print(walk_to_string(walk, vocab))   # e.g. "VOUT--C--VSS, VIN--R--VOUT"
print('valid:', is_valid_walk(walk))  # True/False
```

## Inverse Design (v2 Model)

The v2 model uses an admittance-polynomial encoder to generate circuits from target specifications (frequency, gain, filter type).

```bash
# Generate a band_pass filter at 10 kHz
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type band_pass --fc 10000

# Generate a low_pass filter at 1 kHz with specific gain
.venv/bin/python scripts/generation/generate_inverse_design.py \
  --type low_pass --fc 1000 --gain 0.5

# Frequency-only (no type constraint)
.venv/bin/python scripts/generation/generate_inverse_design.py --fc 50000
```

Available filter types: `band_pass`, `band_stop`, `cl_highpass`, `high_pass`, `lc_lowpass`, `low_pass`, `rl_highpass`, `rl_lowpass`, `rlc_parallel`, `rlc_series`

The pipeline: K-NN interpolation in latent space -> gradient descent on mu through attribute heads -> decode walks. See [`docs/inverse_design.md`](docs/inverse_design.md) for full documentation.

## Setup

### Environment

```bash
./scripts/setup_venv.sh runtime   # or: ./scripts/setup_venv.sh dev
source .venv/bin/activate
make doctor                       # verify dependencies
```

### Required Artifacts

```bash
ls checkpoints/production/best.pt      # v1 trained model
ls checkpoints/production/best_v2.pt   # v2 inverse design model
ls rlc_dataset/filter_dataset.pkl      # dataset (1920 circuits, 8 types)
ls rlc_dataset/rl_dataset.pkl          # RL dataset (480 circuits, 2 types — v2 only)
ls rlc_dataset/stratified_split.pt     # train/val split
```

### Make Targets

```bash
make setup        # runtime venv
make setup-dev    # dev venv
make doctor       # dependency health check
make generate-pz  # sample pole/zero generation
make train        # train from scratch
make eval-pz      # evaluate pole/zero prediction
make test         # run test suite
```

## Training

```bash
.venv/bin/python scripts/training/train.py
```

Trains on `rlc_dataset/filter_dataset.pkl` (1920 circuits, 8 filter types, 240 each). Checkpoints saved to `checkpoints/production/`.

### Validation and Evaluation

```bash
.venv/bin/python scripts/training/validate.py   # topology/component accuracy
.venv/bin/python scripts/eval/eval_pz.py         # pole/zero latent quality
```

## How It Works

The model is a VAE with an 8D latent space:

```
z = [z_topology(2) | z_values(2) | z_pz(4)]
```

- **z[0:4]**: Controls circuit structure (which components, how they connect)
- **z[4:8]**: Encodes pole/zero behavior (frequency response characteristics)

When you use `generate_from_specs.py`, only the decoder runs:
1. Your pole/zero inputs are normalized into z[4:8] via signed-log transform
2. z[0:4] is sampled randomly from N(0, I)
3. The GPT-style sequence decoder generates an Eulerian walk token sequence representing the circuit

**Encoder**: Impedance-aware GNN (3 layers) with hierarchical latent branches
**Decoder**: GPT-style autoregressive transformer (4 layers, 256 dim, 86-token vocabulary)

See `ARCHITECTURE.md` for full model details.

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

- `ARCHITECTURE.md` — model and training design (v1 + v2)
- `docs/inverse_design.md` — v2 admittance encoder and inverse design pipeline
- `USAGE.md` — command reference and workflows
- `GENERATION_RESULTS.md` — generated results and analysis
- `docs/pole_zero_prediction.md` — pole/zero representation details

## License

MIT (see `LICENSE`)
