# Configuration Files

This directory contains YAML configuration files for training the GraphVAE model.

## Available Configurations

### default.yaml
**Purpose**: Production baseline configuration
**Use case**: Full 200-epoch training runs
**Characteristics**:
- Original loss weights (recon=1.0, tf=0.5, kl=0.05)
- Cosine annealing scheduler
- Early stopping patience: 20 epochs
- Suitable for initial training

**When to use**:
- First full training run
- Baseline for comparisons
- Following original research plan

---

### optimized.yaml ⭐ RECOMMENDED
**Purpose**: Optimized configuration with rebalanced loss weights
**Use case**: Best performing configuration after Phase 1 optimizations
**Characteristics**:
- **Optimized loss weights** (recon=1.0, tf=0.01, kl=0.1)
- Reduced learning rate (5e-4 from 1e-3)
- Early stopping patience: 30 epochs
- 53% better loss than baseline

**When to use**:
- Standard training (most cases)
- After completing Phase 1 optimizations
- For best results

**Improvements over default**:
- 50x reduction in TF weight (addresses Chamfer distance scale issue)
- 2x increase in KL weight (better regularization)
- Slightly lower LR for stability

---

### test.yaml
**Purpose**: Quick 2-epoch test for pipeline verification
**Use case**: Testing code changes, debugging
**Characteristics**:
- Only 2 epochs
- No scheduler (faster iteration)
- Same optimized weights as optimized.yaml
- More frequent logging

**When to use**:
- After code changes (verify no errors)
- Quick sanity checks
- Debugging training issues
- CI/CD integration

---

## Configuration Structure

All configs follow this structure:

```yaml
# Data configuration
data:
  dataset_path: "rlc_dataset/filter_dataset.pkl"
  normalize: true              # Normalize impedance features
  log_scale: true              # Log-scale component values
  train_ratio: 0.8
  val_ratio: 0.1
  split_seed: 42

# Model architecture
model:
  node_feature_dim: 4          # [GND, VIN, VOUT, INTERNAL]
  edge_feature_dim: 3          # [log(C), log(G), log(L_inv)]
  gnn_hidden_dim: 64           # GNN hidden layer size
  gnn_num_layers: 3            # Number of GNN layers
  latent_dim: 24               # Hierarchical: 8D topo + 8D values + 8D pz
  decoder_hidden_dim: 128
  dropout: 0.1

# Loss function (KEY DIFFERENCES BETWEEN CONFIGS)
loss:
  recon_weight: 1.0
  tf_weight: 0.01-0.5          # Varies by config
  kl_weight: 0.05-0.1          # Varies by config

# Training
training:
  optimizer: "adamw"
  learning_rate: 5.0e-4 or 1.0e-3
  weight_decay: 1.0e-5
  scheduler: "cosine" or null
  epochs: 2 or 200
  batch_size: 4
  val_interval: 1
  log_interval: 2-5
  early_stopping_patience: 20-30
  checkpoint_dir: "checkpoints"

# Regularization
regularization:
  max_grad_norm: 1.0

# Hardware
hardware:
  num_workers: 0
  pin_memory: false
```

## Creating Custom Configurations

### Example: High Capacity Model
```yaml
# Copy from optimized.yaml, then modify:
model:
  gnn_hidden_dim: 128          # Increased from 64
  gnn_num_layers: 4            # Increased from 3
  latent_dim: 32               # Increased from 24
  decoder_hidden_dim: 256      # Increased from 128
```

### Example: Aggressive Regularization
```yaml
# Copy from optimized.yaml, then modify:
loss:
  recon_weight: 1.0
  tf_weight: 0.01
  kl_weight: 0.5               # Much higher KL for strong regularization

model:
  dropout: 0.2                 # Increased from 0.1
```

### Example: Fast Training
```yaml
# Copy from optimized.yaml, then modify:
training:
  batch_size: 8                # Larger batches
  learning_rate: 1.0e-3        # Higher LR
  early_stopping_patience: 10  # Stop sooner
```

## Loss Weight Guidelines

### Transfer Function Weight (tf_weight)
**Problem**: Chamfer distance on normalized poles/zeros is large (typically 4-8)
**Solution**: Use very low weight (0.01-0.05)

**Guidelines**:
- **0.01**: Recommended (current optimized value)
- **0.05**: If TF loss is critically important
- **0.5**: Original value (causes TF loss to dominate)

### KL Divergence Weight (kl_weight)
**Purpose**: Regularization, prevents latent collapse
**Guidelines**:
- **0.05**: Minimal regularization (original)
- **0.1**: Recommended (current optimized value)
- **0.5**: Strong regularization (β-VAE style)
- **1.0**: Very strong (may hurt reconstruction)

### Reconstruction Weight (recon_weight)
**Purpose**: Base loss, always 1.0
**Guidelines**:
- Keep at 1.0 as baseline
- Other weights are relative to this

## Command Line Overrides

You can override any config parameter:

```bash
# Override epochs
python3 scripts/train.py --config configs/optimized.yaml --epochs 50

# Override batch size
python3 scripts/train.py --config configs/test.yaml --batch-size 8

# Override device
python3 scripts/train.py --config configs/optimized.yaml --device cuda
```

## Experiment-Specific Configs

When running an experiment, the config is automatically saved to the experiment directory:

```
experiments/exp003_my_experiment/
└── config.yaml              # Exact config used for this run
```

This ensures reproducibility - you can always see exactly what settings were used.

## Best Practices

1. **Start with test.yaml** for quick verification
2. **Use optimized.yaml** for full training
3. **Copy and modify** rather than editing in-place
4. **Document changes** in experiment README
5. **Save configs** with experiment results

## Troubleshooting

### Loss is NaN
- Reduce learning rate (try 1e-4)
- Increase gradient clipping (try max_grad_norm: 0.5)
- Check for data issues

### Loss not decreasing
- Increase learning rate
- Reduce regularization (lower kl_weight, dropout)
- Check loss component balance

### TF loss dominates
- Reduce tf_weight (current: 0.01)
- Increase other weights

### Poor clustering
- Increase kl_weight for better regularization
- Train longer (200 epochs)
- Increase latent_dim for more capacity

## Version History

- **v1.0**: Original configs (base_config.yaml)
- **v2.0**: Optimized weights (December 2025)
  - tf_weight: 0.5 → 0.01
  - kl_weight: 0.05 → 0.1
  - 53% loss reduction

## Future Configurations

Planned configurations for Phase 2 optimizations:
- `large_model.yaml`: Increased capacity (128 hidden, 32 latent)
- `curriculum.yaml`: Progressive loss weight scheduling
- `beta_vae.yaml`: Strong disentanglement focus
