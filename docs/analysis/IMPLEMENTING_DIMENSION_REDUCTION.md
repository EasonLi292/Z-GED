# Implementing Latent Dimension Reduction

Based on diffusion map analysis recommendations, this guide shows how to implement reduced latent dimensions in the GraphVAE architecture.

## Analysis Results Summary

**Current architecture:** 24D = 3 × 8D branches (z_topo, z_values, z_pz)

**Recommended architectures:**

| Option | z_topo | z_values | z_pz | Total | Reduction |
|--------|--------|----------|------|-------|-----------|
| Current | 8D | 8D | 8D | 24D | - |
| Conservative | 6D | 4D | 6D | 16D | 33% |
| **Moderate (Recommended)** | **6D** | **3D** | **6D** | **15D** | **37.5%** |
| Aggressive | 6D | 2D | 6D | 14D | 42% |

We'll implement **Option C (Moderate)** as the recommended approach.

---

## Step 1: Update Configuration File

Create `configs/optimized_config.yaml`:

```yaml
# GraphVAE Configuration - Optimized Architecture (15D latent space)
# Based on diffusion map analysis showing intrinsic dimension ~6-14D

# Data configuration
data:
  dataset_path: "rlc_dataset/filter_dataset.pkl"
  normalize: true
  log_scale: true
  train_ratio: 0.8
  val_ratio: 0.1
  split_seed: 42

# Model architecture (MODIFIED - reduced latent dimensions)
model:
  # Node and edge features (unchanged)
  node_feature_dim: 4
  edge_feature_dim: 7

  # GNN encoder (unchanged)
  gnn_hidden_dim: 64
  gnn_num_layers: 3

  # Latent space (REDUCED: 24D → 15D)
  latent_dim: 15  # Split into 6D + 3D + 6D [topo, values, pz]

  # Decoder (unchanged)
  decoder_hidden_dim: 128

  # Regularization (unchanged)
  dropout: 0.1

# Loss function (unchanged)
loss:
  recon_weight: 1.0
  tf_weight: 0.01
  kl_weight: 0.1

  # Curriculum learning
  use_topo_curriculum: true
  topo_curriculum_warmup_epochs: 20
  topo_curriculum_initial_multiplier: 3.0

  # Optional GED loss
  use_ged_loss: false
  ged_weight: 0.5
  ged_matrix_path: "rlc_dataset/ged_matrix.npy"

# Training configuration (unchanged)
training:
  optimizer: "adamw"
  learning_rate: 5.0e-4
  weight_decay: 1.0e-5

  scheduler: "cosine"
  min_lr: 1.0e-6

  epochs: 200
  batch_size: 4

  val_interval: 1
  log_interval: 5

  early_stopping_patience: 30

  checkpoint_dir: "checkpoints"

  use_teacher_forcing: true

# Regularization (unchanged)
regularization:
  max_grad_norm: 1.0
  spectral_norm: false

# Hardware (unchanged)
hardware:
  num_workers: 0
  pin_memory: false
```

**Key change:** `latent_dim: 15` (reduced from 24)

---

## Step 2: Update Encoder (Rebalanced Branches)

Modify `ml/models/encoder.py`:

```python
class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder with rebalanced branch dimensions.

    Based on diffusion map analysis:
      - z_topo: 6D (reduced from 8D)
      - z_values: 3D (reduced from 8D)
      - z_pz: 6D (reduced from 8D)
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        latent_dim: int = 15,  # Changed from 24
        dropout: float = 0.1,
        # NEW: Explicit branch dimensions
        topo_latent_dim: int = 6,    # Reduced from 8
        values_latent_dim: int = 3,  # Reduced from 8
        pz_latent_dim: int = 6       # Reduced from 8
    ):
        super().__init__()

        # Validate branch dimensions sum to total
        assert topo_latent_dim + values_latent_dim + pz_latent_dim == latent_dim, \
            f"Branch dims {topo_latent_dim}+{values_latent_dim}+{pz_latent_dim} != {latent_dim}"

        self.latent_dim = latent_dim
        self.topo_latent_dim = topo_latent_dim
        self.values_latent_dim = values_latent_dim
        self.pz_latent_dim = pz_latent_dim

        # ... (GNN layers unchanged) ...

        # Hierarchical latent projections (MODIFIED dimensions)

        # Branch 1: Topology (6D)
        self.mu_topo = nn.Linear(gnn_hidden_dim, topo_latent_dim)
        self.logvar_topo = nn.Linear(gnn_hidden_dim, topo_latent_dim)

        # Branch 2: Component values (3D)
        self.values_aggregator = nn.Sequential(
            nn.Linear(edge_feature_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu_values = nn.Linear(gnn_hidden_dim // 2, values_latent_dim)
        self.logvar_values = nn.Linear(gnn_hidden_dim // 2, values_latent_dim)

        # Branch 3: Poles/zeros (6D)
        self.pz_encoder = nn.Sequential(
            nn.Linear(4, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu_pz = nn.Linear(gnn_hidden_dim // 2, pz_latent_dim)
        self.logvar_pz = nn.Linear(gnn_hidden_dim // 2, pz_latent_dim)

    def forward(self, ...):
        # ... (unchanged until latent concatenation) ...

        # Concatenate: [6D + 3D + 6D] = 15D
        mu = torch.cat([mu_topo, mu_values, mu_pz], dim=1)
        logvar = torch.cat([logvar_topo, logvar_values, logvar_pz], dim=1)

        # ... (rest unchanged) ...
```

**Changes:**
- Added explicit `topo_latent_dim`, `values_latent_dim`, `pz_latent_dim` parameters
- Updated linear layer output dimensions
- Added assertion to validate dimensions sum correctly

---

## Step 3: Update Decoder (Rebalanced Branches)

Modify `ml/models/decoder.py`:

```python
class HybridDecoder(nn.Module):
    """
    Hybrid decoder with rebalanced branch dimensions.
    """

    def __init__(
        self,
        latent_dim: int = 15,  # Changed from 24
        edge_feature_dim: int = 7,
        hidden_dim: int = 128,
        max_nodes: int = 5,
        max_edges: int = 10,
        dropout: float = 0.1,
        # NEW: Explicit branch dimensions
        topo_latent_dim: int = 6,    # Reduced from 8
        values_latent_dim: int = 3,  # Reduced from 8
        pz_latent_dim: int = 6       # Reduced from 8
    ):
        super().__init__()

        # Validate
        assert topo_latent_dim + values_latent_dim + pz_latent_dim == latent_dim

        self.latent_dim = latent_dim
        self.topo_latent_dim = topo_latent_dim
        self.values_latent_dim = values_latent_dim
        self.pz_latent_dim = pz_latent_dim

        # Stage 1: Topology classifier (6D input)
        self.topo_classifier = nn.Sequential(
            nn.Linear(topo_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(FILTER_TYPES))
        )

        # Stage 2: Component value prediction (3D input)
        self.value_mlp1 = nn.Sequential(
            nn.Linear(values_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # ... (FiLM layers unchanged) ...

        # Stage 3: Poles/zeros prediction (6D input)
        self.pole_decoder = nn.Sequential(
            nn.Linear(pz_latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * 2)
        )
        self.zero_decoder = nn.Sequential(
            nn.Linear(pz_latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * 2)
        )

    def forward(self, z, ...):
        # Split latent vector (6D + 3D + 6D)
        z_topo = z[:, :self.topo_latent_dim]
        z_values = z[:, self.topo_latent_dim:self.topo_latent_dim+self.values_latent_dim]
        z_pz = z[:, self.topo_latent_dim+self.values_latent_dim:]

        # ... (rest unchanged) ...
```

**Changes:**
- Added explicit branch dimension parameters
- Updated layer input dimensions
- Modified latent vector slicing to use dynamic dimensions

---

## Step 4: Update Training Script (Optional)

Modify `scripts/train.py` to pass branch dimensions explicitly:

```python
def create_models(config: dict, device: str):
    """Create encoder and decoder models with explicit branch dimensions."""

    # Calculate branch dimensions (or read from config)
    latent_dim = config['model']['latent_dim']

    # Option 1: Hardcode based on analysis
    if latent_dim == 15:
        topo_dim, values_dim, pz_dim = 6, 3, 6
    elif latent_dim == 16:
        topo_dim, values_dim, pz_dim = 6, 4, 6
    elif latent_dim == 24:
        topo_dim, values_dim, pz_dim = 8, 8, 8
    else:
        # Default: equal split
        topo_dim = values_dim = pz_dim = latent_dim // 3

    # Option 2: Read from config (recommended)
    topo_dim = config['model'].get('topo_latent_dim', latent_dim // 3)
    values_dim = config['model'].get('values_latent_dim', latent_dim // 3)
    pz_dim = config['model'].get('pz_latent_dim', latent_dim // 3)

    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=latent_dim,
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,       # NEW
        values_latent_dim=values_dim,   # NEW
        pz_latent_dim=pz_dim            # NEW
    )

    decoder = HybridDecoder(
        latent_dim=latent_dim,
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,       # NEW
        values_latent_dim=values_dim,   # NEW
        pz_latent_dim=pz_dim            # NEW
    )

    # ... (rest unchanged) ...
```

---

## Step 5: Enhanced Config (Explicit Branch Dims)

Better approach: Add branch dimensions to config file:

```yaml
# configs/optimized_config.yaml
model:
  latent_dim: 15

  # Explicit branch dimensions (NEW - optional but recommended)
  topo_latent_dim: 6
  values_latent_dim: 3
  pz_latent_dim: 6
```

---

## Step 6: Train and Compare

### Baseline (24D)

```bash
python scripts/train.py --config configs/curriculum.yaml --epochs 200
```

Results:
- Total params: 111,359
- Training time: ~5.2s/epoch
- Val loss: TBD

### Optimized (15D)

```bash
python scripts/train.py --config configs/optimized_config.yaml --epochs 200
```

Expected results:
- Total params: ~85,000 (24% reduction)
- Training time: ~4.5s/epoch (15% faster)
- Val loss: Similar to baseline (within 5%)

---

## Step 7: Validation Metrics

After training both models, compare:

```bash
# Evaluate baseline
python scripts/evaluate.py --checkpoint checkpoints/baseline_24d/best.pt

# Evaluate optimized
python scripts/evaluate.py --checkpoint checkpoints/optimized_15d/best.pt
```

**Success criteria:**
- Reconstruction MSE: within 10% of baseline
- Topology accuracy: within 5% of baseline
- Transfer function error: within 10% of baseline
- Model size: 20-30% smaller
- Training speed: 10-20% faster

**If quality loss > 10%:** Use Conservative option (16D: 6+4+6)

---

## Parameter Count Comparison

### Encoder

**Baseline (24D):**
```
mu_topo: 64 → 8 = 512 params
mu_values: 32 → 8 = 256 params
mu_pz: 32 → 8 = 256 params
(plus logvar layers: same)
Total encoder: ~55,000 params
```

**Optimized (15D):**
```
mu_topo: 64 → 6 = 384 params (-25%)
mu_values: 32 → 3 = 96 params (-62%)
mu_pz: 32 → 6 = 192 params (-25%)
Total encoder: ~48,000 params (-13%)
```

### Decoder

**Baseline (24D):**
```
topo_classifier: 8 → 128 → 64 → 6
value_mlp1: 8 → 128
pole_decoder: 8 → 64 → 4
Total decoder: ~56,000 params
```

**Optimized (15D):**
```
topo_classifier: 6 → 128 → 64 → 6
value_mlp1: 3 → 128
pole_decoder: 6 → 64 → 4
Total decoder: ~50,000 params (-11%)
```

**Overall:** ~12% parameter reduction

---

## Alternative: Simpler Approach (Equal Split)

If you don't want to modify encoder/decoder code, just reduce total latent_dim:

```yaml
# configs/simple_optimized.yaml
model:
  latent_dim: 15  # Will split as 5+5+5 automatically
```

Pros:
- No code changes needed
- Simple to implement

Cons:
- Doesn't leverage branch-specific analysis
- Less optimal than rebalanced approach

**Recommendation:** Use rebalanced approach for maximum efficiency.

---

## Monitoring During Training

Track branch utilization:

```python
# In trainer.py, add logging
def log_branch_statistics(self, z, mu, logvar):
    """Log per-branch KL divergence and variance."""
    topo_dim = self.encoder.topo_latent_dim
    values_dim = self.encoder.values_latent_dim

    # Split branches
    mu_topo = mu[:, :topo_dim]
    mu_values = mu[:, topo_dim:topo_dim+values_dim]
    mu_pz = mu[:, topo_dim+values_dim:]

    # Compute variance per branch
    var_topo = mu_topo.var(dim=0).mean()
    var_values = mu_values.var(dim=0).mean()
    var_pz = mu_pz.var(dim=0).mean()

    print(f"  Branch variance - topo: {var_topo:.4f}, "
          f"values: {var_values:.4f}, pz: {var_pz:.4f}")
```

Watch for:
- **Low variance in a branch** → Over-parameterized, reduce further
- **High variance in all branches** → All dimensions being used, appropriate

---

## Rollback Plan

If optimized model performs poorly:

1. **Increase z_values from 3D to 4D**
   - Most likely bottleneck
   - Try 6D + 4D + 6D = 16D (Conservative option)

2. **Check training hyperparameters**
   - May need higher learning rate (5e-4 → 7e-4)
   - May need more epochs for convergence

3. **Verify data quality**
   - Re-run diffusion map analysis on converged baseline
   - Ensure intrinsic dimension is actually low

4. **Revert to baseline if necessary**
   - latent_dim: 24
   - Keep all architectural improvements (curriculum, FiLM, teacher forcing)

---

## Expected Timeline

- **Step 1-5:** 1-2 hours (configuration and code changes)
- **Step 6:** 200 epochs × 5s = ~17 minutes per model
- **Step 7:** 10 minutes (evaluation)

**Total:** 1-2 days for complete implementation and validation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Parameter reduction | >20% |
| Training speedup | >10% |
| Reconstruction quality | <10% degradation |
| Topology accuracy | <5% degradation |
| Transfer function error | <10% degradation |

If all targets met → **Successful optimization** ✅

---

## Future Work

After validating 15D architecture:

1. **Further analysis:** Run diffusion map on 15D model
2. **Ablation studies:** Try 14D, 12D, 10D incrementally
3. **Branch analysis:** Study what each dimension encodes
4. **Conditional generation:** Use learned compact representation

This demonstrates **iterative architecture refinement** using spectral analysis.
