# Latent Space Organization - GraphVAE

## Overview

The GraphVAE uses a **24-dimensional hierarchical latent space** split into 3 independent branches of 8D each. This design separates different types of circuit information into distinct subspaces.

```
Latent Vector z ∈ ℝ²⁴ = [z_topo(8D) | z_values(8D) | z_pz(8D)]
                         ├─────────┼──────────┼──────────┤
                         Topology  Component  Poles/Zeros
                                   Values
```

## Three Branches

### 1. Topology Branch (z_topo) - Dimensions 0-7

**What it encodes**: Discrete circuit structure (which filter type)

**Input source**: Graph structure via GNN
- Node embeddings from 3-layer GNN
- Global pooling (mean + max concatenation)
- MLP encoder

**Processing flow**:
```
Graph (nodes, edges)
  → GNN(x, edge_index, edge_attr)
  → h_nodes [N, 64]
  → GlobalPooling(h_nodes, batch)
  → h_topo [B, 128]  (mean + max)
  → MLP encoder
  → [B, 32]
  → mu_topo [B, 8], logvar_topo [B, 8]
  → z_topo = mu_topo + exp(0.5*logvar_topo) * ε
```

**What it should learn**:
- Distinguishes between 6 filter types (low-pass, high-pass, band-pass, band-stop, RLC series, RLC parallel)
- Captures topological features like number of nodes, connectivity patterns
- Should cluster by filter type in latent space

**Observed behavior** (from evaluation):
- ✅ Perfect cluster purity (100%) - all circuits of same type cluster together
- ✅ High silhouette score (0.62) - good separation between filter types
- This branch is the strongest discriminator

### 2. Component Values Branch (z_values) - Dimensions 8-15

**What it encodes**: Continuous component magnitudes (C, G, L values)

**Input source**: Edge features (impedance values)
- Direct aggregation of edge attributes
- Mean pooling per graph

**Processing flow**:
```
Edge features [E, 3]  (log-scale [C, G, L_inv])
  → Per-graph aggregation
  → edge_attr[edge_batch == i].mean(dim=0)
  → h_values [B, 3]
  → MLP encoder
  → [B, 32]
  → mu_values [B, 8], logvar_values [B, 8]
  → z_values = mu_values + exp(0.5*logvar_values) * ε
```

**What it should learn**:
- Component value scales (high-R vs low-R circuits)
- Impedance ratios
- Frequency scaling (since f ∝ 1/√(LC))

**Observed behavior** (from training):
- Reconstruction loss (edge): ~0.9 after 2 epochs
- Learning slower than topology branch
- Needs more epochs to capture value variations

### 3. Poles/Zeros Branch (z_pz) - Dimensions 16-23

**What it encodes**: Transfer function characteristics

**Input source**: Poles and zeros (complex numbers)
- DeepSets architecture for permutation invariance
- Separate encoders for poles and zeros

**Processing flow**:
```
Poles [num_poles, 2], Zeros [num_zeros, 2]  (real, imag pairs)
  → DeepSets encoder (permutation-invariant)
  → poles: MLP([2] → [16]) → sum/mean → h_poles [16]
  → zeros: MLP([2] → [16]) → sum/mean → h_zeros [16]
  → h_pz = concat(h_poles, h_zeros) [B, 32]
  → MLP combine
  → [B, 32]
  → mu_pz [B, 8], logvar_pz [B, 8]
  → z_pz = mu_pz + exp(0.5*logvar_pz) * ε
```

**What it should learn**:
- Pole/zero locations in complex plane
- Filter order (number of poles/zeros)
- Frequency characteristics (pole magnitudes ≈ cutoff frequency)
- Damping/Q factor (pole imaginary parts)

**Observed behavior** (from training):
- Transfer function loss: ~6-7 (Chamfer distance on normalized poles/zeros)
- High variance - harder to learn than topology
- Most challenging branch (variable-length inputs, complex-valued)

## Hierarchical Combination

All three branches are combined via **concatenation** (not addition):

```python
mu = torch.cat([mu_topo, mu_values, mu_pz], dim=-1)        # [B, 24]
logvar = torch.cat([logvar_topo, logvar_values, logvar_pz], dim=-1)  # [B, 24]

# Reparameterization trick
z = mu + exp(0.5 * logvar) * ε, where ε ~ N(0, I)
```

**Why concatenation?**
- Each branch learns independently
- No interference between discrete topology and continuous values
- Enables controlled generation (modify only one branch)

## Variational Properties

Each branch has its own **mean (μ) and log-variance (log σ²)**:

```
Prior: p(z) = N(0, I)
Posterior: q(z|x) = N(μ, diag(exp(logvar)))
KL divergence: KL[q(z|x) || p(z)] = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
```

**Regularization**: KL divergence encourages:
- μ → 0 (latent means centered)
- σ → 1 (unit variance)
- Prevents collapse (all circuits mapped to same point)

**Current KL behavior** (after optimization):
- KL loss: ~0.05-0.1 (low but non-zero)
- Weight: 0.1 (increased from 0.05 for better regularization)

## Decoder Structure

The decoder mirrors the hierarchical structure:

```
z [B, 24] → split into [z_topo | z_values | z_pz]

1. Topology decoder:
   z_topo [B, 8] → MLP → logits [B, 6]
   → Gumbel-Softmax → one-hot filter type

2. Values decoder:
   z_values [B, 8] → MLP → edge_features [B, max_edges, 3]
   → Predicted component values (log-scale)

3. Poles/Zeros decoder:
   z_pz [B, 8] → MLP_poles → poles [B, 2, 2]
                → MLP_zeros → zeros [B, 2, 2]
   → Predicted transfer function
```

**Current limitation**: Decoder outputs **fixed** 2 poles and 2 zeros
- Many circuits have only 1 pole (first-order filters)
- Model must learn to output near-zero for unused poles
- Future: Variable-length output or filter-type-specific decoders

## Visualization from Evaluation

### Hierarchical Structure (3-branch PCA)

From `evaluation_results/test_run/hierarchical_structure.png`:
- Each 8D branch projected to 2D via PCA
- **z_topo**: Shows strong clustering by filter type (drives overall clustering)
- **z_values**: More diffuse, captures component value variations
- **z_pz**: Intermediate clustering, learning transfer function features

### t-SNE and PCA Projections

From `evaluation_results/test_run/latent_tsne_pca.png`:
- All 24 dimensions projected to 2D
- Clear separation of 6 filter types
- Silhouette score: 0.62 (good clustering)

### Latent Dimension Distributions

From `evaluation_results/test_run/latent_dimensions.png`:
- First 8 dimensions (z_topo): Show distinct modes for each filter type
- Dimensions 8-15 (z_values): More continuous, overlapping distributions
- Dimensions 16-23 (z_pz): Wide distributions, high variance

## Interpretability

### Which dimensions encode what?

**Hypothesis** (needs verification with longer training):
- **Dimensions 0-2**: Primary filter type discriminator (low/high/band)
- **Dimensions 3-7**: Secondary topology features (order, complexity)
- **Dimensions 8-11**: Component magnitude scales
- **Dimensions 12-15**: Impedance ratios (R/L, R/C)
- **Dimensions 16-19**: Pole characteristics (frequency, damping)
- **Dimensions 20-23**: Zero characteristics

**How to verify**:
1. Train for 50+ epochs
2. Compute mutual information between each dimension and known properties
3. Ablation: Zero out specific dimensions, observe reconstruction
4. Linear probe: Train classifier on each dimension → filter type

## Controlled Generation

The hierarchical structure enables **disentangled control**:

### Example 1: Change filter type, keep values
```python
z = encoder(circuit)
z_topo_new = encoder(different_filter_type_circuit)[:, :8]
z_modified = torch.cat([z_topo_new, z[:, 8:16], z[:, 16:24]], dim=-1)
new_circuit = decoder(z_modified)
# Result: Same component values, different topology
```

### Example 2: Scale all component values
```python
z = encoder(circuit)
z_values_scaled = z[:, 8:16] * 2.0  # Shift in latent space
z_modified = torch.cat([z[:, :8], z_values_scaled, z[:, 16:24]], dim=-1)
new_circuit = decoder(z_modified)
# Result: Same topology, larger component values → lower cutoff frequency
```

### Example 3: Interpolate between two circuits
```python
z1, z2 = encoder(circuit1), encoder(circuit2)
# Interpolate only topology
z_interp = torch.cat([
    0.5 * z1[:, :8] + 0.5 * z2[:, :8],  # Average topology
    z1[:, 8:16],                          # Keep values from circuit1
    z1[:, 16:24]                          # Keep poles/zeros from circuit1
], dim=-1)
intermediate = decoder(z_interp)
```

## Design Rationale

### Why 8D per branch?

**Trade-off**:
- Too small (e.g., 4D): Not enough capacity to encode all variations
- Too large (e.g., 16D): Overfitting, redundant dimensions
- **8D chosen**: Reasonable capacity, small dataset (120 circuits)

**Future**: With 1000+ circuits, could increase to 10D or 12D per branch

### Why separate branches?

**Advantages**:
1. **Disentanglement**: Different information types don't interfere
2. **Interpretability**: Can analyze each branch independently
3. **Controlled generation**: Modify one aspect while keeping others fixed
4. **Modular training**: Can freeze/unfreeze branches selectively

**Disadvantages**:
1. **No cross-branch interactions**: Topology and values might be correlated
2. **More parameters**: 3 separate encoders instead of 1
3. **Harder to train**: Must balance all 3 branches

### Alternative architectures considered

**Single 24D latent vector** (no hierarchy):
- Simpler, fewer parameters
- But: entangled representations, harder to interpret
- Rejected: Disentanglement is a core research goal

**Larger hierarchy** (4 or 5 branches):
- Could separate: topology, R values, C values, L values, transfer function
- But: Too fine-grained for small dataset
- Rejected: Risk of overfitting

**Variable-size branches** (e.g., 12D topo, 8D values, 4D pz):
- Could allocate more capacity where needed
- But: Hard to decide sizes a priori
- Considered for future: Based on reconstruction losses

## Current Performance (After 2 Epochs)

| Branch | Loss Component | Value | Status |
|--------|---------------|-------|--------|
| z_topo | Topology cross-entropy | 1.79 | 🟡 Learning (16% accuracy) |
| z_values | Edge MSE | 0.90 | 🟡 Learning |
| z_pz | Pole/zero Chamfer | 6.60 | 🔴 High (but weighted low) |
| Overall | KL divergence | 0.05 | 🟢 Good regularization |
| - | Cluster purity | 100% | 🟢 Perfect separation |
| - | Silhouette score | 0.62 | 🟢 Good clustering |

**Interpretation**:
- **z_topo**: Strongest branch, already clustering perfectly
- **z_values**: Moderate, needs more training
- **z_pz**: Weakest, high loss but contributes little to total (weight 0.01)

## Future Improvements

### 1. Learnable Branch Weighting
```python
log_var_topo = nn.Parameter(torch.zeros(1))
log_var_values = nn.Parameter(torch.zeros(1))
log_var_pz = nn.Parameter(torch.zeros(1))

# Uncertainty-weighted loss
loss = (
    exp(-log_var_topo) * loss_topo + log_var_topo +
    exp(-log_var_values) * loss_values + log_var_values +
    exp(-log_var_pz) * loss_pz + log_var_pz
)
```

### 2. β-VAE for Better Disentanglement
```python
# Increase KL weight for specific branches
kl_topo = compute_kl(mu_topo, logvar_topo)
kl_values = compute_kl(mu_values, logvar_values)
kl_pz = compute_kl(mu_pz, logvar_pz)

loss_kl = β_topo * kl_topo + β_values * kl_values + β_pz * kl_pz
```

### 3. Attention Between Branches
```python
# Allow limited cross-branch communication
h_combined = torch.cat([h_topo, h_values, h_pz], dim=-1)
h_attended = CrossAttention(h_combined)
mu = separate_heads(h_attended)  # Still output 3 separate mus
```

### 4. Variable-Length Pole/Zero Prediction
```python
# Predict number + values
n_poles = predict_count(z_pz)  # [B, 1]
poles = predict_poles(z_pz, max_poles=4)  # [B, 4, 2]
pole_valid = n_poles_sigmoid(z_pz)  # [B, 4]
```

## Summary

The 24D hierarchical latent space is a **key architectural innovation** that:
- ✅ Separates topology, values, and transfer function into independent subspaces
- ✅ Enables disentangled representation learning
- ✅ Supports controlled generation and interpolation
- ✅ Achieves strong clustering (100% purity, 0.62 silhouette)
- 🔄 Still learning after only 2 epochs (needs 50-200 epochs for full performance)

The design reflects the core research goal: **discover intrinsic circuit properties** through learned representations, not just achieve high reconstruction accuracy.
