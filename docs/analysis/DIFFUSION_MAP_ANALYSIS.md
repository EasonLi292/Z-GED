# Diffusion Map Analysis for Latent Space Dimensionality

## Overview

Diffusion maps provide a powerful spectral method for analyzing the intrinsic geometry of high-dimensional data. By examining the eigenvalue spectrum of a diffusion operator, we can estimate the true dimensionality of the latent space and identify redundant dimensions.

This analysis helps answer critical questions:
- **Is 24D the right latent dimension?** Or are we using more capacity than needed?
- **Which dimensions are meaningful?** Which encode distinct circuit properties?
- **Can we reduce the model?** Would a smaller latent space be more efficient?

## Mathematical Background

### Diffusion Maps (Coifman & Lafon, 2006)

Given latent vectors $\{z_1, \ldots, z_N\} \in \mathbb{R}^D$:

**Step 1: Affinity Matrix (Gaussian kernel)**
$$K_{ij} = \exp\left(-\frac{\|z_i - z_j\|^2}{\epsilon}\right)$$

where $\epsilon$ is the kernel bandwidth (auto-selected as median pairwise distance).

**Step 2: Anisotropic Normalization**
$$K_{ij}^{(\alpha)} = \frac{K_{ij}}{(d_i \cdot d_j)^\alpha}$$

where $d_i = \sum_j K_{ij}$ (degree) and $\alpha \in [0, 1]$ controls normalization strength.

**Step 3: Transition Matrix (Markov chain)**
$$P_{ij} = \frac{K_{ij}^{(\alpha)}}{\sum_k K_{ik}^{(\alpha)}}$$

**Step 4: Eigendecomposition**
$$P \phi_k = \lambda_k \phi_k$$

Eigenvalues: $1 = \lambda_0 \geq \lambda_1 \geq \lambda_2 \geq \ldots \geq 0$

**Step 5: Diffusion Coordinates**
$$\Psi_k(t) = \lambda_k^t \phi_k$$

The diffusion distance at time $t$ is:
$$D_t^2(i, j) = \sum_{k=1}^\infty \lambda_k^{2t} (\phi_k(i) - \phi_k(j))^2$$

### Intrinsic Dimensionality Estimation

The **intrinsic dimension** is the minimum number of dimensions needed to represent the data without significant information loss.

**Three estimation methods:**

1. **Spectral Gap Method**
   - Find largest drop in eigenvalue spectrum
   - $d = \arg\max_k (\lambda_k - \lambda_{k+1})$

2. **Cumulative Variance Method**
   - Find minimum $d$ such that $\sum_{k=1}^d \lambda_k / \sum_k \lambda_k \geq 0.95$
   - Uses 95% variance explained threshold

3. **Elbow Method (Maximum Curvature)**
   - Find point of maximum curvature in eigenvalue decay curve
   - Identifies transition from signal to noise

## Implementation

### Core Classes

**DiffusionMap** (`ml/analysis/diffusion_map.py`)
```python
from ml.analysis import DiffusionMap

dmap = DiffusionMap(epsilon=None, n_components=10, alpha=0.5)
dmap.fit(latent_vectors)

# Get diffusion coordinates
coords = dmap.transform(t=1)

# Access eigenvalues/eigenvectors
eigenvalues = dmap.eigenvalues_
eigenvectors = dmap.eigenvectors_
```

Parameters:
- `epsilon`: Kernel bandwidth (default: auto-select using median distance heuristic)
- `n_components`: Number of eigenvectors to compute (default: 10)
- `alpha`: Normalization strength (0=no normalization, 1=Laplace-Beltrami, default: 0.5)

### Main Analysis Function

**estimate_intrinsic_dimension()**
```python
from ml.analysis import estimate_intrinsic_dimension

results = estimate_intrinsic_dimension(
    latent_vectors,       # [N, D] numpy array
    n_components=24,      # Analyze top 24 components
    epsilon=None,         # Auto-select bandwidth
    plot=True,            # Generate eigenspectrum plots
    save_path='results/eigenspectrum.png'
)

# Results
print(f"Intrinsic dimension: {results['recommended_dim']}D")
print(f"Spectral gap: {results['intrinsic_dim_gap']}D")
print(f"95% variance: {results['intrinsic_dim_variance']}D")
print(f"Elbow: {results['intrinsic_dim_elbow']}D")
```

### Visualization

**visualize_diffusion_coordinates()**
```python
from ml.analysis import visualize_diffusion_coordinates

# 2D projection
visualize_diffusion_coordinates(
    latent_vectors,
    labels=filter_types,
    label_names=['Low-pass', 'High-pass', ...],
    n_components=2,
    save_path='diffusion_2d.png'
)

# 3D projection
visualize_diffusion_coordinates(
    latent_vectors,
    labels=filter_types,
    n_components=3,
    save_path='diffusion_3d.png'
)
```

## Usage

### Running the Analysis Script

**Basic usage:**
```bash
python scripts/analyze_latent_space.py --checkpoint checkpoints/best.pt
```

**Full options:**
```bash
python scripts/analyze_latent_space.py \
    --checkpoint checkpoints/20241220_153000/best.pt \
    --dataset rlc_dataset/filter_dataset.pkl \
    --device mps \
    --n-components 24 \
    --output-dir analysis_results \
    --batch-size 32
```

### Output Files

The script generates comprehensive analysis in `analysis_results/<timestamp>/`:

1. **`eigenspectrum.png`** - Full latent space eigenvalue analysis
   - Eigenvalue decay (linear and log scale)
   - Cumulative variance explained
   - Spectral gaps

2. **`eigenspectrum_branch_0.png`** - z_topo (topology) branch analysis
3. **`eigenspectrum_branch_1.png`** - z_values (component values) branch analysis
4. **`eigenspectrum_branch_2.png`** - z_pz (poles/zeros) branch analysis

5. **`diffusion_coords_2d.png`** - 2D diffusion map embedding (colored by filter type)
6. **`diffusion_coords_3d.png`** - 3D diffusion map embedding

7. **`analysis_summary.yaml`** - Numerical results
   ```yaml
   full_latent_space:
     intrinsic_dim_gap: 8
     intrinsic_dim_variance: 12
     intrinsic_dim_elbow: 10
     recommended_dim: 8
     epsilon: 2.3456

   branches:
     z_topo (topology):
       intrinsic_dim_gap: 3
       recommended_dim: 3
     z_values (component values):
       intrinsic_dim_gap: 4
       recommended_dim: 4
     z_pz (poles/zeros):
       intrinsic_dim_gap: 2
       recommended_dim: 2
   ```

## Interpreting Results

### Eigenvalue Spectrum

**Key indicators:**

1. **Sharp drop (spectral gap)**
   - Indicates clear separation between signal and noise
   - Dimensions after the gap are redundant

   Example: λ₁=0.95, λ₂=0.88, λ₃=0.82, **λ₄=0.35**, λ₅=0.31
   → Intrinsic dimension ≈ 3

2. **Gradual decay**
   - Suggests no clear intrinsic dimensionality
   - All dimensions contribute incrementally
   - Current dimension is likely appropriate

3. **Plateau (all eigenvalues similar)**
   - Random/unstructured latent space
   - Poor learning - model hasn't discovered meaningful structure

### Cumulative Variance

```
Component    Eigenvalue    Cumulative Variance
    1          0.95            47.5%
    2          0.88            91.5%
    3          0.82            95.0%  ← 95% threshold
    4          0.35            95.8%
    ...
```

If 95% variance is captured by $d$ dimensions, we can reduce to $d$ without significant loss.

### Branch-Specific Analysis

**Hierarchical latent space (24D = 3 × 8D):**
- **z_topo (topology)**: Should capture discrete filter types (6 classes)
  - Expected intrinsic dimension: 3-4D (enough for 6-class separation)

- **z_values (component values)**: Continuous scales (R, L, C magnitudes)
  - Expected intrinsic dimension: 4-6D (impedance scales, Q factors)

- **z_pz (poles/zeros)**: Transfer function characteristics
  - Expected intrinsic dimension: 2-4D (frequency scales, damping)

**If branch intrinsic dimension < 8D:** That branch is over-parameterized.

## Example Scenarios

### Scenario 1: Over-Parameterized Latent Space

```
Current latent space: 24D
Estimated intrinsic dimension: 12D
Potential reduction: 50%

Recommendation:
  Reduce to 12D latent space (3 × 4D branches)
  This will:
    - Reduce model size by ~30%
    - Speed up training
    - Improve generalization (less overfitting)
```

**Action:** Retrain with `latent_dim: 12` in config.

### Scenario 2: Well-Matched Dimension

```
Current latent space: 24D
Estimated intrinsic dimension: 22D

Recommendation:
  Current dimension (24D) is well-matched to data complexity.
  No reduction recommended.
```

**Action:** Keep current architecture.

### Scenario 3: Under-Parameterized (Rare)

```
Current latent space: 24D
Estimated intrinsic dimension: 24D (all eigenvalues significant)

Interpretation:
  Data complexity may exceed current latent capacity.
  Consider increasing to 32D if reconstruction quality is poor.
```

**Action:** Increase `latent_dim` or investigate data complexity.

### Scenario 4: Branch Imbalance

```
z_topo (8D) → Intrinsic: 3D (over-parameterized)
z_values (8D) → Intrinsic: 7D (well-matched)
z_pz (8D) → Intrinsic: 2D (over-parameterized)

Recommendation:
  Redistribute capacity:
    z_topo: 4D
    z_values: 8D
    z_pz: 4D
  Total: 16D (reduced from 24D)
```

**Action:** Update encoder/decoder branch dimensions in code.

## Integration with GraphVAE

### Step 1: Analyze Trained Model

After completing 200-epoch training run:
```bash
python scripts/train.py --config configs/curriculum.yaml --epochs 200
python scripts/analyze_latent_space.py --checkpoint checkpoints/<timestamp>/best.pt
```

### Step 2: Review Analysis

Examine generated plots and `analysis_summary.yaml`.

### Step 3: Adjust Architecture (If Needed)

If intrinsic dimension < 24D, update `configs/base_config.yaml`:
```yaml
model:
  latent_dim: 16  # Reduced from 24
```

And update encoder/decoder code if branch rebalancing is needed:
```python
# ml/models/encoder.py
self.mu_topo = nn.Linear(hidden_dim, 4)  # Reduced from 8
self.mu_values = nn.Linear(hidden_dim, 8)  # Keep at 8
self.mu_pz = nn.Linear(hidden_dim, 4)  # Reduced from 8
```

### Step 4: Retrain and Compare

Train reduced model and compare:
- Reconstruction quality (MSE)
- Topology accuracy
- Transfer function error
- Training speed
- Model size

## Theoretical Insights

### Why Diffusion Maps?

**Advantages over PCA:**
- **Nonlinear:** Captures curved manifolds (circuits lie on nonlinear constraint surfaces)
- **Metric-aware:** Preserves geodesic distances on the manifold
- **Robust:** Less sensitive to outliers

**Advantages over t-SNE/UMAP:**
- **Intrinsic dimensionality:** Eigenvalues directly estimate dimension
- **Deterministic:** No random initialization
- **Interpretable:** Eigenvalues = "importance" of each dimension

### Connection to Graph Laplacian

Diffusion maps compute eigenvectors of a graph Laplacian:
$$\mathcal{L} = I - P$$

where $P$ is the transition matrix. This reveals:
- **Clustering structure:** Eigenvector values identify clusters
- **Connectivity:** How circuits are connected in latent space
- **Smoothness:** Which properties vary smoothly vs. discretely

### Application to Circuit VAE

**Expected structure:**
1. **Discrete clusters** (6 filter types)
   - Should see 5-6 significant eigenvalues (gaps between clusters)

2. **Continuous variations** (component values, frequency scales)
   - Should see gradual eigenvalue decay

3. **Redundancy** (over-parameterization)
   - Eigenvalues → 0 indicate redundant dimensions

## Advanced Usage

### Custom Kernel Bandwidth

Auto-selection uses median heuristic, but you can manually tune:
```python
# Smaller epsilon → local structure (fine details)
results_local = estimate_intrinsic_dimension(latent_vectors, epsilon=0.5)

# Larger epsilon → global structure (clusters)
results_global = estimate_intrinsic_dimension(latent_vectors, epsilon=5.0)
```

### Multi-Scale Analysis

Analyze at different diffusion times:
```python
dmap = DiffusionMap(n_components=10)
dmap.fit(latent_vectors)

# Short time: local geometry
coords_t1 = dmap.transform(t=1)

# Long time: global geometry
coords_t10 = dmap.transform(t=10)
```

### Comparison with GED

Compare diffusion distance with GED (Graph Edit Distance):
```python
from scipy.spatial.distance import pdist, squareform

# Diffusion distance
dmap = DiffusionMap()
dmap.fit(latent_vectors)
diff_dist = squareform(pdist(dmap.transform(t=1)))

# Load GED matrix
ged_matrix = np.load('rlc_dataset/ged_matrix.npy')

# Correlation
from scipy.stats import pearsonr
correlation, p_value = pearsonr(diff_dist.flatten(), ged_matrix.flatten())
print(f"Diffusion distance vs GED correlation: {correlation:.3f}")
```

High correlation → latent space captures circuit structure well.

## References

1. **Coifman, R. R., & Lafon, S. (2006).** "Diffusion maps." *Applied and computational harmonic analysis*, 21(1), 5-30.
   - Original diffusion maps paper

2. **Nadler, B., et al. (2006).** "Diffusion maps, spectral clustering and reaction coordinates of dynamical systems." *Applied and Computational Harmonic Analysis*, 21(1), 113-127.
   - Applications to dimensionality estimation

3. **Moon, K. R., et al. (2019).** "Visualizing structure and transitions in high-dimensional biological data." *Nature biotechnology*, 37(12), 1482-1492.
   - Modern applications (PHATE algorithm extends diffusion maps)

## FAQ

**Q: What if all three methods disagree?**
A: Use spectral gap as primary estimate. Check eigenspectrum plot visually - clear gap = reliable estimate.

**Q: Should I always reduce if intrinsic dimension < current dimension?**
A: Not necessarily. Small reductions (<25%) may not be worth retraining. Focus on large redundancies (>40%).

**Q: Can I use this during training?**
A: Yes! Compute every 50 epochs to track how intrinsic dimension evolves. Should stabilize by convergence.

**Q: What if intrinsic dimension varies across train/val/test?**
A: Indicates distribution shift. Model may not generalize well. Check for overfitting.

**Q: How does this relate to β-VAE disentanglement?**
A: Orthogonal concerns. Diffusion maps find total dimensions needed; β-VAE encourages independence. Both are useful.

## Next Steps

After completing diffusion map analysis:

1. **Document findings** in experiment logs
2. **Adjust architecture** if significant redundancy found
3. **Retrain and compare** reduced vs. full model
4. **Analyze learned representations** - which dimensions encode which properties?
5. **Generate circuits** from reduced latent space - does it maintain quality?

This analysis provides rigorous justification for architectural decisions, moving from intuition ("24D seems reasonable") to data-driven design ("intrinsic dimension is 12D, reduce by 50%").
