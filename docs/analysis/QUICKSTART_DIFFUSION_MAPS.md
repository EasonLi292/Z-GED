# Quick Start: Diffusion Map Analysis

## TL;DR

Analyze your trained GraphVAE to determine if 24D is the right latent dimension:

```bash
# After training
python scripts/analyze_latent_space.py --checkpoint checkpoints/best.pt
```

This will tell you:
- **Intrinsic dimension**: The true dimensionality of your latent space
- **Recommended changes**: Whether to reduce/keep/increase latent dimension
- **Branch analysis**: Which of the 3 branches (topology, values, poles/zeros) are over/under-parameterized

## Why Use This?

You're currently using a **24D latent space** (3 × 8D branches). But is that optimal?

**Too high?** → Wasted computation, overfitting, slow training
**Too low?** → Information bottleneck, poor reconstruction
**Just right?** → Efficient representation

Diffusion maps analyze the eigenvalue spectrum of your latent space to determine its **intrinsic dimensionality** - the minimum dimensions needed without information loss.

## How It Works

1. **Encode all circuits** to latent space (get z vectors)
2. **Compute diffusion map** (spectral decomposition of distance matrix)
3. **Analyze eigenvalues** to find "spectral gap" (signal vs noise)
4. **Estimate dimension** using 3 methods:
   - Spectral gap (largest eigenvalue drop)
   - 95% variance explained
   - Elbow method (maximum curvature)

## Example Output

```
======================================================================
INTRINSIC DIMENSION ESTIMATES:
======================================================================
  Spectral gap method:     12 dimensions
  Variance method (95%):   14 dimensions
  Elbow method:            10 dimensions

  Recommended:             12 dimensions
----------------------------------------------------------------------

Interpretation:
  Current latent space: 24D
  Intrinsic dimension:  ~12D
  Potential reduction:  50.0%

  The latent space has ~12 redundant dimensions.
  Consider reducing to 12D for more efficient representation.
```

**What this means:**
- Your model is using **twice as many dimensions as needed**
- You can reduce from **24D → 12D** without losing information
- This would **cut model size by ~30%** and speed up training

## Branch-Specific Analysis

The script also analyzes each 8D branch separately:

```
Branch-specific analysis:
  z_topo (topology): 8D → 4D (reduce by 4)
  z_values (component values): 8D → 6D (reduce by 2)
  z_pz (poles/zeros): 8D → 2D (reduce by 6)
```

**Interpretation:**
- **z_topo**: Only needs 4D to encode 6 filter types (makes sense!)
- **z_values**: Needs 6D for component value distributions
- **z_pz**: Only needs 2D for poles/zeros (most redundant branch)

**Recommended architecture:** 4D + 6D + 2D = 12D total

## When to Use This

**During development:**
- After first successful training run (200 epochs)
- Before scaling up to larger datasets
- When debugging poor reconstruction quality

**During research:**
- Investigating learned representations
- Understanding what each latent dimension encodes
- Justifying architectural choices in papers

## Typical Workflow

1. **Train baseline model** (24D latent space)
   ```bash
   python scripts/train.py --config configs/curriculum.yaml --epochs 200
   ```

2. **Analyze latent space**
   ```bash
   python scripts/analyze_latent_space.py --checkpoint checkpoints/best.pt
   ```

3. **Review results** in `analysis_results/<timestamp>/`
   - `eigenspectrum.png` - Check for clear spectral gap
   - `analysis_summary.yaml` - Read recommended dimension
   - `diffusion_coords_2d.png` - Verify filter types cluster well

4. **Adjust architecture** (if needed)
   ```yaml
   # configs/optimized_config.yaml
   model:
     latent_dim: 12  # Reduced from 24
   ```

5. **Retrain and compare**
   ```bash
   python scripts/train.py --config configs/optimized_config.yaml --epochs 200
   ```

6. **Verify no quality loss**
   - Compare reconstruction MSE
   - Compare topology accuracy
   - Compare transfer function error

## Common Results

### Scenario A: Over-parameterized (Most Common)
```
Current: 24D → Intrinsic: 12D
Action: Reduce to 12D, retrain, verify quality maintained
```

### Scenario B: Well-matched
```
Current: 24D → Intrinsic: 22D
Action: Keep 24D, architecture is appropriate
```

### Scenario C: Under-parameterized (Rare)
```
Current: 24D → Intrinsic: 24D (all eigenvalues significant)
Action: Increase to 32D if reconstruction quality is poor
```

### Scenario D: Unstructured (Problem!)
```
All eigenvalues similar (no spectral gap)
Action: Model hasn't learned meaningful structure
        - Increase training epochs
        - Check loss function weights
        - Verify data quality
```

## Interpreting Plots

### eigenspectrum.png

**Top-left: Eigenvalue decay**
- Look for a **sharp drop** (spectral gap)
- Point before the drop = intrinsic dimension

```
  λ₁ ████████████████████ 0.95  ┐
  λ₂ █████████████████    0.88  │ Signal
  λ₃ ████████████         0.82  ┘
  λ₄ ████                 0.35  ┐
  λ₅ ███                  0.32  │ Noise
  λ₆ ███                  0.28  ┘
       ↑ spectral gap (intrinsic dim = 3)
```

**Bottom-left: Cumulative variance**
- Shows how much information is captured
- 95% line = recommended minimum dimension

**Bottom-right: Spectral gaps**
- Peak shows largest eigenvalue drop
- Confirms intrinsic dimensionality

### diffusion_coords_2d.png / 3d.png

- **Well-separated clusters** → Model learned filter type structure ✅
- **Overlapping points** → Confusion between filter types ⚠️
- **Random scatter** → No learned structure ❌

Color-coded by filter type:
- Should see 6 distinct clusters (low-pass, high-pass, etc.)
- Smooth transitions within clusters (component value variations)

## Quick Decisions

**If intrinsic dimension < 0.75 × current dimension:**
→ Reduce latent space (significant redundancy)

**If 0.75 × current < intrinsic < 1.0 × current:**
→ Keep current dimension (reasonable match)

**If intrinsic dimension ≥ current dimension:**
→ Consider increasing (may be bottlenecked)

## Integration with Paper/Thesis

This analysis provides **quantitative justification** for architectural choices:

> "We initially designed a 24-dimensional latent space with three 8D branches.
> However, diffusion map analysis revealed an intrinsic dimensionality of only
> 12D (spectral gap at λ₄ = 0.35, 95% variance at 14D). We therefore reduced
> the architecture to 12D (4D topology + 6D values + 2D poles/zeros), which
> maintained reconstruction quality (MSE: 0.045 vs 0.043) while reducing
> parameters by 28% and training time by 15%."

## Further Reading

See `docs/analysis/DIFFUSION_MAP_ANALYSIS.md` for:
- Mathematical background (Coifman & Lafon, 2006)
- Detailed interpretation guidelines
- Advanced usage (custom kernels, multi-scale analysis)
- Integration with GED metric learning

## Questions?

**Q: How long does analysis take?**
A: ~30 seconds for 120 circuits (encoding + eigendecomposition)

**Q: Can I run this during training?**
A: Yes! Compute every 50 epochs to track dimensionality evolution

**Q: What if methods disagree?**
A: Use spectral gap as primary. Check eigenspectrum plot visually.

**Q: Should I always follow the recommendation?**
A: No. Small reductions (<25%) may not justify retraining. Focus on large gaps (>40%).

**Q: Does this work for other VAE architectures?**
A: Yes! Any latent space can be analyzed this way.
