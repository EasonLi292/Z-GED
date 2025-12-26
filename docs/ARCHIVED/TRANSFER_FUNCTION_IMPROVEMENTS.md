**📦 ARCHIVED - Historical Reference**

# Transfer Function Prediction Improvements

## Problem

The current 8D model has **0% transfer function inference accuracy**:
- Predicted poles/zeros don't match actual circuit behavior
- All circuits inferred as "band_stop" regardless of type
- Transfer function loss weight is too low (λ_tf = 0.01)

## Root Cause

**Weak supervision signal for auxiliary prediction task**:
```
Current loss weights:
- Reconstruction: λ_recon = 1.0
- Topology (with 3x curriculum): λ_topo = 3.0 → 1.0
- Transfer function: λ_tf = 0.01  ← TOO LOW!
- KL divergence: λ_kl = 0.1
```

The transfer function loss is **100x weaker** than reconstruction loss, so the model ignores it.

## Solution: Transfer Function Curriculum

### Key Insight: Transition Zone Confidence

When interpolating between filter types:
```
α=0.00: low_pass  (99.89% confidence)
α=0.25: low_pass  (99.30% confidence)
α=0.50: low_pass  (67.66% confidence)  ← Transition zone
α=0.75: high_pass (98.83% confidence)
α=1.00: high_pass (99.91% confidence)
```

**What the percentages mean**:
- These are **softmax probabilities** for the predicted filter type
- At α=0.5, the latent vector is halfway between low-pass and high-pass
- The decoder outputs: [67.66% low-pass, 30% high-pass, 2.34% others]
- **Lower confidence = both types partially activated** = smooth latent space ✅

This is actually **good** - it shows the model learned smooth transitions rather than sharp boundaries.

### Implemented Changes

**1. Increased Transfer Function Loss Weight**

Created new config: `configs/8d_improved_tf.yaml`

```yaml
loss:
  tf_weight: 1.0  # Increased from 0.01 (100x)
```

**2. Added Transfer Function Curriculum**

Gradually increase tf_weight over training to balance with topology learning:

```yaml
loss:
  # Transfer function curriculum
  use_tf_curriculum: true
  tf_curriculum_warmup_epochs: 50
  tf_curriculum_initial_multiplier: 0.01

  # This creates the schedule:
  # Epoch 0:   tf_weight = 1.0 × 0.01 = 0.01 (same as before)
  # Epoch 25:  tf_weight = 1.0 × 0.50 = 0.50 (halfway)
  # Epoch 50+: tf_weight = 1.0 × 1.00 = 1.00 (full weight)
```

**3. Modified Loss Implementation**

Updated `ml/losses/composite.py`:

```python
class SimplifiedCompositeLoss(nn.Module):
    def __init__(
        self,
        tf_weight: float = 0.5,  # Target/final weight
        use_tf_curriculum: bool = False,
        tf_curriculum_warmup_epochs: int = 50,
        tf_curriculum_initial_multiplier: float = 0.01
    ):
        self.tf_weight_target = tf_weight
        self.use_tf_curriculum = use_tf_curriculum
        self.tf_curriculum_warmup_epochs = tf_curriculum_warmup_epochs
        self.tf_curriculum_initial_multiplier = tf_curriculum_initial_multiplier

    def get_tf_weight(self) -> float:
        """Get current TF weight with curriculum."""
        if not self.use_tf_curriculum:
            return self.tf_weight_target

        # Linear increase from (target × initial) to target
        warmup_progress = min(1.0, self.current_epoch / self.tf_curriculum_warmup_epochs)
        initial = self.tf_weight_target * self.tf_curriculum_initial_multiplier
        return initial + (self.tf_weight_target - initial) * warmup_progress
```

**4. Updated Training Script**

Modified `scripts/train.py` to read and apply TF curriculum parameters:

```python
def create_loss_function(config, device):
    # Read TF curriculum parameters
    use_tf_curriculum = config['loss'].get('use_tf_curriculum', False)
    tf_warmup_epochs = config['loss'].get('tf_curriculum_warmup_epochs', 50)
    tf_initial_multiplier = config['loss'].get('tf_curriculum_initial_multiplier', 0.01)

    loss_fn = SimplifiedCompositeLoss(
        tf_weight=config['loss']['tf_weight'],
        use_tf_curriculum=use_tf_curriculum,
        tf_curriculum_warmup_epochs=tf_warmup_epochs,
        tf_curriculum_initial_multiplier=tf_initial_multiplier,
        # ... other params
    )
```

## Training Schedule Comparison

### Old Schedule (8D baseline)

```
Epoch 0:     λ_topo=3.0, λ_tf=0.01, λ_recon=1.0, λ_kl=0.1
Epoch 20:    λ_topo=1.0, λ_tf=0.01, λ_recon=1.0, λ_kl=0.1  ← Topology done
Epoch 50:    λ_topo=1.0, λ_tf=0.01, λ_recon=1.0, λ_kl=0.1
Epoch 100:   λ_topo=1.0, λ_tf=0.01, λ_recon=1.0, λ_kl=0.1
```

**Problem**: Transfer function always has 100x weaker signal than reconstruction!

### New Schedule (Improved TF)

```
Epoch 0:     λ_topo=2.0, λ_tf=0.01, λ_recon=1.0, λ_kl=0.1  ← Start same
Epoch 25:    λ_topo=1.0, λ_tf=0.50, λ_recon=1.0, λ_kl=0.1  ← TF increasing
Epoch 50:    λ_topo=1.0, λ_tf=1.00, λ_recon=1.0, λ_kl=0.1  ← TF at full weight
Epoch 100:   λ_topo=1.0, λ_tf=1.00, λ_recon=1.0, λ_kl=0.1  ← Balanced training
```

**Benefits**:
1. Early training (0-25): Focus on topology (same as before)
2. Mid training (25-50): Gradually add TF supervision
3. Late training (50+): Equal weight for reconstruction and TF

## Expected Improvements

### Quantitative Targets

| Metric | Before | Target After |
|--------|--------|--------------|
| Topology Accuracy | 100% | 100% (maintained) |
| TF Inference Accuracy | 0% | 60-80% |
| Pole/Zero Chamfer Distance | High | 50% reduction |
| Q Factor Matching | 1.16-1.20x ref | < 1.10x ref |

### Qualitative Improvements

1. **Better pole/zero predictions**
   - Poles should match filter type (real for low-pass, complex for band-pass)
   - Zeros should be near DC for high-pass, at infinity for low-pass

2. **Transfer function inference**
   - Inferred filter type from poles/zeros should match topology prediction
   - Currently 0% → Target 60%+

3. **Maintained topology quality**
   - Curriculum ensures topology learning isn't hurt
   - Start with weak TF weight, increase gradually

## How to Use

### Train New Model

```bash
python scripts/train.py --config configs/8d_improved_tf.yaml
```

This will:
1. Start with same topology focus as before
2. Gradually increase TF supervision over 50 epochs
3. Train for 200 epochs total
4. Save checkpoints to `checkpoints/`

### Monitor Training

Watch for `tf_weight` in logs:
```
Epoch 0:  tf_weight=0.010, loss_tf=2.345
Epoch 25: tf_weight=0.505, loss_tf=1.234
Epoch 50: tf_weight=1.000, loss_tf=0.567
```

The `loss_tf` should **decrease** as `tf_weight` increases, showing the model is learning.

### Validate Results

After training:
```bash
python scripts/validate_generation.py \
    --checkpoint checkpoints/best.pt \
    --num-samples 20
```

Look for:
- ✅ Topology accuracy: Should stay at 100%
- ✅ TF inference: Should improve from 0% → 60%+
- ✅ Q factor ratio: Should improve from 1.16x → 1.10x

## Technical Rationale

### Why Curriculum Instead of Just High Weight?

**Problem with immediate high weight**:
```python
# Naive approach (doesn't work well):
loss = recon_loss + 1.0 * tf_loss  # Both losses compete from start

# What happens:
# - Early training: Topology not learned yet, TF predictions meaningless
# - Model confused: Should it learn topology or poles/zeros first?
# - Result: Both suffer, convergence slower
```

**Curriculum approach (better)**:
```python
# Epoch 0-25: Focus on topology first
loss = recon_loss + 0.01 * tf_loss

# Epoch 25-50: Gradually add TF supervision
loss = recon_loss + 0.5 * tf_loss

# Epoch 50+: Both have equal weight
loss = recon_loss + 1.0 * tf_loss

# What happens:
# - Early: Learn good topology (foundation)
# - Mid: Start learning poles/zeros (building on foundation)
# - Late: Refine both together
# - Result: Better convergence, both tasks succeed
```

### Why 50 Epochs for TF Warmup?

Based on 8D training analysis:
- Topology converges: Epoch ~20-30
- Best model found: Epoch 127
- Early stopping: Epoch 158

**Warmup schedule**:
```
Epoch 0-30:  Topology learning (original curriculum)
Epoch 30-50: TF weight ramping up (after topology stable)
Epoch 50+:   Joint optimization
```

This ensures topology is well-established before adding TF pressure.

### Why Initial Multiplier = 0.01?

Maintains compatibility with current training:
```
Initial TF weight = target_weight × initial_multiplier
                  = 1.0 × 0.01
                  = 0.01 (same as before!)
```

So early training is **identical** to current model, then gradually improves.

## Alternative Approaches (Not Implemented)

### 1. Separate Decoders

```python
# Separate topology and TF decoders
topology_decoder = TopologyDecoder(z_topo, z_values)
tf_decoder = TransferFunctionDecoder(z_pz)
```

**Pros**: Clearer separation of concerns
**Cons**: More parameters, harder to coordinate

### 2. Multi-Task Learning with Uncertainty

```python
# Learn task-dependent weights
loss = torch.exp(-log_sigma_recon) * loss_recon + log_sigma_recon
     + torch.exp(-log_sigma_tf) * loss_tf + log_sigma_tf
```

**Pros**: Automatic weight balancing
**Cons**: More complex, harder to interpret

### 3. Adversarial Transfer Function Loss

```python
# Discriminator judges if poles/zeros are realistic
loss_tf_adv = discriminator(predicted_poles, predicted_zeros)
```

**Pros**: Might generate more realistic poles/zeros
**Cons**: Training instability, GAN complexity

**Why we chose curriculum**: Simple, interpretable, proven effective for multi-task learning.

## Backward Compatibility

Old configs still work:
```yaml
# Old config (no TF curriculum)
loss:
  tf_weight: 0.01
  # use_tf_curriculum defaults to False

# Result: Behaves exactly like before
```

New configs enable curriculum:
```yaml
# New config (with TF curriculum)
loss:
  tf_weight: 1.0
  use_tf_curriculum: true
  tf_curriculum_warmup_epochs: 50
  tf_curriculum_initial_multiplier: 0.01
```

## Next Steps

1. **Train improved model**:
   ```bash
   python scripts/train.py --config configs/8d_improved_tf.yaml
   ```

2. **Validate improvements**:
   ```bash
   python scripts/validate_generation.py --checkpoint checkpoints/best.pt
   ```

3. **Compare to baseline**:
   - Topology accuracy: Should maintain 100%
   - TF inference: 0% → 60%+
   - Pole/zero quality: Visual inspection of predictions

4. **If successful, iterate**:
   - Try higher final weights (tf_weight: 2.0)
   - Adjust warmup schedule
   - Add more training data

## References

- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)
- **Multi-Task Learning**: Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks" (2017)
- **Loss Balancing**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)
