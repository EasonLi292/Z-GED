**✅ CURRENT - Variable-Length Decoder Series**

# Variable-Length Pole/Zero Decoder Design

## Problem Statement

Current decoder outputs **fixed length** (2 poles, 2 zeros) but circuits have **variable length** (0-4 poles, 0-4 zeros).

**Solution**: Encode count information in latent space, decode variable-length outputs.

---

## Design: Count Prediction + Masking

### Architecture Overview

```
Encoder:
  Graph → GNN → z_pz (4D)
           ↓
  Encode: "This circuit has 1 pole, 0 zeros"

Decoder:
  z_pz → [count_poles, count_zeros, pole_features, zero_features]
           ↓
  Predict: num_poles=1, num_zeros=0
  Output:  poles[0] = valid, poles[1:] = masked
           zeros[:] = all masked
```

### Key Components

1. **Count Prediction Head**
   ```python
   # Predict how many poles/zeros (0-4 each)
   num_poles = count_head_poles(z_pz)  # [B, 5] logits for 0,1,2,3,4
   num_zeros = count_head_zeros(z_pz)  # [B, 5] logits for 0,1,2,3,4
   ```

2. **Value Prediction Head** (same as before)
   ```python
   # Predict up to MAX_POLES=4 poles
   poles_all = pole_decoder(z_pz).view(B, 4, 2)  # [B, 4, 2]
   zeros_all = zero_decoder(z_pz).view(B, 4, 2)  # [B, 4, 2]
   ```

3. **Validity Masking**
   ```python
   # Create mask based on predicted count
   pole_mask = create_mask(num_poles_pred, max_len=4)  # [B, 4] boolean
   zero_mask = create_mask(num_zeros_pred, max_len=4)  # [B, 4] boolean

   # Apply mask
   poles_valid = poles_all[pole_mask]
   zeros_valid = zeros_all[zero_mask]
   ```

---

## Implementation Details

### Modified Decoder

```python
class HybridDecoder(nn.Module):
    def __init__(self, ..., max_poles=4, max_zeros=4):
        super().__init__()

        self.max_poles = max_poles
        self.max_zeros = max_zeros

        # Count prediction heads (new)
        self.pole_count_head = nn.Linear(pz_latent_dim, max_poles + 1)  # 0,1,2,3,4
        self.zero_count_head = nn.Linear(pz_latent_dim, max_zeros + 1)

        # Value prediction heads (modified size)
        self.pole_decoder = nn.Sequential(
            nn.Linear(pz_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_poles * 2)  # max_poles * [real, imag]
        )

        self.zero_decoder = nn.Sequential(
            nn.Linear(pz_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_zeros * 2)
        )

    def forward(self, z, hard=True, gt_filter_type=None):
        # ... topology and component value decoding ...

        # Extract pole/zero latent
        z_pz = z[:, -self.pz_latent_dim:]

        # Predict counts
        pole_count_logits = self.pole_count_head(z_pz)  # [B, 5]
        zero_count_logits = self.zero_count_head(z_pz)  # [B, 5]

        if hard:
            num_poles = pole_count_logits.argmax(dim=-1)  # [B]
            num_zeros = zero_count_logits.argmax(dim=-1)  # [B]
        else:
            # Soft counts for training
            pole_count_probs = F.softmax(pole_count_logits, dim=-1)
            zero_count_probs = F.softmax(zero_count_logits, dim=-1)

        # Predict all possible poles/zeros (up to max)
        poles_flat = self.pole_decoder(z_pz)
        zeros_flat = self.zero_decoder(z_pz)

        poles_all = poles_flat.view(batch_size, self.max_poles, 2)  # [B, 4, 2]
        zeros_all = zeros_flat.view(batch_size, self.max_zeros, 2)  # [B, 4, 2]

        return {
            'poles_all': poles_all,           # All predictions [B, 4, 2]
            'zeros_all': zeros_all,           # All predictions [B, 4, 2]
            'num_poles': num_poles,           # Predicted counts [B]
            'num_zeros': num_zeros,           # Predicted counts [B]
            'pole_count_logits': pole_count_logits,  # For loss
            'zero_count_logits': zero_count_logits,
            # ... other outputs ...
        }
```

### Modified Encoder

```python
class HierarchicalEncoder(nn.Module):
    def __init__(self, ...):
        # Existing encoder, but need to encode count information
        # The z_pz branch already gets pole/zero info from DeepSets
        # Just need to ensure it encodes COUNT not just values

        # DeepSets naturally encodes count:
        # - Mean pooling: avg(pole_embeddings) depends on number of poles
        # - Max pooling: less sensitive to count
        # - Sum pooling: directly encodes count (but scale varies)

        # Current: uses mean pooling → should work but could be improved
        pass
```

**Note**: The current encoder uses DeepSets with mean aggregation. This should implicitly encode count information, but we could make it explicit:

```python
# Optional improvement:
h_poles = self.pz_encoder_poles(poles)  # [num_poles, 16]

# Explicitly encode count
pole_count_encoding = encode_count(len(poles), max_count=4)  # [5] one-hot

# Combine value and count information
h_poles_agg = torch.cat([
    h_poles.mean(dim=0),     # Value information
    pole_count_encoding       # Count information
], dim=-1)
```

### Modified Loss Function

```python
class SimplifiedTransferFunctionLoss(nn.Module):
    def forward(self, outputs, targets):
        # 1. Count prediction loss
        loss_pole_count = F.cross_entropy(
            outputs['pole_count_logits'],
            targets['num_poles']  # Ground truth counts
        )
        loss_zero_count = F.cross_entropy(
            outputs['zero_count_logits'],
            targets['num_zeros']
        )

        # 2. Value prediction loss (only on valid poles/zeros)
        batch_size = outputs['poles_all'].size(0)
        pole_losses = []
        zero_losses = []

        for i in range(batch_size):
            # Get predicted count
            n_poles = outputs['num_poles'][i].item()
            n_zeros = outputs['num_zeros'][i].item()

            # Get valid predictions (first n_poles/n_zeros)
            pred_poles = outputs['poles_all'][i, :n_poles]  # [n_poles, 2]
            pred_zeros = outputs['zeros_all'][i, :n_zeros]  # [n_zeros, 2]

            # Get ground truth
            target_poles = targets['poles_list'][i]  # [m_poles, 2]
            target_zeros = targets['zeros_list'][i]  # [m_zeros, 2]

            # Compute Chamfer distance (handles variable lengths)
            pole_loss = chamfer_distance(pred_poles, target_poles)
            zero_loss = chamfer_distance(pred_zeros, target_zeros)

            pole_losses.append(pole_loss)
            zero_losses.append(zero_loss)

        # 3. Total loss
        loss = (
            loss_pole_count + loss_zero_count +  # Count accuracy
            torch.stack(pole_losses).mean() +     # Value accuracy
            torch.stack(zero_losses).mean()
        )

        return loss, {
            'pole_count_loss': loss_pole_count.item(),
            'zero_count_loss': loss_zero_count.item(),
            'pole_value_loss': torch.stack(pole_losses).mean().item(),
            'zero_value_loss': torch.stack(zero_losses).mean().item(),
        }
```

### Modified Dataset

Need to add ground truth counts:

```python
class CircuitDataset:
    def __getitem__(self, idx):
        # ... existing code ...

        # Add counts
        num_poles = len(poles)  # 0-4
        num_zeros = len(zeros)  # 0-4

        return {
            # ... existing returns ...
            'num_poles': torch.tensor(num_poles, dtype=torch.long),
            'num_zeros': torch.tensor(num_zeros, dtype=torch.long),
        }
```

---

## Expected Improvements

### Before (Fixed 2 poles, 2 zeros)

```
Circuit 0 (low_pass, 1 pole, 0 zeros):
  Predicted: 2 poles, 2 zeros ❌
  Structure match: 0%
  Value match: Poor (trying to fit 1 pole into 2 predictions)
```

### After (Variable-length)

```
Circuit 0 (low_pass, 1 pole, 0 zeros):
  Predicted count: 1 pole, 0 zeros ✅
  Predicted values: [[-1.75, 0.01]] ≈ GT [[-1.788, 0.0]] ✅
  Structure match: 100%
  Value match: Good (direct 1-to-1 mapping)
```

---

## Training Strategy

### Phase 1: Count Prediction Warmup

```yaml
# First 50 epochs: Focus on learning correct counts
loss_weights:
  pole_count: 5.0      # High weight - learn structure first
  zero_count: 5.0
  pole_value: 0.1      # Low weight - structure more important
  zero_value: 0.1
```

**Rationale**: Model must learn correct structure before values matter

### Phase 2: Joint Optimization

```yaml
# Epochs 50-200: Balance count and value
loss_weights:
  pole_count: 1.0
  zero_count: 1.0
  pole_value: 1.0
  zero_value: 1.0
```

**Rationale**: Once structure is correct, refine values

---

## Alternative: Filter-Type-Aware Counts

Since we know filter type during generation (100% topology accuracy), we could use that:

```python
# Filter type → expected pole/zero structure
FILTER_STRUCTURES = {
    'low_pass':     {'poles': (1, 2), 'zeros': (0, 1)},  # min, max
    'high_pass':    {'poles': (1, 2), 'zeros': (1, 2)},
    'band_pass':    {'poles': (2, 2), 'zeros': (0, 2)},
    'band_stop':    {'poles': (2, 4), 'zeros': (2, 4)},
    'rlc_series':   {'poles': (2, 2), 'zeros': (0, 0)},
    'rlc_parallel': {'poles': (2, 2), 'zeros': (0, 0)},
}

def decode_with_filter_type(z, filter_type):
    # Know the valid range for this filter
    pole_range = FILTER_STRUCTURES[filter_type]['poles']

    # Only predict within valid range
    num_poles = predict_count(z_pz, min=pole_range[0], max=pole_range[1])
    poles = predict_poles(z_pz, count=num_poles)

    return poles
```

**Pros**:
- More constrained → easier to learn
- Uses known filter characteristics
- Guaranteed valid structure

**Cons**:
- Couples pole/zero prediction to topology
- Less flexible

---

## Implementation Steps

### Step 1: Modify Decoder ✏️

File: `ml/models/decoder.py`

Changes:
- Add `pole_count_head`, `zero_count_head`
- Increase `pole_decoder` output: `4 → 8` (4 poles × 2)
- Increase `zero_decoder` output: `4 → 8` (4 zeros × 2)
- Return counts and all predictions

### Step 2: Modify Encoder 🔍

File: `ml/models/encoder.py`

Changes:
- DeepSets already encodes count implicitly (via mean pooling)
- Optional: Explicitly encode count with one-hot embedding
- Test if current encoder is sufficient first

### Step 3: Modify Loss 📊

File: `ml/losses/transfer_function.py`

Changes:
- Add count prediction cross-entropy loss
- Modify Chamfer distance to use predicted counts
- Add count accuracy metrics

### Step 4: Modify Dataset 💾

File: `ml/data/dataset.py`

Changes:
- Add `num_poles`, `num_zeros` to returned dict
- Compute from `len(poles)`, `len(zeros)`

### Step 5: Update Config 📝

File: `configs/8d_variable_length.yaml`

```yaml
model:
  max_poles: 4
  max_zeros: 4

loss:
  pole_count_weight: 5.0  # High initially
  zero_count_weight: 5.0
  pole_value_weight: 0.1  # Low initially
  zero_value_weight: 0.1

  # Use curriculum
  use_pz_count_curriculum: true
  pz_count_warmup_epochs: 50
```

### Step 6: Train & Validate 🚀

```bash
python scripts/train.py --config configs/8d_variable_length.yaml
python scripts/validate_generation.py --checkpoint checkpoints/best.pt
```

**Expected results**:
- Count accuracy: 80-100%
- Transfer function inference: 60-80% (huge improvement!)
- Structure match: 100% for simple filters

---

## Backward Compatibility

The new decoder is **compatible** with the old:

```python
# Old interface (still works)
outputs = decoder(z, hard=True)
poles = outputs['poles_all'][:, :2]  # Take first 2 (old behavior)

# New interface
outputs = decoder(z, hard=True)
num_poles = outputs['num_poles']     # Get predicted count
poles = outputs['poles_all'][:, :num_poles]  # Use variable count
```

---

## Success Metrics

| Metric | Before | After (Target) |
|--------|--------|----------------|
| **Topology Accuracy** | 100% | 100% (maintained) |
| **Pole Count Accuracy** | 0% | **90%+** |
| **Zero Count Accuracy** | 0% | **90%+** |
| **Pole Value MAE** | 0.69 | **< 0.2** |
| **TF Inference Accuracy** | 0% | **60-80%** |

The key improvement: **Model can now match the structure**, not just approximate values.

---

## Why This Will Work

1. **Addresses root cause**: Variable-length outputs now possible
2. **Adds information**: Count encoded in latent space
3. **Uses correct structure**: Loss only computed on valid poles/zeros
4. **Maintains topology**: Topology branch unchanged
5. **Backward compatible**: Can fall back to old behavior if needed

This is the **proper fix** for the generation failure!
