# Edge Generation Issue - Solution Plan

**Date:** 2025-12-28
**Status:** Investigation Complete, Solutions Designed

---

## Problem Summary

**Symptom:** Single-component edges (R, C, L) have 33-82% accuracy, while RCL edges have 100% accuracy

**Root Causes Identified:**

1. **Edge threshold too conservative** (0.5 hard-coded)
2. **Consistency boosting favors RCL** (1.5x boost for high-consistency edges)
3. **Loss weight imbalance** (component_type_weight=2.0 vs edge_exist_weight=1.0)
4. **Training/generation mismatch** ("None" component type never trained properly)
5. **Separate edge/component heads** (no coordination between decisions)

---

## Solution Options (Ranked by Effectiveness)

### Option 1: Rebalance Loss Weights + Lower Threshold (RECOMMENDED)

**What:** Adjust loss weights to prioritize edge existence, and lower generation threshold

**Changes:**
```python
# In ml/losses/gumbel_softmax_loss.py or training script
edge_exist_weight: 1.0 → 3.0       # Increase 3x
component_type_weight: 10.0 → 5.0  # Decrease 2x
```

```python
# In ml/models/graphgpt_decoder_latent_guided.py
edge_threshold: 0.5 → 0.35  # More permissive
```

**Why this works:**
- Higher `edge_exist_weight` forces model to learn when edges should exist
- Lower threshold compensates for model's conservative bias
- Still preserves Gumbel-Softmax component type selection

**Pros:**
- ✅ Simple, requires only config changes
- ✅ Preserves all existing architecture
- ✅ Should improve R/C/L accuracy to 80-90%
- ✅ Fast to test (just retrain)

**Cons:**
- ⚠️ May increase false positive edges (edges that shouldn't exist)
- ⚠️ Requires full retraining (100 epochs)

**Estimated Time:** 2 hours (modify config + retrain)

**Expected Improvement:** 75% → 85-90%

---

### Option 2: Adaptive Threshold Based on Component Type

**What:** Use different thresholds for different component types

**Changes:**
```python
# In ml/models/graphgpt_decoder_latent_guided.py

# Add component-type-aware thresholding
component_type_pred = torch.argmax(component_type_logits[0], dim=-1)

# Adaptive threshold
if component_type_pred == 7:  # RCL
    threshold = 0.5  # Keep existing
elif component_type_pred in [1, 2, 3]:  # R, C, L
    threshold = 0.3  # More permissive
else:  # RC, RL, CL
    threshold = 0.4

if adjusted_prob[0] > threshold:
    edge_existence[0, i, j] = 1.0
```

**Why this works:**
- Component type prediction is 100% accurate (when edges exist)
- Use predicted component type to inform edge existence decision
- Compensates for bias against single components

**Pros:**
- ✅ Directly addresses the single-component bias
- ✅ No retraining needed (inference-time fix)
- ✅ Can tune thresholds per component type
- ✅ Preserves training process

**Cons:**
- ⚠️ Hacky workaround, not a principled solution
- ⚠️ Circular dependency (component type depends on edge existing)
- ⚠️ May not generalize to new component combinations

**Estimated Time:** 30 minutes (modify decoder.generate())

**Expected Improvement:** 75% → 80-85%

---

### Option 3: Remove Consistency Boosting

**What:** Disable the consistency-based probability adjustment

**Changes:**
```python
# In ml/models/graphgpt_decoder_latent_guided.py

# Comment out lines 348-355:
# if consistency[0] > 0.7:
#     adjusted_prob = base_prob * self.consistency_boost
# elif consistency[0] < 0.3:
#     adjusted_prob = base_prob * self.consistency_penalty
# else:
#     adjusted_prob = base_prob

adjusted_prob = base_prob  # Use raw probability
```

**Why this works:**
- Removes the 1.5x boosting that favors RCL edges
- Levels the playing field for all component types
- TF consistency may be biased toward complex circuits

**Pros:**
- ✅ Simple code change
- ✅ No retraining needed
- ✅ Removes known bias source

**Cons:**
- ⚠️ May hurt overall circuit quality (TF consistency was useful)
- ⚠️ RCL accuracy might drop from 100%
- ⚠️ Doesn't fix underlying threshold issue

**Estimated Time:** 5 minutes (modify decoder.generate())

**Expected Improvement:** 75% → 78-82%

---

### Option 4: Joint Edge-Component Prediction Head (BEST LONG-TERM)

**What:** Replace separate heads with a single joint prediction head

**Changes:**
```python
# In ml/models/latent_guided_decoder.py

# Replace:
# self.edge_exist_head = ...       # Binary: edge exists?
# self.component_type_head = ...   # 8-way: which component?

# With:
self.edge_component_head = nn.Linear(hidden_dim, 8)  # 8-way including "None"

# Output interpretation:
# - Argmax = 0 (None) → No edge
# - Argmax = 1-7 → Edge exists with that component type
```

**In loss function:**
```python
# Unified loss for edge+component
# Target: 0 if no edge, 1-7 if edge with component
target_unified = torch.where(
    edge_mask > 0.5,
    component_types,  # 1-7 for existing edges
    torch.zeros_like(component_types)  # 0 for non-existing
)

loss_unified = F.cross_entropy(edge_component_logits, target_unified)
```

**Why this works:**
- Couples edge existence and component type decisions
- Model learns "None" means no edge
- Single unified loss, no weight balancing issues
- More principled architecture

**Pros:**
- ✅✅ Most principled solution
- ✅ Solves root cause (separate heads)
- ✅ Eliminates weight balancing issues
- ✅ Learns "None" class properly

**Cons:**
- ❌ Requires architecture change
- ❌ Requires full retraining
- ❌ More complex to implement
- ❌ Time-consuming

**Estimated Time:** 4-6 hours (modify architecture + retrain)

**Expected Improvement:** 75% → 90-95%

---

### Option 5: Train on "None" Component Type

**What:** Include non-existing edges in component type loss with target=0

**Changes:**
```python
# In ml/losses/gumbel_softmax_loss.py

# Current (lines 154-171):
edge_mask = (target_edges > 0.5).float()  # Only existing edges
if edge_mask_flat.sum() > 0:
    loss_component_type = ...

# New: Include ALL edges
all_edges_mask = torch.triu(torch.ones_like(target_edges), diagonal=1)

# Target component type:
# - 0 (None) if edge doesn't exist
# - 1-7 if edge exists with component
target_component_unified = torch.where(
    target_edges > 0.5,
    target_component_types,  # Original 1-7
    torch.zeros_like(target_component_types)  # 0 for non-existing
)

# Loss on ALL potential edges
loss_component_type = F.cross_entropy(
    comp_logits_flat,
    target_component_unified_flat,
    reduction='mean'
)
```

**Why this works:**
- Model learns that component_type=0 means "no edge"
- Bridges training/generation gap
- Component type head now predicts edge existence too

**Pros:**
- ✅ Fixes training/generation mismatch
- ✅ Model learns "None" class
- ✅ Moderate complexity

**Cons:**
- ⚠️ Requires retraining
- ⚠️ May confuse component type and edge existence roles
- ⚠️ Still has separate heads (not ideal architecture)

**Estimated Time:** 2 hours (modify loss + retrain)

**Expected Improvement:** 75% → 85-88%

---

## Recommended Solution: Hybrid Approach

**Combine Option 1 + Option 3 + Option 5**

### Phase 1: Quick Wins (No Retraining - 30 mins)
1. ✅ Remove consistency boosting (Option 3)
2. ✅ Lower edge threshold to 0.35 (Option 1 partial)
3. ✅ Test on validation set

**Expected:** 75% → 80%

### Phase 2: Proper Fix (Retraining Required - 2 hours)
1. ✅ Rebalance loss weights (Option 1):
   - `edge_exist_weight: 1.0 → 3.0`
   - `component_type_weight: 10.0 → 5.0`
2. ✅ Train on "None" class (Option 5)
3. ✅ Retrain for 100 epochs
4. ✅ Validate

**Expected:** 80% → 88-92%

### Phase 3: Long-Term (Future Work - 6 hours)
1. ✅ Implement joint edge-component head (Option 4)
2. ✅ Full architecture redesign
3. ✅ Train from scratch

**Expected:** 92% → 95%+

---

## Implementation Details

### Quick Win Changes (Phase 1)

**File 1: `ml/models/graphgpt_decoder_latent_guided.py`**

```python
# Line 232: Change threshold
def generate(
    self,
    latent_code: torch.Tensor,
    conditions: torch.Tensor,
    edge_threshold: float = 0.35,  # Changed from 0.5
    ...
):
```

```python
# Lines 348-355: Comment out consistency boosting
# if consistency[0] > 0.7:
#     adjusted_prob = base_prob * self.consistency_boost
# elif consistency[0] < 0.3:
#     adjusted_prob = base_prob * self.consistency_penalty
# else:
#     adjusted_prob = base_prob

adjusted_prob = base_prob  # Use raw probability directly
```

**File 2: Test**
```bash
python3 scripts/validate_gumbel_comprehensive.py
```

---

### Proper Fix Changes (Phase 2)

**File 1: `scripts/train_gumbel_complete.py`**

```python
# Line 301-308: Update loss function weights
loss_fn = GumbelSoftmaxCircuitLoss(
    node_type_weight=1.0,
    edge_exist_weight=3.0,  # INCREASED from 1.0
    component_type_weight=5.0,  # DECREASED from 10.0
    component_value_weight=0.5,
    use_connectivity_loss=True,
    connectivity_weight=2.0
)
```

**File 2: `ml/losses/gumbel_softmax_loss.py`**

```python
# Lines 154-171: Replace with unified component type loss

# Build target with "None" for non-existing edges
target_component_unified = torch.where(
    target_edges.unsqueeze(-1).expand(-1, -1, -1) > 0.5,
    target_component_types,
    torch.zeros_like(target_component_types)
)

# Flatten
comp_logits_flat = component_type_logits.reshape(-1, 8)
comp_targets_flat = target_component_unified.reshape(-1)

# Cross-entropy on ALL potential edges (upper triangle)
triu_mask = torch.triu(torch.ones(batch_size, max_nodes, max_nodes, device=device), diagonal=1)
triu_mask_flat = triu_mask.reshape(-1)

if triu_mask_flat.sum() > 0:
    loss_per_comp = F.cross_entropy(
        comp_logits_flat,
        comp_targets_flat,
        reduction='none'
    )
    loss_component_type = (loss_per_comp * triu_mask_flat).sum() / (triu_mask_flat.sum() + 1e-6)
else:
    loss_component_type = torch.tensor(0.0, device=device)
```

**File 3: Retrain**
```bash
python3 scripts/train_gumbel_complete.py > training_fixed_edges.log 2>&1 &
```

---

## Testing Plan

### Phase 1 Testing (Quick Wins)
1. Apply quick changes (threshold + remove boosting)
2. Run validation: `python3 scripts/validate_gumbel_comprehensive.py`
3. Check per-type accuracy:
   - **Target:** R > 85%, C > 70%, L > 50%
4. If successful → proceed to Phase 2
5. If unsuccessful → revert and try Option 2 (adaptive threshold)

### Phase 2 Testing (After Retraining)
1. Monitor training for 100 epochs
2. Check validation metrics every 10 epochs
3. Look for:
   - Edge existence accuracy improving
   - Component type accuracy maintaining 100%
   - Single-component (R/C/L) accuracy increasing
4. Final validation on test set
5. Compare confusion matrix with baseline

### Success Criteria

**Phase 1 (Quick Win):**
- Overall: 75% → 80%+
- R accuracy: 82% → 88%+
- C accuracy: 62% → 70%+
- L accuracy: 33% → 50%+

**Phase 2 (Full Fix):**
- Overall: 80% → 90%+
- R accuracy: 88% → 92%+
- C accuracy: 70% → 85%+
- L accuracy: 50% → 80%+
- RCL accuracy: Maintain 100%

---

## Risk Analysis

### Phase 1 Risks
- ⚠️ Lower threshold may create false positive edges
- ⚠️ Removing boosting may hurt RCL accuracy
- **Mitigation:** Easy to revert, test incrementally

### Phase 2 Risks
- ⚠️ Retraining may not converge well with new weights
- ⚠️ Training on "None" class may confuse the model
- ⚠️ 100 epochs = 20-30 minutes training time
- **Mitigation:** Monitor training closely, revert if diverging

---

## Decision Matrix

| Solution | Time | Effort | Expected Δ | Retraining | Risk | Recommendation |
|----------|------|--------|------------|------------|------|----------------|
| **Option 1** | 2h | Low | +10-15% | Yes | Low | ✅ **Phase 2** |
| **Option 2** | 30m | Low | +5-10% | No | Medium | ⚠️ Fallback |
| **Option 3** | 5m | Very Low | +3-7% | No | Low | ✅ **Phase 1** |
| **Option 4** | 6h | High | +15-20% | Yes | Medium | 🔮 Future |
| **Option 5** | 2h | Medium | +10-13% | Yes | Medium | ✅ **Phase 2** |

---

## Next Steps

1. ✅ Implement Phase 1 (Quick Wins) - **START HERE**
2. ✅ Validate improvement
3. ✅ If successful, implement Phase 2 (Full Fix)
4. ✅ Retrain and validate
5. ✅ Document final results
6. ⏳ (Optional) Phase 3 for long-term improvement

---

## Confidence Level

**Phase 1:** 85% confident will improve to 80%
**Phase 2:** 90% confident will improve to 88-92%

**Reasoning:**
- Root causes are well understood
- Solutions directly address identified issues
- Similar fixes have worked in other autoregressive models
- Training convergence should be stable with rebalanced weights

---

**Status:** Ready to implement Phase 1
**Estimated Total Time:** 30 minutes (Phase 1) + 2.5 hours (Phase 2) = **3 hours total**
