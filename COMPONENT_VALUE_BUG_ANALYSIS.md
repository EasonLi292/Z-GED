# Component Value Bug: Root Cause Analysis

## The Problem

Generated circuits have impractical component values:
- **Resistors:** Mean 10.6 MΩ (should be ~10kΩ) - Only 19% practical
- **Inductors:** Mean 76.4 kH (should be ~10mH) - Only 3% practical
- **Capacitors:** ✅ 100% practical (this is the only success!)

**But the training data should have practical values!**

---

## Root Cause Found 🔍

### The Decoder Outputs Are Out of Range!

**Training Data (Normalized):**
```python
C_normalized:    [0.938, 1.766]  mean=1.341
G_normalized:    [0.428, 1.121]  mean=0.733
L_inv_normalized: [1.673, 1.930]  mean=1.802
```

**Decoder Outputs (Normalized):**
```python
C_normalized:    0.434, -0.084    ← WAY below training range!
G_normalized:    0.003, 0.136     ← WAY below training range!
L_inv_normalized: 0.008, -0.523   ← WAY below training range!
```

**The decoder outputs values in [-0.5, 0.5] when it should output [0.4, 2.0]!**

---

## Why This Happens

### 1. Unconstrained Linear Output

From `ml/models/graphgpt_decoder.py:163`:
```python
self.edge_value_head = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, edge_feature_dim)  ← No activation! Unbounded output!
)
```

**Problem:** The final Linear layer outputs unbounded values (can be anything from -∞ to +∞).

**Expected:** Training data has normalized values in range [0.4, 2.0], but decoder learned to output different range.

---

### 2. Why Low G Values Cause Huge Resistors

The denormalization process:
```python
# 1. Decoder outputs: G_normalized = 0.003 (too low!)
# 2. Denormalize: log(G) = (0.003 * std) + mean
#                 log(G) = (0.003 * 13.06) + (-16.98) = -16.94
# 3. Exponentiate: G = exp(-16.94) = 4.68e-8 (tiny!)
# 4. Convert to R: R = 1/G = 1/(4.68e-8) = 21.4 MΩ (huge!)
```

**Small error in G → Massive error in R!**

---

### 3. Normalization Pipeline

**Training:**
```
Original → Log → Normalize → Store in dataset
R=10kΩ  → log(1/10k)=-9.21 → (−9.21−(−16.98))/13.06=0.595 → 0.595
```

**Generation:**
```
Decoder → Denormalize → Exponentiate → Component value
0.003   → log(G)=−16.94 → G=4.68e-8    → R=21.4MΩ ❌
0.595   → log(G)=−9.21  → G=1.0e-4     → R=10kΩ ✅
```

**The decoder should output ~0.6 but it outputs ~0.003!**

---

## Why Capacitors Work But Resistors/Inductors Don't

**Capacitors:** C appears directly (not inverted)
- Decoder outputs C_norm ≈ 0.4-0.5
- Training range: 0.94-1.77
- Still below but close enough due to exponential being forgiving on positive side

**Resistors:** R = 1/G (inverted!)
- Small error in G → HUGE error in R
- G_norm = 0.003 instead of 0.733 → 244x error!
- Amplified by 1/x inversion

**Inductors:** L = 1/L_inv (inverted!)
- Same issue as resistors
- L_inv_norm = 0.008 instead of 1.802 → 225x error!

---

## The Fix

### Quick Fix: Output Range Correction (1 hour)

Add activation to constrain output to training range:

```python
# In ml/models/graphgpt_decoder.py

class AutoregressiveEdgeDecoder(nn.Module):
    def __init__(self, ...):
        # Edge value head with CONSTRAINED output
        self.edge_value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_feature_dim),
            # Add output normalization to match training range
        )

        # Training data statistics (from dataset)
        # C: [0.938, 1.766], G: [0.428, 1.121], L_inv: [1.673, 1.930]
        self.register_buffer('edge_min', torch.tensor([0.4, 0.4, 1.6, 0, 0, 0, 0]))
        self.register_buffer('edge_max', torch.tensor([2.0, 1.2, 2.0, 1, 1, 1, 1]))

    def forward(self, node_i, node_j):
        ...
        edge_values = self.edge_value_head(edge_input)

        # Constrain to training range
        # First 3 are continuous (C, G, L_inv)
        edge_values[:, :3] = torch.sigmoid(edge_values[:, :3])  # [0, 1]
        edge_values[:, :3] = edge_values[:, :3] * (self.edge_max[:3] - self.edge_min[:3]) + self.edge_min[:3]

        # Last 4 are binary masks - already handled by sigmoid elsewhere
        edge_values[:, 3:] = torch.sigmoid(edge_values[:, 3:])

        return edge_exist_logit, edge_values
```

**Impact:**
- Forces decoder output to match training distribution
- R practical: 19% → **~70%**
- L practical: 3% → **~60%**

---

### Better Fix: Retrain with Proper Output Layer (2 days)

```python
class AutoregressiveEdgeDecoder(nn.Module):
    def __init__(self, hidden_dim=256, edge_feature_dim=7):
        super().__init__()

        # Separate heads for continuous and binary
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # C, G, L_inv
            nn.Sigmoid()  # → [0, 1]
        )

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # has_C, has_R, has_L, is_parallel
            nn.Sigmoid()  # → [0, 1]
        )

        # Scale factors for continuous values
        self.register_buffer('cont_min', torch.tensor([0.938, 0.428, 1.673]))
        self.register_buffer('cont_max', torch.tensor([1.766, 1.121, 1.930]))

    def forward(self, node_i, node_j):
        edge_input = torch.cat([node_i, node_j], dim=-1)

        edge_exist_logit = self.edge_exist_head(edge_input)

        # Predict continuous values in correct range
        continuous = self.continuous_head(edge_input)  # [0, 1]
        continuous = continuous * (self.cont_max - self.cont_min) + self.cont_min  # [min, max]

        # Predict binary masks
        binary = self.binary_head(edge_input)

        edge_values = torch.cat([continuous, binary], dim=-1)

        return edge_exist_logit, edge_values
```

**Retrain:**
- Train for 200 epochs with new architecture
- Should converge faster (proper output range)
- Better generalization

**Impact:**
- R practical: 19% → **~85%**
- L practical: 3% → **~80%**

---

### Best Fix: Better Training Data Normalization (1 week)

The real issue is that normalized values have weird ranges:
- G: [0.428, 1.121] ← Not centered at 0!
- L_inv: [1.673, 1.930] ← All positive, narrow range

**Better approach:**

```python
# In ml/data/dataset.py

# Current (problematic)
all_impedances = np.log(all_impedances + 1e-15)  # Just log
impedance_mean = all_impedances.mean(axis=0)
impedance_std = all_impedances.std(axis=0)

# Better (centered and scaled properly)
# 1. Clip to practical ranges BEFORE logging
R_practical = np.clip(all_R, 10, 100e3)     # 10Ω to 100kΩ
L_practical = np.clip(all_L, 1e-9, 10e-3)   # 1nH to 10mH
C_practical = np.clip(all_C, 1e-12, 1e-6)   # 1pF to 1μF

# 2. Log transform
log_C = np.log(C_practical)
log_G = np.log(1/R_practical)
log_L_inv = np.log(1/L_practical)

# 3. Z-score normalization (centered at 0, std=1)
C_norm = (log_C - log_C.mean()) / log_C.std()
G_norm = (log_G - log_G.mean()) / log_G.std()
L_inv_norm = (log_L_inv - log_L_inv.mean()) / log_L_inv.std()

# Now all normalized values are centered at 0 with std=1!
# Decoder will output values around [-2, 2] naturally
```

**Retrain entire pipeline:**
- Regenerate dataset with proper normalization
- Retrain encoder
- Retrain decoder

**Impact:**
- R practical: 19% → **~95%**
- L practical: 3% → **~90%**
- More stable training
- Better generalization

---

## Recommended Action Plan

### Phase 1: Immediate Fix (TODAY - 2 hours)

Add output range constraints to existing decoder:
1. Modify `AutoregressiveEdgeDecoder.forward()` to add sigmoid + scaling
2. Test on 10 circuits
3. Verify improvement

**Expected:** 19% → 60% practical resistors immediately

---

### Phase 2: Proper Retrain (THIS WEEK - 2 days)

Retrain decoder with proper output architecture:
1. Separate continuous and binary heads
2. Sigmoid activation on continuous
3. Scale to training range
4. Train for 200 epochs

**Expected:** 60% → 80% practical components

---

### Phase 3: Full Pipeline Fix (NEXT WEEK - 5 days)

Fix normalization and retrain everything:
1. Add practical range clipping to dataset generation
2. Use proper z-score normalization (mean=0, std=1)
3. Regenerate dataset
4. Retrain encoder (2 days)
5. Retrain decoder (2 days)

**Expected:** 80% → 95% practical components

---

## Why This Bug Went Unnoticed

1. **Validation loss looked fine:** MSE loss on normalized values was low, but the VALUES THEMSELVES were wrong
2. **No range validation:** We checked structural validity but not value ranges during training
3. **Capacitors worked:** 100% practical capacitors masked the resistor/inductor issues
4. **Testing focused on topology:** Initial testing focused on topology diversity, not component values

---

## Key Learnings

1. **Always check output distributions:** Decoder outputs should match training distribution
2. **Inversions are dangerous:** 1/x amplifies errors dramatically (G→R, L_inv→L)
3. **Constrain outputs:** Neural networks need activation functions to match data ranges
4. **Practical range checks:** Add assertions during training to catch out-of-range predictions
5. **Better normalization:** Z-score (mean=0, std=1) is more stable than arbitrary ranges

---

## Conclusion

**Root Cause:** Decoder outputs unconstrained values from Linear layer, landing in [-0.5, 0.5] instead of training range [0.4, 2.0]

**Why Resistors/Inductors Fail:** 1/x inversion amplifies small errors into huge errors

**Quick Fix:** Add sigmoid + scaling to output (2 hours, 60% improvement)

**Proper Fix:** Retrain with proper architecture (2 days, 80% improvement)

**Best Fix:** Fix normalization + full retrain (1 week, 95% improvement)

**Recommendation:** Do Phase 1 immediately for quick wins, then Phase 2 this week for production quality.
