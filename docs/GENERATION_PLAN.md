# Generation Plan: EOS-Style Stopping + Behavior-Centric Latent

**Goal:** Make autoregressive circuit generation stop naturally (LLM-style EOS) while keeping the latent focused on circuit behavior (transfer function), not topology.

---

## 1) Decoder: EOS-Style Stopping (No Fixed Node Count)

**Change:** Use the existing stop head as the only stopping mechanism. Do **not** predict a node count. Keep a large `max_nodes` as a safety cap only.

**Files:**
- `ml/models/node_decoder.py`
- `ml/models/decoder.py`
- `ml/losses/gumbel_softmax_loss.py`

**Specific steps:**
1. **Stop head output is mandatory when `use_stopping_criterion=True`.**
   - Ensure `stop_logit` is returned for every step.
2. **Loss:**
   - Keep BCE loss on `stop_logits` vs `target_stop` (stop when token is `MASK`).
   - Remove any loss terms that explicitly regress node count.
3. **Generation logic:**
   - Always generate GND/VIN/VOUT for the first 3 positions.
   - For each subsequent step:
     - If `sigmoid(stop_logit) > stop_threshold`, stop and mask remaining positions.
     - Else generate the next node and continue.
4. **Heuristic to avoid premature stop:**
   - Add a small bias to continue for the first optional node (e.g., force at least 1 internal node unless `stop_prob` is extremely high).

**Success criteria:**
- Node count varies across specs without collapsing to a single length.
- Generation can exceed training average node count when latent implies complexity.

---

## 2) Latent Emphasis on Behavior, Not Topology

**Change:** Prevent decoder from over-relying on topology latent slices.

**Files:**
- `ml/models/encoder.py`
- `ml/models/decoder.py`
- `configs/*.yaml`

**Specific steps:**
1. **Topo latent dropout:**
   - During training, randomly drop or noise `z_topo` (e.g., 50% of batches).
   - Keep TF latent (`z_pz`) intact.
2. **Latent split usage audit:**
   - Ensure `latent[4:8]` (TF) is always provided to node + edge decoders.
   - Minimize or zero `latent[0:2]` in the node decoder pathway.
3. **Behavioral conditioning:**
   - Encourage the decoder to use `conditions` (cutoff, Q) by injecting them into node decoder and stop head.

**Success criteria:**
- Latent interpolation changes function without deterministically fixing topology.
- Stop decisions correlate with TF complexity (poles/zeros).

---

## 3) Training: Reduce Collapse, Preserve Autoregressive Flavor

**Change:** Make stopping learnable and stable with proper sampling and schedules.

**Files:**
- `scripts/training/train.py`
- `configs/*.yaml`

**Specific steps:**
1. **KL warm-up / beta schedule:**
   - Gradually increase KL weight to keep latent active.
2. **Stop loss weighting:**
   - Increase `stop_weight` if model stops too late.
   - Decrease if it stops too early.
3. **Data balance:**
   - Rebalance mini-batches across node counts (3/4/5+) to prevent length collapse.

**Success criteria:**
- Stop accuracy > 85% on validation.
- No single node-count dominates generation.

---

## 4) Evaluation & Diagnostics

**Files:**
- `validate_stopping_criterion.py`
- `ml/utils/metrics.py`
- `docs/GENERATION_GUIDE.md`

**Specific steps:**
1. **Stop behavior plots:**
   - Plot stop probability vs step index by filter type.
2. **Length distribution:**
   - Compare generated node counts vs ground truth.
3. **Latent sensitivity:**
   - Interpolate `z_pz` only and confirm topology adapts.

**Success criteria:**
- Generated length distribution matches spec complexity.
- `z_pz` interpolation produces smooth changes in topology and values.

---

## 5) Optional: EOS Temperature Controls (Inference Only)

**Change:** Allow creative exploration without retraining by adjusting stopping threshold.

**Files:**
- `ml/models/decoder.py`

**Specific steps:**
1. Expose `stop_threshold` and `min_internal_nodes` in `generate()`.
2. Provide presets:
   - **Conservative:** `stop_threshold=0.5`, `min_internal_nodes=1`
   - **Exploratory:** `stop_threshold=0.7`, `min_internal_nodes=0`

**Success criteria:**
- User can trade off creativity vs stability without retraining.

---

## Milestones

1. **M1:** EOS stop head works end-to-end, no explicit node count.
2. **M2:** Latent TF drives stop decisions; topology not hard-coded.
3. **M3:** Stable training with balanced lengths and no collapse.
4. **M4:** Evaluation shows length diversity and TF-aligned complexity.

