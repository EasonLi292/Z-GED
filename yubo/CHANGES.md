# yubo — Dev Log (2026-03-10)

---

## 1. Standalone experiment: encoder + 2 MLP heads (no decoder)

**Added `yubo/`:**
- `auxiliary_heads.py` — `RegressionMLP`, `ClassificationMLP`
- `dataset.py` — `YuboDataset` wrapping `CircuitDataset`; returns graph, pole_real, pole_imag, filter_type_label
- `train.py` — training loop, KL warmup, checkpoint

**Architecture:**
```
HierarchicalEncoder  z [B, 8]
    ├── RegressionMLP:     8 → 64 → 32 → 2   (pole_real, pole_imag)
    └── ClassificationMLP: 8 → 64 → 32 → 8   (filter type logits)
```
Heads predict from `mu` (not sampled z) for stability.

**Loss:**
```
L = kl_weight · KL(q(z|x) ‖ N(0,I)) + MSE(pole_pred, pole_target) + CE(cls_logits, label)
```
KL weight: 0 → 0.001 over first 10 epochs.

**Results** (best checkpoint, epoch 48):

| Metric | Val |
|--------|-----|
| Total loss | 0.0065 |
| Regression loss (MSE) | 0.0022 |
| Classification loss (CE) | 8.5e-05 |

Classification CE ≈ 0 — all 8 filter types cleanly separated from latent z alone.

---

## 2. MLPs integrated into main training (encoder + decoder)

**Modified `scripts/training/train.py`:**
- Added imports from `yubo.auxiliary_heads`
- Both MLPs trained jointly with encoder + decoder
- `collate_fn` updated: `include_pz_target=True, include_filter_type_label=True`

**Architecture:**
```
HierarchicalEncoder  z [B, 8]
    ├── HierarchicalDecoder          (circuit reconstruction)
    ├── RegressionMLP:     8 → 64 → 32 → 2
    └── ClassificationMLP: 8 → 64 → 32 → 8
```
Params: encoder 237,907 | decoder 7,698,901 | reg MLP 2,722 | cls MLP 2,920

**Loss:**
```
L = CircuitLoss(node_type=1.0, node_count=5.0, edge_component=2.0, connectivity=5.0, kl=0.01)
  + 0.1 · MSE(pole_pred, pole_target)
  + 0.1 · CE(cls_logits, label)
```
KL weight: 0 → 0.01 over first 20 epochs. Train uses sampled z; val uses mu.

**Results** (best checkpoint, epoch 89):

| Metric | Val |
|--------|-----|
| Total loss | 0.9865 |
