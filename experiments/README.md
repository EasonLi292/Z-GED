# Experiments Log

This directory contains all training experiments with organized results.

## Experiment Naming Convention

Format: `exp###_description/`
- `###`: Sequential experiment number (001, 002, 003, ...)
- `description`: Brief description of the experiment

## Experiment Structure

Each experiment directory contains:
```
exp###_description/
├── config.yaml              # Training configuration used
├── checkpoints/             # Saved model weights
│   ├── best.pt             # Best validation model
│   ├── final.pt            # Final epoch model
│   └── epoch_*.pt          # Periodic checkpoints (optional)
├── logs/                    # Training logs
│   └── training_history.json
└── evaluation/              # Evaluation results (optional)
    ├── metrics.json
    └── visualizations/
```

## Completed Experiments

### exp001_baseline_2epochs
**Date**: December 17, 2025
**Config**: `configs/test.yaml` (original weights)
**Duration**: 2 epochs
**Best Val Loss**: 5.9717

**Loss Weights**:
- Reconstruction: 1.0
- Transfer Function: 0.5
- KL Divergence: 0.05

**Results**:
- Total Loss: 5.97
- Reconstruction: 2.65
- Transfer Function: 6.58 (weighted: 3.29, 55% of total)
- KL: 0.65
- Topology Accuracy: 33.33%

**Key Findings**:
- TF loss dominates (55% of total loss)
- Model learning but needs rebalancing

---

### exp002_optimized_2epochs
**Date**: December 19, 2025
**Config**: `configs/test.yaml` (optimized weights)
**Duration**: 2 epochs
**Best Val Loss**: 2.7820
**Improvement**: 53% reduction from exp001

**Loss Weights** (optimized):
- Reconstruction: 1.0
- Transfer Function: 0.01 (reduced 50x)
- KL Divergence: 0.1 (increased 2x)

**Results**:
- Total Loss: 2.78 (**↓ 53%**)
- Reconstruction: 2.71
- Transfer Function: 6.60 (weighted: 0.066, 2% of total)
- KL: 0.049
- Topology Accuracy: 16.67%

**Key Findings**:
- Dramatically improved loss balance
- Better regularization (KL weight increased)
- Ready for full 200-epoch training

**Evaluation Metrics** (test set):
- Silhouette Score: 0.62
- Cluster Purity: 100%
- Pole Chamfer: 4.85
- Zero Chamfer: 4.05

---

## Running New Experiments

### Quick Test (2 epochs)
```bash
python3 scripts/train.py --config configs/test.yaml --epochs 2
```

### Full Training (200 epochs)
```bash
python3 scripts/train.py --config configs/optimized.yaml
```

### Custom Configuration
```bash
python3 scripts/train.py \
  --config configs/default.yaml \
  --epochs 50 \
  --batch-size 8 \
  --device cuda
```

## Evaluating Experiments

```bash
python3 scripts/evaluate.py \
  --checkpoint experiments/exp002_optimized_2epochs/checkpoints/best.pt \
  --output-dir experiments/exp002_optimized_2epochs/evaluation
```

## Comparing Experiments

Use the metrics in `logs/training_history.json` to compare:
- Training curves (loss over epochs)
- Validation performance
- Convergence speed
- Final metrics

## Archived Experiments

Older/failed experiments are moved to `archived/` to keep the main directory clean.

## Next Experiments (Planned)

### exp003_full_training_200epochs
- Config: `optimized.yaml`
- Duration: 200 epochs
- Expected: Much better reconstruction, lower TF loss

### exp004_larger_model
- Increase hidden dimensions (64 → 128)
- Increase latent dim (24 → 32)
- Test if more capacity helps

### exp005_curriculum_learning
- Progressive loss weights
- Start with topology, gradually add TF loss

## Notes

- Always save config.yaml in experiment directory
- Use descriptive experiment names
- Document significant findings in this README
- Archive old experiments regularly
