# File Reorganization Plan

## Current Issues
1. Phase completion docs scattered in root directory (PHASE4_COMPLETE.md, PHASE5_COMPLETE.md, etc.)
2. Checkpoint directories have timestamps - hard to identify (20251217_195419)
3. Config files could use better naming
4. Evaluation results in generic "evaluation_results/" directory
5. No clear separation of documentation vs. code
6. Analysis documents (OPTIMIZATION_ANALYSIS.md) in root

## Proposed Structure

```
Z-GED/
├── docs/                          # All documentation
│   ├── README.md                  # Main project readme
│   ├── PROJECT_STATUS.md          # Current status (moved from root)
│   ├── phases/                    # Phase completion docs
│   │   ├── phase1-3_dataset_models_losses.md
│   │   ├── phase4_training.md     # Renamed from PHASE4_COMPLETE.md
│   │   └── phase5_evaluation.md   # Renamed from PHASE5_COMPLETE.md
│   ├── analysis/                  # Analysis documents
│   │   ├── optimization_analysis.md
│   │   ├── loss_analysis.md
│   │   └── test_organization.md
│   └── guides/                    # User guides
│       ├── training_guide.md
│       ├── evaluation_guide.md
│       └── configuration_guide.md
│
├── configs/                       # Renamed for clarity
│   ├── README.md                  # Config documentation
│   ├── default.yaml               # Renamed from base_config.yaml
│   ├── optimized.yaml             # Renamed from optimized_config.yaml
│   ├── test.yaml                  # Renamed from test_config.yaml
│   └── archived/                  # Old configs
│
├── experiments/                   # Training runs and results
│   ├── exp001_baseline/           # Named experiments instead of timestamps
│   │   ├── config.yaml
│   │   ├── checkpoints/
│   │   │   ├── best.pt
│   │   │   ├── final.pt
│   │   │   └── epoch_*.pt
│   │   ├── logs/
│   │   │   └── training_history.json
│   │   └── evaluation/
│   │       ├── metrics.json
│   │       └── visualizations/
│   ├── exp002_optimized/
│   └── README.md                  # Experiment log
│
├── ml/                            # ML code (keep as-is, well organized)
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   └── utils/
│
├── scripts/                       # Scripts (keep as-is)
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py (future)
│
├── tests/                         # Tests (keep structure)
│   ├── unit/
│   ├── spec_generation/
│   ├── integration/
│   ├── run_tests.py
│   └── README.md
│
├── tools/                         # Utility tools (existing)
│   ├── graph_edit_distance.py
│   ├── circuit_generator.py
│   └── ged_examples.py
│
├── rlc_dataset/                   # Dataset (keep as-is)
│   └── filter_dataset.pkl
│
└── Root files (minimal)
    ├── README.md                  # Quick start guide
    ├── requirements.txt
    ├── setup.py (future)
    └── .gitignore
```

## Reorganization Steps

### Step 1: Create Documentation Structure
```bash
mkdir -p docs/{phases,analysis,guides}
```

### Step 2: Move and Rename Phase Docs
```bash
mv PHASE4_COMPLETE.md docs/phases/phase4_training.md
mv PHASE5_COMPLETE.md docs/phases/phase5_evaluation.md
mv PROJECT_STATUS.md docs/
```

### Step 3: Move Analysis Docs
```bash
mv OPTIMIZATION_ANALYSIS.md docs/analysis/optimization_analysis.md
mv TEST_ORGANIZATION.md docs/analysis/test_organization.md
mv FILE_REORGANIZATION_PLAN.md docs/analysis/file_reorganization_plan.md
```

### Step 4: Reorganize Configs
```bash
# Rename configs
mv configs/base_config.yaml configs/default.yaml
mv configs/optimized_config.yaml configs/optimized.yaml
mv configs/test_config.yaml configs/test.yaml

# Create README
touch configs/README.md
```

### Step 5: Reorganize Experiments
```bash
# Create experiments directory
mkdir -p experiments

# Rename existing checkpoint dirs to experiments
mv checkpoints/test/20251217_195419 experiments/exp001_baseline_2epochs
mv checkpoints/test/20251219_034159 experiments/exp002_optimized_2epochs

# Restructure each experiment
for exp in experiments/exp*; do
    mkdir -p $exp/{checkpoints,logs,evaluation}
    mv $exp/*.pt $exp/checkpoints/ 2>/dev/null || true
    mv $exp/training_history.json $exp/logs/ 2>/dev/null || true
    mv $exp/config.yaml $exp/ 2>/dev/null || true
done

# Create experiment log
touch experiments/README.md
```

### Step 6: Reorganize Evaluation Results
```bash
# Move to corresponding experiments
mv evaluation_results/test_run experiments/exp002_optimized_2epochs/evaluation
```

### Step 7: Create Main README
```bash
# Create comprehensive README.md in root
```

### Step 8: Clean Up Root Directory
```bash
# Remove old scattered files (after backing up)
# Keep only: README.md, requirements.txt, LICENSE (if exists)
```

## File Naming Conventions

### Configs
- `default.yaml` - Production baseline
- `optimized.yaml` - Best performing configuration
- `test.yaml` - Quick 2-epoch test
- `debug.yaml` - Debug settings (verbose logging, small batch)

### Experiments
- Format: `exp###_description/`
- Example: `exp001_baseline/`, `exp002_optimized/`, `exp003_large_model/`
- Each has: `config.yaml`, `checkpoints/`, `logs/`, `evaluation/`

### Documentation
- Use lowercase with underscores: `phase4_training.md`
- Group by category: `docs/phases/`, `docs/analysis/`, `docs/guides/`

### Checkpoints
- `best.pt` - Best validation model
- `final.pt` - Final epoch model
- `epoch_050.pt` - Periodic checkpoint (every 10 epochs)

## Benefits

1. **Clear Documentation**
   - All docs in `docs/` directory
   - Organized by category (phases, analysis, guides)
   - Easy to find and navigate

2. **Experiment Tracking**
   - Named experiments instead of timestamps
   - Self-contained (config + checkpoints + logs + eval)
   - Easy to compare different runs

3. **Better Config Management**
   - Clear naming (default, optimized, test)
   - README documenting each config's purpose
   - Easy to switch between configurations

4. **Cleaner Root**
   - Only essential files (README, requirements)
   - No scattered docs or temp files
   - Professional appearance

5. **Easier Collaboration**
   - Clear structure for new contributors
   - Standard experiment naming
   - Centralized documentation

## Migration Notes

**Backward Compatibility**:
- Old checkpoint paths in evaluation scripts need updating
- Training script checkpoint_dir should point to `experiments/`

**Git Considerations**:
- Add `experiments/` to .gitignore (except README.md)
- Keep only `experiments/README.md` tracked
- Docs should be tracked

## Post-Reorganization Tasks

1. Update README.md with new structure
2. Update training script to use `experiments/` directory
3. Update evaluation script to find checkpoints in experiments
4. Create experiment logging template
5. Add experiment comparison script
6. Update documentation links
