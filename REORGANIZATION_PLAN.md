# Codebase Reorganization Plan

## Current Issues

### 1. Multiple Decoder Versions
- `graphgpt_decoder.py` - Contains `AutoregressiveNodeDecoder` (still used as component)
- `graphgpt_decoder_latent_guided.py` - Current main decoder (used in production)
- `latent_guided_decoder.py` - Helper classes for latent-guided decoder
- `conditional_decoder.py` - **UNUSED** (0 imports found)

### 2. Multiple Encoder Versions
- `encoder.py` - Main encoder (used in production)
- `conditional_encoder.py` - **EXPORTED but NEVER USED**
- `tf_encoder.py` - Transfer function encoder (specialized, used)

### 3. Scattered Scripts (23 files in scripts/)
- Test scripts mixed with analysis scripts
- Multiple similar test scripts (test_comprehensive_specs, test_hybrid_specs, test_single_spec, test_unseen_specs, test_topology_viability)
- No clear organization

### 4. Config Files
- Multiple configs with unclear purposes
- `reduced_overfitting.yaml` - temp experiment config

## Proposed Reorganization

### Phase 1: Archive Unused Models

**Move to `ml/models/archive/`:**
- `conditional_decoder.py` (unused)
- `conditional_encoder.py` (exported but never used)

**Rationale:** Keep codebase clean, but preserve for reference

### Phase 2: Clarify Model Names

**Rename for clarity:**
- `graphgpt_decoder.py` → `node_decoder.py` (since it only contains AutoregressiveNodeDecoder)
- `graphgpt_decoder_latent_guided.py` → `decoder.py` (this is THE production decoder)
- `latent_guided_decoder.py` → `decoder_components.py` (helper classes for main decoder)

**Update imports in:**
- `ml/models/__init__.py`
- `scripts/train.py`
- `scripts/generate_from_specs.py`
- Any other files importing these

### Phase 3: Organize Scripts

**Create script subdirectories:**

```
scripts/
├── training/
│   ├── train.py (main training script)
│   ├── create_stratified_split.py
│   └── validate.py
├── testing/
│   ├── test_comprehensive_specs.py (main test suite)
│   ├── test_single_spec.py
│   └── test_hybrid_specs.py
├── generation/
│   ├── generate_from_specs.py (main generation script)
│   └── extract_circuit_diagrams.py
├── analysis/
│   ├── analyze_overfitting.py
│   ├── analyze_error_sources.py
│   ├── analyze_filter_types.py
│   └── debug_circuit_netlist.py
├── utils/
│   ├── check_gpu.py
│   ├── inspect_dataset.py
│   └── generate_doubled_dataset.py
└── archive/
    └── (existing archived scripts)
```

**Archive redundant test scripts:**
- Move `test_unseen_specs.py` → archive (redundant with test_comprehensive_specs)
- Move `test_topology_viability.py` → archive (one-off analysis)

### Phase 4: Clean Up Configs

**Keep:**
- `latent_guided_decoder.yaml` → rename to `production.yaml` (clearer purpose)
- `optimized_8d.yaml` (documented configuration)
- `test.yaml` (test configuration)

**Remove:**
- `reduced_overfitting.yaml` (temporary experiment, not in use)

### Phase 5: Update Documentation

**Create `ml/models/README.md`:**
```markdown
# Model Architecture

## Production Models

- `encoder.py` - HierarchicalEncoder (8D latent VAE encoder)
- `decoder.py` - LatentGuidedGraphGPTDecoder (main autoregressive decoder)
- `decoder_components.py` - Helper classes (LatentDecomposer, LatentGuidedEdgeDecoder)
- `node_decoder.py` - AutoregressiveNodeDecoder (used by main decoder)

## Specialized Models

- `tf_encoder.py` - Transfer function encoder
- `gnn_layers.py` - Graph neural network layers
- `guided_generation.py` - Guided generation utilities

## Utilities

- `gumbel_softmax_utils.py` - Gumbel-Softmax sampling
- `constants.py` - Filter types and circuit templates

## Archive

Older/experimental models preserved for reference.
```

**Create `scripts/README.md`:**
```markdown
# Scripts Organization

## Training
- `train.py` - Main training script
- `validate.py` - Validation script
- `create_stratified_split.py` - Dataset splitting

## Testing
- `test_comprehensive_specs.py` - Full test suite (USE THIS)
- Other test scripts for specialized scenarios

## Generation
- `generate_from_specs.py` - Generate circuits from specifications

## Analysis
- Various analysis scripts for debugging and insights

## Utils
- Dataset utilities and GPU checking
```

## Implementation Order

1. ✅ Create archive directories
2. ✅ Move unused models to archive
3. ✅ Rename model files (update imports)
4. ✅ Create script subdirectories
5. ✅ Move scripts to appropriate subdirectories
6. ✅ Update imports in moved scripts
7. ✅ Clean up configs
8. ✅ Create README files
9. ✅ Test that everything still works

## Files to Modify

**Models:**
- `ml/models/__init__.py` (update imports)
- Rename 3 model files
- Move 2 models to archive

**Scripts:**
- Create 5 subdirectories
- Move ~20 scripts
- Update imports in moved scripts

**Configs:**
- Rename 1 config
- Remove 1 config

**New files:**
- `ml/models/README.md`
- `scripts/README.md`

## Validation

After reorganization:
```bash
# Test that training still works
python scripts/training/train.py

# Test that generation works
python scripts/generation/generate_from_specs.py

# Run comprehensive tests
python scripts/testing/test_comprehensive_specs.py
```
