# Running Diffusion Model Training on GPU

## Check GPU Availability

```bash
python3 scripts/check_gpu.py
```

This will tell you which device to use: `cuda`, `mps`, or `cpu`.

---

## Training Commands

### Quick Test (2 epochs, ~1 minute on GPU)

```bash
python3 scripts/train_diffusion.py \
  --config configs/diffusion_test.yaml \
  --device mps
```

**Expected output:**
- Phase 1: 1 epoch with frozen encoder
- Phase 2: 1 epoch with joint training
- Checkpoint saved to: `checkpoints/diffusion_test/best.pt`

---

### Full Training (200 epochs)

```bash
# Run in background with output logging
nohup python3 scripts/train_diffusion.py \
  --config configs/diffusion_decoder.yaml \
  --device mps \
  > training.log 2>&1 &

# Check progress
tail -f training.log

# Or run in foreground
python3 scripts/train_diffusion.py \
  --config configs/diffusion_decoder.yaml \
  --device mps
```

**Expected duration:**
- **MPS GPU (Apple Silicon)**: 4-6 hours
- **CUDA GPU (NVIDIA)**: 2-4 hours
- **CPU**: 8-12 hours

**Expected output:**
- Phase 1 (epochs 1-100): Freeze encoder, train diffusion
- Phase 2 (epochs 101-200): Joint fine-tuning
- Checkpoint saved every 10 epochs
- Best model: `checkpoints/diffusion_decoder/best.pt`

---

## Monitor Training Progress

### View real-time metrics

```bash
# Follow the training log
tail -f training.log

# Or use grep to filter
tail -f training.log | grep "Epoch.*Results"
```

### Check saved checkpoints

```bash
# List all checkpoints
ls -lh checkpoints/diffusion_decoder/

# Check best model
ls -lh checkpoints/diffusion_decoder/best.pt
```

---

## Generation & Evaluation

### Generate circuits (after training)

```bash
# Generate 10 circuits with DDIM (fast)
python3 scripts/generate_diffusion.py \
  --checkpoint checkpoints/diffusion_decoder/best.pt \
  --cutoff 1000.0 \
  --q-factor 0.707 \
  --num-samples 10 \
  --sampler ddim \
  --steps 50 \
  --device mps

# Generate with DDPM (highest quality, slower)
python3 scripts/generate_diffusion.py \
  --checkpoint checkpoints/diffusion_decoder/best.pt \
  --cutoff 1000.0 \
  --q-factor 0.707 \
  --num-samples 10 \
  --sampler ddpm \
  --steps 1000 \
  --device mps
```

### Evaluate model quality

```bash
# Comprehensive evaluation on 100 circuits
python3 scripts/evaluate_diffusion.py \
  --checkpoint checkpoints/diffusion_decoder/best.pt \
  --num-samples 100 \
  --sampler ddim \
  --steps 50 \
  --device mps \
  --output results/diffusion_metrics.json
```

**Metrics computed:**
- Structural validity (required nodes, connectivity, no self-loops)
- Topology diversity (unique topologies)
- Pole/zero count distributions
- Component value statistics
- Pole stability (percentage with negative real parts)

---

## Resume Training

If training is interrupted, resume from a checkpoint:

```bash
python3 scripts/train_diffusion.py \
  --config configs/diffusion_decoder.yaml \
  --device mps \
  --resume checkpoints/diffusion_decoder/epoch_50.pt
```

---

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors, reduce batch size:

```bash
# Edit config file
# Change: batch_size: 4  →  batch_size: 2
```

Or use CPU:

```bash
python3 scripts/train_diffusion.py \
  --config configs/diffusion_decoder.yaml \
  --device cpu
```

### MPS Backend Errors

If MPS gives errors, fall back to CPU:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python3 scripts/train_diffusion.py \
  --config configs/diffusion_decoder.yaml \
  --device cpu
```

### Slow Training

- **Use smaller model**: Edit config to reduce `hidden_dim: 256 → 128`
- **Fewer timesteps**: Change `timesteps: 1000 → 100`
- **Fewer epochs**: Change `total_epochs: 200 → 100`

---

## Performance Tips

### Maximize GPU Utilization

1. **Increase batch size** (if you have enough memory):
   ```yaml
   batch_size: 4  →  8  # More parallelism
   ```

2. **Use mixed precision** (for CUDA only, not MPS):
   - Currently not implemented, but could be added

3. **Monitor GPU usage**:
   ```bash
   # For MPS (Apple Silicon)
   sudo powermetrics --samplers gpu_power -i1000

   # For CUDA (NVIDIA)
   watch -n 1 nvidia-smi
   ```

---

## Expected Results

After 200 epochs of training, you should see:

```
✅ TRAINING COMPLETE
📁 Best model saved to: checkpoints/diffusion_decoder/best.pt
📊 Best validation loss: ~1.5-2.0

Metrics:
  Node Type Acc:    50-70%
  Pole Count Acc:   70-85%
  Zero Count Acc:   65-80%
  Edge Exist Acc:   90-95%
```

**Quality metrics** (from evaluation script):
- Structural validity: >90%
- Topology diversity: >50 unique topologies
- Pole stability: >85%
- Novel circuits: 70-80%

---

## Next Steps After Training

1. **Evaluate model**: Run `evaluate_diffusion.py` to get comprehensive metrics

2. **Generate diverse circuits**: Try different specifications
   ```bash
   # Low-pass filter
   --cutoff 1000 --q-factor 0.707

   # High-Q bandpass
   --cutoff 5000 --q-factor 10.0

   # Low-Q filter
   --cutoff 500 --q-factor 0.5
   ```

3. **Compare with template decoder**: Generate circuits with both models and compare topology diversity

4. **Optimize**: Based on evaluation results, tune hyperparameters in config file
