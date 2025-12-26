#!/usr/bin/env python3
"""
Check GPU availability and recommend device.
"""

import torch
import sys

print("=" * 70)
print("GPU AVAILABILITY CHECK")
print("=" * 70)

# Check CUDA (NVIDIA)
print("\n🔍 NVIDIA CUDA:")
if torch.cuda.is_available():
    print(f"  ✅ Available")
    print(f"  Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"\n  Recommended: --device cuda")
else:
    print(f"  ❌ Not available")

# Check MPS (Apple Silicon)
print("\n🔍 Apple Metal (MPS):")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"  ✅ Available")
    print(f"  Built: {torch.backends.mps.is_built()}")
    print(f"\n  Recommended: --device mps")
else:
    print(f"  ❌ Not available")

# CPU (always available)
print("\n🔍 CPU:")
print(f"  ✅ Always available")
print(f"  Recommended: --device cpu")

# Recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if torch.cuda.is_available():
    device = "cuda"
    reason = "NVIDIA GPU detected (fastest)"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    reason = "Apple Silicon GPU detected (fast)"
else:
    device = "cpu"
    reason = "No GPU available (slowest)"

print(f"\n  Use: --device {device}")
print(f"  Reason: {reason}")

# Performance estimates
print("\n" + "=" * 70)
print("ESTIMATED TRAINING TIME (200 epochs)")
print("=" * 70)

if device == "cuda":
    print(f"  CUDA GPU:  ~2-4 hours  ⚡")
elif device == "mps":
    print(f"  MPS GPU:   ~4-6 hours  🚀")
else:
    print(f"  CPU:       ~8-12 hours ⏱️")

print("\n" + "=" * 70)
