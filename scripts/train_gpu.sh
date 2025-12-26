#!/bin/bash
# Convenience script to train diffusion model on GPU with recommended settings

echo "========================================="
echo "Diffusion Model Training (GPU)"
echo "========================================="
echo ""

# Check GPU availability
echo "Checking GPU availability..."
DEVICE=$(python3 scripts/check_gpu.py | grep "Use: --device" | awk '{print $3}')

if [ -z "$DEVICE" ]; then
    echo "❌ Could not detect device. Defaulting to CPU."
    DEVICE="cpu"
fi

echo "✅ Using device: $DEVICE"
echo ""

# Ask user which config to use
echo "Select training mode:"
echo "  1) Full training (200 epochs, ~4-6 hours on GPU)"
echo "  2) Quick test (2 epochs, ~1 minute on GPU)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        CONFIG="configs/diffusion_decoder.yaml"
        echo "Starting full training..."
        ;;
    2)
        CONFIG="configs/diffusion_test.yaml"
        echo "Starting quick test..."
        ;;
    *)
        echo "Invalid choice. Using quick test."
        CONFIG="configs/diffusion_test.yaml"
        ;;
esac

echo ""
echo "Configuration: $CONFIG"
echo "Device: $DEVICE"
echo ""
read -p "Continue? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

# Run training
echo ""
echo "========================================="
echo "Starting training..."
echo "========================================="
echo ""

python3 scripts/train_diffusion.py \
    --config "$CONFIG" \
    --device "$DEVICE"

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
