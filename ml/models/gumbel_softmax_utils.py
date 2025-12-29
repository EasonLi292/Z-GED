"""
Gumbel-Softmax utilities for component selection.

Component types (8 discrete choices):
- 0: None (no component on this edge)
- 1: R (resistor only)
- 2: C (capacitor only)
- 3: L (inductor only)
- 4: RC (resistor + capacitor parallel/series)
- 5: RL (resistor + inductor)
- 6: CL (capacitor + inductor)
- 7: RCL (all three components)
"""

import torch
import torch.nn.functional as F


# Component type to mask mapping
COMPONENT_TYPE_TO_MASKS = torch.tensor([
    [0, 0, 0],  # 0: None
    [0, 1, 0],  # 1: R only (mask_C=0, mask_G=1, mask_L=0)
    [1, 0, 0],  # 2: C only
    [0, 0, 1],  # 3: L only
    [1, 1, 0],  # 4: RC
    [0, 1, 1],  # 5: RL
    [1, 0, 1],  # 6: CL
    [1, 1, 1],  # 7: RCL
], dtype=torch.float32)


def masks_to_component_type(masks: torch.Tensor) -> torch.Tensor:
    """
    Convert binary masks [mask_C, mask_G, mask_L] to component type index.

    Args:
        masks: [..., 3] binary masks (values should be 0 or 1)

    Returns:
        component_type: [...] integer indices (0-7)
    """
    # Round to ensure binary (handle noisy data)
    masks_binary = (masks > 0.5).float()

    # Encode as unique integer: mask_C * 4 + mask_G * 2 + mask_L * 1
    encoding = (
        masks_binary[..., 0] * 4 +  # mask_C
        masks_binary[..., 1] * 2 +  # mask_G
        masks_binary[..., 2] * 1    # mask_L
    ).long()

    # Map encoding to component type
    # Encoding: C G L
    # 0 0 0 → 0 (None)
    # 0 1 0 → 1 (R) - encoding 2
    # 1 0 0 → 2 (C) - encoding 4
    # 0 0 1 → 3 (L) - encoding 1
    # 1 1 0 → 4 (RC) - encoding 6
    # 0 1 1 → 5 (RL) - encoding 3
    # 1 0 1 → 6 (CL) - encoding 5
    # 1 1 1 → 7 (RCL) - encoding 7

    encoding_to_type = torch.tensor([
        0,  # 000 → None
        3,  # 001 → L
        1,  # 010 → R
        5,  # 011 → RL
        2,  # 100 → C
        6,  # 101 → CL
        4,  # 110 → RC
        7   # 111 → RCL
    ], dtype=torch.long, device=masks.device)

    return encoding_to_type[encoding]


def component_type_to_masks(component_type: torch.Tensor, device=None) -> torch.Tensor:
    """
    Convert component type index to binary masks.

    Args:
        component_type: [...] integer indices (0-7)
        device: Device to create tensor on

    Returns:
        masks: [..., 3] binary masks [mask_C, mask_G, mask_L]
    """
    if device is None:
        device = component_type.device

    mask_table = COMPONENT_TYPE_TO_MASKS.to(device)
    return mask_table[component_type]


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Sample from Gumbel-Softmax distribution.

    During training (hard=False):
        Returns soft probabilities (differentiable)
    During inference (hard=True):
        Returns one-hot vectors (discrete choice)

    Args:
        logits: [..., num_classes] unnormalized log probabilities
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, return one-hot; if False, return soft probabilities

    Returns:
        sample: [..., num_classes] Gumbel-Softmax sample
    """
    # Use PyTorch's built-in Gumbel-Softmax
    sample = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
    return sample


def gumbel_softmax_to_component_type(gumbel_sample: torch.Tensor) -> torch.Tensor:
    """
    Convert Gumbel-Softmax sample to component type index.

    Args:
        gumbel_sample: [..., 8] Gumbel-Softmax sample (soft or hard)

    Returns:
        component_type: [...] integer indices (0-7)
    """
    return torch.argmax(gumbel_sample, dim=-1)


def gumbel_softmax_to_masks(gumbel_sample: torch.Tensor, device=None) -> torch.Tensor:
    """
    Convert Gumbel-Softmax sample to binary masks.

    For soft samples (training), this gives weighted mask probabilities.
    For hard samples (inference), this gives binary masks.

    Args:
        gumbel_sample: [..., 8] Gumbel-Softmax sample
        device: Device to create tensor on

    Returns:
        masks: [..., 3] masks [mask_C, mask_G, mask_L]
    """
    if device is None:
        device = gumbel_sample.device

    mask_table = COMPONENT_TYPE_TO_MASKS.to(device)

    # Matrix multiply: [..., 8] @ [8, 3] = [..., 3]
    masks = torch.matmul(gumbel_sample, mask_table)

    return masks


# Component type names for debugging
COMPONENT_TYPE_NAMES = [
    "None",
    "R",
    "C",
    "L",
    "RC",
    "RL",
    "CL",
    "RCL"
]


def get_component_name(component_type: int) -> str:
    """Get human-readable name for component type."""
    return COMPONENT_TYPE_NAMES[component_type]


if __name__ == '__main__':
    """Test Gumbel-Softmax utilities."""
    print("Testing Gumbel-Softmax Component Selection Utilities\n")

    # Test 1: Masks to component type
    print("Test 1: Masks → Component Type")
    test_masks = torch.tensor([
        [0, 0, 0],  # None
        [0, 1, 0],  # R
        [1, 0, 0],  # C
        [0, 0, 1],  # L
        [1, 1, 0],  # RC
        [0, 1, 1],  # RL
        [1, 0, 1],  # CL
        [1, 1, 1],  # RCL
    ], dtype=torch.float32)

    component_types = masks_to_component_type(test_masks)
    print(f"Masks:\n{test_masks}")
    print(f"Component types: {component_types}")
    print(f"Names: {[get_component_name(t.item()) for t in component_types]}\n")

    # Test 2: Component type to masks
    print("Test 2: Component Type → Masks")
    recovered_masks = component_type_to_masks(component_types)
    print(f"Recovered masks:\n{recovered_masks}")
    print(f"Match: {torch.allclose(test_masks, recovered_masks)}\n")

    # Test 3: Gumbel-Softmax sampling
    print("Test 3: Gumbel-Softmax Sampling")
    logits = torch.randn(5, 8)  # 5 edges, 8 component types

    # Soft sample (training)
    soft_sample = gumbel_softmax_sample(logits, temperature=0.5, hard=False)
    print(f"Soft sample (differentiable):")
    print(f"  Shape: {soft_sample.shape}")
    print(f"  Sum per edge: {soft_sample.sum(dim=-1)}")  # Should be ~1.0

    # Hard sample (inference)
    hard_sample = gumbel_softmax_sample(logits, temperature=0.5, hard=True)
    print(f"\nHard sample (one-hot):")
    print(f"  Shape: {hard_sample.shape}")
    print(f"  Sum per edge: {hard_sample.sum(dim=-1)}")  # Should be exactly 1.0
    print(f"  Sample:\n{hard_sample}")

    # Convert to component types
    component_types = gumbel_softmax_to_component_type(hard_sample)
    print(f"\nComponent types: {component_types}")
    print(f"Names: {[get_component_name(t.item()) for t in component_types]}")

    # Convert to masks
    masks = gumbel_softmax_to_masks(hard_sample)
    print(f"\nMasks:\n{masks}")

    print("\n✅ All tests passed!")
