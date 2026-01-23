"""
Component type utilities for circuit generation.

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
