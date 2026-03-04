"""
=============================================================================
X3D Model Builder for Judo Throws Classification
=============================================================================

PURPOSE:
    Build an X3D model pre-trained on Kinetics-400 and adapt it for our
    4-class judo throw classification task.

WHAT IS X3D?
    X3D (Expanding 3D) is a family of efficient video classification models
    from Meta Research. Key ideas:

    1. START SMALL: Begin with a tiny 2D image model (X2D, like MobileNet).

    2. EXPAND ONE AXIS AT A TIME: Progressively expand along 6 axes:
       - Temporal duration (how many frames)
       - Frame rate (temporal stride)
       - Spatial resolution (image size)
       - Network width (channels per layer)
       - Bottleneck width (inner dimensions)
       - Network depth (number of layers)

    3. STEP-WISE SEARCH: At each step, expand the axis that gives the best
       accuracy/complexity tradeoff. This is why X3D is so efficient —
       every parameter "earns its place."

    The result: X3D-M achieves 75.6% on Kinetics-400 with only 3.8M
    parameters and 4.7 GFLOPs — that's 4.8x fewer FLOPs than SlowFast!

ARCHITECTURE DETAILS (X3D-M):
    ┌──────────────────────────────────────────────────────┐
    │  Input: (B, 3, 16, 224, 224)                         │
    │         B=batch, 3=RGB, 16=frames, 224×224=spatial   │
    ├──────────────────────────────────────────────────────┤
    │  Stem: Conv3D → BN → ReLU                            │
    │  (Reduces spatial dimensions, extracts low-level      │
    │   features like edges and motion)                    │
    ├──────────────────────────────────────────────────────┤
    │  Stage 1-4: Residual blocks with:                    │
    │    - Depthwise 3D convolutions (efficient!)          │
    │    - Squeeze-and-Excitation (channel attention)      │
    │    - Temporal convolutions (captures motion)          │
    │  Each stage doubles channels & halves spatial dims    │
    ├──────────────────────────────────────────────────────┤
    │  Stage 5: Final conv block                           │
    ├──────────────────────────────────────────────────────┤
    │  Head: Global Average Pool → FC(2048→400)            │
    │  (We replace 400 → 4 for our judo classes)           │
    └──────────────────────────────────────────────────────┘

HOW TRANSFER LEARNING WORKS HERE:
    1. Load X3D pre-trained on Kinetics-400 (400 action classes)
    2. The backbone (Stem + Stages 1-5) has learned general video features:
       - Low-level: edges, textures, optical flow patterns
       - Mid-level: body parts, limb movements
       - High-level: action primitives (grab, pull, rotate)
    3. Replace the final classification layer: 400 → 4 classes
    4. Fine-tune: the backbone features adapt from "general actions"
       to "specific judo throw differences"
=============================================================================
"""

import torch
import torch.nn as nn

# Import our configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs.config as _cfg


def build_model():
    # Read config values dynamically so callers can override them at runtime
    NUM_CLASSES = _cfg.NUM_CLASSES
    MODEL_NAME = _cfg.MODEL_NAME
    PRETRAINED = _cfg.PRETRAINED
    FREEZE_BACKBONE = _cfg.FREEZE_BACKBONE
    DEVICE = _cfg.DEVICE
    """
    Build and configure the X3D model for judo throw classification.

    STEPS:
        1. Load pre-trained X3D from PyTorchVideo's Torch Hub
        2. Identify and replace the classification head
        3. Optionally freeze backbone layers
        4. Move model to the appropriate device (GPU/CPU)

    Returns:
        model (nn.Module): X3D model ready for training
    """
    print(f"\n{'='*60}")
    print(f"Building Model: {MODEL_NAME.upper()}")
    print(f"{'='*60}")

    # =========================================================================
    # STEP 1: Load pre-trained X3D from Torch Hub
    # =========================================================================
    # torch.hub.load downloads the model architecture + weights from
    # Facebook's PyTorchVideo repository. The model comes pre-trained
    # on Kinetics-400 (400 human action classes).
    print(f"Loading {MODEL_NAME} with pretrained={PRETRAINED}...")

    model = torch.hub.load(
        "facebookresearch/pytorchvideo",  # GitHub repo
        model=MODEL_NAME,                  # e.g., "x3d_m"
        pretrained=PRETRAINED,             # Load Kinetics-400 weights
    )

    # =========================================================================
    # STEP 2: Replace the classification head
    # =========================================================================
    # The original model's final layer outputs 400 classes (Kinetics-400).
    # We need to replace it with a 4-class output for our judo throws.
    #
    # X3D's head structure (from PyTorchVideo):
    #   model.blocks[5].proj = Linear(2048, 400)
    #
    # We replace this with a new Linear layer: 2048 → 4
    # The new layer is randomly initialized — it will be trained from scratch
    # while the backbone provides pre-learned features.

    # Find the final projection layer
    # In PyTorchVideo's X3D, the head is at blocks[5]
    head_block = model.blocks[5]

    # The projection layer maps from the feature dimension to num_classes
    in_features = head_block.proj.in_features  # Typically 2048
    print(f"Original head: Linear({in_features}, 400) [Kinetics-400]")

    # Replace with our 4-class head
    head_block.proj = nn.Linear(in_features, NUM_CLASSES)
    print(f"New head:      Linear({in_features}, {NUM_CLASSES}) [Judo throws]")

    # =========================================================================
    # STEP 3: Optionally freeze the backbone
    # =========================================================================
    # FREEZING means setting requires_grad=False on backbone parameters.
    # Frozen layers won't be updated during training.
    #
    # WHY FREEZE?
    #   - Faster training (fewer gradients to compute)
    #   - Less overfitting risk (fewer trainable parameters)
    #   - The pre-trained features are already good for video understanding
    #
    # WHEN TO UNFREEZE?
    #   - After the head has converged (or from the start with low LR)
    #   - When you have enough data to fine-tune without overfitting

    if FREEZE_BACKBONE:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            # Freeze everything EXCEPT the final projection head
            if "blocks.5.proj" not in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_count += param.numel()
        print(f"\nBackbone FROZEN:")
        print(f"  Frozen params:    {frozen_count:,}")
        print(f"  Trainable params: {trainable_count:,}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nAll layers TRAINABLE (full fine-tuning):")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    # =========================================================================
    # STEP 4: Move to device
    # =========================================================================
    model = model.to(DEVICE)
    print(f"Model moved to: {DEVICE}")

    return model


# =============================================================================
# Quick test: run this file to verify model builds correctly
# =============================================================================
if __name__ == "__main__":
    model = build_model()

    # Test with a dummy input matching X3D-M expected shape
    # (batch=1, channels=3, frames=16, height=224, width=224)
    dummy_input = torch.randn(1, 3, 16, 224, 224).to(DEVICE)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output:       {output.detach().cpu().numpy()}")
    print(f"  Predicted class: {output.argmax(dim=1).item()}")
