"""
=============================================================================
MAIN SCRIPT — Judo Throws Classification with X3D
=============================================================================

DATA.ML.330 Media Analysis — Group 4
Project: Deep Learning-Based Judo Throws Classification
Authors: Alexander Gradov & Oluwaseun Akangbe (Samuel)

USAGE:
    Activate your virtual environment first, then run:
        python main.py

    This script executes the full pipeline:
        1. Set random seeds for reproducibility
        2. Load and preprocess the video dataset
        3. Build and configure the X3D model
        4. Train the model with validation monitoring
        5. Evaluate on the held-out test set
        6. Generate all plots and reports

WHAT HAPPENS BEHIND THE SCENES:
    ┌──────────────────────────────────────────────────────────┐
    │  Video Files (.mp4)                                      │
    │  ├── train/ippon_seoi_nage/vid001.mp4                    │
    │  ├── train/o_goshi/vid002.mp4                            │
    │  └── ...                                                 │
    └────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Dataset (src/dataset.py)                                │
    │  • Decode video with PyAV                                │
    │  • Sample 16 frames uniformly                            │
    │  • Resize → Crop → Normalize                             │
    │  • Output: tensor (3, 16, 224, 224)                      │
    └────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │  X3D Model (src/model.py)                                │
    │  • Pre-trained X3D-M backbone (Kinetics-400)             │
    │  • New classification head: 2048 → 4 classes             │
    │  • Input: (B, 3, 16, 224, 224)                           │
    │  • Output: (B, 4) logits                                 │
    └────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Training Loop (src/train.py)                            │
    │  • Loss: CrossEntropyLoss                                │
    │  • Optimizer: Adam (lr=1e-4)                             │
    │  • Scheduler: ReduceLROnPlateau                          │
    │  • Early stopping: patience=10                           │
    │  • Best model checkpoint saved                           │
    └────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Evaluation (src/evaluate.py)                            │
    │  • Test accuracy                                         │
    │  • Confusion matrix                                      │
    │  • Per-class precision, recall, F1                        │
    │  • Training curves plot                                  │
    │  • Per-class accuracy bar chart                           │
    └──────────────────────────────────────────────────────────┘
=============================================================================
"""

import os
import sys
import time
import random
import numpy as np
import torch

# Ensure our project root is in the Python path
PROJECT_ROOT = os.path.join("e:", os.sep, "media analysis")
sys.path.insert(0, PROJECT_ROOT)

# Import our modules
from configs.config import (
    SEED, DEVICE, OUTPUT_DIR, NUM_CLASSES, CLASS_NAMES,
    MODEL_NAME, NUM_FRAMES, CROP_SIZE, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import train
from src.evaluate import full_evaluation


def set_seed(seed=SEED):
    """
    Set random seeds everywhere for reproducibility.

    WHY THIS MATTERS:
        Deep learning involves many random operations:
        - Weight initialization
        - Data shuffling order
        - Random cropping/flipping during augmentation
        - Dropout mask selection

        By fixing all seeds, you get the EXACT SAME results every run.
        This is crucial for scientific reproducibility in your report —
        you can say "we ran the experiment 3 times with seeds 42, 123, 456
        and report mean ± std accuracy."

    Args:
        seed (int): The random seed to use everywhere
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic (slightly slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def main():
    """
    Execute the full pipeline: data → model → train → evaluate.
    """
    print("=" * 60)
    print("JUDO THROWS CLASSIFICATION WITH X3D")
    print("DATA.ML.330 Media Analysis — Group 4")
    print("=" * 60)
    print(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', None)
        if total_mem:
            print(f"GPU Memory: {total_mem / 1e9:.1f} GB")

    # =========================================================================
    # Step 1: Set seeds for reproducibility
    # =========================================================================
    print(f"\n{'─'*60}")
    print("STEP 1: Setting random seeds")
    print(f"{'─'*60}")
    set_seed(SEED)

    # =========================================================================
    # Step 2: Load dataset
    # =========================================================================
    print(f"\n{'─'*60}")
    print("STEP 2: Loading dataset")
    print(f"{'─'*60}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Frames per clip: {NUM_FRAMES}")
    print(f"  Spatial size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")

    dataloaders = get_dataloaders()

    # =========================================================================
    # Step 3: Build model
    # =========================================================================
    print(f"\n{'─'*60}")
    print("STEP 3: Building X3D model")
    print(f"{'─'*60}")
    print(f"  Architecture: {MODEL_NAME}")
    print(f"  Output classes: {NUM_CLASSES}")

    model = build_model()

    # =========================================================================
    # Step 4: Train
    # =========================================================================
    print(f"\n{'─'*60}")
    print("STEP 4: Training")
    print(f"{'─'*60}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    model, history = train(model, dataloaders)

    # =========================================================================
    # Step 5: Evaluate on test set
    # =========================================================================
    print(f"\n{'─'*60}")
    print("STEP 5: Evaluating on test set")
    print(f"{'─'*60}")

    results = full_evaluation(model, dataloaders["test"], history=history)

    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']:.1%})")
    print(f"\n  Output files saved to: {OUTPUT_DIR}")
    print(f"  ├── best_model.pth           (trained model weights)")
    print(f"  ├── confusion_matrix.png     (confusion matrix plot)")
    print(f"  ├── per_class_accuracy.png   (per-class accuracy chart)")
    print(f"  ├── training_curves.png      (loss & accuracy curves)")
    print(f"  └── classification_report.txt (precision/recall/F1)")


if __name__ == "__main__":
    main()
