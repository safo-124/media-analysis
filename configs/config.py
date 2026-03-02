"""
=============================================================================
Configuration for Judo Throws Classification with X3D
=============================================================================

This file centralizes ALL hyperparameters and paths so you never have to
hunt through multiple files to change a setting.

KEY CONCEPTS:
- X3D expects video clips of a fixed number of frames at a specific resolution.
- We use X3D-M by default: 16 frames at 224×224 spatial resolution.
- The model outputs 400 classes (Kinetics-400 pre-trained), and we replace
  the final head with 4 classes for our judo throws.
=============================================================================
"""

import os
import torch

# =============================================================================
# PATHS
# =============================================================================
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CONFIG_DIR)

# Root of the downloaded dataset (contains train/, val/, test/ subfolders)
DATA_DIR = os.path.join(_PROJECT_ROOT, "judo_throws_dataset")

# Where to save model checkpoints, logs, and plots
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")

# =============================================================================
# DATASET SETTINGS
# =============================================================================
# Class names — automatically derived from folder names, but we fix the order
# so that class indices are consistent across runs.
CLASS_NAMES = ["ippon_seoi_nage", "o_goshi", "osoto_gari", "uchi_mata"]
NUM_CLASSES = len(CLASS_NAMES)  # 4

# =============================================================================
# VIDEO PREPROCESSING
# =============================================================================
# Number of frames to sample from each video clip.
# X3D-M was designed for 16-frame input. This is the temporal "window" the
# model sees. More frames = more temporal context but more memory.
NUM_FRAMES = 16

# Sampling rate: take every Nth frame from the raw video.
# If a video is 30fps and SAMPLING_RATE=2, we effectively see 15fps worth
# of motion over 16 frames = ~1 second of action. For judo throws that
# happen in 1-3 seconds, this is a good starting point.
SAMPLING_RATE = 2

# Spatial resolution for X3D-M. The model expects 224×224 input.
# During training we random-crop to this size from a slightly larger frame.
# During evaluation we center-crop.
CROP_SIZE = 224          # Final crop fed to the model
RESIZE_SHORT_SIDE = 256  # Resize shortest side to this before cropping

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
# These values are the ImageNet/Kinetics mean and std used during X3D
# pre-training. We MUST use the same normalization so the pre-trained
# weights make sense on our data.
MEAN = [0.45, 0.45, 0.45]  # Kinetics normalization mean
STD = [0.225, 0.225, 0.225]  # Kinetics normalization std

# Horizontal flip probability during training. Set to 0.5 for standard
# augmentation. NOTE: judo throws can be performed to both sides, so
# flipping is safe here (unlike, say, reading text).
HORIZONTAL_FLIP_PROB = 0.5

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Which X3D variant to use. Options: "x3d_xs", "x3d_s", "x3d_m", "x3d_l"
# x3d_s is used here for memory efficiency on a 6 GB WDDM GPU.
MODEL_NAME = "x3d_s"

# Whether to load Kinetics-400 pre-trained weights.
# ALWAYS True for fine-tuning on small datasets like ours (805 videos).
PRETRAINED = True

# Whether to freeze the backbone and only train the classification head.
# Stage 1: freeze=True (train only head, fast convergence)
# Stage 2: freeze=False (fine-tune everything with lower LR)
FREEZE_BACKBONE = False

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Batch size: how many video clips per gradient update.
# Video models are memory-hungry. X3D-S with batch_size=2 fits on a 6 GB WDDM GPU.
BATCH_SIZE = 2

# Number of complete passes through the training set.
NUM_EPOCHS = 25

# Learning rate: how big of a step we take during optimization.
# For fine-tuning pre-trained models, use a SMALL LR (1e-4 to 1e-3).
# Too high = destroy pre-trained features. Too low = slow convergence.
LEARNING_RATE = 1e-4

# Weight decay: L2 regularization to prevent overfitting.
# Standard value for Adam-based optimizers with pre-trained models.
WEIGHT_DECAY = 1e-4

# Learning rate scheduler: reduce LR when validation loss plateaus.
# 'patience' = how many epochs to wait before reducing.
# 'factor' = multiply LR by this when reducing (0.1 = divide by 10).
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.1

# Early stopping: stop training if val loss hasn't improved for N epochs.
# Prevents wasting time on overfitting.
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# SYSTEM
# =============================================================================
# Automatically use GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of parallel workers for data loading.
# On Windows, multiprocessing can be problematic — use 0 for safety.
NUM_WORKERS = 0

# Random seed for reproducibility.
SEED = 42

# =============================================================================
# LOGGING
# =============================================================================
# How often to print training progress (every N batches).
LOG_INTERVAL = 10
