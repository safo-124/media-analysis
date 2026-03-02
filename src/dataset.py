"""
=============================================================================
Dataset & DataLoader for Judo Throws Video Classification
=============================================================================

PURPOSE:
    Load video files from disk, sample a fixed number of frames, apply
    spatial/temporal augmentations, and serve them as PyTorch tensors
    for training, validation, and testing.

HOW VIDEO LOADING WORKS (step by step):
    1. We scan each split folder (train/val/test) for video files.
    2. Each subfolder name IS the class label (e.g., "ippon_seoi_nage").
    3. For each video, we:
       a) Open it with PyAV (fast C-based video decoder)
       b) Uniformly sample NUM_FRAMES frames spread across the video
       c) Convert frames to a tensor of shape (C, T, H, W)
          - C = 3 color channels (RGB)
          - T = NUM_FRAMES (temporal dimension)
          - H, W = spatial height and width
       d) Apply transforms (resize, crop, normalize, flip)
    4. Return (video_tensor, class_index) pairs to the model.

WHY THIS APPROACH:
    - PyAV is ~3x faster than OpenCV for video decoding
    - Uniform temporal sampling ensures we see the full throw action
      regardless of video length
    - Kinetics-normalization matches X3D's pre-training distribution
=============================================================================
"""

import os
import av
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import our configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_DIR, CLASS_NAMES, NUM_CLASSES, NUM_FRAMES, SAMPLING_RATE,
    CROP_SIZE, RESIZE_SHORT_SIDE, MEAN, STD,
    HORIZONTAL_FLIP_PROB, BATCH_SIZE, NUM_WORKERS
)


def decode_video(video_path, num_frames=NUM_FRAMES):
    """
    Decode a video file and uniformly sample `num_frames` frames.

    UNIFORM SAMPLING STRATEGY:
        If a video has 90 frames and we want 16, we pick frames at
        indices [0, 5, 11, 16, 22, 28, 33, 39, 45, 50, 56, 61, 67, 73, 78, 84]
        This ensures we "see" the full temporal extent of the throw,
        not just the beginning or end.

    Args:
        video_path (str): Full path to the video file.
        num_frames (int): Number of frames to sample.

    Returns:
        frames (np.ndarray): Array of shape (T, H, W, C) with uint8 values.
                             T = num_frames, C = 3 (RGB)
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Collect all frames from the video
    all_frames = []
    for frame in container.decode(video=0):
        # frame.to_ndarray(format="rgb24") converts to H×W×3 numpy array
        all_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total_frames = len(all_frames)

    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Uniformly sample `num_frames` indices across the video
    if total_frames >= num_frames:
        # Spread indices evenly: linspace gives us num_frames values
        # from 0 to total_frames-1, then we cast to int for indexing
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Video is shorter than requested frames — repeat the last frame
        # to pad up to num_frames. This handles very short clips.
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (num_frames - total_frames)
        indices = np.array(indices)

    sampled_frames = np.array([all_frames[i] for i in indices])
    return sampled_frames  # Shape: (T, H, W, C)


class JudoThrowsDataset(Dataset):
    """
    PyTorch Dataset for loading judo throw video clips.

    FOLDER STRUCTURE EXPECTED:
        judo_throws_dataset/
        ├── train/
        │   ├── ippon_seoi_nage/
        │   │   ├── video001.mp4
        │   │   └── ...
        │   ├── o_goshi/
        │   ├── osoto_gari/
        │   └── uchi_mata/
        ├── val/
        └── test/

    Each video is loaded, temporally sampled, spatially transformed,
    and returned as a (tensor, label) pair.

    Args:
        split (str): One of "train", "val", "test".
        transform (callable, optional): Spatial/temporal transforms to apply.
    """

    def __init__(self, split="train", transform=None):
        """
        Scan the split directory and build a list of (video_path, label) pairs.
        """
        self.split = split
        self.transform = transform

        # Build class-to-index mapping: {"ippon_seoi_nage": 0, "o_goshi": 1, ...}
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        # Scan directories for video files
        self.samples = []  # List of (video_path, class_index) tuples
        split_dir = os.path.join(DATA_DIR, split)

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"WARNING: Directory not found: {class_dir}")
                continue

            for filename in sorted(os.listdir(class_dir)):
                # Skip macOS metadata files and hidden files
                if filename.startswith('.'):
                    continue
                # Only include video files
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_dir, filename)
                    label = self.class_to_idx[class_name]
                    self.samples.append((video_path, label))

        print(f"[{split.upper()}] Loaded {len(self.samples)} videos "
              f"across {NUM_CLASSES} classes")

    def __len__(self):
        """Return total number of videos in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and process a single video.

        Returns:
            video (torch.Tensor): Shape (C, T, H, W) — channels first,
                                  ready for the X3D model.
            label (int): Class index (0-3).
        """
        video_path, label = self.samples[idx]

        # Step 1: Decode video and sample frames
        # Result shape: (T, H, W, C) = (16, orig_H, orig_W, 3)
        try:
            frames = decode_video(video_path, NUM_FRAMES)
        except Exception as e:
            print(f"WARNING: Failed to load {video_path}: {e}")
            # Return a black video as fallback for corrupt files
            frames = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

        # Step 2: Convert to tensor and rearrange dimensions
        # From numpy (T, H, W, C) uint8 → torch (T, C, H, W) float32 [0, 1]
        video = torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]
        video = video.permute(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)

        # Step 3: Apply spatial transforms (resize, crop, normalize)
        if self.transform:
            # Apply transform to each frame independently
            # transform expects (C, H, W) input
            video = torch.stack([self.transform(frame) for frame in video])

        # Step 4: Rearrange to (C, T, H, W) — the format X3D expects
        # Currently: (T, C, H, W) → need (C, T, H, W)
        video = video.permute(1, 0, 2, 3)

        return video, label


def get_transforms(split="train"):
    """
    Build the appropriate transform pipeline for each split.

    TRAINING TRANSFORMS:
        1. Resize shortest side to 256 pixels (maintain aspect ratio)
        2. Random crop to 224×224 (data augmentation — model sees different
           regions each epoch)
        3. Random horizontal flip (50% chance — judo throws go both ways)
        4. Normalize with Kinetics mean/std (CRITICAL for pre-trained weights)

    VALIDATION/TEST TRANSFORMS:
        1. Resize shortest side to 256 pixels
        2. Center crop to 224×224 (deterministic — no randomness)
        3. Normalize with Kinetics mean/std

    WHY DIFFERENT TRANSFORMS?
        Training: We want variety (augmentation) to reduce overfitting.
        Val/Test: We want deterministic results for fair comparison.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(RESIZE_SHORT_SIDE, antialias=True),
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(RESIZE_SHORT_SIDE, antialias=True),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


def get_dataloaders():
    """
    Create DataLoaders for train, val, and test splits.

    A DataLoader wraps a Dataset and handles:
        - Batching: groups videos into batches of BATCH_SIZE
        - Shuffling: randomize order each epoch (train only)
        - Multi-processing: load videos in parallel (NUM_WORKERS)
        - Pin memory: faster GPU transfer when using CUDA

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        transform = get_transforms(split)
        dataset = JudoThrowsDataset(split=split, transform=transform)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            # Shuffle training data so the model doesn't memorize order
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            # pin_memory speeds up CPU→GPU transfer
            pin_memory=torch.cuda.is_available(),
            # Drop incomplete last batch during training to keep batch size
            # consistent (important for BatchNorm layers)
            drop_last=(split == "train"),
        )

    return dataloaders


# =============================================================================
# Quick test: run this file directly to verify data loading works
# =============================================================================
if __name__ == "__main__":
    from configs.config import NUM_CLASSES

    print("=" * 60)
    print("Testing Dataset & DataLoader")
    print("=" * 60)

    loaders = get_dataloaders()

    for split, loader in loaders.items():
        batch = next(iter(loader))
        videos, labels = batch
        print(f"\n{split.upper()}:")
        print(f"  Batch video shape: {videos.shape}")
        print(f"  Expected shape:    (B, C=3, T={NUM_FRAMES}, H={CROP_SIZE}, W={CROP_SIZE})")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Label names: {[CLASS_NAMES[l] for l in labels.tolist()]}")
