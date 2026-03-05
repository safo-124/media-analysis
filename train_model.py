"""
=============================================================================
Train a specific X3D variant — standalone runner with gradient accumulation
=============================================================================

USAGE:
    .\.venv\Scripts\python.exe train_model.py --model x3d_m --batch-size 1 --accum-steps 4

    This trains X3D-M with effective batch size 4 (1 × 4 accumulation steps)
    on a 6 GB GPU that can't fit batch_size=2 for X3D-M.

    Outputs go to: outputs/<model_name>/
=============================================================================
"""

import os
import sys
import argparse
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    SEED, DEVICE, NUM_CLASSES, CLASS_NAMES,
    NUM_FRAMES, CROP_SIZE, MEAN, STD,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.evaluate import full_evaluation


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch_accum(model, dataloader, criterion, optimizer, epoch,
                          num_epochs, accum_steps=1):
    """Train one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]",
                leave=False)

    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        loss = criterion(outputs, labels)
        # Scale loss by accumulation steps so the effective gradient
        # magnitude matches a larger batch
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        # Step optimizer every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        batch_size = videos.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{running_corrects/total_samples:.4f}'
        })

    return running_loss / total_samples, running_corrects / total_samples


def validate(model, dataloader, criterion, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]",
                leave=False)
    with torch.no_grad():
        for videos, labels in pbar:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            batch_size = videos.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += batch_size
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{running_corrects/total_samples:.4f}'
            })
    return running_loss / total_samples, running_corrects / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train an X3D variant")
    parser.add_argument("--model", type=str, default="x3d_m",
                        choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size per GPU step")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch-size × accum-steps)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "outputs", args.model.replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)

    effective_batch = args.batch_size * args.accum_steps

    print("=" * 60)
    print(f"TRAINING {args.model.upper()}")
    print("=" * 60)
    print(f"  Device:           {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    print(f"  Micro batch:      {args.batch_size}")
    print(f"  Accum steps:      {args.accum_steps}")
    print(f"  Effective batch:  {effective_batch}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Output dir:       {output_dir}")
    print()

    # ── Seed ──
    set_seed(SEED)

    # ── Dataset ──
    # Temporarily override BATCH_SIZE in config module
    import configs.config as cfg
    original_batch = cfg.BATCH_SIZE
    cfg.BATCH_SIZE = args.batch_size
    dataloaders = get_dataloaders()
    cfg.BATCH_SIZE = original_batch  # restore

    # ── Model ──
    # Override MODEL_NAME so build_model uses the requested variant
    cfg.MODEL_NAME = args.model
    model = build_model()

    # ── Training setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR
    )

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch_accum(
            model, dataloaders["train"], criterion, optimizer,
            epoch, args.epochs, args.accum_steps
        )
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, epoch, args.epochs
        )
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            ckpt_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'model_name': args.model,
            }, ckpt_path)
            print(f"  * New best! Val Acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n!! Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time/60:.1f} min — Best Val Acc: {best_val_acc:.4f}")

    # ── Load best weights & evaluate ──
    model.load_state_dict(best_model_weights)

    # Override OUTPUT_DIR for evaluation so plots go to the model subfolder
    cfg.OUTPUT_DIR = output_dir
    results = full_evaluation(model, dataloaders["test"], history=history)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']:.1%})")
    print(f"  Outputs:       {output_dir}")


if __name__ == "__main__":
    main()
