"""
=============================================================================
Training Engine for Judo Throws Classification
=============================================================================

PURPOSE:
    Orchestrate the full training loop: forward pass, loss computation,
    backpropagation, validation, learning rate scheduling, early stopping,
    and checkpoint saving.

TRAINING LOOP EXPLAINED:
    For each epoch:
      1. TRAINING PHASE:
         - Set model to train mode (enables dropout, BatchNorm updates)
         - For each batch of videos:
           a) Forward pass: video → model → class predictions
           b) Compute loss: CrossEntropyLoss(predictions, true_labels)
           c) Backward pass: compute gradients via backpropagation
           d) Optimizer step: update model weights
         - Track running loss and accuracy

      2. VALIDATION PHASE:
         - Set model to eval mode (disables dropout, uses running BN stats)
         - For each batch: forward pass only (no gradients needed)
         - Compute validation loss and accuracy
         - This tells us if the model generalizes or is overfitting

      3. LEARNING RATE SCHEDULING:
         - If val loss hasn't improved for `patience` epochs, reduce LR
         - This helps the model converge to a better minimum

      4. EARLY STOPPING:
         - If val loss hasn't improved for a longer patience, stop training
         - Prevents wasting compute on an overfitting model

      5. CHECKPOINTING:
         - Save model weights whenever we get the best val accuracy
         - At the end, we load the best checkpoint for evaluation
=============================================================================
"""

import os
import time
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Import our configuration
import sys
sys.path.insert(0, os.path.join("e:", os.sep, "media analysis"))
from configs.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE, OUTPUT_DIR, LOG_INTERVAL,
    CLASS_NAMES
)


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs):
    """
    Train the model for one complete pass through the training data.

    DETAILED STEPS FOR EACH BATCH:
        1. Move data to GPU/CPU
        2. Zero gradients (clear from previous batch)
        3. Forward pass: compute predictions
        4. Compute loss: how wrong are the predictions?
        5. Backward pass: compute gradients (∂loss/∂weights)
        6. Optimizer step: adjust weights in the direction that reduces loss

    Args:
        model: The X3D model
        dataloader: Training DataLoader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Adam optimizer
        epoch: Current epoch number
        num_epochs: Total epochs

    Returns:
        epoch_loss (float): Average loss over all batches
        epoch_acc (float): Accuracy over all samples
    """
    model.train()  # Enable training mode (dropout ON, BatchNorm updates)

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # tqdm gives us a progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]",
                leave=False)

    for batch_idx, (videos, labels) in enumerate(pbar):
        # -----------------------------------------------------------------
        # Step 1: Move data to device (GPU if available)
        # -----------------------------------------------------------------
        # videos shape: (B, C=3, T=16, H=224, W=224)
        # labels shape: (B,) — integer class indices
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        # -----------------------------------------------------------------
        # Step 2: Zero gradients
        # -----------------------------------------------------------------
        # PyTorch ACCUMULATES gradients by default. We must clear them
        # before each batch, otherwise gradients from previous batches
        # would add up and corrupt the update.
        optimizer.zero_grad()

        # -----------------------------------------------------------------
        # Step 3: Forward pass
        # -----------------------------------------------------------------
        # The model processes the video and outputs raw scores (logits)
        # for each class. Shape: (B, NUM_CLASSES) = (B, 4)
        outputs = model(videos)

        # -----------------------------------------------------------------
        # Step 4: Compute loss
        # -----------------------------------------------------------------
        # CrossEntropyLoss combines LogSoftmax + NLLLoss:
        #   loss = -log(softmax(output)[correct_class])
        # It penalizes wrong predictions more when the model is confident.
        loss = criterion(outputs, labels)

        # -----------------------------------------------------------------
        # Step 5: Backward pass (backpropagation)
        # -----------------------------------------------------------------
        # Compute gradients: how should each weight change to reduce loss?
        # This is the chain rule applied through the entire network.
        loss.backward()

        # -----------------------------------------------------------------
        # Step 6: Optimizer step
        # -----------------------------------------------------------------
        # Update weights: w_new = w_old - lr * gradient
        # Adam also uses momentum and adaptive learning rates per parameter.
        optimizer.step()

        # -----------------------------------------------------------------
        # Track metrics
        # -----------------------------------------------------------------
        batch_size = videos.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)  # Get predicted class index
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{running_corrects/total_samples:.4f}'
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, epoch, num_epochs, split="VAL"):
    """
    Evaluate the model on validation or test data (no gradient computation).

    KEY DIFFERENCE from training:
        - model.eval() disables dropout and uses running BatchNorm stats
        - torch.no_grad() skips gradient computation → faster & less memory
        - No optimizer.step() — we're just measuring performance

    Args:
        model: The X3D model
        dataloader: Validation/Test DataLoader
        criterion: Loss function
        epoch: Current epoch number
        num_epochs: Total epochs
        split: Label for the progress bar ("VAL" or "TEST")

    Returns:
        epoch_loss (float): Average loss
        epoch_acc (float): Accuracy
    """
    model.eval()  # Evaluation mode (dropout OFF, BatchNorm frozen)

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [{split}]",
                leave=False)

    # torch.no_grad() tells PyTorch not to track operations for gradient
    # computation. This saves memory and speeds up inference.
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

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train(model, dataloaders):
    """
    Full training pipeline with validation, scheduling, and early stopping.

    OPTIMIZER CHOICE — Adam:
        Adam (Adaptive Moment Estimation) is ideal for fine-tuning because:
        - Per-parameter learning rates adapt automatically
        - Handles sparse gradients well (common in video models)
        - Less sensitive to LR choice than SGD
        We use a small LR (1e-4) to avoid destroying pre-trained features.

    LOSS FUNCTION — CrossEntropyLoss:
        Standard for multi-class classification. Combines softmax + negative
        log likelihood. The model learns to maximize the probability of the
        correct class.

    LR SCHEDULER — ReduceLROnPlateau:
        Monitors validation loss. If it stops improving for `patience` epochs,
        multiply LR by `factor` (0.1 = divide by 10). This gives the model
        a chance to escape local minima with a smaller step size.

    EARLY STOPPING:
        If val loss doesn't improve for `patience` epochs, stop training.
        We restore the best model weights, so we always keep the best
        version even if we stop early.

    Args:
        model: The X3D model (already on device)
        dataloaders: Dict with "train" and "val" DataLoaders

    Returns:
        model: The best model (loaded from checkpoint)
        history: Dict with training metrics for plotting
    """
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Device:          {DEVICE}")
    print(f"  Epochs:          {NUM_EPOCHS}")
    print(f"  Learning Rate:   {LEARNING_RATE}")
    print(f"  Weight Decay:    {WEIGHT_DECAY}")
    print(f"  LR Scheduler:    ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
    print(f"  Early Stopping:  patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Train batches:   {len(dataloaders['train'])}")
    print(f"  Val batches:     {len(dataloaders['val'])}")

    # =========================================================================
    # Setup optimizer, loss, and scheduler
    # =========================================================================

    # CrossEntropyLoss: the standard loss for classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with weight decay (L2 regularization)
    optimizer = Adam(
        # Only optimize parameters that require gradients
        # (if backbone is frozen, this only includes the head)
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # We want to MINIMIZE val loss
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR,
        verbose=True          # Print when LR changes
    )

    # =========================================================================
    # Training state tracking
    # =========================================================================
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    # History for plotting training curves later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- Training Phase ---
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, epoch, NUM_EPOCHS
        )

        # --- Validation Phase ---
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, epoch, NUM_EPOCHS
        )

        # --- Learning Rate Scheduling ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)  # Reduce LR if val_loss plateaus

        # --- Record History ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # --- Epoch Summary ---
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # --- Checkpoint: Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            # Save checkpoint to disk
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
            }, checkpoint_path)
            print(f"  ★ New best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        # --- Early Stopping ---
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    # =========================================================================
    # Final summary
    # =========================================================================
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"  Best Val Acc:  {best_val_acc:.4f}")
    print(f"  Checkpoint:    {os.path.join(OUTPUT_DIR, 'best_model.pth')}")

    # Load the best model weights
    model.load_state_dict(best_model_weights)

    return model, history
