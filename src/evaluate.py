"""
=============================================================================
Evaluation & Visualization for Judo Throws Classification
=============================================================================

PURPOSE:
    After training, we need to rigorously evaluate the model and generate
    visualizations for the project report. This module provides:

    1. TEST SET EVALUATION  — Final accuracy on held-out test data
    2. CONFUSION MATRIX     — Shows per-class errors (which throws are confused)
    3. CLASSIFICATION REPORT — Precision, Recall, F1-score per class
    4. TRAINING CURVES      — Loss and accuracy over epochs
    5. PER-CLASS ACCURACY   — Bar chart comparing class performance

WHY THESE METRICS MATTER:
    - ACCURACY alone can be misleading (e.g., 75% accuracy could mean one
      class is always wrong)
    - CONFUSION MATRIX reveals which throws the model confuses — this is
      extremely important for judo because some throws look similar
      (e.g., ippon seoi nage vs o goshi both involve hip contact)
    - PRECISION: Of all videos predicted as class X, what fraction was correct?
    - RECALL: Of all actual class X videos, what fraction did we find?
    - F1-SCORE: Harmonic mean of precision and recall (balanced metric)
=============================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
from tqdm import tqdm

# Import our configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DEVICE, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES


def evaluate_model(model, dataloader, split="TEST"):
    """
    Run the model on a dataset and collect all predictions and labels.

    This is similar to the validation function in train.py, but here we
    collect ALL predictions (not just accuracy) for detailed analysis.

    Args:
        model: Trained X3D model
        dataloader: Test DataLoader
        split: Label for display ("TEST" or "VAL")

    Returns:
        all_preds (np.array): Predicted class indices for every sample
        all_labels (np.array): True class indices for every sample
        all_probs (np.array): Softmax probabilities (B, NUM_CLASSES)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc=f"Evaluating [{split}]")

    with torch.no_grad():
        for videos, labels in pbar:
            videos = videos.to(DEVICE)

            # Forward pass: get raw logits
            outputs = model(videos)

            # Convert logits to probabilities with softmax
            probs = torch.softmax(outputs, dim=1)

            # Get predicted class (highest probability)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(all_labels, all_preds, save_path=None):
    """
    Generate and display a confusion matrix heatmap.

    HOW TO READ A CONFUSION MATRIX:
        - Rows = TRUE class (what the video actually is)
        - Columns = PREDICTED class (what the model thinks)
        - Diagonal = CORRECT predictions (higher is better)
        - Off-diagonal = ERRORS (shows which classes get confused)

    EXAMPLE:
        If row "ippon_seoi_nage" has high values in column "o_goshi",
        it means the model frequently mistakes seoi nage for o goshi.
        This makes sense because both throws involve close hip contact.

    Args:
        all_labels: True labels
        all_preds: Predicted labels
        save_path: Where to save the plot (None = just display)
    """
    cm = confusion_matrix(all_labels, all_preds)

    # Use shorter display names for readability
    display_names = [name.replace("_", " ").title() for name in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Add labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=display_names,
           yticklabels=display_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix — Judo Throws Classification')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add count text inside each cell
    thresh = cm.max() / 2.0  # Threshold for text color (light/dark)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy over epochs.

    WHAT TO LOOK FOR:
        - GOOD: Train and val curves are close → model generalizes well
        - BAD: Train acc high but val acc low → OVERFITTING
          (model memorized training data but doesn't generalize)
        - BAD: Both curves plateau at low accuracy → UNDERFITTING
          (model is too simple or learning rate is wrong)
        - GOOD: Val loss decreasing steadily → model is still learning

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Where to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss Plot ---
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy Plot ---
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('X3D-M Judo Throws Classification — Training Curves', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.show()
    plt.close()


def plot_per_class_accuracy(all_labels, all_preds, save_path=None):
    """
    Bar chart showing accuracy for each throw class.

    WHY THIS MATTERS:
        Overall accuracy might be 85%, but if one throw (e.g., o goshi)
        is only 60% while others are 90%+, we know where the model struggles.
        This is actionable information — we could:
        - Collect more data for weak classes
        - Apply class-specific augmentation
        - Use class-weighted loss to penalize mistakes on rare classes

    Args:
        all_labels: True labels
        all_preds: Predicted labels
        save_path: Where to save the plot
    """
    display_names = [name.replace("_", " ").title() for name in CLASS_NAMES]

    per_class_acc = []
    for i in range(NUM_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
        else:
            acc = 0.0
        per_class_acc.append(acc)

    # Color bars by performance (green=good, red=bad)
    colors = ['#2ecc71' if a >= 0.8 else '#e67e22' if a >= 0.6 else '#e74c3c'
              for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(display_names, per_class_acc, color=colors, edgecolor='black', alpha=0.8)

    # Add percentage labels on top of each bar
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy — Judo Throws')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(per_class_acc), color='blue', linestyle='--',
               label=f'Mean: {np.mean(per_class_acc):.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy saved to: {save_path}")

    plt.show()
    plt.close()


def full_evaluation(model, dataloader, history=None):
    """
    Run complete evaluation pipeline and generate all outputs.

    This is the function you call after training to get everything:
    - Test accuracy
    - Confusion matrix
    - Classification report
    - Training curves
    - Per-class accuracy

    Args:
        model: Trained X3D model
        dataloader: Test DataLoader
        history: Training history dict (optional, for plotting curves)

    Returns:
        results: Dict with all metrics
    """
    print(f"\n{'='*60}")
    print("FULL EVALUATION ON TEST SET")
    print(f"{'='*60}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # 1. Get predictions
    # =========================================================================
    all_preds, all_labels, all_probs = evaluate_model(model, dataloader)

    # =========================================================================
    # 2. Overall accuracy
    # =========================================================================
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n>> TEST ACCURACY: {test_acc:.4f} ({test_acc:.1%})")

    # =========================================================================
    # 3. Classification report (precision, recall, F1)
    # =========================================================================
    display_names = [name.replace("_", " ").title() for name in CLASS_NAMES]
    report = classification_report(all_labels, all_preds,
                                   target_names=display_names, digits=4)
    print(f"\nClassification Report:")
    print(report)

    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # =========================================================================
    # 4. Confusion matrix
    # =========================================================================
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # =========================================================================
    # 5. Per-class accuracy
    # =========================================================================
    plot_per_class_accuracy(
        all_labels, all_preds,
        save_path=os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    )

    # =========================================================================
    # 6. Training curves (if history provided)
    # =========================================================================
    if history:
        plot_training_curves(
            history,
            save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
        )

    # =========================================================================
    # Compile results
    # =========================================================================
    results = {
        'test_accuracy': test_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
    }

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}")

    return results
