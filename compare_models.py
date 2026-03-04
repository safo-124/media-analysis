"""
=============================================================================
Compare X3D-S vs X3D-M — side-by-side performance comparison
=============================================================================

USAGE:
    .venv/Scripts/python.exe compare_models.py

    Reads classification reports + training curves from outputs/x3d_s/ and
    outputs/x3d_m/, then produces a combined comparison in outputs/comparison/.
=============================================================================
"""

import os
import sys
import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import CLASS_NAMES, DEVICE, NUM_FRAMES, CROP_SIZE
from src.dataset import get_dataloaders
from src.model import build_model
import configs.config as cfg

MODELS = {
    "x3d_s": os.path.join(PROJECT_ROOT, "outputs", "x3d_s"),
    "x3d_m": os.path.join(PROJECT_ROOT, "outputs", "x3d_m"),
}

COMPARE_DIR = os.path.join(PROJECT_ROOT, "outputs", "comparison")
os.makedirs(COMPARE_DIR, exist_ok=True)

COLORS = {"x3d_s": "#3b82f6", "x3d_m": "#f59e0b"}  # blue, amber
NICE_NAMES = {"x3d_s": "X3D-S", "x3d_m": "X3D-M"}


# ─────────────────────────────────────────────────────────────
# 1.  Parse classification reports
# ─────────────────────────────────────────────────────────────
def parse_report(path):
    """Parse a sklearn-style classification_report.txt into a dict."""
    data = {"classes": {}, "accuracy": 0.0}
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Test Accuracy line
            m = re.match(r"Test Accuracy:\s+([\d.]+)", line)
            if m:
                data["accuracy"] = float(m.group(1))
                continue
            # Per-class line:  Ippon Seoi Nage     0.81    0.84    0.82      25
            parts = re.split(r"\s{2,}", line)
            if len(parts) == 5:
                name, prec, rec, f1, sup = parts
                if name.lower() not in ("accuracy", "macro avg", "weighted avg", ""):
                    data["classes"][name] = {
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                        "support": int(sup),
                    }
            # macro/weighted avg
            if parts[0].lower() in ("macro avg", "weighted avg") and len(parts) == 5:
                data[parts[0].lower().replace(" ", "_")] = {
                    "precision": float(parts[1]),
                    "recall": float(parts[2]),
                    "f1": float(parts[3]),
                }
    return data


# ─────────────────────────────────────────────────────────────
# 2.  Load checkpoint histories
# ─────────────────────────────────────────────────────────────
def load_history(ckpt_path):
    """Try to extract training history from the checkpoint or return None."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("history", None)


# ─────────────────────────────────────────────────────────────
# 3.  Comparison plots
# ─────────────────────────────────────────────────────────────
def plot_accuracy_comparison(reports):
    """Bar chart comparing test accuracy across models."""
    names = [NICE_NAMES[k] for k in reports]
    accs = [reports[k]["accuracy"] * 100 for k in reports]
    colors = [COLORS[k] for k in reports]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("X3D-S vs X3D-M — Test Accuracy")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARE_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved accuracy_comparison.png")


def plot_per_class_f1(reports):
    """Grouped bar chart comparing per-class F1 scores."""
    model_keys = list(reports.keys())
    class_names = list(reports[model_keys[0]]["classes"].keys())
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mk in enumerate(model_keys):
        f1s = [reports[mk]["classes"][c]["f1"] * 100 for c in class_names]
        bars = ax.bar(x + i * width, f1s, width, label=NICE_NAMES[mk],
                      color=COLORS[mk], edgecolor="white")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 — X3D-S vs X3D-M")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARE_DIR, "per_class_f1_comparison.png"), dpi=150)
    plt.close()
    print("  Saved per_class_f1_comparison.png")


def plot_precision_recall(reports):
    """Side-by-side precision vs recall per class."""
    model_keys = list(reports.keys())
    class_names = list(reports[model_keys[0]]["classes"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, metric in zip(axes, ["precision", "recall"]):
        x = np.arange(len(class_names))
        width = 0.35
        for i, mk in enumerate(model_keys):
            vals = [reports[mk]["classes"][c][metric] * 100 for c in class_names]
            ax.bar(x + i * width, vals, width, label=NICE_NAMES[mk],
                   color=COLORS[mk], edgecolor="white")
        ax.set_ylabel(f"{metric.capitalize()} (%)")
        ax.set_title(metric.capitalize())
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(class_names, rotation=15, ha="right")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Precision & Recall — X3D-S vs X3D-M", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARE_DIR, "precision_recall_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved precision_recall_comparison.png")


def write_summary_report(reports):
    """Write a text comparison table."""
    path = os.path.join(COMPARE_DIR, "comparison_report.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON: X3D-S vs X3D-M\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Metric':<25} {'X3D-S':>10} {'X3D-M':>10} {'Diff':>10}\n")
        f.write("-" * 55 + "\n")

        for mk in ["accuracy"]:
            s_val = reports["x3d_s"].get(mk, 0)
            m_val = reports["x3d_m"].get(mk, 0)
            diff = m_val - s_val
            f.write(f"{'Test Accuracy':<25} {s_val:>9.4f} {m_val:>9.4f} {diff:>+9.4f}\n")

        f.write("\n" + "-" * 55 + "\n")
        f.write(f"\n{'Class':<20} {'Metric':<12} {'X3D-S':>8} {'X3D-M':>8} {'Diff':>8}\n")
        f.write("-" * 58 + "\n")

        class_names = list(reports["x3d_s"]["classes"].keys())
        for cls in class_names:
            for metric in ["precision", "recall", "f1"]:
                s = reports["x3d_s"]["classes"][cls][metric]
                m = reports["x3d_m"]["classes"][cls][metric]
                d = m - s
                label = cls if metric == "precision" else ""
                f.write(f"{label:<20} {metric:<12} {s:>7.4f} {m:>7.4f} {d:>+7.4f}\n")
            f.write("\n")

    print(f"  Saved comparison_report.txt")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MODEL COMPARISON: X3D-S vs X3D-M")
    print("=" * 60)

    # Check both outputs exist
    reports = {}
    for name, path in MODELS.items():
        report_file = os.path.join(path, "classification_report.txt")
        if not os.path.exists(report_file):
            print(f"\n⚠ Missing {report_file}")
            print(f"  Train {name} first:  python train_model.py --model {name}")
            return
        reports[name] = parse_report(report_file)
        print(f"\n{NICE_NAMES[name]}: Test Accuracy = {reports[name]['accuracy']:.4f}")

    # Generate comparison outputs
    print(f"\nGenerating comparison plots in {COMPARE_DIR}/ ...")
    plot_accuracy_comparison(reports)
    plot_per_class_f1(reports)
    plot_precision_recall(reports)
    write_summary_report(reports)

    # Print quick summary
    s_acc = reports["x3d_s"]["accuracy"] * 100
    m_acc = reports["x3d_m"]["accuracy"] * 100
    diff = m_acc - s_acc
    winner = "X3D-M" if diff > 0 else "X3D-S" if diff < 0 else "Tie"
    print(f"\n{'='*60}")
    print(f"RESULT:  X3D-S {s_acc:.1f}%  vs  X3D-M {m_acc:.1f}%  →  {winner} wins by {abs(diff):.1f}%")
    print(f"{'='*60}")
    print(f"\nAll comparison files in: {COMPARE_DIR}")


if __name__ == "__main__":
    main()
