"""Plotting utilities: training curves, confusion matrix, energy comparison."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Skip if no training history (evaluate-only mode)
    if not history.get("train_loss") or len(history["train_loss"]) == 0:
        axes[0].text(0.5, 0.5, "No training history\n(evaluate-only mode)", 
                     ha="center", va="center", fontsize=12)
        axes[0].set_title("Training and Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        
        axes[1].text(0.5, 0.5, "No training history\n(evaluate-only mode)", 
                     ha="center", va="center", fontsize=12)
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        
        plt.tight_layout()
        plt.savefig(save_path / "training_curves.png", dpi=100, bbox_inches="tight")
        plt.close()
        return
    
    # Loss
    axes[0].plot(history["train_loss"], marker="o", label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], marker="s", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], marker="o", label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], marker="s", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, label_names, save_path):
    """Plot normalized confusion matrix heatmap."""
    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={"label": "Normalized Count"},
        ax=ax
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix (Normalized)")
    
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_energy_comparison(energy_results, save_path):
    """Bar chart: Baseline vs Proposed energy."""
    baseline = energy_results["baseline_mJ"]
    proposed = energy_results["proposed_mJ"]
    savings = energy_results["savings_pct"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    systems = ["Baseline\n(No Prediction)", "Proposed\n(Prediction-Guided)"]
    energies = [baseline, proposed]
    colors = ["#ff6b6b", "#51cf66"]
    
    bars = ax.bar(systems, energies, color=colors, width=0.6, edgecolor="black", linewidth=1.5)
    
    # Annotate savings
    ax.text(
        0.5, max(energies) * 0.95,
        f"Energy Savings: {savings:.1f}%",
        ha="center", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7)
    )
    
    ax.set_ylabel("Energy Consumption (mJ)", fontsize=12)
    ax.set_title("Energy Consumption: Baseline vs Proposed System", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(energies) * 1.15)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f"{energy:.1f} mJ",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    
    plt.tight_layout()
    plt.savefig(save_path / "energy_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_per_activity_energy(energy_results, save_path):
    """Horizontal bar chart of per-activity energy breakdown."""
    breakdown = energy_results.get("per_activity_breakdown", {})
    
    if not breakdown:
        return  # Skip if no breakdown available
    
    activities = list(breakdown.keys())
    energies = list(breakdown.values())
    
    # Sort by energy
    sorted_data = sorted(zip(activities, energies), key=lambda x: x[1])
    activities, energies = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(activities, energies, color=plt.cm.Set3(np.linspace(0, 1, len(activities))))
    
    ax.set_xlabel("Energy Consumption (mJ)", fontsize=12)
    ax.set_title("Per-Activity Energy Breakdown (Proposed System)", fontsize=13, fontweight="bold")
    
    # Add value labels
    for bar, energy in zip(bars, energies):
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height()/2.,
            f" {energy:.2f} mJ",
            ha="left", va="center", fontsize=10, fontweight="bold"
        )
    
    plt.tight_layout()
    plt.savefig(save_path / "per_activity_energy.png", dpi=100, bbox_inches="tight")
    plt.close()


def generate_all_graphs(history, metrics, energy_results, label_names, config):
    """Generate all 4 plots."""
    graphs_path = Path(config["paths"]["graphs"])
    graphs_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating graphs...")
    
    plot_training_curves(history, graphs_path)
    print("  [OK] training_curves.png")
    
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size > 0:
        plot_confusion_matrix(cm, label_names, graphs_path)
        print("  [OK] confusion_matrix.png")
    
    plot_energy_comparison(energy_results, graphs_path)
    print("  [OK] energy_comparison.png")
    
    plot_per_activity_energy(energy_results, graphs_path)
    print("  [OK] per_activity_energy.png")
