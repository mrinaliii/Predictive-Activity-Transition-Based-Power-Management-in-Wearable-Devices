"""Evaluation: metrics, confusion matrix, per-class performance."""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from src.model import get_model


def evaluate_model(model, X_test, y_test, label_names, config):
    """Evaluate model on test set and compute metrics."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load best model weights
    saved_model_path = Path(config["paths"]["saved_model"])
    if saved_model_path.exists():
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
    
    # Inference
    model.eval()
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    
    # Per-class metrics
    precisions = precision_score(y_test, predictions, average=None, zero_division=0)
    recalls = recall_score(y_test, predictions, average=None, zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    
    # Build metrics dict
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {}
    }
    
    for i, label in enumerate(label_names):
        metrics["per_class"][label] = {
            "precision": float(precisions[i]) if i < len(precisions) else 0.0,
            "recall": float(recalls[i]) if i < len(recalls) else 0.0,
            "f1": float(f1_score(
                y_test == i, predictions == i, average="binary", zero_division=0
            ))
        }
    
    metrics["confusion_matrix"] = cm.tolist()
    
    # Save metrics
    metrics_path = Path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics
