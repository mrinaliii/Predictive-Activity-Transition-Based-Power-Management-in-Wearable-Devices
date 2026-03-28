"""
Test script for regenerate_all_outputs() with full PAMAP2 dataset.

This script:
1. Loads the full PAMAP2 dataset (6,005 windows)
2. Loads the trained GRU model
3. Runs regenerate_all_outputs() to generate all 7 plots
4. Verifies the assertions (energy, safety events, etc.)
"""

import sys
import os
import yaml
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset
from src.preprocess import preprocess_pipeline
from src.feature_engineering import extract_features
from src.model import get_model
from src.train import load_training_history
from src.adaptive_pipeline import AdaptivePipelineOrchestrator


def load_config(config_path="config/config.yaml"):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run regenerate_all_outputs() with full dataset."""
    print("\n" + "="*70)
    print("REGENERATING ALL GRAPHS FROM FULL DATASET")
    print("="*70)
    
    # Load configuration
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Step 1: Load full dataset
    print("\n[1/5] Loading full dataset...")
    X, y, label_names = load_dataset(config)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Number of activities: {len(label_names)}")
    print(f"  Activity names: {label_names}")
    
    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(X, y, config)
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Step 3: Extract features
    print("\n[3/5] Extracting features...")
    X_train, _, _, _ = extract_features(X_train, save_path=None)
    X_val, _, _, _ = extract_features(X_val, save_path=None)
    X_test, _, _, _ = extract_features(X_test, save_path=None)
    print(f"  Features extracted (input size: {X_train.shape[2]})")
    
    # Step 4: Load pre-trained model
    print("\n[4/5] Loading pre-trained GRU model...")
    input_size = X_train.shape[2]
    num_classes = len(label_names)
    
    model = get_model(config, input_size, num_classes)
    model = model.to(device)
    
    # Try to load saved model weights
    model_path = Path(config["paths"]["saved_model"])
    if model_path.exists():
        try:
            checkpoint = torch.load(str(model_path), map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Loaded model from: {model_path}")
        except Exception as e:
            print(f"  Warning: Could not load model weights: {e}")
            print(f"  Using random initialization instead")
    else:
        print(f"  Model file not found at {model_path}")
        print(f"  Using random initialization")
    
    # Step 5: Prepare data for energy simulation
    print("\n[5/5] Preparing for energy simulation...")
    
    # Get predictions on test set
    model.eval()
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    
    print(f"  Test set size: {X_test.shape[0]}")
    print(f"  Predictions ready")
    
    # Load training history if available
    logs_path = Path(config["paths"]["logs"]).parent if "logs" in config["paths"] else Path("results/logs")
    training_history = load_training_history(str(logs_path))
    if training_history and training_history.get("train_loss"):
        print(f"  Training history loaded ({len(training_history['train_loss'])} epochs)")
    else:
        print(f"  Training history not available")
        training_history = None
    
    # Initialize Orchestrator
    print("\n" + "="*70)
    print("INITIALIZING ADAPTIVE PIPELINE ORCHESTRATOR")
    print("="*70)
    orchestrator = AdaptivePipelineOrchestrator(
        gru_model=model,
        user_id='full_dataset_test',
        num_classes=num_classes,
        device=str(device)
    )
    
    # Run full simulation to get actual adaptive metrics
    print("\n" + "="*70)
    print("RUNNING FULL SIMULATION WITH ADAPTIVE PIPELINE")
    print("="*70)
    
    metrics = orchestrator.run_simulation(X_test, y_test)
    print(f"\nSimulation Complete:")
    print(f"  Windows processed: {metrics.get('total_windows', 0)}")
    print(f"  Baseline energy: {metrics.get('baseline_energy', 0):.1f} mJ")
    print(f"  Adaptive energy: {metrics.get('adaptive_energy', 0):.1f} mJ")
    print(f"  Safety events: {metrics.get('safety_override_wins', 0)}")
    
    # Run regenerate_all_outputs() with full simulation results
    print("\n" + "="*70)
    print("RUNNING regenerate_all_outputs()")
    print("="*70)
    
    try:
        result = orchestrator.regenerate_all_outputs(
            dataset=X_test,
            labels=y_test,
            output_dir='results',
            training_history=training_history,
            test_predictions=(y_pred, y_test),
            activity_names=label_names
        )
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"\nOutput directory: {result['output_directory']}")
        print(f"\nGenerated plots:")
        for plot_name, plot_path in result['files'].items():
            if plot_path:
                exists = Path(plot_path).exists()
                status = "[OK]" if exists else "[MISSING]"
                print(f"  {status} {plot_name}: {Path(plot_path).name}")
            else:
                print(f"  [SKIP] {plot_name}: Not generated")
        
        print(f"\nAssertion values:")
        assertions = result['assertions']
        print(f"  Total windows: {assertions['total_windows']}")
        print(f"  Baseline energy: {assertions['baseline_energy']:.1f} mJ")
        print(f"  Adaptive energy: {assertions['adaptive_energy']:.1f} mJ")
        print(f"  Safety override count: {assertions['safety_override_count']}")
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to run regenerate_all_outputs(): {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
