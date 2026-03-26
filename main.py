"""Main entry point: Complete activity recognition pipeline with energy simulation."""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from src.data_loader import load_dataset
from src.preprocess import preprocess_pipeline
from src.feature_engineering import extract_features
from src.model import get_model
from src.train import train_model, load_training_history
from src.evaluate import evaluate_model
from src.energy_simulation import run_energy_simulation
from src.plot_utils import generate_all_graphs
from src.utils import update_readme, print_summary_table
from src.adaptive_pipeline import AdaptivePipelineOrchestrator


def load_config(config_path="config/config.yaml"):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Full pipeline execution."""
    
    # Configuration
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load dataset
    print("\n[1/7] Loading dataset...")
    X, y, label_names = load_dataset(config)
    print(f"  Shape: {X.shape}, Classes: {len(label_names)}")
    
    # Step 2: Preprocess
    print("\n[2/7] Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(X, y, config)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 3: Feature engineering
    print("\n[3/7] Extracting features...")
    X_train, stat_train, sma_train, mag_train = extract_features(
        X_train, save_path=None
    )
    X_val, stat_val, sma_val, mag_val = extract_features(X_val, save_path=None)
    X_test, stat_test, sma_test, mag_test = extract_features(X_test, save_path=None)
    print("  Features extracted (mean, var, std, SMA, magnitude)")
    
    # Step 4: Create and train model
    print("\n[4/7] Creating and training model...")
    input_size = X_train.shape[2]
    num_classes = len(label_names)
    
    model = get_model(config, input_size, num_classes)
    model = model.to(device)
    
    print(f"  Model: {config['model']['type']}")
    print(f"  Input size: {input_size}, Output classes: {num_classes}")
    
    history = None
    if args.train or args.all:
        history = train_model(model, X_train, y_train, X_val, y_val, config)
        print(f"  Best validation accuracy: {max(history['val_acc']):.2f}%")
    else:
        # For evaluate-only mode, load training history from log file
        history = load_training_history(config["paths"]["logs"])
        if history["train_loss"]:
            print(f"  Loaded training history ({len(history['train_loss'])} epochs)")
            print(f"  Best validation accuracy: {max(history['val_acc']):.2f}%")
        else:
            print("  No previous training history found (will show blank training curves)")
    
    # Step 5: Evaluate
    print("\n[5/7] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, label_names, config)
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    
    # Step 6: Energy simulation
    print("\n[6/7] Running energy simulation...")
    # Get predictions on test set for energy simulation
    model.eval()
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    
    energy_results = run_energy_simulation(y_pred, y_test, 
                                          sequence_length=config["dataset"]["sequence_length"],
                                          config=config)
    print(f"  Baseline: {energy_results['baseline_mJ']:.1f} mJ")
    print(f"  Proposed: {energy_results['proposed_mJ']:.1f} mJ")
    print(f"  Savings:  {energy_results['savings_pct']:.1f}%")
    
    # PHASE 2-8: Integrated Predictive-Adaptive Pipeline
    print("\n[6.5/7] Running Integrated Predictive-Adaptive Pipeline...")
    print("  Initializing orchestrator with all 5 components...")
    print("  (Note: Using first 9 channels from 52-dimensional PAMAP2 features)")
    
    orchestrator = AdaptivePipelineOrchestrator(
        gru_model=model,
        user_id="test_user",
        num_classes=num_classes,
        device=device
    )
    
    # Run simulation on test set (orchestrator will extract 9-channel data internally)
    adaptive_metrics = orchestrator.run_simulation(X_test, y_test, duration_per_window_seconds=1.28)
    
    # Print extended comparison table
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON: Baseline vs Adaptive Pipeline")
    print("="*70)
    print(f"\nEnerGY CONSUMPTION (mJ):")
    print(f"  {'Baseline (all sensors @ 100 Hz):':<45} {energy_results['baseline_mJ']:>10.2f}")
    print(f"  {'Adaptive Pipeline (dynamic):':<45} {adaptive_metrics['total_adaptive_energy_mj']:>10.2f}")
    print(f"  {'Energy Saved (mJ):':<45} {energy_results['baseline_mJ'] - adaptive_metrics['total_adaptive_energy_mj']:>10.2f}")
    print(f"  {'Energy Saved (%):':<45} {adaptive_metrics['energy_saved_percent']:>10.1f}")
    
    print(f"\nACCURACY:")
    print(f"  {'Baseline GRU:':<45} {metrics['accuracy']*100:>10.2f} %")
    print(f"  {'Adaptive Pipeline:':<45} {adaptive_metrics['accuracy']*100:>10.2f} %")
    
    print(f"\nEVENTS & TRIGGERS:")
    print(f"  {'Retraining events:':<45} {adaptive_metrics['retraining_events']:>10}")
    print(f"  {'Safety override events:':<45} {adaptive_metrics['override_events']:>10}")
    
    print(f"\nCONFIDENCE TIER DISTRIBUTION:")
    tier_dist = adaptive_metrics['tier_distribution']
    print(f"  {'High (25 Hz) - >=0.85 confidence:':<45} {tier_dist['high_percent']:>10.1f} %")
    print(f"  {'Medium (50 Hz) - 0.50 to 0.85:':<45} {tier_dist['medium_percent']:>10.1f} %")
    print(f"  {'Low (100 Hz) - <0.50 confidence:':<45} {tier_dist['low_percent']:>10.1f} %")
    
    print("\n" + "="*70 + "\n")
    
    # Step 7: Generate graphs and README
    print("[7/7] Generating visualizations and documentation...")
    
    # Use orchestrator energy results instead of basic energy simulation
    # Update energy_results with orchestrator values for accurate graphs
    orchestrator_energy_results = {
        "baseline_mJ": energy_results['baseline_mJ'],
        "proposed_mJ": adaptive_metrics['total_adaptive_energy_mj'],
        "savings_pct": adaptive_metrics['energy_saved_percent'],
        "per_activity_breakdown": adaptive_metrics.get('per_activity_energy', {})
    }
    
    generate_all_graphs(history, metrics, orchestrator_energy_results, label_names, config)
    update_readme(metrics, orchestrator_energy_results, config)
    print("  [OK] README.md updated")
    
    # Print summary with orchestrator results
    print_summary_table(metrics, orchestrator_energy_results, config)
    print("[OK] Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Activity Recognition + Energy Simulation Pipeline"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model only (no evaluation)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate saved model only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline (default)"
    )
    
    args = parser.parse_args()
    
    # Default to --all if no arguments
    if not (args.train or args.evaluate or args.all):
        args.all = True
    
    main(args)
