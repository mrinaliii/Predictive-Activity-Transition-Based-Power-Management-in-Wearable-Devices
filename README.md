# Predictive Activity Transition-Based Power Management

## Project Overview

This project implements a machine learning-based power management system for wearable devices that predicts activity transitions and adapts sensor sampling and wireless transmission rates accordingly. By leveraging temporal sequence models (GRU, LSTM, or CNN-LSTM), the system achieves significant energy savings while maintaining high classification accuracy.

## Problem Statement

Wearable health monitoring devices require continuous sensor sampling and frequent data transmission, consuming substantial battery power. Our approach uses activity prediction to reduce sampling rates and transmission frequency during low-power activities (lying, sitting) while maintaining full monitoring during active states (walking, running). This adaptive strategy can extend device battery life by 20-40%.

## Dataset: PAMAP2

The project supports three major activity recognition datasets:
- **UCI HAR**: 6 activities, 561 features, smartphone sensors
- **PAMAP2**: 6 activities, ~50 IMU features, multiple sensors
- **WISDM**: 6 activities, 3-axis accelerometer, smartphone-based

If real data is not available, the system generates synthetic data matching the dataset structure for demonstration purposes.

## Model Architecture: GRU

The selected model type is **GRU**. Three architectures are available:

1. **GRU**: Gated Recurrent Unit with LayerNorm and dropout for efficient sequence processing
2. **LSTM**: Long Short-Term Memory with cell state tracking for longer temporal dependencies
3. **CNN_LSTM**: Hybrid approach combining convolutional feature extraction with LSTM sequence modeling

All models:
- Take sequences of shape [batch_size, sequence_length, n_features]
- Use the last hidden state for classification
- Support GPU acceleration (CUDA)

## Training Process

- **Optimizer**: Adam with configurable learning rate
- **Loss Function**: Cross-Entropy
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: Patience=5 epochs
- **Batch Size**: 64
- **Epochs**: 50

The training pipeline:
1. Loads raw sensor data (with synthetic fallback)
2. Preprocesses: fills NaN, normalizes, creates sliding windows
3. Extracts engineered features (mean, variance, SMA, magnitude)
4. Trains model with validation monitoring
5. Saves best checkpoint based on validation loss

## Results

### Classification Performance
- **Accuracy**: 98.8%
- **Macro F1**: 0.9871
- **Weighted F1**: 0.9880

### Energy Efficiency
- **Baseline Energy** (no prediction): 43236.0 mJ
- **Proposed Energy** (prediction-guided): 32959.7 mJ
- **Energy Savings**: **23.8%**

The energy model includes:
- Sensor read cost: 0.05 mJ per sample
- BLE transmission: 0.8 mJ per transmission
- CPU inference: 0.02 mJ per window

Adaptive sampling by activity:
- Walking/Running: 100% sampling, 100% transmission
- Sitting/Standing: 40% sampling, 30% transmission
- Lying/Transitions: 20% sampling, 10% transmission

## Graphs

Generated visualizations are saved to `results/graphs/`:

1. **training_curves.png** - Loss and accuracy across training epochs
2. **confusion_matrix.png** - Normalized prediction errors by activity
3. **energy_comparison.png** - Baseline vs proposed system energy with savings %
4. **per_activity_energy.png** - Energy breakdown by predicted activity class

## Folder Structure

```
Embedded System Project/
+-- data/
|   +-- raw/                    # Download raw datasets here
|   +-- processed/              # Preprocessed data (.npy)
+-- src/
|   +-- __init__.py
|   +-- data_loader.py          # Dataset loading with synthetic fallback
|   +-- preprocess.py           # Normalization, splitting, windowing
|   +-- feature_engineering.py  # Statistical & motion features
|   +-- model.py                # GRU, LSTM, CNN_LSTM architectures
|   +-- train.py                # Training loop with early stopping
|   +-- evaluate.py             # Inference and metrics
|   +-- energy_simulation.py    # Baseline vs proposed energy calculation
|   +-- plot_utils.py           # Graph generation
|   +-- utils.py                # README generation
+-- models/
|   +-- saved_models/
|       +-- best_model.pt       # Best checkpoint
|       +-- scaler.pkl          # Fitted StandardScaler
+-- results/
|   +-- graphs/                 # PNG visualizations
|   +-- metrics/                # JSON results
|   +-- logs/                   # Training logs (JSON lines)
+-- notebooks/                  # Jupyter exploration (optional)
+-- config/
|   +-- config.yaml             # Configuration (all hyperparams)
+-- main.py                     # CLI entry point
+-- scaffold.py                 # Setup script (already run)
+-- requirements.txt            # Python dependencies
+-- README.md                   # This file
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Default)
```bash
python main.py --all
```

This will:
- Load or generate dataset
- Preprocess and engineer features
- Train the model
- Evaluate on test set
- Simulate energy consumption
- Generate all graphs
- Update README with results

### 3. Training Only
```bash
python main.py --train
```

Trains model and saves best checkpoint without evaluation.

### 4. Evaluation Only
```bash
python main.py --evaluate
```

Loads saved model and runs evaluation + energy simulation.

### 5. View Configuration
Edit `config/config.yaml` to:
- Switch dataset (UCI_HAR, PAMAP2, WISDM)
- Change model type (GRU, LSTM, CNN_LSTM)
- Adjust hyperparameters (batch_size, learning_rate, epochs)
- Modify paths

## Implementation Notes

- **Zero hardcoded paths**: All paths and hyperparams from config.yaml
- **Synthetic fallback**: Project runs end-to-end without real dataset
- **Device agnostic**: Automatically uses GPU if available, falls back to CPU
- **Importable modules**: No circular dependencies, each file independently usable
- **Robust I/O**: Creates parent directories automatically

## Future Enhancements

- Multi-device ensemble predictions
- Online learning with streaming data
- Activity transition probability modeling
- Firmware deployment optimization
- Real-world battery life validation

---

Generated automatically. Last updated: 2026-03-24
