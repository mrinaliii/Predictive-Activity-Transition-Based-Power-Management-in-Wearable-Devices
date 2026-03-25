"""Preprocessing: fill missing values, normalize, sliding windows, train/val/test split."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def fill_missing(X):
    """Fill NaN values: forward-fill then zero-fill."""
    # X shape: [N, T, F] or [T, F]
    if len(X.shape) == 3:  # [N, T, F]
        X_filled = X.copy()
        for n in range(X.shape[0]):
            for f in range(X.shape[2]):
                series = X_filled[n, :, f]
                mask = np.isnan(series)
                if mask.any():
                    # Forward fill
                    for t in range(1, len(series)):
                        if np.isnan(series[t]) and not np.isnan(series[t-1]):
                            series[t] = series[t-1]
                    # Zero fill remaining
                    series[np.isnan(series)] = 0
                X_filled[n, :, f] = series
    else:  # [T, F]
        X_filled = X.copy()
        for f in range(X.shape[1]):
            series = X_filled[:, f]
            mask = np.isnan(series)
            if mask.any():
                # Forward fill
                for t in range(1, len(series)):
                    if np.isnan(series[t]) and not np.isnan(series[t-1]):
                        series[t] = series[t-1]
                # Zero fill remaining
                series[np.isnan(series)] = 0
            X_filled[:, f] = series
    
    return X_filled


def normalize(X_train, X_val, X_test, scaler_path=None):
    """Normalize using StandardScaler fitted on train set."""
    # Reshape to [N*T, F] for fitting
    N, T, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(N, T, F)
    
    # Transform val and test
    if X_val is not None:
        N_val, T_val, F_val = X_val.shape
        X_val_flat = X_val.reshape(-1, F)
        X_val_flat = scaler.transform(X_val_flat)
        X_val = X_val_flat.reshape(N_val, T_val, F_val)
    
    if X_test is not None:
        N_test, T_test, F_test = X_test.shape
        X_test_flat = X_test.reshape(-1, F)
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(N_test, T_test, F_test)
    
    # Save scaler
    if scaler_path:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    
    return X_train, X_val, X_test, scaler


def sliding_window(X, y, window_size, step=None):
    """Create sliding windows from sequences with 50% overlap (default)."""
    if step is None:
        step = window_size // 2
    
    # X already in sequence format [N, T, F]
    # Just split strategically
    X_windows, y_windows = [], []
    
    # If X is already windowed correctly, return as-is
    if X.shape[1] == window_size:
        return X, y
    
    # Otherwise create windows from raw data
    for i in range(0, X.shape[0] - window_size + 1, step):
        window = X[i:i + window_size]
        label = y[i + window_size - 1]  # Last timestep label (transition target)
        X_windows.append(window)
        y_windows.append(label)
    
    return np.array(X_windows, dtype=np.float32), np.array(y_windows, dtype=np.int64)


def train_val_test_split(X, y, config):
    """Stratified split into train, val, test."""
    test_split = config["dataset"]["test_split"]
    val_split = config["dataset"]["val_split"]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=42
    )
    
    # Second split: train vs val (from temp set)
    val_split_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split_adjusted, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(X_raw, y_raw, config):
    """Full preprocessing pipeline: fill missing → split → normalize."""
    
    # Fill missing values
    X_filled = fill_missing(X_raw)
    
    # Stratified train/val/test split
    if len(X_filled.shape) == 3:  # Already sequenced [N, T, F]
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X_filled, y_raw, config
        )
    else:
        # If flat [T, F], need to create sequences first
        X_filled, y_raw = sliding_window(X_filled, y_raw, config["dataset"]["sequence_length"])
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X_filled, y_raw, config
        )
    
    # Normalize (fit on train, apply to all)
    scaler_path = Path(config["paths"]["saved_model"]).parent / "scaler.pkl"
    X_train, X_val, X_test, scaler = normalize(
        X_train, X_val, X_test, scaler_path=scaler_path
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
