"""Feature engineering: extract statistical and motion features from sensor windows."""

import numpy as np
from pathlib import Path


def extract_statistical_features(X):
    """Extract mean, variance, std per window per axis."""
    # X shape: [N, T, F]
    N, T, F = X.shape
    
    # Per-window statistics: [N, 3*F] (mean, var, std for each feature)
    features = []
    
    for n in range(N):
        window = X[n]  # [T, F]
        
        means = np.nanmean(window, axis=0)  # [F,]
        vars_ = np.nanvar(window, axis=0)   # [F,]
        stds = np.nanstd(window, axis=0)    # [F,]
        
        stat_vector = np.concatenate([means, vars_, stds])  # [3*F,]
        features.append(stat_vector)
    
    return np.array(features, dtype=np.float32)  # [N, 3*F]


def extract_sma(X):
    """Signal Magnitude Area: sum(|x|+|y|+|z|) / T per window."""
    # Assumes X has at least 3 features (x, y, z accelerometer)
    N, T, F = X.shape
    
    sma_list = []
    for n in range(N):
        window = X[n]  # [T, F]
        
        # Use first 3 features as x, y, z (if available)
        if F >= 3:
            acc_axes = window[:, :3]  # [T, 3]
            sma = np.nansum(np.abs(acc_axes)) / T
        else:
            # Fallback: use all features
            sma = np.nansum(np.abs(window)) / T
        
        sma_list.append(sma)
    
    return np.array(sma_list, dtype=np.float32).reshape(-1, 1)  # [N, 1]


def extract_acceleration_magnitude(X):
    """Acceleration magnitude: mean(sqrt(x²+y²+z²)) per window."""
    N, T, F = X.shape
    
    mag_list = []
    for n in range(N):
        window = X[n]  # [T, F]
        
        # Use first 3 features for magnitude
        if F >= 3:
            acc_axes = window[:, :3]  # [T, 3]
            mag = np.sqrt(np.nansum(acc_axes ** 2, axis=1))  # [T,]
            mag_mean = np.nanmean(mag)
        else:
            mag_mean = np.nanmean(np.linalg.norm(window, axis=1))
        
        mag_list.append(mag_mean)
    
    return np.array(mag_list, dtype=np.float32).reshape(-1, 1)  # [N, 1]


def extract_features(X, save_path=None):
    """Extract all engineered features and append to X."""
    # X shape: [N, T, F]
    
    # Extract features
    stat_features = extract_statistical_features(X)  # [N, 3*F]
    sma_features = extract_sma(X)                    # [N, 1]
    mag_features = extract_acceleration_magnitude(X) # [N, 1]
    
    # Combine: append engineered features to original sequences
    # Option 1: Append as additional timestep features (broadcast to all timesteps)
    # Option 2: Store separately and concatenate at model input
    
    # We'll flatten and augment: X remains [N, T, F], engineered features stored separately
    X_enhanced = X.copy()  # Keep original structure
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / "X_enhanced.npy", X_enhanced)
        np.save(save_path / "stat_features.npy", stat_features)
        np.save(save_path / "sma_features.npy", sma_features)
        np.save(save_path / "mag_features.npy", mag_features)
    
    return X_enhanced, stat_features, sma_features, mag_features
