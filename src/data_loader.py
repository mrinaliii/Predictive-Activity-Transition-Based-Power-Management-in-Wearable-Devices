"""Data loader: Support UCI_HAR, PAMAP2, WISDM datasets with synthetic fallback."""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_uci_har(raw_path):
    """Load UCI Human Activity Recognition dataset."""
    raw_path = Path(raw_path) / "UCI_HAR"
    
    # Expected files
    x_train_path = raw_path / "X_train.txt"
    y_train_path = raw_path / "y_train.txt"
    x_test_path = raw_path / "X_test.txt"
    y_test_path = raw_path / "y_test.txt"
    
    # Check if files exist
    if not all([x_train_path.exists(), y_train_path.exists(), 
                x_test_path.exists(), y_test_path.exists()]):
        return None
    
    # Load data
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path, dtype=int) - 1  # Convert to 0-indexed
    X_test = np.loadtxt(x_test_path)
    y_test = np.loadtxt(y_test_path, dtype=int) - 1
    
    # Concatenate
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    label_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                   "SITTING", "STANDING", "LAYING"]
    
    return X, y, label_names


def load_pamap2(raw_path):
    """Load PAMAP2 Physical Activity Monitoring dataset."""
    raw_path = Path(raw_path) / "PAMAP2"
    
    # Expected subject files
    subject_files = [f"subject10{i}.dat" for i in range(1, 10)]  # subject101-109
    files_exist = all((raw_path / f).exists() for f in subject_files)
    
    if not files_exist:
        return None
    
    X_list, y_list = [], []
    
    # PAMAP2 activity mapping (from protocol documentation)
    # Activity IDs: 1=lying, 2=sitting, 3=standing, 4=walking, 5=running, 6=cycling,
    # 7=Nordic Walking, 12=watching TV, 13=computer work, 16=car driving, 17=ascending stairs, 
    # 18=descending stairs, 19=vacuum cleaning, 20=ironing, 24=rope jumping
    label_map = {
        1: "LYING", 2: "SITTING", 3: "STANDING", 4: "WALKING", 5: "RUNNING", 
        6: "CYCLING", 7: "NORDIC_WALK", 12: "TV", 13: "COMPUTER", 16: "CAR",
        17: "STAIRS_UP", 18: "STAIRS_DOWN", 19: "VACUUM", 20: "IRONING", 24: "ROPE_JUMP"
    }
    
    for subject_file in subject_files:
        filepath = raw_path / subject_file
        print(f"    Loading {subject_file}...", end=" ", flush=True)
        
        # Read with optimized parameters for speed
        df = pd.read_csv(filepath, delimiter=" ", header=None, dtype={1: int})
        
        # PAMAP2: columns 0=timestamp, 1=activityID, 2+=IMU data
        # Drop rows with >30% NaN
        df = df.dropna(thresh=df.shape[1] * 0.7)
        
        # Extract activity label (column 1, 0-indexed)
        y = df.iloc[:, 1].values.astype(int)
        
        # Filter to only known activities
        mask = np.isin(y, list(label_map.keys()))
        if mask.sum() == 0:
            print("skipped (no valid activities)")
            continue
        
        y = y[mask]
        
        # Extract IMU features (skip timestamp and activityID)
        X = df.iloc[:, 2:].values.astype(np.float32)[mask]
        
        X_list.append(X)
        y_list.append(y)
        print(f"[OK] {len(X)} samples")
    
    # Concatenate all subjects
    if not X_list:
        return None
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Remap activity IDs to 0-indexed classes
    unique_activities = sorted(np.unique(y))
    activity_to_class = {act_id: idx for idx, act_id in enumerate(unique_activities)}
    y = np.array([activity_to_class[act_id] for act_id in y])
    
    label_names = [label_map[act_id] for act_id in unique_activities]
    
    return X, y, label_names


def load_wisdm(raw_path):
    """Load WISDM Activity Recognition dataset."""
    raw_path = Path(raw_path) / "WISDM"
    
    data_file = raw_path / "WISDM_at_v1.1_raw.txt"
    if not data_file.exists():
        return None
    
    df = pd.read_csv(data_file, header=None, names=["user", "activity", "timestamp", "x", "y", "z"])
    df = df.dropna()
    
    label_map = {"Walking": 0, "Jogging": 1, "Upstairs": 2, "Downstairs": 3, 
                 "Sitting": 4, "Standing": 5}
    label_names = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
    
    # Filter only recognized activities
    df = df[df["activity"].isin(label_map.keys())]
    df["activity_code"] = df["activity"].map(label_map)
    
    X = df[["x", "y", "z"]].values.astype(float)
    y = df["activity_code"].values
    
    if len(X) == 0:
        return None
    
    return X, y, label_names


def generate_synthetic_data(dataset_name, sequence_length=128, n_windows=500):
    """Generate synthetic data matching real dataset structure."""
    
    config_map = {
        "UCI_HAR": {"n_features": 561, "n_classes": 6},
        "PAMAP2": {"n_features": 52, "n_classes": 6},
        "WISDM": {"n_features": 3, "n_classes": 6}
    }
    
    dataset_info = config_map.get(dataset_name, {"n_features": 561, "n_classes": 6})
    n_features = dataset_info["n_features"]
    n_classes = dataset_info["n_classes"]
    
    # Generate random sensor data normalized to [-1, 1]
    X = np.random.randn(n_windows, sequence_length, n_features).astype(np.float32) * 0.5
    
    # Generate labels with some class balance
    y = np.random.randint(0, n_classes, n_windows).astype(np.int64)
    
    label_names = ["CLASS_0", "CLASS_1", "CLASS_2", "CLASS_3", "CLASS_4", "CLASS_5"][:n_classes]
    
    return X, y, label_names


def reshape_to_sequences(X, y, sequence_length, step=None):
    """Create sliding windows from flat time series."""
    if step is None:
        step = sequence_length // 2  # 50% overlap
    
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - sequence_length + 1, step):
        window = X[i:i + sequence_length]
        # Label = last timestep label (transition target)
        label = y[i + sequence_length - 1]
        
        X_seq.append(window)
        y_seq.append(label)
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def load_dataset(config):
    """Load dataset (UCI_HAR, PAMAP2, or WISDM) with synthetic fallback."""
    
    dataset_name = config["dataset"]["name"]
    raw_path = config["dataset"]["raw_path"]
    sequence_length = config["dataset"]["sequence_length"]
    
    # Try loading real data
    if dataset_name == "UCI_HAR":
        result = load_uci_har(raw_path)
    elif dataset_name == "PAMAP2":
        result = load_pamap2(raw_path)
    elif dataset_name == "WISDM":
        result = load_wisdm(raw_path)
    else:
        result = None
    
    # Fallback to synthetic if real data not found
    if result is None:
        print("[WARNING] Real data not found - using synthetic data for demo.")
        X_flat, y_flat, label_names = generate_synthetic_data(dataset_name, sequence_length)
        X, y = X_flat, y_flat
    else:
        X_flat, y_flat, label_names = result
        # Reshape flat time series to sequences (only if not already in sequence format)
        if len(X_flat.shape) == 2:  # [total_samples, features]
            X, y = reshape_to_sequences(X_flat, y_flat, sequence_length)
        else:  # already [N, T, F]
            X, y = X_flat, y_flat
    
    return X, y, label_names
