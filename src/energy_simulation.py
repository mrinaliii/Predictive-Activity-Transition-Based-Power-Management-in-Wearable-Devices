"""Energy simulation: Baseline vs Proposed system energy consumption."""

import json
from pathlib import Path
from collections import defaultdict


# Energy constants (in millijoules)
SENSOR_READ_COST = 0.05  # mJ per sample
BLE_TX_COST = 0.8       # mJ per transmission
CPU_COST = 0.02         # mJ per inference

# Activity-based sampling rates (multipliers of baseline)
ACTIVITY_ENERGY_MAP = {
    0: {"name": "CLASS_0", "sampling_rate": 1.0, "ble_rate": 1.0},      # Default: full power
    1: {"name": "CLASS_1", "sampling_rate": 0.4, "ble_rate": 0.3},      # Sitting/Standing
    2: {"name": "CLASS_2", "sampling_rate": 0.2, "ble_rate": 0.1},      # Lying/Transitions
    3: {"name": "CLASS_3", "sampling_rate": 1.0, "ble_rate": 1.0},      # Walking/Running
    4: {"name": "CLASS_4", "sampling_rate": 0.4, "ble_rate": 0.3},
    5: {"name": "CLASS_5", "sampling_rate": 0.2, "ble_rate": 0.1},
}


def run_energy_simulation(y_pred, y_true, sequence_length=128, config=None):
    """Simulate energy consumption for baseline vs proposed system."""
    
    num_windows = len(y_pred)
    
    # Baseline system: Fixed high-rate sampling + constant BLE
    baseline_energy = 0.0
    for window_idx in range(num_windows):
        # Sensor reads for entire window
        baseline_energy += sequence_length * SENSOR_READ_COST
        # Single BLE transmission per window
        baseline_energy += BLE_TX_COST
    
    # Proposed system: Prediction-guided adaptive sampling
    proposed_energy = 0.0
    per_activity_energy = defaultdict(float)
    
    for window_idx in range(num_windows):
        predicted_class = int(y_pred[window_idx])
        
        # Get activity parameters
        activity_info = ACTIVITY_ENERGY_MAP.get(predicted_class, ACTIVITY_ENERGY_MAP[0])
        sampling_rate = activity_info["sampling_rate"]
        ble_rate = activity_info["ble_rate"]
        
        # Adaptive sensor reads
        effective_samples = int(sequence_length * sampling_rate)
        sensor_energy = effective_samples * SENSOR_READ_COST
        
        # Adaptive BLE
        ble_energy = BLE_TX_COST * ble_rate
        
        # Model inference overhead
        inference_energy = CPU_COST
        
        window_energy = sensor_energy + ble_energy + inference_energy
        proposed_energy += window_energy
        per_activity_energy[predicted_class] += window_energy
    
    # Calculate savings
    energy_saved = baseline_energy - proposed_energy
    savings_pct = (energy_saved / baseline_energy * 100) if baseline_energy > 0 else 0.0
    
    # Build results
    results = {
        "baseline_mJ": round(baseline_energy, 4),
        "proposed_mJ": round(proposed_energy, 4),
        "energy_saved_mJ": round(energy_saved, 4),
        "savings_pct": round(savings_pct, 2),
        "per_activity_breakdown": {}
    }
    
    for class_id, energy in per_activity_energy.items():
        activity_name = ACTIVITY_ENERGY_MAP.get(class_id, {}).get("name", f"CLASS_{class_id}")
        results["per_activity_breakdown"][activity_name] = round(energy, 4)
    
    # Save results
    if config:
        energy_results_path = Path(config["paths"]["energy_results"])
        energy_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(energy_results_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results
