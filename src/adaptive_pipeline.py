"""
Adaptive Pipeline Orchestrator: Unified control of all 5 predictive-adaptive components.

Requires: torch, numpy, pandas

Combines TransitionWatchdog, TransitionProbabilityMatrix, ConfidenceController,
SafetyOverride, and RetrainingManager into a single unified decision pipeline.

Processes sensor windows sequentially and tracks energy, accuracy, and system events.
"""

from typing import Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from src.transition_watchdog import TransitionWatchdog, TransitionProbabilityMatrix
from src.sensor_profiles import SensorActivationProfile
from src.confidence_controller import ConfidenceController
from src.safety_override import SafetyOverride
from src.retraining_manager import RetrainingManager


class AdaptivePipelineOrchestrator:
    """
    Master orchestrator for the Integrated Predictive-Adaptive Architecture.
    
    Coordinates all 5 components to make dynamic sampling decisions:
    1. SafetyOverride: Detect critical safety events
    2. TransitionWatchdog: Predict upcoming activity transitions
    3. GRU inference: Classify current activity with confidence
    4. ConfidenceController: Map confidence to sampling tier
    5. RetrainingManager: Fine-tune on flagged low-confidence samples
    
    Tracks energy consumption, retraining events, and override triggers.
    
    Attributes:
        gru_model (nn.Module): Trained GRU activity classifier.
        user_id (str): User identifier.
        sensor_profile (SensorActivationProfile): Sensor configuration reference.
        num_classes (int): Number of activity classes.
        device (str): Device to use ('cpu' or 'cuda').
    """
    
    def __init__(self, 
                 gru_model: nn.Module,
                 user_id: str = "default",
                 num_classes: int = 12,
                 device: str = "cpu"):
        """
        Initialize AdaptivePipelineOrchestrator.
        
        Creates all 5 component instances and sets up tracking.
        
        Args:
            gru_model (nn.Module): Trained GRU or other PyTorch activity classifier.
            user_id (str): User identifier. Default "default".
            num_classes (int): Number of activity classes. Default 12 (PAMAP2).
            device (str): Device to use ('cpu' or 'cuda'). Default 'cpu'.
        """
        self.gru_model = gru_model
        self.user_id = user_id
        self.num_classes = num_classes
        self.device = device
        
        # Initialize all 5 components
        self.safety_override = SafetyOverride(entropy_threshold=0.98)  # Increased from 0.90 to reduce false positives
        self.transition_watchdog = TransitionWatchdog(num_classes=num_classes, device=device).to(device)
        self.transition_matrix = TransitionProbabilityMatrix(num_classes=num_classes, user_id=user_id)
        self.sensor_profile = SensorActivationProfile(num_classes=num_classes)
        self.confidence_controller = ConfidenceController(user_id=user_id, num_classes=num_classes)
        self.retraining_manager = RetrainingManager(gru_model, trigger_threshold=200, 
                                                    user_id=user_id, device=device)
        
        # Tracking variables
        self.window_count = 0
        self.last_activity = None
        self.override_events = []
        self.retraining_events = []
        self.window_decisions = []  # Log each window's decision
    
    def process_window(self, sensor_window: np.ndarray, true_label: Optional[int] = None) -> Dict:
        """
        Process a single sensor window through the entire adaptive pipeline.
        
        Execution flow:
        1. SafetyOverride: Check for critical events (falls, etc.)
        2. TransitionWatchdog: Detect imminent transitions
        3. GRU inference: Get activity probabilities
        4. ConfidenceController: Map confidence to tier
        5. RetrainingManager: Buffer and potentially retrain
        6. Update TransitionProbabilityMatrix
        
        Args:
            sensor_window (np.ndarray): Sensor data of shape [128, 9] or [32, 9].
            true_label (int, optional): Ground-truth label for supervised updates.
        
        Returns:
            Dict with keys:
                'window_id': int
                'override_active': bool
                'override_reason': str or None
                'transition_detected': bool
                'predicted_activity': int
                'confidence': float
                'tier': str ('high', 'medium', 'low')
                'active_axes': list of str
                'sampling_rate': int (Hz)
                'flag_for_retraining': bool
                'energy_mj': float
                'retraining_triggered': bool
                'last_activity': int or None
        
        Example:
            >>> import torch
            >>> model = torch.nn.Linear(9, 12)  # Dummy model (128, 9) -> (12,) not realistic
            >>> orchestrator = AdaptivePipelineOrchestrator(model, num_classes=12)
            >>> window = np.random.randn(128, 9)
            >>> result = orchestrator.process_window(window, true_label=3)
            >>> assert 'sampling_rate' in result
        """
        self.window_count += 1
        window_id = self.window_count
        
        # ===== STEP 1: Safety Override Check =====
        # If input has more than 9 channels, use only first 9 (core sensors)
        safety_window = sensor_window
        if safety_window.shape[1] > 9:
            safety_window = safety_window[:, :9]
        
        safety_result = self.safety_override.check(safety_window)
        if safety_result['override_active']:
            self.override_events.append({
                'window_id': window_id,
                'reason': safety_result['reason'],
                'timestamp': datetime.now()
            })
            
            result = {
                'window_id': window_id,
                'override_active': True,
                'override_reason': safety_result['reason'],
                'transition_detected': False,
                'predicted_activity': self.last_activity,
                'confidence': 1.0,
                'tier': 'high',
                'active_axes': self.sensor_profile.ALL_AXES,
                'sampling_rate': safety_result['recommended_sampling_rate'],
                'flag_for_retraining': False,
                'energy_mj': self.sensor_profile.compute_energy_cost(
                    self.sensor_profile.ALL_AXES,
                    safety_result['recommended_sampling_rate'],
                    1.28
                ),
                'retraining_triggered': False,
                'last_activity': self.last_activity
            }
            self.window_decisions.append(result)
            return result
        
        # ===== STEP 2: Transition Detection =====
        # Use last 32 timesteps if available
        if sensor_window.shape[0] >= 32:
            watchdog_window = sensor_window[-32:, :]  # [32, num_channels]
            # If input has more than 9 channels, use only first 9 (core sensors)
            if watchdog_window.shape[1] > 9:
                watchdog_window = watchdog_window[:, :9]
        
        transition_detected, transition_probs = self.transition_watchdog.predict(watchdog_window)
        probable_targets = None
        if transition_detected and self.last_activity is not None:
            probable_targets = self.transition_matrix.get_probable_targets(self.last_activity, top_k=3)
        
        # ===== STEP 3: GRU Inference =====
        self.gru_model.eval()
        with torch.no_grad():
            # Prepare input: [seq_len, num_channels] -> [1, seq_len, num_channels] -> tensor
            # Use full sensor data (all channels) for GRU which was trained on this
            X_tensor = torch.from_numpy(sensor_window).unsqueeze(0).float().to(self.device)
            logits = self.gru_model(X_tensor)  # [1, num_classes]
            activity_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [num_classes]
        
        # ===== STEP 4: Confidence Controller =====
        predicted_activity = int(np.argmax(activity_probs))
        confidence_decision = self.confidence_controller.decide(activity_probs, predicted_activity)
        
        # ===== STEP 5: Retraining Buffer Management =====
        flag_for_retraining = confidence_decision['flag_for_retraining']
        if flag_for_retraining and true_label is not None:
            self.retraining_manager.add_flagged_sample(sensor_window, true_label)
        
        # ===== STEP 6: Check Retraining Trigger =====
        retraining_triggered = False
        if self.retraining_manager.should_retrain():
            self.retraining_manager.retrain(epochs=3, lr=1e-4)
            retraining_triggered = True
            self.retraining_events.append({
                'window_id': window_id,
                'timestamp': datetime.now()
            })
        
        # ===== STEP 7: Update Transition Matrix =====
        if self.last_activity is not None and self.last_activity != predicted_activity:
            self.transition_matrix.update(self.last_activity, predicted_activity)
        
        # Update accuracy tracking if ground truth available
        if true_label is not None:
            self.confidence_controller.add_prediction(predicted_activity, true_label)
        
        # ===== STEP 8: Compute Energy =====
        active_axes = confidence_decision['active_axes']
        sampling_rate = confidence_decision['sampling_rate']
        duration_seconds = 1.28  # Standard window duration for 128-sample @ ~100Hz
        energy_mj = self.sensor_profile.compute_energy_cost(active_axes, sampling_rate, duration_seconds)
        
        # ===== Build Result Dict =====
        result = {
            'window_id': window_id,
            'override_active': False,
            'override_reason': None,
            'transition_detected': transition_detected,
            'predicted_activity': predicted_activity,
            'confidence': confidence_decision['confidence'],
            'tier': confidence_decision['tier'],
            'active_axes': active_axes,
            'sampling_rate': sampling_rate,
            'flag_for_retraining': flag_for_retraining,
            'energy_mj': energy_mj,
            'retraining_triggered': retraining_triggered,
            'last_activity': self.last_activity
        }
        
        self.window_decisions.append(result)
        self.last_activity = predicted_activity
        
        return result
    
    def run_simulation(self, dataset: np.ndarray, labels: np.ndarray, 
                      duration_per_window_seconds: float = 1.28) -> Dict:
        """
        Run the adaptive pipeline on a full dataset and compute energy metrics.
        
        Processes every window in the dataset, tracks energy consumption, and
        compares against a baseline (all sensors at 100 Hz).
        
        Args:
            dataset (np.ndarray): Full dataset of shape [num_windows, 128, 9].
            labels (np.ndarray): Ground-truth labels of shape [num_windows].
            duration_per_window_seconds (float): Time per window for energy calculation. Default 1.28.
        
        Returns:
            Dict with simulation metrics:
                'total_adaptive_energy_mj': float
                'baseline_energy_mj': float
                'energy_saved_mj': float
                'energy_saved_percent': float
                'accuracy': float
                'num_windows': int
                'override_events': int
                'retraining_events': int
                'tier_distribution': dict
                'window_log': list (each window's decision)
        
        Example:
            >>> dataset = np.random.randn(50, 128, 9)
            >>> labels = np.random.randint(0, 12, 50)
            >>> metrics = orchestrator.run_simulation(dataset, labels)
            >>> print(f"Energy saved: {metrics['energy_saved_percent']:.1f}%")
        """
        num_windows = len(dataset)
        print(f"\n{'='*70}")
        print(f"Running adaptive simulation on {num_windows} windows...")
        print(f"{'='*70}")
        
        total_adaptive_energy = 0.0
        correct_predictions = 0
        tier_counts = defaultdict(int)
        
        for i in range(num_windows):
            sensor_window = dataset[i]
            true_label = labels[i]
            
            result = self.process_window(sensor_window, true_label=true_label)
            
            total_adaptive_energy += result['energy_mj']
            tier_counts[result['tier']] += 1
            
            if result['predicted_activity'] == true_label:
                correct_predictions += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{num_windows} windows...")
        
        # Compute baseline energy (all axes, 100 Hz for all windows)
        baseline_energy = self.sensor_profile.compute_energy_cost(
            self.sensor_profile.ALL_AXES,
            100,
            num_windows * duration_per_window_seconds
        )
        
        # Calculate metrics
        energy_saved = baseline_energy - total_adaptive_energy
        energy_saved_percent = (energy_saved / baseline_energy) * 100 if baseline_energy > 0 else 0
        accuracy = (correct_predictions / num_windows) if num_windows > 0 else 0
        
        # Tier distribution percentages
        tier_distribution = {
            'high_percent': (tier_counts['high'] / num_windows) * 100,
            'medium_percent': (tier_counts['medium'] / num_windows) * 100,
            'low_percent': (tier_counts['low'] / num_windows) * 100
        }
        
        metrics = {
            'total_adaptive_energy_mj': total_adaptive_energy,
            'baseline_energy_mj': baseline_energy,
            'energy_saved_mj': energy_saved,
            'energy_saved_percent': energy_saved_percent,
            'accuracy': accuracy,
            'num_windows': num_windows,
            'override_events': len(self.override_events),
            'retraining_events': len(self.retraining_events),
            'tier_distribution': tier_distribution,
            'window_log': self.window_decisions
        }
        
        # Print summary table
        self._print_summary_table(metrics)
        
        return metrics
    
    def _print_summary_table(self, metrics: Dict) -> None:
        """Print formatted summary table of simulation results."""
        print(f"\n{'='*70}")
        print(f"ADAPTIVE PIPELINE SIMULATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nENERGY METRICS:")
        print(f"  {'Baseline energy (all sensors @ 100 Hz):':<40} {metrics['baseline_energy_mj']:>10.2f} mJ")
        print(f"  {'Adaptive energy (dynamic sampling):':<40} {metrics['total_adaptive_energy_mj']:>10.2f} mJ")
        print(f"  {'Energy saved:':<40} {metrics['energy_saved_mj']:>10.2f} mJ")
        print(f"  {'Energy saved (%):':<40} {metrics['energy_saved_percent']:>10.1f} %")
        
        print(f"\nACCURACY & EVENTS:")
        print(f"  {'Accuracy:':<40} {metrics['accuracy']*100:>10.2f} %")
        print(f"  {'Safety override events:':<40} {metrics['override_events']:>10} ")
        print(f"  {'Retraining events:':<40} {metrics['retraining_events']:>10} ")
        
        tier_dist = metrics['tier_distribution']
        print(f"\nCONFIDENCE TIER DISTRIBUTION:")
        print(f"  {'High confidence (25 Hz):':<40} {tier_dist['high_percent']:>10.1f} %")
        print(f"  {'Medium confidence (50 Hz):':<40} {tier_dist['medium_percent']:>10.1f} %")
        print(f"  {'Low confidence (100 Hz):':<40} {tier_dist['low_percent']:>10.1f} %")
        
        print(f"\n{'='*70}\n")
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"AdaptivePipelineOrchestrator(user='{self.user_id}', "
            f"num_classes={self.num_classes}, "
            f"device='{self.device}', "
            f"windows_processed={self.window_count}, "
            f"overrides={len(self.override_events)}, "
            f"retrainings={len(self.retraining_events)})"
        )
