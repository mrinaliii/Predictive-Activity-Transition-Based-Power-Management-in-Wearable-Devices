"""
Confidence-Aware Controller: Adaptive sampling based on prediction confidence.
Requires: numpy

Dependencies: src.sensor_profiles.SensorActivationProfile
Dynamically adjusts sensor sampling rates and active axes based on model confidence
and accuracy performance. Flags uncertain predictions for potential retraining.
"""

from collections import deque
from typing import Dict
import numpy as np
from src.sensor_profiles import SensorActivationProfile


class ConfidenceController:
    """
    Controls sensor sampling parameters based on prediction confidence and accuracy.
    
    Maps model confidence scores to sensor activation tiers (high/medium/low) and
    adjusts thresholds dynamically based on system accuracy. Tracks predictions
    for online accuracy estimation.
    
    Attributes:
        high_threshold (float): Confidence threshold for 'high' tier (default 0.85).
        low_threshold (float): Confidence threshold for 'low' tier (default 0.50).
        user_id (str): User identifier for per-user tracking.
        sensor_profile (SensorActivationProfile): Reference to sensor configuration.
    """
    
    def __init__(self, high_threshold: float = 0.85, low_threshold: float = 0.50, 
                 user_id: str = "default", num_classes: int = 12):
        """
        Initialize ConfidenceController.
        
        Args:
            high_threshold (float): Confidence threshold for 'high' confidence tier.
                                   Predictions >= this value use minimal sensors.
                                   Default 0.85.
            low_threshold (float): Confidence threshold for 'low' confidence tier.
                                  Predictions < this value use all sensors.
                                  Default 0.50.
            user_id (str): User identifier. Default "default".
            num_classes (int): Number of activity classes (default 12 for PAMAP2).
        
        Raises:
            ValueError: If thresholds are not in valid range: low_threshold < high_threshold.
        """
        if not (low_threshold < high_threshold):
            raise ValueError(f"low_threshold ({low_threshold}) must be < high_threshold ({high_threshold})")
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.user_id = user_id
        self.num_classes = num_classes
        
        # Reference to sensor profiles for axis and rate lookup
        self.sensor_profile = SensorActivationProfile(num_classes)
        
        # Rolling window of recent predictions and their correctness
        # Each item: (predicted_activity, actual_activity, is_correct)
        self.prediction_history = deque(maxlen=100)
    
    def decide(self, activity_probs: np.ndarray, current_activity: int) -> Dict:
        """
        Decide sensor configuration based on prediction confidence.
        
        Classifies prediction into confidence tier (high/medium/low) and returns
        corresponding sensor axes, sampling rate, and retraining flag.
        
        Args:
            activity_probs (np.ndarray): Probability distribution over activities,
                                        shape [num_classes].
            current_activity (int): Current/assumed activity ID (for output context).
        
        Returns:
            Dict with keys:
                'tier': str - One of ['high', 'medium', 'low']
                'confidence': float - Maximum probability (in range [0, 1])
                'predicted_activity': int - Activity with highest probability
                'active_axes': list - Sensor axes to activate
                'sampling_rate': int - Sampling rate in Hz
                'flag_for_retraining': bool - True if tier=='low' (low confidence)
        
        Raises:
            ValueError: If activity_probs shape is incorrect.
        
        Example:
            >>> controller = ConfidenceController()
            >>> probs = np.random.dirichlet(np.ones(12))  # Random probabilities
            >>> decision = controller.decide(probs, current_activity=3)
            >>> print(f"Tier: {decision['tier']}, Rate: {decision['sampling_rate']} Hz")
        """
        if activity_probs.shape != (self.num_classes,):
            raise ValueError(f"activity_probs must have shape ({self.num_classes},), "
                           f"got {activity_probs.shape}")
        
        # Determine confidence and predicted activity
        confidence = float(np.max(activity_probs))
        predicted_activity = int(np.argmax(activity_probs))
        
        # Classify into tier based on confidence thresholds
        if confidence >= self.high_threshold:
            tier = 'high'
        elif confidence >= self.low_threshold:
            tier = 'medium'
        else:
            tier = 'low'
        
        # Retrieve sensor configuration for this activity and confidence tier
        active_axes = self.sensor_profile.get_active_axes(predicted_activity, tier)
        sampling_rate = self.sensor_profile.get_sampling_rate(predicted_activity, tier)
        
        # Flag for retraining if confidence is very low
        flag_for_retraining = (tier == 'low')
        
        decision = {
            'tier': tier,
            'confidence': confidence,
            'predicted_activity': predicted_activity,
            'active_axes': active_axes,
            'sampling_rate': sampling_rate,
            'flag_for_retraining': flag_for_retraining
        }
        
        return decision
    
    def update_thresholds(self, recent_accuracy: float) -> None:
        """
        Dynamically adjust confidence thresholds based on model accuracy.
        
        Strategy:
        - High accuracy (>95%): Tighten thresholds (increase by 0.02) to conservatively
                                classify most predictions as 'low' confidence (use more sensors)
        - Low accuracy (<85%):  Loosen thresholds (decrease by 0.02) to liberally
                                classify predictions as 'high' confidence (use fewer sensors)
        - Otherwise: No change
        
        Thresholds are clamped to safe ranges:
        - high_threshold: [0.70, 0.95]
        - low_threshold: [0.40, 0.70]
        
        Args:
            recent_accuracy (float): Recent accuracy metric (0.0 to 1.0).
        
        Example:
            >>> controller = ConfidenceController()
            >>> controller.update_thresholds(0.96)  # High accuracy
            >>> print(f"New high_threshold: {controller.high_threshold}")  # ~0.87
        """
        old_high = self.high_threshold
        old_low = self.low_threshold
        
        if recent_accuracy > 0.95:
            # High accuracy: be more conservative, tighten thresholds
            self.high_threshold += 0.02
            self.low_threshold += 0.02
        elif recent_accuracy < 0.85:
            # Low accuracy: be more liberal, loosen thresholds
            self.high_threshold -= 0.02
            self.low_threshold -= 0.02
        
        # Clamp to valid ranges
        self.high_threshold = np.clip(self.high_threshold, 0.70, 0.95)
        self.low_threshold = np.clip(self.low_threshold, 0.40, 0.70)
        
        # Ensure invariant: low < high
        if self.low_threshold >= self.high_threshold:
            self.low_threshold = self.high_threshold - 0.10
            self.low_threshold = np.clip(self.low_threshold, 0.40, 0.70)
        
        if (self.high_threshold != old_high) or (self.low_threshold != old_low):
            print(f"Thresholds updated for user '{self.user_id}': "
                  f"high {old_high:.3f}->{self.high_threshold:.3f}, "
                  f"low {old_low:.3f}->{self.low_threshold:.3f}")
    
    def add_prediction(self, predicted_activity: int, actual_activity: int) -> None:
        """
        Record a prediction outcome in the rolling history.
        
        Updates the rolling window of recent predictions for accuracy computation.
        
        Args:
            predicted_activity (int): Predicted activity ID.
            actual_activity (int): Ground-truth activity ID.
        """
        is_correct = (predicted_activity == actual_activity)
        self.prediction_history.append((predicted_activity, actual_activity, is_correct))
    
    def get_recent_accuracy(self) -> float:
        """
        Compute accuracy over recent predictions in rolling window.
        
        Returns:
            float: Accuracy as fraction correct, in range [0.0, 1.0].
                  Returns 0.0 if history is empty.
        
        Example:
            >>> controller = ConfidenceController()
            >>> controller.add_prediction(3, 3)  # Correct
            >>> controller.add_prediction(4, 3)  # Incorrect
            >>> acc = controller.get_recent_accuracy()
            >>> print(f"{acc:.2f}")  # 0.50
        """
        if len(self.prediction_history) == 0:
            return 0.0
        
        num_correct = sum(1 for pred, actual, is_correct in self.prediction_history 
                         if is_correct)
        accuracy = num_correct / len(self.prediction_history)
        
        return accuracy
    
    def get_prediction_history_size(self) -> int:
        """
        Get current size of prediction history buffer.
        
        Returns:
            int: Number of predictions currently in rolling window (max 100).
        """
        return len(self.prediction_history)
    
    def reset_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        print(f"Prediction history cleared for user '{self.user_id}'")
    
    def get_status(self) -> str:
        """
        Get a status summary of the controller.
        
        Returns:
            str: Multi-line summary of thresholds, accuracy, and state.
        """
        accuracy = self.get_recent_accuracy()
        history_size = len(self.prediction_history)
        
        status = (
            f"ConfidenceController Status (user: {self.user_id}):\n"
            f"  High threshold: {self.high_threshold:.3f}\n"
            f"  Low threshold: {self.low_threshold:.3f}\n"
            f"  Recent accuracy: {accuracy:.3f} ({history_size}/100 predictions)\n"
        )
        
        return status
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"ConfidenceController(user='{self.user_id}', "
            f"high_threshold={self.high_threshold:.3f}, "
            f"low_threshold={self.low_threshold:.3f}, "
            f"num_classes={self.num_classes})"
        )
