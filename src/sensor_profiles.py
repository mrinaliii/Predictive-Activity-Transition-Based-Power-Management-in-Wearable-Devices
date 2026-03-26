"""
Sensor Activation Profiles: Activity-dependent sensor selection and sampling rates.
Requires: numpy

Dependencies:
Defines which sensor axes should be active for each activity class at different
confidence levels, along with corresponding sampling rates and energy costs.
"""

from typing import Dict, List
import numpy as np


class SensorActivationProfile:
    """
    Stores sensor activation patterns for each activity at different confidence tiers.
    
    Maps activity IDs to sensor configurations. For each activity, specifies:
    - Which sensor axes are active at high/medium/low confidence levels
    - Appropriate sampling rate for each confidence tier
    - Energy consumption estimates
    
    The 9 sensor axes are:
        ['chest_x', 'chest_y', 'chest_z', 'arm_x', 'arm_y', 'arm_z', 
         'ankle_x', 'ankle_y', 'ankle_z']
    
    Confidence tiers:
        - high_confidence: Model is highly confident; use minimal sensors
        - medium_confidence: Moderate confidence; use standard sensors
        - low_confidence: Low confidence; use all sensors for robustness
    
    Sampling rates:
        - high: 25 Hz (minimal battery usage)
        - medium: 50 Hz (balanced)
        - low: 100 Hz (high fidelity, high power)
    """
    
    # All possible sensor axes
    ALL_AXES = [
        'chest_x', 'chest_y', 'chest_z',
        'arm_x', 'arm_y', 'arm_z',
        'ankle_x', 'ankle_y', 'ankle_z'
    ]
    
    # Sampling rate (Hz) for each confidence tier
    SAMPLING_RATES = {
        'high': 25,
        'medium': 50,
        'low': 100
    }
    
    # Energy cost: mJ per axis-sample (0.015 mJ base unit)
    ENERGY_COST_PER_AXIS_SAMPLE = 0.015
    
    def __init__(self, num_classes: int = 12):
        """
        Initialize SensorActivationProfile with sensible defaults.
        
        Args:
            num_classes (int): Number of activity classes. Default 12 (PAMAP2).
        """
        self.num_classes = num_classes
        
        # Map: activity_id -> {confidence_tier -> list of active axes}
        self.profiles = self._initialize_default_profiles()
    
    def _initialize_default_profiles(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Pre-populate sensor profiles with sensible defaults based on activity type.
        
        Returns:
            Dict with activity-specific sensor configurations.
        """
        profiles = {}
        
        # Define activity types with their default patterns
        # PAMAP2 activities: 0=LYING, 1=SITTING, 2=STANDING, 3=WALKING, 4=RUNNING,
        #                   5=CYCLING, 6=NORDIC_WALK, 7=TV, 8=COMPUTER, 
        #                   9=CAR, 10=STAIRS_UP, 11=STAIRS_DOWN
        
        # Stationary activities (LYING, SITTING, STANDING, TV, COMPUTER)
        stationary = {
            'high_confidence': ['arm_x', 'arm_y', 'arm_z'],
            'medium_confidence': self.ALL_AXES,
            'low_confidence': self.ALL_AXES
        }
        
        # Moving activities (WALKING, RUNNING, NORDIC_WALK)
        moving = {
            'high_confidence': self.ALL_AXES,
            'medium_confidence': self.ALL_AXES,
            'low_confidence': self.ALL_AXES
        }
        
        # Cycling
        cycling = {
            'high_confidence': ['arm_x', 'arm_y', 'arm_z', 'ankle_x', 'ankle_y', 'ankle_z'],
            'medium_confidence': self.ALL_AXES,
            'low_confidence': self.ALL_AXES
        }
        
        # Car driving
        car = {
            'high_confidence': ['arm_x', 'arm_y', 'arm_z'],
            'medium_confidence': self.ALL_AXES,
            'low_confidence': self.ALL_AXES
        }
        
        # Stairs (need all sensors for balance/safety)
        stairs = {
            'high_confidence': self.ALL_AXES,
            'medium_confidence': self.ALL_AXES,
            'low_confidence': self.ALL_AXES
        }
        
        # Assign patterns to activities
        profiles[0] = stationary  # LYING
        profiles[1] = stationary  # SITTING
        profiles[2] = stationary  # STANDING
        profiles[3] = moving      # WALKING
        profiles[4] = moving      # RUNNING
        profiles[5] = cycling     # CYCLING
        profiles[6] = moving      # NORDIC_WALK
        profiles[7] = stationary  # TV
        profiles[8] = stationary  # COMPUTER
        profiles[9] = car         # CAR
        profiles[10] = stairs     # STAIRS_UP
        profiles[11] = stairs     # STAIRS_DOWN
        
        return profiles
    
    def get_active_axes(self, activity_id: int, confidence_tier: str) -> List[str]:
        """
        Get list of active sensor axes for a given activity and confidence tier.
        
        Args:
            activity_id (int): Activity ID (0 to num_classes-1).
            confidence_tier (str): One of ['high', 'medium', 'low'].
        
        Returns:
            List[str]: List of active axis names.
        
        Raises:
            ValueError: If activity_id or confidence_tier is invalid.
        
        Example:
            >>> profile = SensorActivationProfile()
            >>> profile.get_active_axes(activity_id=3, confidence_tier='high')
            ['chest_x', 'chest_y', 'chest_z', 'arm_x', 'arm_y', 'arm_z', 
             'ankle_x', 'ankle_y', 'ankle_z']  # All axes for walking
        """
        if not (0 <= activity_id < self.num_classes):
            raise ValueError(f"activity_id {activity_id} out of range [0, {self.num_classes-1}]")
        if confidence_tier not in ['high', 'medium', 'low']:
            raise ValueError(f"confidence_tier must be one of ['high', 'medium', 'low'], "
                           f"got '{confidence_tier}'")
        
        tier_key = f"{confidence_tier}_confidence"
        return self.profiles[activity_id][tier_key]
    
    def get_sampling_rate(self, activity_id: int, confidence_tier: str) -> int:
        """
        Get sampling rate (Hz) for a given activity and confidence tier.
        
        Args:
            activity_id (int): Activity ID (0 to num_classes-1).
            confidence_tier (str): One of ['high', 'medium', 'low'].
        
        Returns:
            int: Sampling rate in Hz.
                - high: 25 Hz (minimal power consumption)
                - medium: 50 Hz (balanced)
                - low: 100 Hz (maximum fidelity)
        
        Raises:
            ValueError: If confidence_tier is invalid.
        
        Example:
            >>> profile = SensorActivationProfile()
            >>> profile.get_sampling_rate(activity_id=1, confidence_tier='high')
            25  # Sitting is stationary, so low sampling rate is acceptable
        """
        if confidence_tier not in ['high', 'medium', 'low']:
            raise ValueError(f"confidence_tier must be one of ['high', 'medium', 'low'], "
                           f"got '{confidence_tier}'")
        
        return self.SAMPLING_RATES[confidence_tier]
    
    def compute_energy_cost(self, active_axes: List[str], sampling_rate: int, 
                           duration_seconds: float) -> float:
        """
        Compute energy consumption for a sensor configuration.
        
        Formula: energy (mJ) = num_active_axes * sampling_rate (Hz) * duration (s) * 0.015 (mJ per axis-sample)
        
        Args:
            active_axes (List[str]): List of active sensor axis names.
            sampling_rate (int): Sampling rate in Hz.
            duration_seconds (float): Duration in seconds.
        
        Returns:
            float: Energy consumption in millijoules (mJ).
        
        Raises:
            ValueError: If any axis name is invalid.
        
        Example:
            >>> profile = SensorActivationProfile()
            >>> axes = profile.get_active_axes(3, 'high')  # Walking, high confidence
            >>> rate = profile.get_sampling_rate(3, 'high')  # 25 Hz
            >>> energy = profile.compute_energy_cost(axes, rate, 60)  # 60 seconds
            >>> print(f"Energy for 1 min of walking: {energy:.2f} mJ")
            Energy for 1 min of walking: 202.50 mJ
        """
        # Validate axis names
        for axis in active_axes:
            if axis not in self.ALL_AXES:
                raise ValueError(f"Invalid axis name '{axis}'. Must be one of {self.ALL_AXES}")
        
        num_active = len(active_axes)
        num_samples = sampling_rate * duration_seconds
        energy_mj = num_active * num_samples * self.ENERGY_COST_PER_AXIS_SAMPLE
        
        return energy_mj
    
    def set_custom_profile(self, activity_id: int, profile_dict: Dict[str, List[str]]) -> None:
        """
        Set a custom sensor profile for a specific activity.
        
        Args:
            activity_id (int): Activity ID to configure.
            profile_dict (Dict): Dictionary with keys ['high_confidence', 'medium_confidence', 
                                'low_confidence'], each mapping to a list of axis names.
        
        Raises:
            ValueError: If activity_id is invalid or profile_dict has incorrect structure.
        
        Example:
            >>> profile = SensorActivationProfile()
            >>> custom = {
            ...     'high_confidence': ['ankle_x', 'ankle_y', 'ankle_z'],
            ...     'medium_confidence': ['arm_x', 'arm_y', 'arm_z', 'ankle_x', 'ankle_y', 'ankle_z'],
            ...     'low_confidence': SensorActivationProfile.ALL_AXES
            ... }
            >>> profile.set_custom_profile(activity_id=4, profile_dict=custom)
        """
        if not (0 <= activity_id < self.num_classes):
            raise ValueError(f"activity_id {activity_id} out of range [0, {self.num_classes-1}]")
        
        required_keys = {'high_confidence', 'medium_confidence', 'low_confidence'}
        if set(profile_dict.keys()) != required_keys:
            raise ValueError(f"profile_dict must have keys {required_keys}, "
                           f"got {set(profile_dict.keys())}")
        
        # Validate all axes
        for tier, axes in profile_dict.items():
            for axis in axes:
                if axis not in self.ALL_AXES:
                    raise ValueError(f"Invalid axis '{axis}' in tier '{tier}'. "
                                   f"Must be one of {self.ALL_AXES}")
        
        self.profiles[activity_id] = profile_dict
        print(f"Custom profile set for activity {activity_id}")
    
    def get_summary(self, activity_id: int) -> str:
        """
        Get a summary string of all sensor configurations for an activity.
        
        Args:
            activity_id (int): Activity ID to summarize.
        
        Returns:
            str: Formatted summary of the activity's sensor configuration.
        """
        profile = self.profiles[activity_id]
        summary = f"Activity {activity_id}:\n"
        for tier in ['high', 'medium', 'low']:
            tier_key = f"{tier}_confidence"
            axes = profile[tier_key]
            rate = self.SAMPLING_RATES[tier]
            summary += f"  {tier}_confidence: {axes} @ {rate} Hz\n"
        return summary
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"SensorActivationProfile(num_classes={self.num_classes}, "
            f"all_axes={len(self.ALL_AXES)})"
        )
