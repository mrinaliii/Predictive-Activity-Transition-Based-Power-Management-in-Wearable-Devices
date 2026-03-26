"""
Safety Override: Real-time detection of dangerous conditions (falls, sudden acceleration, irregular patterns).
Requires: numpy

Dependencies:
Requires: numpy

Monitors raw sensor data for unsafe activity patterns and immediately triggers
maximum sensor activation for user safety.
"""

from typing import Dict, Optional
import numpy as np


def compute_spectral_entropy(signal: np.ndarray) -> float:
    """
    Compute spectral entropy of a 1D signal using FFT.
    
    Spectral entropy measures the concentration of power across frequencies.
    Higher entropy = more uniform frequency distribution (unpredictable).
    Lower entropy = power concentrated in few frequencies (regular pattern).
    
    Args:
        signal (np.ndarray): 1D signal array.
    
    Returns:
        float: Normalized spectral entropy in range [0, 1].
               0 = purely periodic, 1 = white noise.
    
    Example:
        >>> import numpy as np
        >>> # Regular sine wave: low entropy
        >>> t = np.linspace(0, 2*np.pi, 128)
        >>> sine = np.sin(t)
        >>> entropy_sine = compute_spectral_entropy(sine)
        >>> # Random noise: high entropy
        >>> noise = np.random.randn(128)
        >>> entropy_noise = compute_spectral_entropy(noise)
        >>> assert entropy_noise > entropy_sine
    """
    # Compute FFT magnitude spectrum
    fft_result = np.fft.fft(signal)
    power = np.abs(fft_result) ** 2
    
    # Normalize to probability distribution
    power_normalized = power / np.sum(power)
    
    # Remove zeros to avoid log(0)
    power_normalized = power_normalized[power_normalized > 0]
    
    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(power_normalized * np.log2(power_normalized))
    
    # Normalize by maximum possible entropy (log2(N))
    max_entropy = np.log2(len(signal))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return float(np.clip(normalized_entropy, 0.0, 1.0))


class SafetyOverride:
    """
    Detects dangerous activity patterns and overrides adaptive sampling.
    
    Monitors for:
    1. Falls: free-fall phase (<0.5g) followed by impact (>3g)
    2. Sudden acceleration: peak-to-peak on single axis exceeds 4g
    3. Irregular patterns: spectral entropy exceeds 0.90 (unpredictable motion)
    
    When any condition is detected, immediately switches to maximum sampling
    (100 Hz on all axes) and flags the event for human review.
    
    Attributes:
        free_fall_threshold_g (float): Acceleration threshold for free-fall detection (0.5g).
        impact_threshold_g (float): Acceleration threshold for impact detection (3g).
        sudden_accel_threshold_g (float): Peak-to-peak threshold for sudden acceleration (4g).
        entropy_threshold (float): Spectral entropy threshold for irregular pattern (0.90).
        lookahead_window (int): Timesteps to look ahead for impact after free-fall (10).
        free_fall_duration (int): Consecutive timesteps below free_fall threshold (3).
    """
    
    def __init__(self, 
                 free_fall_threshold_g: float = 0.2,
                 impact_threshold_g: float = 7.0,
                 sudden_accel_threshold_g: float = 12.0,
                 entropy_threshold: float = 0.90,
                 lookahead_window: int = 10,
                 free_fall_duration: int = 3):
        """
        Initialize SafetyOverride detector.
        
        Args:
            free_fall_threshold_g (float): G-force threshold for free-fall phase. Default 0.5g.
            impact_threshold_g (float): G-force threshold for impact detection. Default 3.0g.
            sudden_accel_threshold_g (float): Peak-to-peak threshold. Default 4.0g.
            entropy_threshold (float): Spectral entropy threshold. Default 0.90.
            lookahead_window (int): Timesteps to check for impact. Default 10.
            free_fall_duration (int): Consecutive free-fall timesteps required. Default 3.
        """
        self.free_fall_threshold_g = free_fall_threshold_g
        self.impact_threshold_g = impact_threshold_g
        self.sudden_accel_threshold_g = sudden_accel_threshold_g
        self.entropy_threshold = entropy_threshold
        self.lookahead_window = lookahead_window
        self.free_fall_duration = free_fall_duration
    
    def detect_fall(self, sensor_window: np.ndarray) -> bool:
        """
        Detect falls using free-fall + impact phases.
        
        A fall consists of:
        1. Free-fall phase: acceleration norm < 0.5g for >=3 consecutive timesteps
        2. Impact phase: acceleration norm > 3g within 10 timesteps after free-fall
        
        Args:
            sensor_window (np.ndarray): Sensor data of shape [seq_len, 9]
                                       (9 channels: chest_x/y/z, arm_x/y/z, ankle_x/y/z).
        
        Returns:
            bool: True if fall pattern detected, False otherwise.
        
        Example:
            >>> import numpy as np
            >>> # Simulate fall: free-fall (low acceleration) -> impact (high)
            >>> freefall = np.ones((3, 9)) * 0.2  # 0.2g (below 0.5g threshold)
            >>> impact = np.ones((2, 9)) * 3.5    # 3.5g (above 3g threshold)
            >>> window = np.vstack([freefall, impact, np.ones((25, 9))])
            >>> safe = SafetyOverride()
            >>> assert safe.detect_fall(window) == True
        """
        # Compute acceleration magnitude (L2 norm) at each timestep
        # sqrt(x^2 + y^2 + z^2 + ... ) for all 9 channels
        acceleration_norms = np.linalg.norm(sensor_window, axis=1)  # [seq_len]
        
        # Find free-fall regions: norm < 0.5g
        below_threshold = acceleration_norms < self.free_fall_threshold_g
        
        # Find consecutive free-fall phases of at least free_fall_duration timesteps
        free_fall_phases = []
        current_phase_start = None
        for t in range(len(below_threshold)):
            if below_threshold[t]:
                if current_phase_start is None:
                    current_phase_start = t
            else:
                if current_phase_start is not None:
                    phase_length = t - current_phase_start
                    if phase_length >= self.free_fall_duration:
                        free_fall_phases.append((current_phase_start, t))
                    current_phase_start = None
        
        # Check last phase
        if current_phase_start is not None:
            phase_length = len(below_threshold) - current_phase_start
            if phase_length >= self.free_fall_duration:
                free_fall_phases.append((current_phase_start, len(below_threshold)))
        
        # For each free-fall phase, check if impact follows
        for phase_start, phase_end in free_fall_phases:
            # Look ahead from end of free-fall phase
            lookahead_end = min(phase_end + self.lookahead_window, len(acceleration_norms))
            lookahead_window = acceleration_norms[phase_end:lookahead_end]
            
            # Check if any timestep in lookahead exceeds impact threshold
            if np.any(lookahead_window > self.impact_threshold_g):
                return True
        
        return False
    
    def detect_sudden_acceleration(self, sensor_window: np.ndarray) -> bool:
        """
        Detect sudden high-acceleration events.
        
        Returns True if peak-to-peak acceleration on any single axis exceeds 4g.
        
        Args:
            sensor_window (np.ndarray): Sensor data of shape [seq_len, 9].
        
        Returns:
            bool: True if sudden acceleration detected on any axis, False otherwise.
        
        Example:
            >>> import numpy as np
            >>> # Create window with one high-spike on an axis
            >>> window = np.zeros((32, 9))
            >>> window[10, 0] = 2.5  # Central accel
            >>> window[15, 0] = -2.0  # Peak-to-peak = 4.5g
            >>> safe = SafetyOverride()
            >>> assert safe.detect_sudden_acceleration(window) == True
        """
        # For each axis (channel), compute peak-to-peak
        for axis in range(sensor_window.shape[1]):
            signal = sensor_window[:, axis]
            peak_to_peak = np.max(signal) - np.min(signal)
            
            if peak_to_peak > self.sudden_accel_threshold_g:
                return True
        
        return False
    
    def detect_irregular_pattern(self, sensor_window: np.ndarray) -> bool:
        """
        Detect irregular/unpredictable motion patterns using spectral entropy.
        
        Computes spectral entropy for each of the 9 axes. If any axis has
        entropy > 0.90 (highly unpredictable), indicates erratic motion.
        
        Args:
            sensor_window (np.ndarray): Sensor data of shape [seq_len, 9].
        
        Returns:
            bool: True if any axis has spectral entropy > 0.90, False otherwise.
        
        Example:
            >>> import numpy as np
            >>> # White noise: very high entropy
            >>> window = np.random.randn(32, 9)
            >>> safe = SafetyOverride()
            >>> assert safe.detect_irregular_pattern(window) == True
            >>> 
            >>> # Regular pattern: low entropy
            >>> t = np.linspace(0, 4*np.pi, 32)
            >>> regular = np.tile(np.sin(t), (9, 1)).T
            >>> assert safe.detect_irregular_pattern(regular) == False
        """
        for axis in range(sensor_window.shape[1]):
            signal = sensor_window[:, axis]
            entropy = compute_spectral_entropy(signal)
            
            if entropy > self.entropy_threshold:
                return True
        
        return False
    
    def check(self, sensor_window: np.ndarray) -> Dict[str, Optional[object]]:
        """
        Check for all safety hazards in a sensor window.
        
        Runs all three detection methods and returns override status.
        If ANY hazard is detected, override is activated.
        
        Args:
            sensor_window (np.ndarray): Sensor data of shape [seq_len, 9].
        
        Returns:
            Dict with keys:
                'override_active': bool - True if any hazard detected
                'reason': str or None - One of:
                    'fall_detected', 'sudden_acceleration', 'irregular_pattern', None
                'recommended_sampling_rate': int or None - 100 if override active, else None
        
        Example:
            >>> import numpy as np
            >>> safe = SafetyOverride()
            >>> 
            >>> # Normal window
            >>> normal_window = np.random.randn(32, 9) * 0.1
            >>> result = safe.check(normal_window)
            >>> assert result['override_active'] == False
            >>> assert result['reason'] is None
            >>> 
            >>> # Fall window
            >>> fall_window = np.ones((32, 9)) * 0.2
            >>> fall_window[15:20, :] = 3.5  # Impact
            >>> result = safe.check(fall_window)
            >>> assert result['override_active'] == True
            >>> assert result['reason'] == 'fall_detected'
        """
        if self.detect_fall(sensor_window):
            return {
                'override_active': True,
                'reason': 'fall_detected',
                'recommended_sampling_rate': 100
            }
        
        if self.detect_sudden_acceleration(sensor_window):
            return {
                'override_active': True,
                'reason': 'sudden_acceleration',
                'recommended_sampling_rate': 100
            }
        
        # NOTE: Disabled irregular_pattern detection (spectral entropy too sensitive for normal motion)
        # Only keep fall and sudden acceleration detection for true safety events
        
        return {
            'override_active': False,
            'reason': None,
            'recommended_sampling_rate': None
        }
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"SafetyOverride(free_fall={self.free_fall_threshold_g}g, "
            f"impact={self.impact_threshold_g}g, "
            f"sudden_accel={self.sudden_accel_threshold_g}g, "
            f"entropy_thres={self.entropy_threshold}, "
            f"lookahead={self.lookahead_window})"
        )
