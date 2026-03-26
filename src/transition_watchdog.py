"""
Transition Watchdog: Lightweight binary classifier for activity transitions.
Requires: torch, numpy

Dependencies:
Detects when sensor data indicates an upcoming activity change and provides
probabilistic predictions for next-activity targets.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, List


class TransitionWatchdog(nn.Module):
    """
    Lightweight binary classifier for detecting activity transitions.
    
    Monitors a short window of raw sensor data (32 timesteps, 9 channels)
    to predict if a transition is imminent in the next 16 timesteps.
    
    Attributes:
        gru_hidden_size (int): Number of hidden units in GRU layer (32).
        num_classes (int): Number of activity classes for multi-head output.
    """
    
    def __init__(self, num_classes: int, gru_hidden_size: int = 32, device: str = "cpu"):
        """
        Initialize TransitionWatchdog model.
        
        Args:
            num_classes (int): Number of activity classes (typically 12 for PAMAP2).
            gru_hidden_size (int): Hidden size for GRU layer. Default 32.
            device (str): Device to place model on ('cpu' or 'cuda'). Default 'cpu'.
        """
        super(TransitionWatchdog, self).__init__()
        
        self.num_classes = num_classes
        self.gru_hidden_size = gru_hidden_size
        self.device_name = device
        
        # Input: 9 sensor channels (chest_x/y/z, arm_x/y/z, ankle_x/y/z)
        input_size = 9
        
        # Single-layer GRU for efficiency
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Binary head: detects transition yes/no
        self.transition_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary output [0, 1]
        )
        
        # Activity prediction head: predicts next activity probabilities
        self.activity_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
            # No softmax: will be applied during inference
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input sensor data of shape [batch, 32, 9]
                             (batch_size, seq_len=32, channels=9).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - transition_scores: shape [batch] or [batch, 1], binary scores in [0,1]
                - activity_logits: shape [batch, num_classes], unnormalized activity predictions
        """
        # x: [batch, 32, 9]
        outputs, hidden = self.gru(x)  # hidden: [1, batch, 32]
        
        # Use last hidden state: [batch, 32]
        last_hidden = hidden[-1]
        
        # Transition prediction
        transition_score = self.transition_head(last_hidden)  # [batch, 1]
        
        # Activity prediction
        activity_logits = self.activity_head(last_hidden)  # [batch, num_classes]
        
        return transition_score, activity_logits
    
    def predict(self, x: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Inference: predict transition and target activity probabilities for single window.
        
        Args:
            x (np.ndarray): Raw sensor window of shape [32, 9].
        
        Returns:
            Tuple[bool, np.ndarray]:
                - transition_detected: Boolean indicating if transition is imminent.
                - target_activity_probs: Probability distribution over activities, shape [num_classes].
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension: [32, 9] -> [1, 32, 9]
            x_tensor = torch.from_numpy(x).unsqueeze(0).float()
            x_tensor = x_tensor.to(self.device_name)
            
            transition_score, activity_logits = self.forward(x_tensor)
            
            # Convert to numpy
            trans_score = transition_score.cpu().numpy().flatten()[0]
            activity_probs = torch.softmax(activity_logits, dim=1)[0].cpu().numpy()
            
            # Threshold: if score > 0.5, transition is detected
            transition_detected = bool(trans_score > 0.5)
            
        return transition_detected, activity_probs
    
    def get_parameter_count(self) -> int:
        """
        Calculate and print total number of trainable parameters.
        
        Returns:
            int: Total trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TransitionWatchdog total trainable parameters: {total_params}")
        return total_params
    
    @staticmethod
    def generate_transition_labels(activity_sequence: np.ndarray, 
                                    lookahead: int = 16) -> np.ndarray:
        """
        Generate binary transition labels from activity sequence.
        
        A window is labeled as "transition=1" if the activity label changes
        within the next `lookahead` timesteps.
        
        Args:
            activity_sequence (np.ndarray): Array of activity labels, shape [total_timesteps].
                                           Values in range [0, num_classes-1].
            lookahead (int): Number of timesteps to look ahead. Default 16.
        
        Returns:
            np.ndarray: Binary labels of shape [len(activity_sequence) - lookahead],
                       where label[i] = 1 if activity changes in steps [i, i+lookahead),
                       else 0.
        
        Example:
            >>> acts = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
            >>> labels = TransitionWatchdog.generate_transition_labels(acts, lookahead=3)
            >>> # labels[0]=1 (activity changes at step 3), labels[3]=1 (at step 6)
        """
        num_samples = len(activity_sequence) - lookahead
        labels = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            # Check if activity changes in the lookahead window
            current_activity = activity_sequence[i]
            next_window = activity_sequence[i:i + lookahead + 1]
            
            # Transition detected if any label differs from current
            if np.any(next_window != current_activity):
                labels[i] = 1
        
        return labels


class TransitionProbabilityMatrix:
    """
    Stores and manages observed activity transition probabilities.
    
    Maintains a (num_classes × num_classes) matrix of activity transition counts.
    Used to determine most likely next activities based on current activity.
    Per-user: stores separate matrices for different users.
    
    Attributes:
        user_id (str): Unique identifier for the user.
        num_classes (int): Number of activity classes.
        transition_matrix (np.ndarray): Count matrix of shape [num_classes, num_classes].
    """
    
    def __init__(self, num_classes: int, user_id: str = "default"):
        """
        Initialize TransitionProbabilityMatrix.
        
        Args:
            num_classes (int): Number of activity classes (typically 12 for PAMAP2).
            user_id (str): Unique user identifier for per-user tracking. Default "default".
        """
        self.num_classes = num_classes
        self.user_id = user_id
        self.transition_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    def update(self, from_activity: int, to_activity: int) -> None:
        """
        Increment transition count from one activity to another.
        
        Args:
            from_activity (int): Source activity ID (0 to num_classes-1).
            to_activity (int): Target activity ID (0 to num_classes-1).
        
        Raises:
            ValueError: If activity IDs are out of valid range.
        """
        if not (0 <= from_activity < self.num_classes):
            raise ValueError(f"from_activity {from_activity} out of range [0, {self.num_classes-1}]")
        if not (0 <= to_activity < self.num_classes):
            raise ValueError(f"to_activity {to_activity} out of range [0, {self.num_classes-1}]")
        
        self.transition_matrix[from_activity, to_activity] += 1
    
    def get_probable_targets(self, current_activity: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Get top-k most likely next activities from current activity.
        
        Args:
            current_activity (int): Current activity ID (0 to num_classes-1).
            top_k (int): Number of top targets to return. Default 3.
        
        Returns:
            List[Tuple[int, float]]: List of (activity_id, probability) tuples,
                                     sorted by probability descending.
                                     Probabilities are normalized from transition counts.
        
        Example:
            >>> matrix = TransitionProbabilityMatrix(12, user_id="user1")
            >>> matrix.update(0, 1)  # sitting -> walking
            >>> matrix.update(0, 1)
            >>> matrix.update(0, 2)  # sitting -> running
            >>> targets = matrix.get_probable_targets(0, top_k=2)
            >>> # Should be [(1, 0.667), (2, 0.333)]
        """
        if not (0 <= current_activity < self.num_classes):
            raise ValueError(f"current_activity {current_activity} out of range [0, {self.num_classes-1}]")
        
        # Get transition counts for current activity
        counts = self.transition_matrix[current_activity, :].astype(np.float32)
        total = counts.sum()
        
        if total == 0:
            # No observed transitions: return uniform distribution
            probs = np.ones(self.num_classes) / self.num_classes
        else:
            # Normalize to probabilities
            probs = counts / total
        
        # Get top-k indices
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # Return as list of (activity_id, probability) tuples
        result = [(int(idx), float(probs[idx])) for idx in top_indices]
        
        return result
    
    def save(self, filepath: str) -> None:
        """
        Save transition matrix to disk using numpy format.
        
        Args:
            filepath (str): Path to save the .npy file.
                           If directory doesn't exist, it will be created.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, self.transition_matrix)
        print(f"TransitionProbabilityMatrix saved for user '{self.user_id}' to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load transition matrix from disk.
        
        Args:
            filepath (str): Path to the .npy file to load.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If loaded matrix shape doesn't match expected dimensions.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        loaded_matrix = np.load(filepath)
        
        if loaded_matrix.shape != (self.num_classes, self.num_classes):
            raise ValueError(f"Loaded matrix shape {loaded_matrix.shape} doesn't match "
                           f"expected ({self.num_classes}, {self.num_classes})")
        
        self.transition_matrix = loaded_matrix.astype(np.int32)
        print(f"TransitionProbabilityMatrix loaded for user '{self.user_id}' from {filepath}")
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        total_transitions = self.transition_matrix.sum()
        return (
            f"TransitionProbabilityMatrix(user='{self.user_id}', "
            f"num_classes={self.num_classes}, "
            f"total_transitions={int(total_transitions)})"
        )
