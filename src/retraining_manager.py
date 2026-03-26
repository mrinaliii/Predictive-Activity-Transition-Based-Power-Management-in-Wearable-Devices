"""
Retraining Manager: Incremental fine-tuning on flagged samples without full dataset.
Requires: torch, numpy

Dependencies:
Requires: torch, numpy

Buffers low-confidence samples and fine-tunes the final classification head
on demand without requiring the full PAMAP2 dataset.
"""

from typing import Optional
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class RetrainingManager:
    """
    Manages incremental retraining on flagged low-confidence samples.
    
    Buffers samples flagged by the ConfidenceController and fine-tunes the
    final classification head of the GRU model when sufficient samples accumulate.
    
    Does NOT require the full PAMAP2 dataset — operates only on buffered samples.
    Freezes all layers except the final linear head for efficient retraining.
    
    Attributes:
        base_model: PyTorch nn.Module (the trained GRU).
        trigger_threshold (int): Number of samples to accumulate before retraining.
        user_id (str): User identifier.
        device (str): 'cpu' or 'cuda'.
        sample_buffer (deque): Buffer of (X, y) tuples.
        retraining_history (list): Log of retraining events.
    """
    
    def __init__(self, base_model: nn.Module, trigger_threshold: int = 200, 
                 user_id: str = "default", device: str = "cpu"):
        """
        Initialize RetrainingManager.
        
        Args:
            base_model (nn.Module): Trained GRU model (or any PyTorch model).
            trigger_threshold (int): Number of samples to accumulate before retraining.
                                    Default 200.
            user_id (str): User identifier. Default "default".
            device (str): Device to use ('cpu' or 'cuda'). Default 'cpu'.
        """
        self.base_model = base_model
        self.trigger_threshold = trigger_threshold
        self.user_id = user_id
        self.device = device
        
        # Buffer for flagged samples
        self.sample_buffer = deque()
        
        # Log of retraining events
        self.retraining_history = []
    
    def add_flagged_sample(self, X: np.ndarray, y: int) -> None:
        """
        Add a flagged low-confidence sample to the retraining buffer.
        
        Args:
            X (np.ndarray): Input sensor sequence of shape [128, 9]
                           (or compatible with the model's input_size).
            y (int): Ground-truth activity label (0 to num_classes-1).
        
        Example:
            >>> import numpy as np
            >>> import torch.nn as nn
            >>> # Dummy model
            >>> model = nn.Linear(9, 12)
            >>> manager = RetrainingManager(model, trigger_threshold=2)
            >>> X_sample = np.random.randn(128, 9)
            >>> manager.add_flagged_sample(X_sample, y=3)
            >>> assert manager.sample_buffer.__len__() == 1
        """
        self.sample_buffer.append((X, y))
    
    def should_retrain(self) -> bool:
        """
        Check if accumulated samples meet retraining threshold.
        
        Returns:
            bool: True if buffer size >= trigger_threshold, False otherwise.
        """
        return len(self.sample_buffer) >= self.trigger_threshold
    
    def retrain(self, epochs: int = 3, lr: float = 1e-4, batch_size: int = 16) -> None:
        """
        Fine-tune the model's final classification head on buffered samples.
        
        Strategy:
        1. Freeze all layers of base_model
        2. Unfreeze only the final linear classification head
        3. Create DataLoader from buffered samples
        4. Train for specified epochs
        5. Clear buffer after completion
        
        Logs training info and prints summary.
        
        Args:
            epochs (int): Number of retraining epochs. Default 3.
            lr (float): Learning rate for fine-tuning. Default 1e-4.
            batch_size (int): Batch size for DataLoader. Default 16.
        
        Example:
            >>> # After adding samples with add_flagged_sample():
            >>> if manager.should_retrain():
            ...     manager.retrain(epochs=5, lr=1e-4)
            ...     print(manager.get_status())
        """
        if len(self.sample_buffer) == 0:
            print(f"[{self.user_id}] No samples in buffer. Skipping retraining.")
            return
        
        # Convert buffer to tensors
        X_list, y_list = [], []
        for X, y in self.sample_buffer:
            X_list.append(X)
            y_list.append(y)
        
        X_tensor = torch.from_numpy(np.array(X_list)).float().to(self.device)
        y_tensor = torch.from_numpy(np.array(y_list)).long().to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        num_samples = len(self.sample_buffer)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[RETRAINING] User '{self.user_id}' at {timestamp}")
        print(f"  Samples in buffer: {num_samples}")
        print(f"  Epochs: {epochs}, Learning rate: {lr}, Batch size: {batch_size}")
        
        # Freeze all parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classification head
        # Assuming the model has a 'dense' attribute for the final layer
        if hasattr(self.base_model, 'dense'):
            for param in self.base_model.dense.parameters():
                param.requires_grad = True
        else:
            # If 'dense' doesn't exist, try to find the last linear layer
            for module in self.base_model.modules():
                if isinstance(module, nn.Linear):
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Setup optimizer (only for unfrozen parameters)
        optimizer = torch.optim.Adam([p for p in self.base_model.parameters() if p.requires_grad], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.base_model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                logits = self.base_model(X_batch)
                loss = criterion(logits, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")
        
        # Log the retraining event
        self.retraining_history.append({
            'timestamp': timestamp,
            'num_samples': num_samples,
            'epochs': epochs,
            'final_loss': epoch_losses[-1],
            'learning_rate': lr
        })
        
        # Clear buffer
        self.sample_buffer.clear()
        print(f"  Buffer cleared. Next retraining at {self.trigger_threshold} samples.")
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the model's state_dict to disk.
        
        Args:
            filepath (str): Path to save .pt file.
                           Directories will be created if needed.
        
        Example:
            >>> manager.save_checkpoint('models/checkpoints/retrained_model.pt')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.base_model.state_dict(), filepath)
        print(f"Model checkpoint saved for user '{self.user_id}' to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a previously saved state_dict into the base_model.
        
        Args:
            filepath (str): Path to load .pt file from.
        
        Raises:
            FileNotFoundError: If file does not exist.
        
        Example:
            >>> manager.load_checkpoint('models/checkpoints/retrained_model.pt')
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        state_dict = torch.load(filepath, map_location=self.device)
        self.base_model.load_state_dict(state_dict)
        print(f"Model checkpoint loaded for user '{self.user_id}' from {filepath}")
    
    def get_buffer_size(self) -> int:
        """
        Get current size of sample buffer.
        
        Returns:
            int: Number of samples currently buffered.
        """
        return len(self.sample_buffer)
    
    def get_status(self) -> str:
        """
        Get a status summary of the retraining manager.
        
        Returns:
            str: Multi-line status including buffer size, threshold, and history.
        """
        buffer_size = len(self.sample_buffer)
        percent_to_threshold = (buffer_size / self.trigger_threshold) * 100
        
        status = (
            f"RetrainingManager Status (user: '{self.user_id}'):\n"
            f"  Buffer size: {buffer_size}/{self.trigger_threshold} "
            f"({percent_to_threshold:.1f}%)\n"
            f"  Should retrain now: {self.should_retrain()}\n"
            f"  Total retraining events: {len(self.retraining_history)}\n"
        )
        
        if self.retraining_history:
            last_event = self.retraining_history[-1]
            status += (
                f"  Last retraining: {last_event['timestamp']}\n"
                f"    - Samples used: {last_event['num_samples']}\n"
                f"    - Final loss: {last_event['final_loss']:.4f}\n"
            )
        
        return status
    
    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"RetrainingManager(trigger={self.trigger_threshold}, "
            f"buffer={len(self.sample_buffer)}, "
            f"user='{self.user_id}', "
            f"device='{self.device}', "
            f"retrainings={len(self.retraining_history)})"
        )
