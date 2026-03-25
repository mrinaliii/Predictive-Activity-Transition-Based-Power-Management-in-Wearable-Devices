"""Model architectures: GRU, LSTM, CNN-LSTM for activity recognition."""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """GRU-based model for activity recognition."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Forward pass: x shape [batch, seq_len, input_size]."""
        # GRU: outputs [batch, seq_len, hidden_size], hidden [num_layers, batch, hidden_size]
        outputs, hidden = self.gru(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Layer norm and dropout
        normalized = self.layer_norm(last_hidden)
        dropped = self.dropout(normalized)
        
        # Classification
        logits = self.dense(dropped)  # [batch, num_classes]
        
        return logits


class LSTMModel(nn.Module):
    """LSTM-based model for activity recognition."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Forward pass: x shape [batch, seq_len, input_size]."""
        # LSTM: outputs [batch, seq_len, hidden_size], (hidden, cell) states
        outputs, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Layer norm and dropout
        normalized = self.layer_norm(last_hidden)
        dropped = self.dropout(normalized)
        
        # Classification
        logits = self.dense(dropped)  # [batch, num_classes]
        
        return logits


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for activity recognition."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        """Initialize CNN-LSTM model."""
        super(CNNLSTMModel, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Forward pass: x shape [batch, seq_len, input_size]."""
        # Transpose for Conv1d: [batch, input_size, seq_len]
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        
        # Transpose back for LSTM: [batch, seq_len//2, 128]
        x = x.transpose(1, 2)
        
        # LSTM
        outputs, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Classification
        dropped = self.dropout(last_hidden)
        logits = self.dense(dropped)  # [batch, num_classes]
        
        return logits


def get_model(config, input_size, num_classes):
    """Factory function to create model based on config."""
    model_type = config["model"]["type"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    
    if model_type == "GRU":
        model = GRUModel(input_size, hidden_size, num_layers, num_classes, dropout)
    elif model_type == "LSTM":
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)
    elif model_type == "CNN_LSTM":
        model = CNNLSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
