"""Training pipeline: DataLoader, optimizer, loss, early stopping, checkpointing."""

import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm


def train_model(model, X_train, y_train, X_val, y_val, config):
    """Train model with early stopping and checkpointing."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Training loop
    num_epochs = config["training"]["epochs"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    log_path = Path(config["paths"]["logs"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    
    saved_model_path = Path(config["paths"]["saved_model"])
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss_total += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss = train_loss_total / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss_total += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss = val_loss_total / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        # Log to file (JSON lines)
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), saved_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    log_file.close()
    
    return history
