"""
Training script for TrumpetJudge ML pipeline.

Trains the regression head on frozen PANNs embeddings to predict 5 performance scores.

Usage:
    python ml/train.py --train_csv data/train.csv --val_csv data/val.csv

The encoder (PANNs CNN14) is frozen - only the regression head is trained.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder_panns import PANNsEncoder
from models.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores
from ml.dataset import TrumpetDataset, create_dataloaders


def train_epoch(
    encoder: PANNsEncoder,
    head: RegressionHead,
    train_loader,
    optimizer,
    criterion,
    device: str,
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss for the epoch
    """
    head.train()
    total_loss = 0.0
    num_batches = 0
    
    for waveforms, labels in train_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # Forward pass through frozen encoder
        with torch.no_grad():
            embeddings = encoder(waveforms)
        
        # Forward pass through trainable head
        predictions = head(embeddings)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    encoder: PANNsEncoder,
    head: RegressionHead,
    val_loader,
    criterion,
    device: str,
) -> dict:
    """
    Validate the model.
    
    Returns:
        Dict with loss and per-score MAE
    """
    head.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    for waveforms, labels in val_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        embeddings = encoder(waveforms)
        predictions = head(embeddings)
        
        # Compute loss
        loss = criterion(predictions, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # Collect predictions for metrics
        all_preds.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Unscale to original 1-5 range for interpretable metrics
    all_preds_unscaled = unscale_scores(all_preds)
    all_labels_unscaled = unscale_scores(all_labels)
    
    # Compute MAE per score
    mae_per_score = {}
    for i, name in enumerate(SCORE_NAMES):
        mae = torch.abs(all_preds_unscaled[:, i] - all_labels_unscaled[:, i]).mean().item()
        mae_per_score[name] = mae
    
    # Overall MAE
    overall_mae = sum(mae_per_score.values()) / len(mae_per_score)
    
    return {
        "loss": total_loss / num_batches,
        "mae": overall_mae,
        "mae_per_score": mae_per_score,
    }


def train(
    train_csv: str,
    val_csv: str,
    output_dir: str = "checkpoints",
    duration: float = 20.0,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
):
    """
    Main training function.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        output_dir: Directory to save checkpoints
        duration: Audio duration in seconds
        batch_size: Training batch size
        learning_rate: Initial learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to use (None for auto-detect)
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TrumpetJudge Training")
    print("=" * 60)
    
    # Initialize encoder (frozen)
    print("\nInitializing PANNs encoder (frozen)...")
    encoder = PANNsEncoder(duration=duration, device=device)
    device = encoder.device  # Use the device encoder selected
    print(f"  Device: {device}")
    
    # Initialize regression head (trainable)
    print("\nInitializing regression head (trainable)...")
    head = RegressionHead(embedding_dim=encoder.embedding_dim)
    head = head.to(device)
    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, _ = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        duration=duration,
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = Adam(head.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(encoder, head, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(encoder, head, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]
        
        # Update scheduler (based on MAE now)
        scheduler.step(val_mae)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.3f}")
        
        # Check for improvement (based on MAE - lower is better)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_mae_per_score": val_metrics["mae_per_score"],
            }
            torch.save(checkpoint, run_dir / "best_model.pt")
            print(f"  → Saved best model (MAE: {val_mae:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val MAE: {best_val_mae:.3f}")
    
    # Load best model and report final metrics
    checkpoint = torch.load(run_dir / "best_model.pt")
    head.load_state_dict(checkpoint["head_state_dict"])
    
    print(f"\nFinal MAE per score (on validation set):")
    for name, mae in checkpoint["val_mae_per_score"].items():
        print(f"  {name}: {mae:.3f}")
    
    # Save training history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        "train_csv": train_csv,
        "val_csv": val_csv,
        "duration": duration,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCheckpoint saved to: {run_dir}")
    
    return head, history


def main():
    parser = argparse.ArgumentParser(description="Train TrumpetJudge regression head")
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default="data/val.csv",
                        help="Path to validation CSV")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detect if not specified.")
    
    args = parser.parse_args()
    
    train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        duration=args.duration,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()

