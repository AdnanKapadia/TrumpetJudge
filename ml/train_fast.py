"""
Fast Training on Precomputed Embeddings for TrumpetJudge.

Trains the regression head on precomputed PANNs embeddings.
This is ~10-15x faster than train.py since we skip the encoder.

Usage:
    python ml/train_fast.py --train data/embeddings/train.pt --val data/embeddings/val.pt

    # With augmented data (recommended):
    python ml/train_fast.py \
        --train data/embeddings/train.pt data/embeddings/train_augmented.pt \
        --val data/embeddings/val.pt

Prerequisites:
    1. Run augment_offline.py to create augmented audio (optional)
    2. Run precompute.py to create embedding files
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EmbeddingDataset(Dataset):
    """
    Dataset for precomputed embeddings.
    
    Much faster than loading audio since embeddings are already computed.
    """
    
    def __init__(self, embedding_files: list):
        """
        Initialize from one or more embedding files.
        
        Args:
            embedding_files: List of paths to .pt embedding files
        """
        self.embeddings = []
        self.labels = []
        self.ids = []
        
        for path in embedding_files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            self.embeddings.append(data["embeddings"])
            self.labels.append(data["labels"])
            self.ids.extend(data["ids"])
        
        # Concatenate all
        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        
        print(f"Loaded {len(self.embeddings)} embeddings from {len(embedding_files)} file(s)")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train_epoch(head, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    head.train()
    total_loss = 0.0
    num_batches = 0
    
    for embeddings, labels in train_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        predictions = head(embeddings)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate(head, val_loader, criterion, device):
    """Validate the model."""
    head.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    for embeddings, labels in val_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        predictions = head(embeddings)
        loss = criterion(predictions, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_preds.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Unscale to [1, 5] for interpretable metrics
    all_preds_unscaled = unscale_scores(all_preds)
    all_labels_unscaled = unscale_scores(all_labels)
    
    # Compute MAE per score
    mae_per_score = {}
    for i, name in enumerate(SCORE_NAMES):
        mae = torch.abs(all_preds_unscaled[:, i] - all_labels_unscaled[:, i]).mean().item()
        mae_per_score[name] = mae
    
    overall_mae = sum(mae_per_score.values()) / len(mae_per_score)
    
    return {
        "loss": total_loss / num_batches,
        "mae": overall_mae,
        "mae_per_score": mae_per_score,
    }


def train_fast(
    train_files: list,
    val_files: list,
    output_dir: str = "checkpoints",
    embedding_dim: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    device: str = None,
    seed: int = 42,
):
    """
    Fast training on precomputed embeddings.
    
    Args:
        train_files: List of training embedding .pt files
        val_files: List of validation embedding .pt files
        output_dir: Directory to save checkpoints
        embedding_dim: Embedding dimension (2048 for PANNs CNN14)
        batch_size: Training batch size (can be larger since no audio loading)
        learning_rate: Initial learning rate
        epochs: Maximum epochs
        patience: Early stopping patience
        device: Device to use
        seed: Random seed
    """
    # Set seed
    set_seed(seed)
    
    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"fast_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TrumpetJudge Fast Training (Precomputed Embeddings)")
    print("=" * 60)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    
    # Load datasets
    print("\nLoading precomputed embeddings...")
    train_dataset = EmbeddingDataset(train_files)
    val_dataset = EmbeddingDataset(val_files)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Embeddings are already in memory
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing regression head...")
    head = RegressionHead(embedding_dim=embedding_dim)
    head = head.to(device)
    num_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")
    
    # Training setup
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
        train_loss = train_epoch(head, train_loader, optimizer, criterion, device)
        val_metrics = validate(head, val_loader, criterion, device)
        
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]
        
        scheduler.step(val_mae)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.3f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            
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
    
    # Load best and report
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head_state_dict"])
    
    print(f"\nFinal MAE per score:")
    for name, mae in checkpoint["val_mae_per_score"].items():
        print(f"  {name}: {mae:.3f}")
    
    # Save history and config
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    config = {
        "train_files": [str(f) for f in train_files],
        "val_files": [str(f) for f in val_files],
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCheckpoint saved to: {run_dir}")
    
    return head, history


def main():
    parser = argparse.ArgumentParser(description="Fast training on precomputed embeddings")
    
    parser.add_argument("--train", type=str, nargs="+", required=True,
                        help="Training embedding file(s) (.pt)")
    parser.add_argument("--val", type=str, nargs="+", required=True,
                        help="Validation embedding file(s) (.pt)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64, can be larger)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train_fast(
        train_files=args.train,
        val_files=args.val,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

