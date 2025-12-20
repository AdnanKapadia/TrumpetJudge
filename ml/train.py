"""
Training script for TrumpetJudge ML pipeline.

Trains the regression head on frozen PANNs embeddings to predict 5 performance scores.

Usage (standard train/val split):
    python ml/train.py --train_csv data/prepared/train.csv --val_csv data/prepared/val.csv

Usage (k-fold cross-validation):
    python ml/train.py --cv --data_csv data/prepared/all_data.csv --n_folds 6

Usage (with gating head - learns to reject non-trumpet audio):
    python ml/train.py --gated --train_csv data/prepared/train_gated.csv --val_csv data/prepared/val_gated.csv

The encoder (PANNs CNN14) is frozen - only the regression/gating heads are trained.
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.encoder_panns import PANNsEncoder
from ml.head_regressor import RegressionHead, GatingHead, SCORE_NAMES, unscale_scores
from ml.dataset import (
    TrumpetDataset, TrumpetDatasetWithGating,
    create_dataloaders, create_dataloader_from_df, create_gated_dataloader_from_df,
    create_augmented_dataloaders, create_augmented_dataloader_from_df,
)


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
    output_dir: str = "models/checkpoints",
    duration: float = 20.0,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
    augment: bool = False,
    seed: int = 42,
    num_workers: int = 6,
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
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
    """
    # Set seed for reproducibility
    set_seed(seed)
    
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
    if augment:
        print("  Using data augmentation for training set")
        train_loader, val_loader, _ = create_augmented_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=batch_size,
            duration=duration,
            augment_train=True,
            num_workers=num_workers,
        )
    else:
        train_loader, val_loader, _ = create_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=batch_size,
            duration=duration,
            num_workers=num_workers,
        )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Data loading workers: {num_workers}")
    
    # Setup training
    criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers than MSE
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
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
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
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCheckpoint saved to: {run_dir}")
    
    return head, history


def train_cv(
    data_csv: str,
    output_dir: str = "models/checkpoints",
    n_folds: int = 6,
    duration: float = 20.0,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
    augment: bool = False,
    seed: int = 42,
    num_workers: int = 6,
):
    """
    K-fold cross-validation training.
    
    Args:
        data_csv: Path to CSV with all data (will be split into folds)
        output_dir: Directory to save checkpoints
        n_folds: Number of folds for cross-validation (default: 6)
        duration: Audio duration in seconds
        batch_size: Training batch size
        learning_rate: Initial learning rate
        epochs: Maximum number of epochs per fold
        patience: Early stopping patience
        device: Device to use (None for auto-detect)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"cv_{n_folds}fold_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"TrumpetJudge {n_folds}-Fold Cross-Validation")
    print("=" * 60)
    
    # Load all data
    full_df = pd.read_csv(data_csv)
    print(f"\nLoaded {len(full_df)} total samples from {data_csv}")
    
    # Initialize encoder (frozen) - shared across all folds
    print("\nInitializing PANNs encoder (frozen)...")
    encoder = PANNsEncoder(duration=duration, device=device)
    device = encoder.device
    print(f"  Device: {device}")
    
    # Setup K-Fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Track metrics across all folds
    fold_results = []
    all_val_maes = []
    
    print("\n" + "=" * 60)
    print("Starting Cross-Validation...")
    print("=" * 60)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(full_df), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'='*60}")
        
        # Create fold-specific dataframes
        train_df = full_df.iloc[train_indices].reset_index(drop=True)
        val_df = full_df.iloc[val_indices].reset_index(drop=True)
        
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        
        # Create data loaders for this fold
        if augment:
            train_loader = create_augmented_dataloader_from_df(
                train_df, batch_size=batch_size, duration=duration, shuffle=True, augment=True,
                num_workers=num_workers
            )
        else:
            train_loader = create_dataloader_from_df(
                train_df, batch_size=batch_size, duration=duration, shuffle=True,
                num_workers=num_workers
            )
        val_loader = create_dataloader_from_df(
            val_df, batch_size=batch_size, duration=duration, shuffle=False,
            num_workers=num_workers
        )
        
        # Initialize fresh regression head for this fold
        head = RegressionHead(embedding_dim=encoder.embedding_dim)
        head = head.to(device)
        
        # Setup training
        criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers than MSE
        optimizer = Adam(head.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        # Training loop for this fold
        best_val_mae = float("inf")
        best_epoch = 0
        patience_counter = 0
        fold_history = {"train_loss": [], "val_loss": [], "val_mae": []}
        best_mae_per_score = None
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_epoch(encoder, head, train_loader, optimizer, criterion, device)
            
            # Validate
            val_metrics = validate(encoder, head, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            val_mae = val_metrics["mae"]
            
            # Update scheduler
            scheduler.step(val_mae)
            
            # Record history
            fold_history["train_loss"].append(train_loss)
            fold_history["val_loss"].append(val_loss)
            fold_history["val_mae"].append(val_mae)
            
            # Print progress
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.3f}")
            
            # Check for improvement
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                patience_counter = 0
                best_mae_per_score = val_metrics["mae_per_score"]
                
                # Save best model for this fold
                fold_dir = run_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "epoch": epoch,
                    "fold": fold_idx,
                    "head_state_dict": head.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_mae_per_score": val_metrics["mae_per_score"],
                }
                torch.save(checkpoint, fold_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        # Record fold results
        fold_result = {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "mae_per_score": best_mae_per_score,
            "train_size": len(train_df),
            "val_size": len(val_df),
        }
        fold_results.append(fold_result)
        all_val_maes.append(best_val_mae)
        
        # Save fold history
        with open(fold_dir / "history.json", "w") as f:
            json.dump(fold_history, f, indent=2)
        
        print(f"\n  Fold {fold_idx} Best: Epoch {best_epoch}, MAE: {best_val_mae:.3f}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("Cross-Validation Complete!")
    print("=" * 60)
    
    mean_mae = np.mean(all_val_maes)
    std_mae = np.std(all_val_maes)
    
    print(f"\n{n_folds}-Fold CV Results:")
    print(f"  Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
    print(f"\nPer-fold MAE:")
    for i, mae in enumerate(all_val_maes, 1):
        print(f"  Fold {i}: {mae:.3f}")
    
    # Aggregate per-score metrics
    print(f"\nMean MAE per score (across all folds):")
    aggregated_scores = {name: [] for name in SCORE_NAMES}
    for result in fold_results:
        for name in SCORE_NAMES:
            aggregated_scores[name].append(result["mae_per_score"][name])
    
    for name in SCORE_NAMES:
        mean_score = np.mean(aggregated_scores[name])
        std_score = np.std(aggregated_scores[name])
        print(f"  {name}: {mean_score:.3f} ± {std_score:.3f}")
    
    # Save aggregate results
    cv_summary = {
        "n_folds": n_folds,
        "data_csv": data_csv,
        "total_samples": len(full_df),
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "fold_maes": [float(m) for m in all_val_maes],
        "mean_mae_per_score": {name: float(np.mean(aggregated_scores[name])) for name in SCORE_NAMES},
        "std_mae_per_score": {name: float(np.std(aggregated_scores[name])) for name in SCORE_NAMES},
        "config": {
            "duration": duration,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
        },
        "fold_details": fold_results,
    }
    
    with open(run_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    
    return cv_summary


def train_gated_epoch(
    encoder: PANNsEncoder,
    gating_head: GatingHead,
    regression_head: RegressionHead,
    train_loader,
    optimizer,
    gating_criterion,
    regression_criterion,
    device: str,
    gating_weight: float = 1.0,
) -> dict:
    """
    Train for one epoch with both gating and regression heads.
    
    Returns:
        Dict with gating_loss, regression_loss, and total_loss
    """
    gating_head.train()
    regression_head.train()
    
    total_gating_loss = 0.0
    total_regression_loss = 0.0
    num_batches = 0
    
    for waveforms, labels, is_valid in train_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        is_valid = is_valid.to(device)
        
        # Forward pass through frozen encoder
        with torch.no_grad():
            embeddings = encoder(waveforms)
        
        # Gating head prediction (all samples)
        gating_pred = gating_head(embeddings)
        gating_loss = gating_criterion(gating_pred, is_valid)
        
        # Regression head prediction (only valid samples)
        valid_mask = is_valid.squeeze(-1) == 1
        if valid_mask.any():
            valid_embeddings = embeddings[valid_mask]
            valid_labels = labels[valid_mask]
            regression_pred = regression_head(valid_embeddings)
            regression_loss = regression_criterion(regression_pred, valid_labels)
        else:
            regression_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = gating_weight * gating_loss + regression_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_gating_loss += gating_loss.item()
        total_regression_loss += regression_loss.item()
        num_batches += 1
    
    return {
        "gating_loss": total_gating_loss / num_batches,
        "regression_loss": total_regression_loss / num_batches,
        "total_loss": (gating_weight * total_gating_loss + total_regression_loss) / num_batches,
    }


@torch.no_grad()
def validate_gated(
    encoder: PANNsEncoder,
    gating_head: GatingHead,
    regression_head: RegressionHead,
    val_loader,
    gating_criterion,
    regression_criterion,
    device: str,
) -> dict:
    """
    Validate both gating and regression heads.
    
    Returns:
        Dict with losses, gating accuracy, and regression MAE
    """
    gating_head.eval()
    regression_head.eval()
    
    total_gating_loss = 0.0
    total_regression_loss = 0.0
    num_batches = 0
    
    # For gating metrics
    all_gating_preds = []
    all_gating_labels = []
    
    # For regression metrics
    all_reg_preds = []
    all_reg_labels = []
    
    for waveforms, labels, is_valid in val_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        is_valid = is_valid.to(device)
        
        # Forward pass
        embeddings = encoder(waveforms)
        
        # Gating
        gating_pred = gating_head(embeddings)
        gating_loss = gating_criterion(gating_pred, is_valid)
        total_gating_loss += gating_loss.item()
        
        all_gating_preds.append(gating_pred.cpu())
        all_gating_labels.append(is_valid.cpu())
        
        # Regression (only valid samples)
        valid_mask = is_valid.squeeze(-1) == 1
        if valid_mask.any():
            valid_embeddings = embeddings[valid_mask]
            valid_labels = labels[valid_mask]
            regression_pred = regression_head(valid_embeddings)
            regression_loss = regression_criterion(regression_pred, valid_labels)
            total_regression_loss += regression_loss.item()
            
            all_reg_preds.append(regression_pred.cpu())
            all_reg_labels.append(valid_labels.cpu())
        
        num_batches += 1
    
    # Gating metrics
    all_gating_preds = torch.cat(all_gating_preds, dim=0)
    all_gating_labels = torch.cat(all_gating_labels, dim=0)
    gating_binary = (all_gating_preds >= 0.5).float()
    gating_accuracy = (gating_binary == all_gating_labels).float().mean().item()
    
    # Regression metrics (if any valid samples)
    if all_reg_preds:
        all_reg_preds = torch.cat(all_reg_preds, dim=0)
        all_reg_labels = torch.cat(all_reg_labels, dim=0)
        
        # Unscale for interpretable MAE
        all_reg_preds_unscaled = unscale_scores(all_reg_preds)
        all_reg_labels_unscaled = unscale_scores(all_reg_labels)
        
        mae_per_score = {}
        for i, name in enumerate(SCORE_NAMES):
            mae = torch.abs(all_reg_preds_unscaled[:, i] - all_reg_labels_unscaled[:, i]).mean().item()
            mae_per_score[name] = mae
        overall_mae = sum(mae_per_score.values()) / len(mae_per_score)
    else:
        mae_per_score = {name: 0.0 for name in SCORE_NAMES}
        overall_mae = 0.0
    
    return {
        "gating_loss": total_gating_loss / num_batches,
        "regression_loss": total_regression_loss / max(1, num_batches),
        "gating_accuracy": gating_accuracy,
        "mae": overall_mae,
        "mae_per_score": mae_per_score,
    }


def train_gated(
    train_csv: str,
    val_csv: str,
    output_dir: str = "models/checkpoints",
    duration: float = 20.0,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    gating_weight: float = 1.0,
    device: str = None,
    seed: int = 42,
    num_workers: int = 6,
):
    """
    Train both gating and regression heads together.
    
    Args:
        train_csv: Path to training CSV (with is_valid column)
        val_csv: Path to validation CSV (with is_valid column)
        output_dir: Directory to save checkpoints
        duration: Audio duration in seconds
        batch_size: Training batch size
        learning_rate: Initial learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
        gating_weight: Weight for gating loss in combined loss
        device: Device to use (None for auto-detect)
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"gated_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TrumpetJudge Gated Training")
    print("(Training both gating + regression heads)")
    print("=" * 60)
    
    # Initialize encoder (frozen)
    print("\nInitializing PANNs encoder (frozen)...")
    encoder = PANNsEncoder(duration=duration, device=device)
    device = encoder.device
    print(f"  Device: {device}")
    
    # Initialize heads (trainable)
    print("\nInitializing gating head (trainable)...")
    gating_head = GatingHead(embedding_dim=encoder.embedding_dim)
    gating_head = gating_head.to(device)
    gating_params = sum(p.numel() for p in gating_head.parameters() if p.requires_grad)
    print(f"  Gating head parameters: {gating_params:,}")
    
    print("\nInitializing regression head (trainable)...")
    regression_head = RegressionHead(embedding_dim=encoder.embedding_dim)
    regression_head = regression_head.to(device)
    reg_params = sum(p.numel() for p in regression_head.parameters() if p.requires_grad)
    print(f"  Regression head parameters: {reg_params:,}")
    
    # Create data loaders
    print("\nLoading datasets (with gating labels)...")
    train_dataset = TrumpetDatasetWithGating(train_csv, duration=duration)
    val_dataset = TrumpetDatasetWithGating(val_csv, duration=duration)
    
    from torch.utils.data import DataLoader
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent if num_workers > 0 else False
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Data loading workers: {num_workers}")
    
    # Setup training
    gating_criterion = nn.BCELoss()
    regression_criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers than MSE
    
    # Combined optimizer for both heads
    all_params = list(gating_head.parameters()) + list(regression_head.parameters())
    optimizer = Adam(all_params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_combined_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_gating_loss": [], "train_regression_loss": [],
        "val_gating_loss": [], "val_regression_loss": [],
        "val_gating_accuracy": [], "val_mae": [],
    }
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_gated_epoch(
            encoder, gating_head, regression_head, train_loader,
            optimizer, gating_criterion, regression_criterion, device, gating_weight
        )
        
        # Validate
        val_metrics = validate_gated(
            encoder, gating_head, regression_head, val_loader,
            gating_criterion, regression_criterion, device
        )
        
        # Combined metric for early stopping: (1 - gating_acc) + mae
        combined_metric = (1 - val_metrics["gating_accuracy"]) + val_metrics["mae"]
        
        # Update scheduler
        scheduler.step(combined_metric)
        
        # Record history
        history["train_gating_loss"].append(train_metrics["gating_loss"])
        history["train_regression_loss"].append(train_metrics["regression_loss"])
        history["val_gating_loss"].append(val_metrics["gating_loss"])
        history["val_regression_loss"].append(val_metrics["regression_loss"])
        history["val_gating_accuracy"].append(val_metrics["gating_accuracy"])
        history["val_mae"].append(val_metrics["mae"])
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Gate Acc: {val_metrics['gating_accuracy']:.3f} | "
              f"Reg MAE: {val_metrics['mae']:.3f} | "
              f"Gate Loss: {val_metrics['gating_loss']:.4f} | "
              f"Reg Loss: {val_metrics['regression_loss']:.4f}")
        
        # Check for improvement
        if combined_metric < best_combined_metric:
            best_combined_metric = combined_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "gating_head_state_dict": gating_head.state_dict(),
                "regression_head_state_dict": regression_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_gating_accuracy": val_metrics["gating_accuracy"],
                "val_mae": val_metrics["mae"],
                "val_mae_per_score": val_metrics["mae_per_score"],
            }
            torch.save(checkpoint, run_dir / "best_model.pt")
            print(f"  → Saved best model (Gate Acc: {val_metrics['gating_accuracy']:.3f}, MAE: {val_metrics['mae']:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    gating_head.load_state_dict(checkpoint["gating_head_state_dict"])
    regression_head.load_state_dict(checkpoint["regression_head_state_dict"])
    
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best gating accuracy: {checkpoint['val_gating_accuracy']:.3f}")
    print(f"  Best regression MAE: {checkpoint['val_mae']:.3f}")
    
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
        "gating_weight": gating_weight,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_gating_accuracy": checkpoint["val_gating_accuracy"],
        "best_val_mae": checkpoint["val_mae"],
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCheckpoint saved to: {run_dir}")
    
    return gating_head, regression_head, history


def main():
    parser = argparse.ArgumentParser(description="Train TrumpetJudge regression head")
    
    # Training modes
    parser.add_argument("--cv", action="store_true",
                        help="Enable k-fold cross-validation mode")
    parser.add_argument("--gated", action="store_true",
                        help="Enable gated training (with rejection detection)")
    parser.add_argument("--n_folds", type=int, default=6,
                        help="Number of folds for cross-validation (default: 6)")
    parser.add_argument("--data_csv", type=str, default=None,
                        help="Path to full dataset CSV (for CV mode)")
    
    # Standard train/val mode
    parser.add_argument("--train_csv", type=str, default="data/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default="data/val.csv",
                        help="Path to validation CSV")
    
    # Common arguments
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
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
    parser.add_argument("--gating_weight", type=float, default=1.0,
                        help="Weight for gating loss (gated mode only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detect if not specified.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation for training set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of data loading workers (default: 4, use 0 for single-threaded)")
    
    args = parser.parse_args()
    
    if args.cv:
        # Cross-validation mode
        if args.data_csv is None:
            parser.error("--data_csv is required when using --cv mode")
        
        train_cv(
            data_csv=args.data_csv,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            duration=args.duration,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            augment=args.augment,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    elif args.gated:
        # Gated training mode (with rejection detection)
        train_gated(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            output_dir=args.output_dir,
            duration=args.duration,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            gating_weight=args.gating_weight,
            device=args.device,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        # Standard train/val mode
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
            augment=args.augment,
            seed=args.seed,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()

