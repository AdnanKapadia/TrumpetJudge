"""
Precompute PANNs Embeddings for TrumpetJudge.

Extracts embeddings from all audio files once and saves them to disk.
This allows training the MLP head without running PANNs every epoch.

Usage:
    # Precompute for train, val, and augmented train
    python ml/precompute.py --csv data/prepared/train.csv --output data/embeddings/train.pt
    python ml/precompute.py --csv data/prepared/val.csv --output data/embeddings/val.pt
    python ml/precompute.py --csv data/prepared/train_augmented.csv --output data/embeddings/train_augmented.pt

    # Or use the convenience command to do all at once:
    python ml/precompute.py --all

Output:
    - .pt files containing {embeddings, labels, ids} tensors
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder_panns import PANNsEncoder
from models.head_regressor import SCORE_NAMES, scale_scores


def precompute_embeddings(
    csv_path: str,
    output_path: str,
    duration: float = 20.0,
    batch_size: int = 8,
    device: str = None,
):
    """
    Precompute embeddings for all samples in a CSV.
    
    Args:
        csv_path: Path to CSV with audio paths and labels
        output_path: Path to save embeddings (.pt file)
        duration: Audio duration in seconds
        batch_size: Batch size for processing
        device: Device to use (None for auto-detect)
    
    Saves a dict with:
        - embeddings: Tensor of shape (N, 2048)
        - labels: Tensor of shape (N, 5) - scaled to [0, 1]
        - ids: List of sample IDs
        - paths: List of audio paths
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loading {len(df)} samples from {csv_path}")
    
    # Initialize encoder
    print("\nInitializing PANNs encoder...")
    encoder = PANNsEncoder(duration=duration, device=device)
    device = encoder.device
    print(f"  Device: {device}")
    
    # Prepare data loading
    from ml.dataset import TrumpetDataset
    dataset = TrumpetDataset(dataframe=df, duration=duration)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for ID matching
        num_workers=4,
        pin_memory=True,
    )
    
    # Extract embeddings
    all_embeddings = []
    all_labels = []
    
    print(f"\nExtracting embeddings...")
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Processing"):
            waveforms = waveforms.to(device)
            embeddings = encoder(waveforms)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Get IDs and paths
    ids = df["id"].tolist()
    paths = df["path"].tolist()
    
    # Create output dict
    output = {
        "embeddings": all_embeddings,
        "labels": all_labels,
        "ids": ids,
        "paths": paths,
        "duration": duration,
        "embedding_dim": all_embeddings.shape[1],
        "num_samples": len(df),
        "score_names": SCORE_NAMES,
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    
    print(f"\n✅ Saved embeddings to {output_path}")
    print(f"   Samples: {len(df)}")
    print(f"   Embedding shape: {all_embeddings.shape}")
    print(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    return output


def precompute_all(
    train_csv: str = "data/prepared/train.csv",
    val_csv: str = "data/prepared/val.csv",
    train_aug_csv: str = "data/prepared/train_augmented.csv",
    output_dir: str = "data/embeddings",
    duration: float = 20.0,
    batch_size: int = 8,
    device: str = None,
):
    """
    Precompute embeddings for train, val, and augmented train sets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoder once (reuse across all sets)
    print("=" * 60)
    print("Precomputing PANNs Embeddings")
    print("=" * 60)
    
    print("\nInitializing PANNs encoder...")
    encoder = PANNsEncoder(duration=duration, device=device)
    device = encoder.device
    print(f"  Device: {device}")
    
    from ml.dataset import TrumpetDataset
    from torch.utils.data import DataLoader
    
    def process_csv(csv_path, output_name):
        if not os.path.exists(csv_path):
            print(f"\n⚠️  Skipping {csv_path} (not found)")
            return None
            
        df = pd.read_csv(csv_path)
        print(f"\n📁 Processing {csv_path} ({len(df)} samples)...")
        
        dataset = TrumpetDataset(dataframe=df, duration=duration)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for waveforms, labels in tqdm(loader, desc=f"  {output_name}"):
                waveforms = waveforms.to(device)
                embeddings = encoder(waveforms)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        output = {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "ids": df["id"].tolist(),
            "paths": df["path"].tolist(),
            "duration": duration,
            "embedding_dim": all_embeddings.shape[1],
            "num_samples": len(df),
            "score_names": SCORE_NAMES,
        }
        
        output_path = output_dir / f"{output_name}.pt"
        torch.save(output, output_path)
        print(f"   ✅ Saved: {output_path} ({all_embeddings.shape[0]} × {all_embeddings.shape[1]})")
        
        return output
    
    # Process each set
    results = {}
    results["train"] = process_csv(train_csv, "train")
    results["val"] = process_csv(val_csv, "val")
    results["train_augmented"] = process_csv(train_aug_csv, "train_augmented")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Precomputation Complete!")
    print("=" * 60)
    print(f"\nEmbeddings saved to: {output_dir}/")
    
    for name, data in results.items():
        if data:
            print(f"  {name}.pt: {data['num_samples']} samples")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute PANNs embeddings")
    
    parser.add_argument("--all", action="store_true",
                        help="Precompute for train, val, and train_augmented")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to single CSV to process")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single CSV mode")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="Output directory for --all mode")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.all:
        precompute_all(
            output_dir=args.output_dir,
            duration=args.duration,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.csv:
        if args.output is None:
            # Generate output path from input
            stem = Path(args.csv).stem
            args.output = f"data/embeddings/{stem}.pt"
        
        precompute_embeddings(
            csv_path=args.csv,
            output_path=args.output,
            duration=args.duration,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python ml/precompute.py --all")
        print("  python ml/precompute.py --csv data/prepared/train.csv --output data/embeddings/train.pt")


if __name__ == "__main__":
    main()

