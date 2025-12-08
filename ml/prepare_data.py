"""
Data preparation script for TrumpetJudge training.

Links label CSVs with the master to_label.csv to create train/val splits.
Ensures samples from the same video_id stay together in train or val.

Usage:
    python ml/prepare_data.py --labels data/labels/labels_adnan.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_data(
    labels_csv: str,
    to_label_csv: str = "data/to_label.csv",
    output_dir: str = "data/prepared",
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Prepare training data by linking labels with audio paths.
    
    Args:
        labels_csv: Path to labels CSV (e.g., labels_adnan.csv)
        to_label_csv: Path to master CSV with audio paths and video_ids
        output_dir: Directory to save prepared train/val CSVs
        val_fraction: Fraction of videos to use for validation
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("TrumpetJudge Data Preparation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print(f"\nLoading labels from: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    print(f"  Total entries: {len(labels_df)}")
    
    # Filter out rejected samples
    valid_labels = labels_df[labels_df['rejected'] == False].copy()
    rejected_count = len(labels_df) - len(valid_labels)
    print(f"  Rejected samples: {rejected_count}")
    print(f"  Valid samples: {len(valid_labels)}")
    
    # Load master CSV with paths
    print(f"\nLoading audio metadata from: {to_label_csv}")
    master_df = pd.read_csv(to_label_csv)
    print(f"  Total entries: {len(master_df)}")
    
    # Merge labels with master (to get path and video_id)
    # labels has 'sample_id', master has 'id'
    merged = valid_labels.merge(
        master_df[['id', 'video_id', 'path']],
        left_on='sample_id',
        right_on='id',
        how='inner'
    )
    print(f"\nMerged dataset: {len(merged)} samples")
    
    if len(merged) < len(valid_labels):
        missing = len(valid_labels) - len(merged)
        print(f"  Warning: {missing} samples could not be matched to audio files")
    
    # Get unique video IDs
    unique_videos = merged['video_id'].unique()
    n_videos = len(unique_videos)
    print(f"\nUnique videos: {n_videos}")
    
    # Split videos into train/val
    np.random.seed(seed)
    np.random.shuffle(unique_videos)
    
    n_val = max(1, int(n_videos * val_fraction))
    n_train = n_videos - n_val
    
    val_videos = set(unique_videos[:n_val])
    train_videos = set(unique_videos[n_val:])
    
    print(f"  Train videos: {n_train}")
    print(f"  Val videos: {n_val}")
    
    # Split samples based on video assignment
    train_df = merged[merged['video_id'].isin(train_videos)].copy()
    val_df = merged[merged['video_id'].isin(val_videos)].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Prepare output columns (matching expected format)
    output_cols = ['id', 'path', 'overall', 'intonation', 'tone', 'timing', 'technique']
    
    # Rename sample_id to id for consistency (using the matched 'id' from master)
    train_out = train_df[output_cols].copy()
    val_out = val_df[output_cols].copy()
    
    # Prepend 'data/' to paths since training runs from project root
    train_out['path'] = 'data/' + train_out['path'].astype(str)
    val_out['path'] = 'data/' + val_out['path'].astype(str)
    
    # Save CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    
    print(f"\nSaved train CSV: {train_path}")
    print(f"Saved val CSV: {val_path}")
    
    # Print score distribution summary
    print("\n" + "=" * 60)
    print("Score Distribution Summary")
    print("=" * 60)
    
    for split_name, df in [("Train", train_out), ("Val", val_out)]:
        print(f"\n{split_name}:")
        for col in ['overall', 'intonation', 'tone', 'timing', 'technique']:
            mean = df[col].mean()
            std = df[col].std()
            print(f"  {col}: mean={mean:.2f}, std={std:.2f}")
    
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from labels")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels CSV (e.g., data/labels/labels_adnan.csv)")
    parser.add_argument("--master", type=str, default="data/to_label.csv",
                        help="Path to master CSV with audio paths")
    parser.add_argument("--output_dir", type=str, default="data/prepared",
                        help="Directory to save prepared CSVs")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of videos for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_data(
        labels_csv=args.labels,
        to_label_csv=args.master,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

