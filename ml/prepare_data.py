"""
Data preparation script for TrumpetJudge training.

Links label CSVs with the master to_label.csv to create train/val splits.
Ensures samples from the same video_id stay together in train or val.

Usage (standard - valid samples only):
    python ml/prepare_data.py --labels data/labels/labels_adnan.csv
    python ml/prepare_data.py --labels data/labels/labels_adnan.csv data/labels/labels_unc.csv

Usage (with gating - includes rejected samples for training the gating head):
    python ml/prepare_data.py --labels data/labels/*.csv --include_rejected

Usage (produce combined CSV for k-fold CV):
    python ml/prepare_data.py --labels data/labels/*.csv --save_all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCORE_COLS = ['overall', 'intonation', 'tone', 'timing', 'technique']


def prepare_data(
    labels_csvs: List[str],
    to_label_csv: str = "data/to_label.csv",
    output_dir: str = "data/prepared",
    val_fraction: float = 0.2,
    seed: int = 42,
    include_rejected: bool = False,
    save_all: bool = False,
    all_filename: str = None,
):
    """
    Prepare training data by linking labels with audio paths.
    
    Args:
        labels_csvs: List of paths to label CSVs (e.g., [labels_adnan.csv, labels_unc.csv])
        to_label_csv: Path to master CSV with audio paths and video_ids
        output_dir: Directory to save prepared train/val CSVs
        val_fraction: Fraction of videos to use for validation
        seed: Random seed for reproducibility
        include_rejected: If True, include rejected samples with is_valid=0 for gating head training
        save_all: If True, also save a single combined CSV (for k-fold CV)
        all_filename: Optional custom filename for the combined CSV
    """
    print("=" * 60)
    print("TrumpetJudge Data Preparation")
    if include_rejected:
        print("(Including rejected samples for gating head)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and combine labels from all files
    all_labels = []
    for labels_csv in labels_csvs:
        print(f"\nLoading labels from: {labels_csv}")
        df = pd.read_csv(labels_csv)
        print(f"  Total entries: {len(df)}")
        all_labels.append(df)
    
    labels_df = pd.concat(all_labels, ignore_index=True)
    print(f"\nCombined total entries: {len(labels_df)}")
    
    # Separate valid and rejected samples
    valid_labels = labels_df[labels_df['rejected'] == False].copy()
    rejected_labels = labels_df[labels_df['rejected'] == True].copy()
    print(f"  Valid samples: {len(valid_labels)}")
    print(f"  Rejected samples: {len(rejected_labels)}")
    
    # Average duplicate sample_ids for valid samples (when same clip labeled by multiple users)
    duplicates = valid_labels['sample_id'].duplicated(keep=False).sum()
    if duplicates > 0:
        print(f"  Duplicate valid sample_ids found: {duplicates} entries")
        # Group by sample_id and average the scores
        valid_labels = valid_labels.groupby('sample_id', as_index=False).agg({
            **{col: 'mean' for col in SCORE_COLS},
            'user_id': 'first',
        })
        for col in SCORE_COLS:
            valid_labels[col] = valid_labels[col].round().astype(int)
        print(f"  After averaging duplicates: {len(valid_labels)} unique valid samples")
    
    # Deduplicate rejected samples
    if include_rejected and len(rejected_labels) > 0:
        rejected_labels = rejected_labels.drop_duplicates(subset=['sample_id'], keep='first')
        print(f"  Unique rejected samples: {len(rejected_labels)}")
    
    # Load master CSV with paths
    print(f"\nLoading audio metadata from: {to_label_csv}")
    master_df = pd.read_csv(to_label_csv)
    print(f"  Total entries: {len(master_df)}")
    
    # Merge valid labels with master (to get path and video_id)
    merged_valid = valid_labels.merge(
        master_df[['id', 'video_id', 'path']],
        left_on='sample_id',
        right_on='id',
        how='inner'
    )
    merged_valid['is_valid'] = 1
    print(f"\nMerged valid samples: {len(merged_valid)}")
    
    if len(merged_valid) < len(valid_labels):
        missing = len(valid_labels) - len(merged_valid)
        print(f"  Warning: {missing} valid samples could not be matched to audio files")
    
    # Merge rejected labels if including them
    if include_rejected and len(rejected_labels) > 0:
        merged_rejected = rejected_labels.merge(
            master_df[['id', 'video_id', 'path']],
            left_on='sample_id',
            right_on='id',
            how='inner'
        )
        merged_rejected['is_valid'] = 0
        # Fill NaN scores with 0 (they won't be used in training)
        for col in SCORE_COLS:
            merged_rejected[col] = 0
        print(f"Merged rejected samples: {len(merged_rejected)}")
        
        # Combine valid and rejected
        merged = pd.concat([merged_valid, merged_rejected], ignore_index=True)
        print(f"\nTotal merged (valid + rejected): {len(merged)}")
    else:
        merged = merged_valid
    
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
    if include_rejected:
        output_cols = ['id', 'path', 'overall', 'intonation', 'tone', 'timing', 'technique', 'is_valid']
    else:
        output_cols = ['id', 'path', 'overall', 'intonation', 'tone', 'timing', 'technique']
    
    # Rename sample_id to id for consistency (using the matched 'id' from master)
    train_out = train_df[output_cols].copy()
    val_out = val_df[output_cols].copy()
    
    # Prepend 'data/' to paths since training runs from project root
    train_out['path'] = 'data/' + train_out['path'].astype(str)
    val_out['path'] = 'data/' + val_out['path'].astype(str)
    
    # Save CSVs
    if include_rejected:
        train_path = output_dir / "train_gated.csv"
        val_path = output_dir / "val_gated.csv"
    else:
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
    
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    
    print(f"\nSaved train CSV: {train_path}")
    print(f"Saved val CSV: {val_path}")

    # Optionally save combined CSV (useful for k-fold CV)
    if save_all:
        if all_filename:
            all_path = output_dir / all_filename
        else:
            all_path = output_dir / ("all_gated.csv" if include_rejected else "all_data.csv")
        merged_out = merged[output_cols].copy()
        merged_out['path'] = 'data/' + merged_out['path'].astype(str)
        merged_out.to_csv(all_path, index=False)
        print(f"Saved combined CSV for CV: {all_path}")
    
    # Print distribution summary
    print("\n" + "=" * 60)
    print("Data Distribution Summary")
    print("=" * 60)
    
    for split_name, df in [("Train", train_out), ("Val", val_out)]:
        print(f"\n{split_name}:")
        if include_rejected:
            n_valid = (df['is_valid'] == 1).sum()
            n_rejected = (df['is_valid'] == 0).sum()
            print(f"  Valid samples: {n_valid}")
            print(f"  Rejected samples: {n_rejected}")
            # Only show score stats for valid samples
            valid_df = df[df['is_valid'] == 1]
            if len(valid_df) > 0:
                print(f"  Score distribution (valid only):")
                for col in SCORE_COLS:
                    mean = valid_df[col].mean()
                    std = valid_df[col].std()
                    print(f"    {col}: mean={mean:.2f}, std={std:.2f}")
        else:
            for col in SCORE_COLS:
                mean = df[col].mean()
                std = df[col].std()
                print(f"  {col}: mean={mean:.2f}, std={std:.2f}")
    
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from labels")
    parser.add_argument("--labels", type=str, nargs='+', required=True,
                        help="Path(s) to labels CSV(s). Can specify multiple files.")
    parser.add_argument("--master", type=str, default="data/to_label.csv",
                        help="Path to master CSV with audio paths")
    parser.add_argument("--output_dir", type=str, default="data/prepared",
                        help="Directory to save prepared CSVs")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of videos for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--include_rejected", action="store_true",
                        help="Include rejected samples for gating head training")
    parser.add_argument("--save_all", action="store_true",
                        help="Also save a combined CSV for CV")
    parser.add_argument("--all_filename", type=str, default=None,
                        help="Custom filename for the combined CSV (default: all_data.csv or all_gated.csv)")
    
    args = parser.parse_args()
    
    prepare_data(
        labels_csvs=args.labels,
        to_label_csv=args.master,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        include_rejected=args.include_rejected,
        save_all=args.save_all,
        all_filename=args.all_filename,
    )


if __name__ == "__main__":
    main()

