"""
Offline Data Augmentation for TrumpetJudge.

Creates augmented copies of training audio files and saves them to disk.
This allows precomputing embeddings once and training much faster.

Usage:
    python ml/augment_offline.py --input_csv data/prepared/train.csv --num_augments 5

Output:
    - Augmented audio files in data/audio_augmented/
    - New CSV manifest: data/prepared/train_augmented.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.augment import WaveformAugmentPipeline


def create_augmented_dataset(
    input_csv: str,
    output_dir: str = "data/audio_augmented",
    output_csv: str = "data/prepared/train_augmented.csv",
    num_augments: int = 5,
    sample_rate: int = 32000,
    augment_config: dict = None,
):
    """
    Create augmented copies of training audio files.
    
    Args:
        input_csv: Path to input CSV (train.csv)
        output_dir: Directory to save augmented audio files
        output_csv: Path to output CSV manifest
        num_augments: Number of augmented versions per original sample
        sample_rate: Sample rate for audio files
        augment_config: Optional augmentation configuration
    """
    # Load input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup augmentation pipeline
    default_config = {
        "gain_p": 0.9,
        "gain_min_db": -12.0,
        "gain_max_db": 12.0,
        "shift_p": 0.9,
        "shift_min_ms": -100.0,
        "shift_max_ms": 100.0,
        "shift_mode": "zero",
        "noise_p": 0.7,
        "noise_types": ["white", "pink", "brown"],
        "noise_min_snr": 20.0,
        "noise_max_snr": 40.0,
        "reverb_p": 0.6,
        "reverb_min_wet": 0.05,
        "reverb_max_wet": 0.25,
        "reverb_ir_dir": "data/impulse_response",
        "sample_rate": sample_rate,
    }
    
    if augment_config:
        default_config.update(augment_config)
    
    pipeline = WaveformAugmentPipeline(**default_config)
    
    # Create augmented samples
    augmented_rows = []
    
    print(f"\nCreating {num_augments} augmented versions per sample...")
    print(f"Output directory: {output_dir}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        # Load original audio
        audio_path = row["path"]
        audio_data, sr = sf.read(audio_path, dtype='float32')
        
        # Resample if needed
        if sr != sample_rate:
            import torchaudio
            import torch
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        # Create augmented versions
        for aug_idx in range(num_augments):
            # Apply augmentation
            augmented_audio = pipeline(audio_data.copy())
            
            # Generate output filename
            original_name = Path(audio_path).stem
            aug_filename = f"{original_name}_aug{aug_idx:02d}.wav"
            aug_path = output_dir / aug_filename
            
            # Save augmented audio
            sf.write(aug_path, augmented_audio, sample_rate)
            
            # Add to manifest
            aug_row = row.copy()
            aug_row["id"] = f"{row['id']}_aug{aug_idx:02d}"
            aug_row["path"] = str(aug_path)
            aug_row["original_id"] = row["id"]
            aug_row["augment_idx"] = aug_idx
            augmented_rows.append(aug_row)
    
    # Create augmented CSV
    aug_df = pd.DataFrame(augmented_rows)
    
    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Created {len(aug_df)} augmented samples")
    print(f"   Audio files: {output_dir}/")
    print(f"   Manifest: {output_csv}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Original samples: {len(df)}")
    print(f"   Augments per sample: {num_augments}")
    print(f"   Total augmented: {len(aug_df)}")
    print(f"   Combined (orig + aug): {len(df) + len(aug_df)}")
    
    return aug_df


def main():
    parser = argparse.ArgumentParser(description="Create augmented training data offline")
    
    parser.add_argument("--input_csv", type=str, default="data/prepared/train.csv",
                        help="Path to input CSV (training data)")
    parser.add_argument("--output_dir", type=str, default="data/audio_augmented",
                        help="Directory to save augmented audio files")
    parser.add_argument("--output_csv", type=str, default="data/prepared/train_augmented.csv",
                        help="Path to output CSV manifest")
    parser.add_argument("--num_augments", type=int, default=5,
                        help="Number of augmented versions per sample (default: 5)")
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Sample rate for output audio (default: 32000)")
    
    args = parser.parse_args()
    
    create_augmented_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        num_augments=args.num_augments,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()

