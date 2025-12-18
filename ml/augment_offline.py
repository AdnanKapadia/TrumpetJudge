"""
Offline Data Augmentation for TrumpetJudge.

Creates augmented copies of training audio files and saves them to disk.
This allows precomputing embeddings once and training much faster.

Usage:
    python ml/augment_offline.py --input_csv data/prepared/train.csv --num_augments 5
    
    # Fast mode with parallel processing:
    python ml/augment_offline.py --input_csv data/to_label.csv --num_augments 5 --workers 8 --fast

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.augment import WaveformAugmentPipeline


# Global pipeline for worker processes
_worker_pipeline = None
_worker_config = None


def _init_worker(config):
    """Initialize pipeline in worker process."""
    global _worker_pipeline, _worker_config
    _worker_config = config
    _worker_pipeline = WaveformAugmentPipeline(**config)


def _augment_one_file(args):
    """Process one file - called by worker."""
    row_dict, num_augments, output_dir, sample_rate = args
    global _worker_pipeline
    
    audio_path = row_dict["path"]
    
    # Handle path
    if not os.path.exists(audio_path):
        if os.path.exists(f"data/{audio_path}"):
            audio_path = f"data/{audio_path}"
        else:
            return []
    
    try:
        audio_data, sr = sf.read(audio_path, dtype='float32')
        
        # Resample if needed
        if sr != sample_rate:
            import torchaudio
            import torch
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        results = []
        for aug_idx in range(num_augments):
            augmented_audio = _worker_pipeline(audio_data.copy())
            
            original_name = Path(audio_path).stem
            aug_filename = f"{original_name}_aug{aug_idx:02d}.wav"
            aug_path = Path(output_dir) / aug_filename
            
            sf.write(aug_path, augmented_audio, sample_rate)
            
            aug_row = row_dict.copy()
            aug_row["id"] = f"{row_dict['id']}_aug{aug_idx:02d}"
            aug_row["path"] = str(aug_path)
            aug_row["original_id"] = row_dict["id"]
            aug_row["augment_idx"] = aug_idx
            results.append(aug_row)
        
        return results
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []


def create_augmented_dataset(
    input_csv: str,
    output_dir: str = "data/audio_augmented",
    output_csv: str = "data/prepared/train_augmented.csv",
    num_augments: int = 5,
    sample_rate: int = 32000,
    augment_config: dict = None,
    workers: int = 1,
    fast: bool = False,
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
        workers: Number of parallel workers (default: 1)
        fast: Use fast augmentation (no reverb, simpler transforms)
    """
    # Load input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup augmentation config
    if fast:
        # Fast mode: skip reverb (slowest), use simpler augments
        default_config = {
            "gain_p": 0.9,
            "gain_min_db": -8.0,
            "gain_max_db": 8.0,
            "shift_p": 0.8,
            "shift_min_ms": -50.0,
            "shift_max_ms": 50.0,
            "shift_mode": "zero",
            "noise_p": 0.5,
            "noise_types": ["white"],
            "noise_min_snr": 25.0,
            "noise_max_snr": 40.0,
            "reverb_p": 0.0,  # Skip reverb - it's slow!
            "sample_rate": sample_rate,
        }
        print("⚡ Fast mode: simplified augmentation (no reverb)")
    else:
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
    
    print(f"\nCreating {num_augments} augmented versions per sample...")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {workers}")
    
    augmented_rows = []
    
    if workers > 1:
        # Parallel processing
        print(f"🚀 Using {workers} parallel workers...")
        
        # Prepare args for workers
        args_list = [
            (row.to_dict(), num_augments, str(output_dir), sample_rate)
            for _, row in df.iterrows()
        ]
        
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(default_config,)
        ) as executor:
            futures = {executor.submit(_augment_one_file, args): i for i, args in enumerate(args_list)}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
                results = future.result()
                augmented_rows.extend(results)
    else:
        # Single-threaded (original behavior)
        pipeline = WaveformAugmentPipeline(**default_config)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            audio_path = row["path"]
            
            if not os.path.exists(audio_path):
                if os.path.exists(f"data/{audio_path}"):
                    audio_path = f"data/{audio_path}"
                else:
                    print(f"\n  Warning: Skipping {audio_path} (not found)")
                    continue
            
            audio_data, sr = sf.read(audio_path, dtype='float32')
            
            if sr != sample_rate:
                import torchaudio
                import torch
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze(0).numpy()
            
            for aug_idx in range(num_augments):
                augmented_audio = pipeline(audio_data.copy())
                
                original_name = Path(audio_path).stem
                aug_filename = f"{original_name}_aug{aug_idx:02d}.wav"
                aug_path = output_dir / aug_filename
                
                sf.write(aug_path, augmented_audio, sample_rate)
                
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
    
    print(f"\n📊 Summary:")
    print(f"   Original samples: {len(df)}")
    print(f"   Augments per sample: {num_augments}")
    print(f"   Total augmented: {len(aug_df)}")
    
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, try 4-8 for speedup)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip reverb and use simpler augments (~5x faster)")
    
    args = parser.parse_args()
    
    create_augmented_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        num_augments=args.num_augments,
        sample_rate=args.sample_rate,
        workers=args.workers,
        fast=args.fast,
    )


if __name__ == "__main__":
    main()

