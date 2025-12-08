"""
TrumpetDataset - PyTorch Dataset for loading trumpet audition audio and labels.

Expects CSV files with columns:
    id, path, overall, intonation, tone, timing, technique

Scores should be integers from 1-5.
"""

import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.head_regressor import SCORE_NAMES, scale_scores


class TrumpetDataset(Dataset):
    """
    PyTorch Dataset for trumpet audition recordings.
    
    Loads audio files and their corresponding human-labeled scores from a CSV file.
    Audio is preprocessed (mono, resampled, padded/trimmed) to fixed length.
    
    Attributes:
        csv_path (str): Path to CSV file with labels
        sample_rate (int): Target sample rate (32000 for PANNs)
        duration (float): Fixed audio duration in seconds
    """
    
    SAMPLE_RATE = 32000  # Must match PANNs encoder
    
    def __init__(
        self,
        csv_path: str,
        duration: float = 20.0,
        data_root: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with columns: id, path, overall, intonation, tone, timing, technique
            duration: Fixed audio duration in seconds (pad/trim to this length)
            data_root: Optional root directory for audio paths. If None, paths in CSV are used as-is.
        """
        self.csv_path = csv_path
        self.duration = duration
        self.data_root = data_root
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ["id", "path"] + SCORE_NAMES
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        # Validate score ranges
        for col in SCORE_NAMES:
            if not self.df[col].between(1, 5).all():
                bad_rows = self.df[~self.df[col].between(1, 5)]
                raise ValueError(f"Score '{col}' must be between 1-5. Invalid rows:\n{bad_rows}")
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, labels):
                - waveform: Preprocessed audio tensor of shape (num_samples,)
                - labels: Score tensor of shape (5,) with values scaled to [0, 1]
        """
        row = self.df.iloc[idx]
        
        # Build audio path
        audio_path = row["path"]
        if self.data_root:
            audio_path = os.path.join(self.data_root, audio_path)
        
        # Load and preprocess audio
        waveform = self._load_and_preprocess(audio_path)
        
        # Get labels and scale to [0, 1]
        labels = torch.tensor([row[name] for name in SCORE_NAMES], dtype=torch.float32)
        labels = scale_scores(labels)
        
        return waveform, labels
    
    def _load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and preprocess for PANNs encoder.
        
        Steps:
            1. Load audio
            2. Convert to mono
            3. Resample to 32kHz
            4. Pad or trim to fixed duration
            
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor of shape (num_samples,)
        """
        # Load audio using soundfile directly (avoids torchcodec dependency)
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
        else:
            waveform = waveform.permute(1, 0)  # (samples, channels) -> (channels, samples)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 32kHz if needed
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        
        # Pad or trim to fixed length
        current_samples = waveform.shape[1]
        
        if current_samples < self.num_samples:
            # Pad with zeros
            padding = self.num_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > self.num_samples:
            # Trim from start
            waveform = waveform[:, :self.num_samples]
        
        # Remove channel dimension: (1, samples) -> (samples,)
        waveform = waveform.squeeze(0)
        
        return waveform
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample (for debugging/display)."""
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "path": row["path"],
            "scores": {name: row[name] for name in SCORE_NAMES},
        }


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train, val, and optionally test sets.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Optional path to test CSV
        batch_size: Batch size for DataLoader
        duration: Fixed audio duration in seconds
        data_root: Optional root directory for audio paths
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader is None if test_csv not provided
    """
    train_dataset = TrumpetDataset(train_csv, duration=duration, data_root=data_root)
    val_dataset = TrumpetDataset(val_csv, duration=duration, data_root=data_root)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = None
    if test_csv:
        test_dataset = TrumpetDataset(test_csv, duration=duration, data_root=data_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Quick test with dummy data."""
    import tempfile
    import numpy as np
    
    print("Creating temporary test data...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy audio files
        audio_dir = os.path.join(tmpdir, "audio")
        os.makedirs(audio_dir)
        
        csv_rows = []
        for i in range(5):
            # Create 3-second dummy audio
            duration = 3.0
            sr = 32000
            audio = torch.randn(1, int(sr * duration)) * 0.1
            audio_path = os.path.join(audio_dir, f"{i:04d}.wav")
            torchaudio.save(audio_path, audio, sr)
            
            # Random scores 1-5
            scores = np.random.randint(1, 6, size=5)
            csv_rows.append({
                "id": f"{i:04d}",
                "path": audio_path,
                "overall": scores[0],
                "intonation": scores[1],
                "tone": scores[2],
                "timing": scores[3],
                "technique": scores[4],
            })
        
        # Write CSV
        csv_path = os.path.join(tmpdir, "test.csv")
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)
        print(f"  Created {len(csv_rows)} dummy samples")
        
        # Test dataset
        print("\nTesting TrumpetDataset...")
        dataset = TrumpetDataset(csv_path, duration=5.0)
        print(f"  Dataset length: {len(dataset)}")
        
        # Get a sample
        waveform, labels = dataset[0]
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels (scaled 0-1): {labels.tolist()}")
        
        # Test DataLoader
        print("\nTesting DataLoader...")
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_waveform, batch_labels = next(iter(loader))
        print(f"  Batch waveform shape: {batch_waveform.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        
        print("\n✓ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()

