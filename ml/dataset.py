"""
TrumpetDataset - PyTorch Dataset for loading trumpet audition audio and labels.

Expects CSV files with columns:
    id, path, overall, intonation, tone, timing, technique

For gated training (with rejection detection):
    id, path, overall, intonation, tone, timing, technique, is_valid

Scores should be integers from 1-5 for valid samples.
Rejected samples have is_valid=0 and scores can be any value (ignored in training).
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

from ml.head_regressor import SCORE_NAMES, scale_scores


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
        csv_path: Optional[str] = None,
        duration: float = 20.0,
        data_root: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with columns: id, path, overall, intonation, tone, timing, technique
            duration: Fixed audio duration in seconds (pad/trim to this length)
            data_root: Optional root directory for audio paths. If None, paths in CSV are used as-is.
            dataframe: Optional DataFrame to use directly instead of loading from CSV.
        """
        self.csv_path = csv_path
        self.duration = duration
        self.data_root = data_root
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        
        # Load from DataFrame or CSV
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Must provide either csv_path or dataframe")
        
        # Validate columns
        required_cols = ["id", "path"] + SCORE_NAMES
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Validate score ranges
        for col in SCORE_NAMES:
            if not self.df[col].between(1, 5).all():
                bad_rows = self.df[~self.df[col].between(1, 5)]
                raise ValueError(f"Score '{col}' must be between 1-5. Invalid rows:\n{bad_rows}")
        
        source = csv_path if csv_path else "DataFrame"
        print(f"Loaded {len(self.df)} samples from {source}")
    
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


class TrumpetDatasetWithGating(Dataset):
    """
    PyTorch Dataset for trumpet audition recordings WITH gating labels.
    
    Returns (waveform, labels, is_valid) tuples where:
    - waveform: Preprocessed audio tensor
    - labels: Score tensor (5,) - only meaningful when is_valid=1
    - is_valid: Binary label (1=valid trumpet, 0=rejected/invalid)
    
    Rejected samples (is_valid=0) include:
    - Talking/speech
    - Non-trumpet instruments
    - Silence/empty audio
    - Very bad quality recordings
    """
    
    SAMPLE_RATE = 32000  # Must match PANNs encoder
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        duration: float = 20.0,
        data_root: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV with columns: id, path, overall, intonation, tone, timing, technique, is_valid
            duration: Fixed audio duration in seconds
            data_root: Optional root directory for audio paths
            dataframe: Optional DataFrame to use directly instead of loading from CSV
        """
        self.csv_path = csv_path
        self.duration = duration
        self.data_root = data_root
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        
        # Load from DataFrame or CSV
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Must provide either csv_path or dataframe")
        
        # Validate columns
        required_cols = ["id", "path", "is_valid"] + SCORE_NAMES
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Count valid/invalid samples
        n_valid = (self.df['is_valid'] == 1).sum()
        n_invalid = (self.df['is_valid'] == 0).sum()
        
        # Validate score ranges only for valid samples
        valid_df = self.df[self.df['is_valid'] == 1]
        for col in SCORE_NAMES:
            if len(valid_df) > 0 and not valid_df[col].between(1, 5).all():
                bad_rows = valid_df[~valid_df[col].between(1, 5)]
                raise ValueError(f"Score '{col}' must be between 1-5 for valid samples. Invalid rows:\n{bad_rows}")
        
        source = csv_path if csv_path else "DataFrame"
        print(f"Loaded {len(self.df)} samples from {source} ({n_valid} valid, {n_invalid} rejected)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, labels, is_valid):
                - waveform: Preprocessed audio tensor of shape (num_samples,)
                - labels: Score tensor of shape (5,) scaled to [0, 1]
                - is_valid: Binary tensor of shape (1,) - 1=valid, 0=rejected
        """
        row = self.df.iloc[idx]
        
        # Build audio path
        audio_path = row["path"]
        if self.data_root:
            audio_path = os.path.join(self.data_root, audio_path)
        
        # Load and preprocess audio
        waveform = self._load_and_preprocess(audio_path)
        
        # Get validity label
        is_valid = torch.tensor([float(row["is_valid"])], dtype=torch.float32)
        
        # Get labels - for rejected samples, use zeros (will be masked in loss)
        if row["is_valid"] == 1:
            labels = torch.tensor([row[name] for name in SCORE_NAMES], dtype=torch.float32)
            labels = scale_scores(labels)
        else:
            # Rejected sample - scores don't matter, use zeros
            labels = torch.zeros(len(SCORE_NAMES), dtype=torch.float32)
        
        return waveform, labels, is_valid
    
    def _load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file (same as TrumpetDataset)."""
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.permute(1, 0)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        
        current_samples = waveform.shape[1]
        if current_samples < self.num_samples:
            padding = self.num_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        
        waveform = waveform.squeeze(0)
        return waveform
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        row = self.df.iloc[idx]
        info = {
            "id": row["id"],
            "path": row["path"],
            "is_valid": bool(row["is_valid"]),
        }
        if row["is_valid"] == 1:
            info["scores"] = {name: row[name] for name in SCORE_NAMES}
        return info


def create_gated_dataloader_from_df(
    df: pd.DataFrame,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 6,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with gating labels from a DataFrame.
    
    Args:
        df: DataFrame with columns: id, path, overall, intonation, tone, timing, technique, is_valid
        batch_size: Batch size for DataLoader
        duration: Fixed audio duration in seconds
        data_root: Optional root directory for audio paths
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the gated dataset
    """
    dataset = TrumpetDatasetWithGating(dataframe=df, duration=duration, data_root=data_root)
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return loader


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 6,
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
    
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    
    test_loader = None
    if test_csv:
        test_dataset = TrumpetDataset(test_csv, duration=duration, data_root=data_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
    
    return train_loader, val_loader, test_loader


def create_dataloader_from_df(
    df: pd.DataFrame,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 6,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a DataFrame.
    
    Args:
        df: DataFrame with columns: id, path, overall, intonation, tone, timing, technique
        batch_size: Batch size for DataLoader
        duration: Fixed audio duration in seconds
        data_root: Optional root directory for audio paths
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the dataset
    """
    dataset = TrumpetDataset(dataframe=df, duration=duration, data_root=data_root)
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return loader


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


class AugmentedTrumpetDataset(Dataset):
    """
    Augmented version of TrumpetDataset with blind data augmentation.
    
    Applies waveform augmentations during training to improve generalization:
    - Gain/loudness variation
    - Time shift
    - Additive noise
    - Convolution reverb
    
    Note: SpecAugment should be applied separately if using mel spectrograms.
    """
    
    SAMPLE_RATE = 32000
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        duration: float = 20.0,
        data_root: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        augment: bool = True,
        augment_config: Optional[Dict] = None,
    ):
        """
        Initialize the augmented dataset.
        
        Args:
            csv_path: Path to CSV file with labels
            duration: Fixed audio duration in seconds
            data_root: Optional root directory for audio paths
            dataframe: Optional DataFrame to use directly
            augment: Whether to apply augmentations
            augment_config: Optional config dict to override default augmentation settings
        """
        self.csv_path = csv_path
        self.duration = duration
        self.data_root = data_root
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        self.augment = augment
        
        # Load from DataFrame or CSV
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Must provide either csv_path or dataframe")
        
        # Validate columns
        required_cols = ["id", "path"] + SCORE_NAMES
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Setup augmentation pipeline
        if augment:
            from ml.augment import WaveformAugmentPipeline
            
            # Default augmentation config
            default_config = {
                "gain_p": 0.8,
                "gain_min_db": -12.0,
                "gain_max_db": 12.0,
                "shift_p": 0.8,
                "shift_min_ms": -50.0,
                "shift_max_ms": 50.0,
                "shift_mode": "zero",
                "noise_p": 0.5,
                "noise_types": ["white", "pink", "brown"],
                "noise_min_snr": 20.0,
                "noise_max_snr": 35.0,
                "reverb_p": 0.5,
                "reverb_min_wet": 0.05,
                "reverb_max_wet": 0.20,
                "reverb_ir_dir": None,
                "sample_rate": self.SAMPLE_RATE,
            }
            
            # Override with user config
            if augment_config:
                default_config.update(augment_config)
            
            self.augment_pipeline = WaveformAugmentPipeline(**default_config)
        else:
            self.augment_pipeline = None
        
        source = csv_path if csv_path else "DataFrame"
        aug_str = " (augmented)" if augment else ""
        print(f"Loaded {len(self.df)} samples from {source}{aug_str}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with augmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, labels):
                - waveform: Augmented audio tensor of shape (num_samples,)
                - labels: Score tensor of shape (5,) scaled to [0, 1]
        """
        row = self.df.iloc[idx]
        
        # Build audio path
        audio_path = row["path"]
        if self.data_root:
            audio_path = os.path.join(self.data_root, audio_path)
        
        # Load and preprocess audio
        waveform = self._load_and_preprocess(audio_path)
        
        # Apply augmentation if enabled
        if self.augment and self.augment_pipeline is not None:
            # Convert to numpy for augmentation, then back to torch
            waveform_np = waveform.numpy()
            waveform_np = self.augment_pipeline(waveform_np)
            waveform = torch.from_numpy(waveform_np).to(torch.float32)
        
        # Get labels and scale to [0, 1]
        labels = torch.tensor([row[name] for name in SCORE_NAMES], dtype=torch.float32)
        labels = scale_scores(labels)
        
        return waveform, labels
    
    def _load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.permute(1, 0)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        
        current_samples = waveform.shape[1]
        if current_samples < self.num_samples:
            padding = self.num_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        
        waveform = waveform.squeeze(0)
        return waveform
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "path": row["path"],
            "scores": {name: row[name] for name in SCORE_NAMES},
            "augmented": self.augment,
        }


class AugmentedTrumpetDatasetWithGating(Dataset):
    """
    Augmented TrumpetDataset with gating labels for rejection detection.
    
    Combines augmentation with the gating head architecture.
    """
    
    SAMPLE_RATE = 32000
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        duration: float = 20.0,
        data_root: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        augment: bool = True,
        augment_config: Optional[Dict] = None,
    ):
        """
        Initialize the augmented gated dataset.
        
        Args:
            csv_path: Path to CSV with is_valid column
            duration: Fixed audio duration in seconds
            data_root: Optional root directory for audio paths
            dataframe: Optional DataFrame to use directly
            augment: Whether to apply augmentations
            augment_config: Optional config dict for augmentation settings
        """
        self.csv_path = csv_path
        self.duration = duration
        self.data_root = data_root
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        self.augment = augment
        
        # Load from DataFrame or CSV
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Must provide either csv_path or dataframe")
        
        # Validate columns
        required_cols = ["id", "path", "is_valid"] + SCORE_NAMES
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Setup augmentation pipeline
        if augment:
            from ml.augment import WaveformAugmentPipeline
            
            default_config = {
                "gain_p": 0.8,
                "gain_min_db": -12.0,
                "gain_max_db": 12.0,
                "shift_p": 0.8,
                "shift_min_ms": -50.0,
                "shift_max_ms": 50.0,
                "shift_mode": "zero",
                "noise_p": 0.5,
                "noise_types": ["white", "pink", "brown"],
                "noise_min_snr": 20.0,
                "noise_max_snr": 35.0,
                "reverb_p": 0.5,
                "reverb_min_wet": 0.05,
                "reverb_max_wet": 0.20,
                "reverb_ir_dir": None,
                "sample_rate": self.SAMPLE_RATE,
            }
            
            if augment_config:
                default_config.update(augment_config)
            
            self.augment_pipeline = WaveformAugmentPipeline(**default_config)
        else:
            self.augment_pipeline = None
        
        # Count valid/invalid
        n_valid = (self.df['is_valid'] == 1).sum()
        n_invalid = (self.df['is_valid'] == 0).sum()
        
        source = csv_path if csv_path else "DataFrame"
        aug_str = " (augmented)" if augment else ""
        print(f"Loaded {len(self.df)} samples from {source}{aug_str} ({n_valid} valid, {n_invalid} rejected)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample with augmentation.
        
        Returns:
            Tuple of (waveform, labels, is_valid)
        """
        row = self.df.iloc[idx]
        
        # Build audio path
        audio_path = row["path"]
        if self.data_root:
            audio_path = os.path.join(self.data_root, audio_path)
        
        # Load and preprocess audio
        waveform = self._load_and_preprocess(audio_path)
        
        # Apply augmentation if enabled
        if self.augment and self.augment_pipeline is not None:
            waveform_np = waveform.numpy()
            waveform_np = self.augment_pipeline(waveform_np)
            waveform = torch.from_numpy(waveform_np).to(torch.float32)
        
        # Get validity label
        is_valid = torch.tensor([float(row["is_valid"])], dtype=torch.float32)
        
        # Get labels
        if row["is_valid"] == 1:
            labels = torch.tensor([row[name] for name in SCORE_NAMES], dtype=torch.float32)
            labels = scale_scores(labels)
        else:
            labels = torch.zeros(len(SCORE_NAMES), dtype=torch.float32)
        
        return waveform, labels, is_valid
    
    def _load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.permute(1, 0)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        
        current_samples = waveform.shape[1]
        if current_samples < self.num_samples:
            padding = self.num_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        
        waveform = waveform.squeeze(0)
        return waveform


def create_augmented_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 6,
    augment_train: bool = True,
    augment_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders with augmentation for training set.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Optional path to test CSV
        batch_size: Batch size for DataLoader
        duration: Fixed audio duration in seconds
        data_root: Optional root directory for audio paths
        num_workers: Number of worker processes
        augment_train: Whether to augment training data
        augment_config: Optional config for augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training set WITH augmentation
    train_dataset = AugmentedTrumpetDataset(
        train_csv,
        duration=duration,
        data_root=data_root,
        augment=augment_train,
        augment_config=augment_config,
    )
    
    # Validation and test sets WITHOUT augmentation
    val_dataset = TrumpetDataset(val_csv, duration=duration, data_root=data_root)
    
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    
    test_loader = None
    if test_csv:
        test_dataset = TrumpetDataset(test_csv, duration=duration, data_root=data_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
    
    return train_loader, val_loader, test_loader


def create_augmented_dataloader_from_df(
    df: pd.DataFrame,
    batch_size: int = 8,
    duration: float = 20.0,
    data_root: Optional[str] = None,
    num_workers: int = 6,
    shuffle: bool = False,
    augment: bool = True,
    augment_config: Optional[Dict] = None,
) -> DataLoader:
    """
    Create an augmented DataLoader from a DataFrame.
    
    Args:
        df: DataFrame with required columns
        batch_size: Batch size
        duration: Audio duration in seconds
        data_root: Optional root directory for audio paths
        num_workers: Number of workers
        shuffle: Whether to shuffle
        augment: Whether to apply augmentation
        augment_config: Optional augmentation config
        
    Returns:
        DataLoader for the augmented dataset
    """
    dataset = AugmentedTrumpetDataset(
        dataframe=df,
        duration=duration,
        data_root=data_root,
        augment=augment,
        augment_config=augment_config,
    )
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return loader


if __name__ == "__main__":
    test_dataset()

