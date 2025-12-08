"""
ML package for TrumpetJudge training pipeline.

Contains:
    - TrumpetDataset: PyTorch Dataset for loading audio and labels
    - create_dataloaders: Helper to create train/val/test DataLoaders
    - train: Main training function
"""

from .dataset import TrumpetDataset, create_dataloaders
from .train import train

__all__ = ["TrumpetDataset", "create_dataloaders", "train"]

