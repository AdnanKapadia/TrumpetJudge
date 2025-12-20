"""
ML package for TrumpetJudge training pipeline.

Contains:
    - encoder_panns: PANNs CNN14 audio encoder (frozen)
    - head_regressor: Trainable regression/gating heads
    - dataset: PyTorch Dataset for loading audio and labels
    - train: Main training script
    - ensemble: Ensemble model for robust predictions
    - augment: Data augmentation utilities
"""

from .encoder_panns import PANNsEncoder
from .head_regressor import RegressionHead, GatingHead, SCORE_NAMES, scale_scores, unscale_scores
from .dataset import TrumpetDataset, create_dataloaders
from .train import train

__all__ = [
    "PANNsEncoder",
    "RegressionHead",
    "GatingHead",
    "SCORE_NAMES",
    "scale_scores",
    "unscale_scores",
    "TrumpetDataset",
    "create_dataloaders",
    "train",
]

