"""
Models package for TrumpetJudge ML pipeline.

Contains:
    - PANNsEncoder: Pretrained audio encoder for extracting embeddings
    - RegressionHead: Small MLP that maps embeddings to 5 scores (1-5 scale)
"""

from .encoder_panns import PANNsEncoder
from .head_regressor import RegressionHead, SCORE_NAMES, scale_scores, unscale_scores

__all__ = ["PANNsEncoder", "RegressionHead", "SCORE_NAMES", "scale_scores", "unscale_scores"]

