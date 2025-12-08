"""
Regression Head for Trumpet Score Prediction

This module implements a small MLP that takes audio embeddings from the PANNs encoder
and predicts 5 performance scores. This is the only part that gets trained.

Scores (1-5 scale):
    - Overall: General performance quality
    - Intonation: Pitch accuracy and tuning
    - Tone: Sound quality, warmth, clarity
    - Timing: Rhythmic accuracy and steadiness
    - Technique: Articulation, dynamics, control
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# Score names in order (matches output tensor indices)
SCORE_NAMES = ["overall", "intonation", "tone", "timing", "technique"]

# Score range
SCORE_MIN = 1
SCORE_MAX = 5


class RegressionHead(nn.Module):
    """
    MLP regression head that maps audio embeddings to performance scores.
    
    Architecture:
        Linear(embedding_dim → 256) → ReLU → Dropout
        Linear(256 → 64) → ReLU → Dropout  
        Linear(64 → 5) → Sigmoid → scale to [1, 5]
    
    The model predicts in [0, 1] internally and rescales to [1, 5] for output.
    Training targets should be scaled to [0, 1] using scale_scores().
    
    Attributes:
        embedding_dim (int): Input embedding dimension (2048 for CNN14)
        num_scores (int): Number of output scores (5)
    """
    
    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize the regression head.
        
        Args:
            embedding_dim: Dimension of input embeddings (2048 for PANNs CNN14)
            hidden_dim: Dimension of first hidden layer
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_scores = len(SCORE_NAMES)
        
        # MLP layers
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, self.num_scores),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embedding → scores in [0, 1].
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            Scores tensor of shape (batch, 5) with values in [0, 1]
            Use unscale_scores() to convert to [1, 10] range.
        """
        return self.network(embedding)
    
    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict scores in the original [1, 5] scale.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            Scores tensor of shape (batch, 5) with values in [1, 5]
        """
        scaled_scores = self.forward(embedding)
        return unscale_scores(scaled_scores)
    
    def predict_dict(self, embedding: torch.Tensor) -> List[Dict[str, float]]:
        """
        Predict scores and return as list of dictionaries.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            List of dicts, one per sample, with score names as keys.
            Example: [{"overall": 7.5, "intonation": 8.2, ...}, ...]
        """
        scores = self.predict(embedding)
        
        results = []
        for i in range(scores.shape[0]):
            sample_scores = {
                name: scores[i, j].item()
                for j, name in enumerate(SCORE_NAMES)
            }
            results.append(sample_scores)
        
        return results


def scale_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Scale scores from [1, 5] to [0, 1] for training.
    
    Args:
        scores: Tensor with values in [1, 10]
        
    Returns:
        Tensor with values in [0, 1]
    """
    return (scores - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def unscale_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Unscale scores from [0, 1] back to [1, 5] for inference.
    
    Args:
        scores: Tensor with values in [0, 1]
        
    Returns:
        Tensor with values in [1, 10]
    """
    return scores * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


def test_head():
    """Quick test to verify regression head works."""
    print("Initializing regression head...")
    head = RegressionHead(embedding_dim=2048)
    
    print(f"  Input dim: {head.embedding_dim}")
    print(f"  Output scores: {SCORE_NAMES}")
    print(f"  Score range: [{SCORE_MIN}, {SCORE_MAX}]")
    
    # Count parameters
    num_params = sum(p.numel() for p in head.parameters())
    print(f"  Trainable parameters: {num_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_embedding = torch.randn(4, 2048)  # Batch of 4
    
    # Raw output (scaled 0-1)
    raw_scores = head(dummy_embedding)
    print(f"  Input shape: {dummy_embedding.shape}")
    print(f"  Raw output shape: {raw_scores.shape}")
    print(f"  Raw output range: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
    
    # Predicted scores (1-5)
    pred_scores = head.predict(dummy_embedding)
    print(f"  Predicted scores shape: {pred_scores.shape}")
    print(f"  Predicted range: [{pred_scores.min():.1f}, {pred_scores.max():.1f}]")
    
    # Dict output
    print("\nExample prediction (first sample):")
    pred_dicts = head.predict_dict(dummy_embedding)
    for name, score in pred_dicts[0].items():
        print(f"    {name}: {score:.1f}")
    
    # Test scaling functions
    print("\nTesting scale/unscale...")
    original = torch.tensor([1.0, 3.0, 5.0])
    scaled = scale_scores(original)
    recovered = unscale_scores(scaled)
    print(f"  Original: {original.tolist()}")
    print(f"  Scaled:   {scaled.tolist()}")
    print(f"  Recovered: {recovered.tolist()}")
    
    assert torch.allclose(original, recovered), "Scale/unscale mismatch!"
    print("\n✓ Regression head test passed!")
    
    return head


if __name__ == "__main__":
    test_head()

