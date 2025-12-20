"""
Ensemble Model for TrumpetJudge.

Combines predictions from multiple fold models to get more robust estimates.
Typically averages predictions from all K folds of a cross-validation run.

Usage:
    # Load ensemble from best sweep result
    ensemble = EnsembleModel.from_sweep("models/sweeps/sweep_20251218_004330")
    
    # Load ensemble from specific CV run
    ensemble = EnsembleModel.from_cv_run("models/sweeps/sweep_20251218_004330/lr0.01_bs32/cv_6fold_20251218_004342")
    
    # Predict with ensemble
    scores = ensemble.predict(embeddings)  # Returns averaged predictions
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores


class EnsembleModel(nn.Module):
    """
    Ensemble of regression heads for robust trumpet score prediction.
    
    Loads multiple fold models and averages their predictions.
    """
    
    def __init__(
        self,
        models: List[RegressionHead],
        weights: Optional[List[float]] = None,
        embedding_dim: int = 2048,
    ):
        """
        Initialize ensemble from list of models.
        
        Args:
            models: List of RegressionHead models (one per fold)
            weights: Optional weights for weighted averaging (default: equal weights)
            embedding_dim: Input embedding dimension
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        
        # Set up weights (normalized to sum to 1)
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Register as buffer so it moves with model to device
        self.register_buffer('weights', torch.tensor(weights))
        
        # Put all models in eval mode
        for model in self.models:
            model.eval()
    
    @classmethod
    def from_cv_run(
        cls,
        cv_dir: str,
        device: str = None,
        weight_by_performance: bool = False,
    ) -> "EnsembleModel":
        """
        Load ensemble from a cross-validation run directory.
        
        Args:
            cv_dir: Path to CV run directory (contains fold_1, fold_2, etc.)
            device: Device to load models to (None for auto-detect)
            weight_by_performance: If True, weight folds inversely by their MAE
            
        Returns:
            EnsembleModel with all fold models loaded
        """
        cv_dir = Path(cv_dir)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CV summary to get fold info
        summary_path = cv_dir / "cv_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"CV summary not found: {summary_path}")
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        n_folds = summary["n_folds"]
        fold_maes = summary.get("fold_maes", [])
        
        print(f"Loading ensemble from {cv_dir}")
        print(f"  Folds: {n_folds}")
        print(f"  Mean MAE: {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
        
        # Load each fold model
        models = []
        for fold_idx in range(1, n_folds + 1):
            fold_dir = cv_dir / f"fold_{fold_idx}"
            model_path = fold_dir / "best_model.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Infer hidden dims from state dict layer shapes
            state_dict = checkpoint["head_state_dict"]
            hidden_dim = state_dict["network.0.weight"].shape[0]   # First layer output
            hidden_dim2 = state_dict["network.3.weight"].shape[0]  # Second layer output
            
            # Create model with correct architecture and load state dict
            model = RegressionHead(embedding_dim=2048, hidden_dim=hidden_dim, hidden_dim2=hidden_dim2)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            models.append(model)
            
            mae = fold_maes[fold_idx - 1] if fold_maes else checkpoint.get("val_mae", "?")
            print(f"  Loaded fold {fold_idx} (MAE: {mae:.4f})" if isinstance(mae, float) else f"  Loaded fold {fold_idx}")
        
        # Calculate weights
        weights = None
        if weight_by_performance and fold_maes:
            # Inverse MAE weighting (lower MAE = higher weight)
            inverse_maes = [1.0 / mae for mae in fold_maes]
            weights = inverse_maes
            print(f"  Using performance-weighted ensemble")
        
        ensemble = cls(models, weights=weights, embedding_dim=2048)
        ensemble = ensemble.to(device)
        
        # Store metadata
        ensemble.cv_dir = str(cv_dir)
        ensemble.summary = summary
        
        print(f"  ✓ Ensemble ready ({len(models)} models)")
        return ensemble
    
    @classmethod
    def from_sweep(
        cls,
        sweep_dir: str,
        device: str = None,
        weight_by_performance: bool = False,
    ) -> "EnsembleModel":
        """
        Load ensemble from the best run of a sweep.
        
        Args:
            sweep_dir: Path to sweep directory (contains sweep_results.json)
            device: Device to load models to (None for auto-detect)
            weight_by_performance: If True, weight folds inversely by their MAE
            
        Returns:
            EnsembleModel with all fold models from best config
        """
        sweep_dir = Path(sweep_dir)
        
        # Load sweep results
        results_path = sweep_dir / "sweep_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Sweep results not found: {results_path}")
        
        with open(results_path) as f:
            results = json.load(f)
        
        best = results["best_config"]
        print(f"Loading ensemble from sweep: {sweep_dir}")
        print(f"  Best config: lr={best['lr']}, batch_size={best['batch_size']}")
        print(f"  Best MAE: {best['mean_mae']:.4f} ± {best['std_mae']:.4f}")
        
        # Find the CV run directory for the best config
        config_dir_name = f"lr{best['lr']}_bs{best['batch_size']}"
        config_dir = sweep_dir / config_dir_name
        
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        # Find the CV run (should be only one subdirectory)
        cv_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith("cv_")]
        if not cv_dirs:
            raise FileNotFoundError(f"No CV run found in: {config_dir}")
        
        cv_dir = cv_dirs[0]  # Take the first (usually only) CV run
        
        return cls.from_cv_run(str(cv_dir), device=device, weight_by_performance=weight_by_performance)
    
    @classmethod
    def from_optuna_sweep(
        cls,
        sweep_dir: str,
        device: str = None,
        weight_by_performance: bool = False,
    ) -> "EnsembleModel":
        """
        Load ensemble from the best trial of an Optuna sweep.
        
        Args:
            sweep_dir: Path to Optuna sweep directory (contains optuna_results.json)
            device: Device to load models to (None for auto-detect)
            weight_by_performance: If True, weight folds inversely by their MAE
            
        Returns:
            EnsembleModel with all fold models from best trial
        """
        sweep_dir = Path(sweep_dir)
        
        # Load optuna results
        results_path = sweep_dir / "optuna_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Optuna results not found: {results_path}")
        
        with open(results_path) as f:
            results = json.load(f)
        
        best_trial = results["best_trial"]
        best_mae = results["best_mae"]
        best_params = results["best_params"]
        
        print(f"Loading ensemble from Optuna sweep: {sweep_dir}")
        print(f"  Best trial: {best_trial}")
        print(f"  Best MAE: {best_mae:.4f}")
        print(f"  Params: lr={best_params['lr']:.6f}, bs={best_params['batch_size']}, "
              f"dropout={best_params['dropout']:.2f}, wd={best_params['weight_decay']:.2e}")
        
        # Find the trial directory
        trial_dir = sweep_dir / f"trial_{best_trial:03d}"
        
        if not trial_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
        
        # Find the CV run (should be only one subdirectory)
        cv_dirs = [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith("cv_")]
        if not cv_dirs:
            raise FileNotFoundError(f"No CV run found in: {trial_dir}")
        
        cv_dir = cv_dirs[0]  # Take the first (usually only) CV run
        
        return cls.from_cv_run(str(cv_dir), device=device, weight_by_performance=weight_by_performance)
    
    @classmethod
    def from_any_sweep(
        cls,
        sweep_dir: str,
        device: str = None,
        weight_by_performance: bool = False,
    ) -> "EnsembleModel":
        """
        Auto-detect sweep type and load ensemble from best run.
        
        Works with both grid sweeps (sweep_results.json) and Optuna sweeps (optuna_results.json).
        
        Args:
            sweep_dir: Path to sweep directory
            device: Device to load models to (None for auto-detect)
            weight_by_performance: If True, weight folds inversely by their MAE
            
        Returns:
            EnsembleModel with all fold models from best config/trial
        """
        sweep_dir = Path(sweep_dir)
        
        # Try Optuna first
        if (sweep_dir / "optuna_results.json").exists():
            return cls.from_optuna_sweep(str(sweep_dir), device=device, weight_by_performance=weight_by_performance)
        
        # Try grid sweep
        if (sweep_dir / "sweep_results.json").exists():
            return cls.from_sweep(str(sweep_dir), device=device, weight_by_performance=weight_by_performance)
        
        # Maybe it's a CV run directly?
        if (sweep_dir / "cv_summary.json").exists():
            return cls.from_cv_run(str(sweep_dir), device=device, weight_by_performance=weight_by_performance)
        
        raise FileNotFoundError(
            f"Could not find sweep_results.json, optuna_results.json, or cv_summary.json in: {sweep_dir}"
        )
    
    def save(self, path: str):
        """
        Save ensemble to a single file for easy deployment.
        
        Args:
            path: Output path for the ensemble checkpoint (.pt file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all model state dicts
        model_states = [model.state_dict() for model in self.models]
        
        checkpoint = {
            "num_models": self.num_models,
            "embedding_dim": self.embedding_dim,
            "weights": self.weights.cpu().tolist(),
            "model_states": model_states,
            "metadata": {
                "cv_dir": getattr(self, 'cv_dir', None),
                "summary": getattr(self, 'summary', None),
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Saved ensemble to: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> "EnsembleModel":
        """
        Load ensemble from a saved checkpoint.
        
        Args:
            path: Path to ensemble checkpoint (.pt file)
            device: Device to load models to (None for auto-detect)
            
        Returns:
            EnsembleModel loaded from checkpoint
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Recreate models
        models = []
        for state_dict in checkpoint["model_states"]:
            # Infer hidden dims from state dict layer shapes
            hidden_dim = state_dict["network.0.weight"].shape[0]   # First layer output
            hidden_dim2 = state_dict["network.3.weight"].shape[0]  # Second layer output
            
            model = RegressionHead(
                embedding_dim=checkpoint["embedding_dim"],
                hidden_dim=hidden_dim,
                hidden_dim2=hidden_dim2,
            )
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            models.append(model)
        
        # Create ensemble
        ensemble = cls(
            models=models,
            weights=checkpoint["weights"],
            embedding_dim=checkpoint["embedding_dim"],
        )
        ensemble = ensemble.to(device)
        
        # Restore metadata
        if checkpoint.get("metadata"):
            ensemble.cv_dir = checkpoint["metadata"].get("cv_dir")
            ensemble.summary = checkpoint["metadata"].get("summary")
        
        print(f"Loaded ensemble from: {path}")
        print(f"  Models: {ensemble.num_models}")
        
        return ensemble
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns averaged predictions in [0, 1] range.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            Scores tensor of shape (batch, 5) with values in [0, 1]
        """
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(embedding)
                predictions.append(pred)
        
        # Stack: (num_models, batch, 5)
        stacked = torch.stack(predictions, dim=0)
        
        # Weighted average
        # weights is (num_models,), need to broadcast to (num_models, 1, 1)
        weights = self.weights.view(-1, 1, 1)
        averaged = (stacked * weights).sum(dim=0)
        
        return averaged
    
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
    
    def predict_with_uncertainty(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scores with uncertainty estimates from model disagreement.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            Tuple of:
                - mean_scores: (batch, 5) mean predictions in [1, 5]
                - std_scores: (batch, 5) standard deviation across models
        """
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model.predict(embedding)  # Already in [1, 5]
                predictions.append(pred)
        
        # Stack: (num_models, batch, 5)
        stacked = torch.stack(predictions, dim=0)
        
        mean_scores = stacked.mean(dim=0)
        std_scores = stacked.std(dim=0)
        
        return mean_scores, std_scores
    
    def predict_dict(self, embedding: torch.Tensor) -> List[Dict[str, float]]:
        """
        Predict scores and return as list of dictionaries.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            List of dicts with score names as keys
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
    
    def predict_dict_with_uncertainty(
        self, embedding: torch.Tensor
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Predict scores with uncertainty, returned as dictionaries.
        
        Args:
            embedding: Audio embedding tensor of shape (batch, embedding_dim)
            
        Returns:
            List of dicts with score names mapping to {"mean": x, "std": y}
        """
        mean_scores, std_scores = self.predict_with_uncertainty(embedding)
        
        results = []
        for i in range(mean_scores.shape[0]):
            sample_scores = {
                name: {
                    "mean": mean_scores[i, j].item(),
                    "std": std_scores[i, j].item(),
                }
                for j, name in enumerate(SCORE_NAMES)
            }
            results.append(sample_scores)
        
        return results


def infer_audio_ensemble(
    audio_path: str,
    ensemble: EnsembleModel,
    duration: float = 20.0,
    device: str = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run inference on a single audio file using ensemble.
    
    Args:
        audio_path: Path to audio file
        ensemble: EnsembleModel instance
        duration: Audio duration to use (seconds)
        device: Device to run on
        
    Returns:
        Dict with score names mapping to {"mean": x, "std": y}
    """
    from ml.encoder_panns import PANNsEncoder
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load encoder
    encoder = PANNsEncoder(duration=duration, device=device)
    
    # Get embedding
    embedding = encoder.encode_file(audio_path)
    embedding = embedding.unsqueeze(0)  # Add batch dim
    
    # Predict with uncertainty
    result = ensemble.predict_dict_with_uncertainty(embedding)[0]
    
    return result


def infer_embeddings_ensemble(
    embeddings: torch.Tensor,
    ensemble: EnsembleModel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on precomputed embeddings.
    
    Args:
        embeddings: Tensor of shape (batch, embedding_dim)
        ensemble: EnsembleModel instance
        
    Returns:
        Tuple of (mean_scores, std_scores), each of shape (batch, 5)
    """
    return ensemble.predict_with_uncertainty(embeddings)


def main():
    """Test ensemble loading and inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble inference")
    parser.add_argument("--sweep", type=str, help="Path to sweep directory (auto-detects type)")
    parser.add_argument("--cv_run", type=str, help="Path to CV run directory")
    parser.add_argument("--load", type=str, help="Load ensemble from saved checkpoint")
    parser.add_argument("--save", type=str, help="Save ensemble to file (e.g. models/ensemble.pt)")
    parser.add_argument("--audio", type=str, help="Audio file to test on")
    parser.add_argument("--embeddings", type=str, help="Embeddings file to test on")
    parser.add_argument("--weighted", action="store_true", help="Use performance-weighted ensemble")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cpu', 'cuda')")
    args = parser.parse_args()
    
    # Load ensemble
    if args.load:
        ensemble = EnsembleModel.load(args.load, device=args.device)
    elif args.sweep:
        # Auto-detect sweep type (grid sweep or Optuna)
        ensemble = EnsembleModel.from_any_sweep(args.sweep, device=args.device, weight_by_performance=args.weighted)
    elif args.cv_run:
        ensemble = EnsembleModel.from_cv_run(args.cv_run, device=args.device, weight_by_performance=args.weighted)
    else:
        print("Please provide --sweep, --cv_run, or --load")
        return
    
    print(f"\nEnsemble loaded: {ensemble.num_models} models")
    
    # Save if requested
    if args.save:
        ensemble.save(args.save)
    
    # Test inference
    if args.audio:
        print(f"\nRunning inference on: {args.audio}")
        result = infer_audio_ensemble(args.audio, ensemble, device=args.device)
        
        print("\nPredictions:")
        for name, scores in result.items():
            print(f"  {name}: {scores['mean']:.2f} ± {scores['std']:.2f}")
    
    elif args.embeddings:
        print(f"\nLoading embeddings from: {args.embeddings}")
        data = torch.load(args.embeddings, map_location="cpu", weights_only=False)
        embeddings = data["embeddings"]
        
        print(f"  Loaded {len(embeddings)} embeddings")
        
        # Run inference on all
        mean_scores, std_scores = infer_embeddings_ensemble(embeddings, ensemble)
        
        print(f"\nResults summary (mean ± std across all samples):")
        for i, name in enumerate(SCORE_NAMES):
            mean_mean = mean_scores[:, i].mean().item()
            mean_std = std_scores[:, i].mean().item()
            print(f"  {name}: {mean_mean:.2f} (avg uncertainty: ±{mean_std:.2f})")
    
    else:
        # Quick test with random embeddings
        print("\nTesting with random embeddings...")
        dummy = torch.randn(4, 2048)
        
        if next(ensemble.parameters()).is_cuda:
            dummy = dummy.cuda()
        
        mean_scores, std_scores = ensemble.predict_with_uncertainty(dummy)
        
        print(f"  Input shape: {dummy.shape}")
        print(f"  Mean scores shape: {mean_scores.shape}")
        print(f"  Std scores shape: {std_scores.shape}")
        
        print("\n  Sample prediction (first item):")
        result = ensemble.predict_dict_with_uncertainty(dummy)[0]
        for name, scores in result.items():
            print(f"    {name}: {scores['mean']:.2f} ± {scores['std']:.2f}")
        
        print("\n✓ Ensemble test passed!")


if __name__ == "__main__":
    main()

