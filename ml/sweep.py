#!/usr/bin/env python3
"""
Hyperparameter optimization for TrumpetJudge using Optuna.

Uses Bayesian optimization (TPE) to intelligently search the hyperparameter space.
Much more efficient than grid search - learns from previous trials.

Usage:
    python ml/sweep.py                    # 30 trials (default)
    python ml/sweep.py --n_trials 50      # More trials
    python ml/sweep.py --n_trials 10      # Quick test

I"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from train_fast import train_cv


def create_objective(
    embeddings_file: str,
    labels_csv: str,
    aug_embeddings: str,
    aug_csv: str,
    n_folds: int,
    epochs: int,
    patience: int,
    output_dir: Path,
    device: str,
    seed: int,
):
    """Create an objective function for Optuna."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64, 96, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number + 1}")
        print(f"  lr={lr:.6f}, batch={batch_size}, dropout={dropout:.2f}, wd={weight_decay:.6f}")
        print("=" * 60)
        
        try:
            # Run CV with these hyperparameters
            summary = train_cv(
                embeddings_file=embeddings_file,
                labels_csv=labels_csv,
                n_folds=n_folds,
                output_dir=str(output_dir / f"trial_{trial.number:03d}"),
                batch_size=batch_size,
                learning_rate=lr,
                dropout=dropout,
                weight_decay=weight_decay,
                epochs=epochs,
                patience=patience,
                aug_embeddings=aug_embeddings,
                aug_csv=aug_csv,
                seed = seed,
                device = device,
            )
            
            mean_mae = summary["mean_mae"]
            std_mae = summary["std_mae"]
            
            print(f"\n  → MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            
            # Store additional info in trial
            trial.set_user_attr("std_mae", std_mae)
            trial.set_user_attr("fold_maes", summary["fold_maes"])
            
            return mean_mae
            
        except Exception as e:
            print(f"\n  ❌ Trial failed: {e}")
            raise optuna.TrialPruned()
    
    return objective


def run_sweep(
    embeddings_file: str = "data/embeddings/all_audio.pt",
    labels_csv: str = "data/prepared/all_data.csv",
    aug_embeddings: str = "data/embeddings/all_augmented.pt",
    aug_csv: str = "data/prepared/all_augmented.csv",
    output_dir: str = "sweeps",
    n_folds: int = 6,
    n_trials: int = 30,
    epochs: int = 100,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
):
    """Run Optuna hyperparameter optimization."""
    
    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"optuna_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TrumpetJudge Hyperparameter Optimization (Optuna)")
    print("=" * 60)
    print(f"\nSearch space:")
    print(f"  lr:           [1e-4, 5e-2] (log scale)")
    print(f"  batch_size:   [16, 24, 32, 48, 64, 96, 128]")
    print(f"  dropout:      [0.1, 0.6]")
    print(f"  weight_decay: [1e-6, 1e-2] (log scale)")
    print(f"\nTrials: {n_trials}")
    print(f"Output: {sweep_dir}")
    
    # Create study with TPE sampler (Bayesian optimization)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",  # Minimize MAE
        sampler=sampler,
        study_name="trumpet_judge_hpo",
    )
    
    # Create objective
    objective = create_objective(
        embeddings_file=embeddings_file,
        labels_csv=labels_csv,
        aug_embeddings=aug_embeddings,
        aug_csv=aug_csv,
        n_folds=n_folds,
        epochs=epochs,
        patience=patience,
        output_dir=sweep_dir,
        device=device,
        seed=seed,
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\n🏆 Best trial: #{study.best_trial.number + 1}")
    print(f"   MAE: {study.best_value:.4f}")
    print(f"\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.6f}")
        else:
            print(f"     {key}: {value}")
    
    # Top 5 trials
    print(f"\n📊 Top 5 trials:")
    print(f"{'Rank':<5} {'MAE':<10} {'LR':<12} {'Batch':<7} {'Dropout':<9} {'WD':<12}")
    print("-" * 60)
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))
    for rank, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is not None:
            p = trial.params
            print(f"{rank:<5} {trial.value:<10.4f} {p['lr']:<12.6f} {p['batch_size']:<7} {p['dropout']:<9.2f} {p['weight_decay']:<12.6f}")
    
    # Save results
    results = {
        "timestamp": timestamp,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "best_trial": study.best_trial.number,
        "best_mae": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "mae": t.value,
                "std_mae": t.user_attrs.get("std_mae"),
                "fold_maes": t.user_attrs.get("fold_maes"),
            }
            for t in study.trials
            if t.value is not None
        ],
    }
    
    with open(sweep_dir / "optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {sweep_dir / 'optuna_results.json'}")
    
    return study, results


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/all_audio.pt",
                        help="Embeddings file")
    parser.add_argument("--labels", type=str, default="data/prepared/all_data.csv",
                        help="Labels CSV")
    parser.add_argument("--aug_embeddings", type=str, default="data/embeddings/all_augmented.pt",
                        help="Augmented embeddings")
    parser.add_argument("--aug_csv", type=str, default="data/prepared/all_augmented.csv",
                        help="Augmented CSV")
    parser.add_argument("--output", type=str, default="sweeps",
                        help="Output directory")
    parser.add_argument("--n_folds", type=int, default=6,
                        help="Number of CV folds")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    
    args = parser.parse_args()
    
    run_sweep(
        embeddings_file=args.embeddings,
        labels_csv=args.labels,
        aug_embeddings=args.aug_embeddings,
        aug_csv=args.aug_csv,
        output_dir=args.output,
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
