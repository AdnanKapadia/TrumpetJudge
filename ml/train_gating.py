"""
Train the GatingHead on precomputed embeddings + rejected flags.

This uses:
- data/embeddings/all_audio.pt
- data/embeddings/all_augmented.pt (optional, for extra negative examples if desired)
- data/prepared/all_data.csv or all_gated.csv (with is_valid column)

Usage (basic):
    python -m ml.train_gating \
        --embeddings data/embeddings/all_audio.pt \
        --labels_csv data/prepared/all_data.csv \
        --output_dir checkpoints_gating

If you re-run prepare_data with --include_rejected --save_all, you can instead use:
    python -m ml.train_gating \
        --embeddings data/embeddings/all_audio.pt \
        --labels_csv data/prepared/all_gated.csv \
        --output_dir checkpoints_gating
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.head_regressor import GatingHead  # noqa: E402


class GatingDataset(Dataset):
    """
    Dataset for training the GatingHead.

    - Loads embeddings from all_audio.pt
    - Loads labels_csv (must contain columns: id, is_valid)
    - Matches by id to create (embedding, is_valid) pairs
    """

    def __init__(
        self,
        embeddings_file: str,
        labels_csv: str,
        aug_embeddings_file: str | None = None,
        aug_csv: str | None = None,
    ):
        import pandas as pd

        print(f"Loading embeddings from {embeddings_file}...")
        emb_data = torch.load(embeddings_file, map_location="cpu", weights_only=False)
        emb_ids = emb_data["ids"]
        all_embeddings = emb_data["embeddings"]

        id_to_idx = {id_: idx for idx, id_ in enumerate(emb_ids)}

        print(f"  {len(emb_ids)} embeddings loaded")

        print(f"Loading gating labels from {labels_csv}...")
        df = pd.read_csv(labels_csv)

        if "is_valid" not in df.columns:
            raise ValueError(
                f"labels_csv '{labels_csv}' must contain an 'is_valid' column. "
                "Run ml/prepare_data.py with --include_rejected --save_all to create all_gated.csv."
            )

        matched_embeddings = []
        matched_targets = []
        matched_ids = []

        missing = 0
        for _, row in df.iterrows():
            sample_id = row["id"]
            if sample_id in id_to_idx:
                idx = id_to_idx[sample_id]
                matched_embeddings.append(all_embeddings[idx])
                # is_valid: 1 for valid trumpet, 0 for rejected / non-trumpet etc.
                target = float(row["is_valid"])
                matched_targets.append(target)
                matched_ids.append(sample_id)
            else:
                missing += 1

        if missing > 0:
            print(f"  Warning: {missing} rows in labels_csv did not match any embedding id")

        # Optionally add augmented embeddings, inheriting is_valid from original_id
        if aug_embeddings_file and aug_csv:
            print(f"\nLoading augmented embeddings from {aug_embeddings_file}...")
            aug_emb_data = torch.load(aug_embeddings_file, map_location="cpu", weights_only=False)
            aug_ids = aug_emb_data["ids"]
            aug_embeddings = aug_emb_data["embeddings"]
            aug_id_to_idx = {id_: idx for idx, id_ in enumerate(aug_ids)}
            print(f"  {len(aug_ids)} augmented embeddings loaded")

            print(f"Loading augmented CSV from {aug_csv}...")
            aug_df = pd.read_csv(aug_csv)
            print(f"  {len(aug_df)} augmented rows")

            # Map original_id -> is_valid from labels CSV
            orig_to_is_valid = {row["id"]: float(row["is_valid"]) for _, row in df.iterrows()}

            missing_aug_emb = 0
            missing_orig = 0
            added_aug = 0

            for _, row in aug_df.iterrows():
                aug_id = row["id"]
                original_id = row["original_id"]

                if aug_id not in aug_id_to_idx:
                    missing_aug_emb += 1
                    continue
                if original_id not in orig_to_is_valid:
                    missing_orig += 1
                    continue

                idx = aug_id_to_idx[aug_id]
                matched_embeddings.append(aug_embeddings[idx])
                matched_targets.append(orig_to_is_valid[original_id])
                matched_ids.append(aug_id)
                added_aug += 1

            print(
                f"  Added {added_aug} augmented samples for gating "
                f"(missing_emb={missing_aug_emb}, missing_original={missing_orig})"
            )

        if not matched_embeddings:
            raise ValueError("No samples matched between embeddings and labels (including augmented).")

        self.embeddings = torch.stack(matched_embeddings)
        self.targets = torch.tensor(matched_targets, dtype=torch.float32).unsqueeze(-1)
        self.ids = matched_ids

        print(f"  Total matched samples for gating (orig + aug): {len(self.embeddings)}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx]


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        preds = model(embeddings)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Returns (loss, accuracy, pos_ratio)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_targets = []
    all_preds = []

    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        probs = model(embeddings)
        loss = criterion(probs, targets)

        total_loss += loss.item()
        n_batches += 1

        all_targets.append(targets.cpu())
        all_preds.append(probs.cpu())

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Binary accuracy at threshold 0.5
    preds_binary = (all_preds >= 0.5).float()
    correct = (preds_binary == all_targets).float().mean().item()

    pos_ratio = all_targets.mean().item()

    return total_loss / n_batches, correct, pos_ratio


def train_gating(
    embeddings_file: str,
    labels_csv: str,
    aug_embeddings_file: str | None = None,
    aug_csv: str | None = None,
    output_dir: str = "checkpoints_gating",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
    seed: int = 42,
):
    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TrumpetJudge GatingHead Training")
    print("=" * 60)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # Dataset (use simple random split, since this is just gating)
    full_dataset = GatingDataset(
        embeddings_file=embeddings_file,
        labels_csv=labels_csv,
        aug_embeddings_file=aug_embeddings_file,
        aug_csv=aug_csv,
    )

    n_samples = len(full_dataset)
    n_val = max(1, int(0.2 * n_samples))
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"\n  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")

    model = GatingHead(embedding_dim=full_dataset.embeddings.shape[1])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  GatingHead parameters: {n_params:,}")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_ckpt_path = output_dir / "best_gating.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, pos_ratio = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:5.1f}% | "
            f"Pos ratio (val): {pos_ratio:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "pos_ratio": pos_ratio,
                    "embeddings_file": embeddings_file,
                    "labels_csv": labels_csv,
                },
                best_ckpt_path,
            )
            print(f"  → Saved best gating model to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    print("\n" + "=" * 60)
    print("GatingHead Training Complete")
    print("=" * 60)
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {best_ckpt_path}")

    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train GatingHead on embeddings + rejected flags")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings/all_audio.pt",
        help="Embeddings file (from precompute.py --no_labels)",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="data/prepared/all_gated.csv",
        help="CSV with columns id and is_valid (from prepare_data.py --include_rejected --save_all)",
    )
    parser.add_argument(
        "--aug_embeddings",
        type=str,
        default=None,
        help="Augmented embeddings file (e.g., data/embeddings/all_augmented.pt)",
    )
    parser.add_argument(
        "--aug_csv",
        type=str,
        default=None,
        help="Augmented CSV with original_id column (e.g., data/prepared/all_augmented.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_gating",
        help="Directory to save gating checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    train_gating(
        embeddings_file=args.embeddings,
        labels_csv=args.labels_csv,
        aug_embeddings_file=args.aug_embeddings,
        aug_csv=args.aug_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


