"""
PANNs Setup Module

This module ensures PANNs data files are available before importing panns_inference.
It handles:
1. Creating ~/panns_data directory
2. Copying class_labels_indices.csv from repo
3. Downloading CNN14 model weights if needed (using urllib, not wget)

IMPORTANT: Import this module BEFORE importing panns_inference!
"""

import os
import shutil
from pathlib import Path
import urllib.request
import sys


# PANNs data directory (where panns_inference expects files)
PANNS_DATA_DIR = Path.home() / "panns_data"

# Model weights URL and filename
MODEL_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
MODEL_FILENAME = "Cnn14_mAP=0.431.pth"

# Labels file (bundled in repo)
LABELS_FILENAME = "class_labels_indices.csv"


def get_repo_weights_dir() -> Path:
    """Get the path to the weights directory in the repo."""
    return Path(__file__).parent.parent / "models" / "weights"


def setup_panns_data():
    """
    Ensure PANNs data files are available.
    
    This function:
    1. Creates ~/panns_data if it doesn't exist
    2. Copies class_labels_indices.csv from the repo if not present
    3. Downloads CNN14 weights if not present (with progress indicator)
    
    Call this BEFORE importing panns_inference.
    """
    # Create panns_data directory
    PANNS_DATA_DIR.mkdir(exist_ok=True)
    
    # Copy labels CSV from repo if needed
    labels_dest = PANNS_DATA_DIR / LABELS_FILENAME
    if not labels_dest.exists():
        labels_src = get_repo_weights_dir() / LABELS_FILENAME
        if labels_src.exists():
            shutil.copy(labels_src, labels_dest)
            print(f"  Copied {LABELS_FILENAME} to {PANNS_DATA_DIR}")
        else:
            raise FileNotFoundError(
                f"Labels file not found in repo: {labels_src}\n"
                "Please ensure models/weights/class_labels_indices.csv exists."
            )
    
    # Download model weights if needed
    model_dest = PANNS_DATA_DIR / MODEL_FILENAME
    if not model_dest.exists():
        print(f"  Downloading PANNs CNN14 model weights (~300MB)...")
        print(f"  This is a one-time download.")
        _download_with_progress(MODEL_URL, model_dest)
        print(f"  Saved to {model_dest}")
    
    return True


def _download_with_progress(url: str, dest: Path):
    """Download a file with a progress indicator."""
    
    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # Newline after progress
    except Exception as e:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download model weights: {e}")


def check_panns_ready() -> bool:
    """Check if PANNs data files are available."""
    labels_ok = (PANNS_DATA_DIR / LABELS_FILENAME).exists()
    model_ok = (PANNS_DATA_DIR / MODEL_FILENAME).exists()
    return labels_ok and model_ok


# Auto-setup when this module is imported
if not check_panns_ready():
    print("Setting up PANNs data files...")
    setup_panns_data()
    print("PANNs setup complete!")

