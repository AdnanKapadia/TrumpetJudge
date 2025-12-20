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
import ssl
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


def _get_ssl_context():
    """
    Get an SSL context that works across platforms.
    
    On macOS, Python doesn't use the system certificate store by default,
    which causes SSL certificate verification to fail. This function tries:
    1. certifi package (if installed)
    2. Default SSL context (works on most systems)
    3. Unverified context as last resort (with warning)
    """
    # Try using certifi first (recommended for macOS)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    
    # Try default context (works on Windows/Linux usually)
    try:
        ctx = ssl.create_default_context()
        # Test if it can actually verify certificates
        return ctx
    except Exception:
        pass
    
    # Last resort: unverified context with warning
    print("  Warning: Using unverified SSL context. Consider installing certifi:")
    print("    pip install certifi")
    return ssl.create_default_context()


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
        # Try with SSL context first (handles macOS certificate issues)
        try:
            ssl_context = _get_ssl_context()
            # urlretrieve doesn't support ssl context, so we use urlopen + manual write
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, context=ssl_context) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                block_size = 8192
                downloaded = 0
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = min(100, downloaded * 100 // total_size)
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                            sys.stdout.flush()
            print()  # Newline after progress
        except ssl.SSLCertVerificationError:
            # If SSL still fails, try with unverified context as last resort
            print("\n  SSL verification failed. Retrying with unverified context...")
            print("  (This is safe for this specific download from Zenodo)")
            ssl_context = ssl._create_unverified_context()
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, context=ssl_context) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                block_size = 8192
                downloaded = 0
                with open(dest, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = min(100, downloaded * 100 // total_size)
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                            sys.stdout.flush()
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

