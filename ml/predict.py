"""
Inference script for TrumpetJudge.

Run the trained model on an audio file and get scores for each 20s chunk.

Usage:
    python ml/predict.py --audio path/to/audio.wav
    python ml/predict.py --audio path/to/audio.wav --checkpoint checkpoints/run_XXXX/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import soundfile as sf
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder_panns import PANNsEncoder
from models.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores


def load_audio(audio_path: str, target_sr: int = 32000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(audio_path, dtype='float32')
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.squeeze(0).numpy()
    
    return audio


def chunk_audio(audio: np.ndarray, chunk_duration: float, sample_rate: int) -> list:
    """Split audio into fixed-duration chunks."""
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            # Skip if chunk is too short (less than half)
            if len(chunk) < chunk_samples // 2:
                continue
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        chunks.append(chunk)
    
    return chunks


def find_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> str:
    """Find the most recent checkpoint."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    runs = sorted(checkpoints_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No training runs found in {checkpoints_dir}")
    
    latest_run = runs[-1]
    checkpoint_path = latest_run / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {latest_run}")
    
    return str(checkpoint_path)


@torch.no_grad()
def predict(
    audio_path: str,
    checkpoint_path: str = None,
    chunk_duration: float = 20.0,
    device: str = None,
):
    """
    Run inference on an audio file.
    
    Args:
        audio_path: Path to audio file
        checkpoint_path: Path to model checkpoint (auto-finds latest if None)
        chunk_duration: Duration of each chunk in seconds
        device: Device to use (auto-detect if None)
    """
    print("=" * 60)
    print("TrumpetJudge Inference")
    print("=" * 60)
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    print(f"\nUsing checkpoint: {checkpoint_path}")
    
    # Load models
    print("\nLoading models...")
    encoder = PANNsEncoder(duration=chunk_duration, device=device)
    device = encoder.device
    
    head = RegressionHead(embedding_dim=encoder.embedding_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head = head.to(device)
    head.eval()
    
    print(f"  Device: {device}")
    print(f"  Model trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  Val MAE: {checkpoint.get('val_mae', '?'):.3f}")
    
    # Load and chunk audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path, target_sr=32000)
    duration_sec = len(audio) / 32000
    print(f"  Duration: {duration_sec:.1f}s")
    
    chunks = chunk_audio(audio, chunk_duration, 32000)
    print(f"  Chunks: {len(chunks)} x {chunk_duration}s")
    
    # Run inference on each chunk
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)
    
    all_scores = []
    
    for i, chunk in enumerate(chunks):
        # Prepare input
        waveform = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        
        # Forward pass
        embedding = encoder(waveform)
        prediction = head(embedding)
        
        # Unscale to 1-5 range
        scores = unscale_scores(prediction).squeeze(0).cpu().numpy()
        all_scores.append(scores)
        
        # Print chunk results
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration_sec)
        print(f"\nChunk {i+1} ({start_time:.0f}s - {end_time:.0f}s):")
        for j, name in enumerate(SCORE_NAMES):
            score = scores[j]
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"  {name:12s}: {score:.2f} {bar}")
    
    # Print average scores
    if len(all_scores) > 1:
        avg_scores = np.mean(all_scores, axis=0)
        print("\n" + "=" * 60)
        print("Average Scores")
        print("=" * 60)
        for j, name in enumerate(SCORE_NAMES):
            score = avg_scores[j]
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"  {name:12s}: {score:.2f} {bar}")
    
    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Run TrumpetJudge inference on audio")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (uses latest if not specified)")
    parser.add_argument("--chunk_duration", type=float, default=20.0,
                        help="Duration of each chunk in seconds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detect if not specified.")
    
    args = parser.parse_args()
    
    predict(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        chunk_duration=args.chunk_duration,
        device=args.device,
    )


if __name__ == "__main__":
    main()

