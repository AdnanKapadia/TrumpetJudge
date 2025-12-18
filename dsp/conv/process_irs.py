"""
Process impulse response files:
1. Filter by selection rules
2. Randomly select 50 files
3. Delete the rest
4. Process selected files: upscale to 32kHz, trim leading silence, normalize with headroom, cap tail length, ensure mono
"""

import os
import random
import fnmatch
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import torchaudio

# Configuration
# Point to data/impulse_response folder (go up from dsp/conv to project root, then to data/impulse_response)
IR_DIR = Path(__file__).parent.parent.parent / "data" / "impulse_response"
TARGET_SAMPLE_RATE = 32000
NUM_SELECT = 50
SILENCE_THRESHOLD_DB = -40  # dB threshold for silence detection
SILENCE_WINDOW_MS = 50  # Window size in ms for silence detection
MAX_LENGTH_SECONDS = 3.0  # Maximum length for IR tail cap (in seconds)


def matches_selection_rules(filename: str) -> bool:
    """
    Check if file matches selection rules:
    - Include: *rir*.wav and air_type1_air_phone_*.wav
    - Exclude: *noise*.wav and *cirline*.wav
    """
    filename_lower = filename.lower()
    
    # Exclude noise files
    if 'noise' in filename_lower:
        return False
    
    # Exclude cirline files (these don't work)
    if 'cirline' in filename_lower:
        return False
    
    # Include rir files
    if 'rir' in filename_lower:
        return True
    
    # Include air_type1_air_phone_ files
    if fnmatch.fnmatch(filename_lower, 'air_type1_air_phone_*.wav'):
        return True
    
    return False


def trim_leading_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40) -> np.ndarray:
    """
    Trim leading silence from audio.
    
    Args:
        audio: Audio array (mono, 1D)
        sample_rate: Sample rate
        threshold_db: RMS threshold in dB for silence detection
        
    Returns:
        Trimmed audio array
    """
    if len(audio) == 0:
        return audio
    
    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20.0)
    
    # Calculate RMS in windows
    window_samples = int(SILENCE_WINDOW_MS * sample_rate / 1000)
    window_samples = max(1, window_samples)  # Ensure at least 1 sample
    
    # Find first non-silent window
    start_idx = 0
    for i in range(0, len(audio) - window_samples, window_samples):
        window = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > threshold_linear:
            start_idx = i
            break
    
    return audio[start_idx:]


def process_ir_file(filepath: Path) -> bool:
    """
    Process a single IR file:
    - Load audio
    - Convert to mono
    - Resample to 32kHz
    - Trim leading silence
    - Normalize with headroom (0.95)
    - Trim tail to max length
    - Save back to same file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Processing: {filepath.name}")
        
        # Load audio
        audio, sr = sf.read(filepath, dtype='float32')
        
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 32kHz if needed
        if sr != TARGET_SAMPLE_RATE:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=TARGET_SAMPLE_RATE
            )
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sr = TARGET_SAMPLE_RATE
        
        # Trim leading silence (before normalizing)
        audio = trim_leading_silence(audio, sr, SILENCE_THRESHOLD_DB)
        
        # Normalize with headroom (0.95 to avoid clipping)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            audio *= 0.95
        
        # Trim tail to max length (cap IR duration)
        max_samples = int(MAX_LENGTH_SECONDS * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"    Trimmed tail to {MAX_LENGTH_SECONDS}s")
        
        # Save processed file
        sf.write(filepath, audio, sr)
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {filepath.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Impulse Response Processing Script")
    print("=" * 60)
    
    # Verify IR directory exists
    if not IR_DIR.exists():
        print(f"\nERROR: Impulse response directory not found: {IR_DIR}")
        print(f"       Please ensure the directory exists.")
        return
    
    # Step 1: Find all WAV files
    print(f"\n[1/5] Scanning {IR_DIR} for WAV files...")
    all_wav_files = list(IR_DIR.glob("*.wav"))
    print(f"       Found {len(all_wav_files)} WAV files")
    
    # Step 2: Filter by selection rules
    print(f"\n[2/5] Filtering by selection rules...")
    matching_files = [f for f in all_wav_files if matches_selection_rules(f.name)]
    print(f"       {len(matching_files)} files match selection rules")
    
    if len(matching_files) == 0:
        print("       ERROR: No files match the selection rules!")
        return
    
    # Step 3: Randomly select 50 files
    print(f"\n[3/5] Randomly selecting {NUM_SELECT} files...")
    if len(matching_files) <= NUM_SELECT:
        selected_files = matching_files
        print(f"       Only {len(matching_files)} files available, keeping all")
    else:
        random.seed(42)  # For reproducibility
        selected_files = random.sample(matching_files, NUM_SELECT)
        print(f"       Selected {NUM_SELECT} files")
    
    # Step 4: Delete non-selected files
    print(f"\n[4/5] Deleting non-selected files...")
    deleted_count = 0
    for file in matching_files:
        if file not in selected_files:
            try:
                file.unlink()
                deleted_count += 1
                print(f"       Deleted: {file.name}")
            except Exception as e:
                print(f"       ERROR deleting {file.name}: {e}")
    
    # Also delete any WAV files that didn't match the rules
    for file in all_wav_files:
        if file not in matching_files:
            try:
                file.unlink()
                deleted_count += 1
                print(f"       Deleted (didn't match rules): {file.name}")
            except Exception as e:
                print(f"       ERROR deleting {file.name}: {e}")
    
    print(f"       Deleted {deleted_count} files total")
    
    # Step 5: Process selected files
    print(f"\n[5/5] Processing {len(selected_files)} selected files...")
    success_count = 0
    for file in selected_files:
        if process_ir_file(file):
            success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"  - Selected: {len(selected_files)} files")
    print(f"  - Successfully processed: {success_count} files")
    print(f"  - Failed: {len(selected_files) - success_count} files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

