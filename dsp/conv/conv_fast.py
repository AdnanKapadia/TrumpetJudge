"""
Fast convolution-based reverb using FFT.

Provides reusable functions for applying impulse responses to audio signals.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
from typing import Union, Tuple, Optional
import random


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def convolve_fast(
    signal: np.ndarray,
    impulse_response: np.ndarray,
    wet_mix: float = 0.15,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply convolution reverb to a signal using FFT.
    
    Args:
        signal: Input audio signal (1D numpy array)
        impulse_response: Impulse response (1D numpy array)
        wet_mix: Mix ratio of wet signal (0.0 = dry only, 1.0 = wet only)
        normalize: If True, normalize output to prevent clipping
        
    Returns:
        Processed audio with reverb applied
    """
    # Ensure 1D
    if signal.ndim > 1:
        signal = signal.mean(axis=1) if signal.shape[1] > 1 else signal.flatten()
    if impulse_response.ndim > 1:
        impulse_response = impulse_response.mean(axis=1) if impulse_response.shape[1] > 1 else impulse_response.flatten()
    
    # Calculate FFT size (next power of 2 for efficiency)
    fft_size = next_power_of_2(max(len(signal), len(impulse_response)))
    
    # Convolve using FFT
    wet = np.fft.irfft(np.fft.rfft(signal, fft_size) * np.fft.rfft(impulse_response, fft_size))
    
    # Trim to original length
    wet = wet[:len(signal)]
    
    # Mix dry and wet signals
    dry_mix = 1.0 - wet_mix
    output = dry_mix * signal + wet_mix * wet
    
    # Normalize to prevent clipping
    if normalize:
        max_val = np.abs(output).max()
        if max_val > 0.95:
            output = output * (0.95 / max_val)
    
    return output


def load_impulse_response(
    ir_path: Union[str, Path],
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an impulse response from file.
    
    Args:
        ir_path: Path to impulse response WAV file
        target_sr: If provided, resample to this sample rate
        
    Returns:
        Tuple of (impulse_response, sample_rate)
    """
    ir, sr = sf.read(ir_path, dtype='float32')
    
    # Convert to mono if needed
    if ir.ndim > 1:
        ir = ir.mean(axis=1)
    
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        try:
            import torchaudio
            import torch
            ir_tensor = torch.from_numpy(ir).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            ir_tensor = resampler(ir_tensor)
            ir = ir_tensor.squeeze(0).numpy()
            sr = target_sr
        except ImportError:
            # Fallback: simple interpolation (not ideal for audio)
            import warnings
            warnings.warn("torchaudio not available, using linear interpolation for resampling")
            from scipy import signal as scipy_signal
            num_samples = int(len(ir) * target_sr / sr)
            ir = scipy_signal.resample(ir, num_samples)
            sr = target_sr
    
    return ir, sr


def get_random_ir(
    ir_dir: Optional[Union[str, Path]] = None,
    target_sr: int = 32000,
) -> Tuple[np.ndarray, int, str]:
    """
    Load a random impulse response from the IR directory.
    
    Args:
        ir_dir: Directory containing IR files. If None, uses default location.
        target_sr: Target sample rate for the IR
        
    Returns:
        Tuple of (impulse_response, sample_rate, filename)
    """
    if ir_dir is None:
        # Default to project's impulse_response folder
        ir_dir = Path(__file__).parent.parent.parent / "data" / "impulse_response"
    
    ir_dir = Path(ir_dir)
    
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")
    
    # Get all WAV files
    ir_files = list(ir_dir.glob("*.wav"))
    if not ir_files:
        raise FileNotFoundError(f"No WAV files found in {ir_dir}")
    
    # Select random IR
    ir_path = random.choice(ir_files)
    ir, sr = load_impulse_response(ir_path, target_sr=target_sr)
    
    return ir, sr, ir_path.name


def apply_reverb(
    audio: np.ndarray,
    sample_rate: int = 32000,
    wet_mix: float = 0.15,
    ir_path: Optional[Union[str, Path]] = None,
    ir_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Apply reverb to audio using a random or specified impulse response.
    
    This is the main entry point for augmentation use.
    
    Args:
        audio: Input audio (1D numpy array)
        sample_rate: Sample rate of the input audio
        wet_mix: Wet/dry mix ratio (0.0-1.0)
        ir_path: Specific IR file to use. If None, random IR is selected.
        ir_dir: Directory to select random IR from (if ir_path not specified)
        
    Returns:
        Audio with reverb applied
    """
    if ir_path is not None:
        ir, _ = load_impulse_response(ir_path, target_sr=sample_rate)
    else:
        ir, _, _ = get_random_ir(ir_dir=ir_dir, target_sr=sample_rate)
    
    return convolve_fast(audio, ir, wet_mix=wet_mix)


def main():
    """Demo: apply reverb to a test file."""
    print("conv_fast.py - fast convolution based reverberation")
    
    # Default paths
    test_audio_path = os.path.join('data', 'impulse_response', 'Toms_diner.wav')
    ir_path = os.path.join('data', 'impulse_response', 'IR.wav')
    
    if not os.path.exists(test_audio_path):
        print(f"Test audio not found: {test_audio_path}")
        print("Usage: provide audio file or use apply_reverb() function directly")
        return
    
    if not os.path.exists(ir_path):
        # Try random IR
        print(f"Specific IR not found, using random IR from directory")
        ir, sr_ir, ir_name = get_random_ir(target_sr=32000)
        print(f"Selected IR: {ir_name}")
    else:
        ir, sr_ir = load_impulse_response(ir_path, target_sr=32000)
    
    # Load test audio
    x, sr_x = sf.read(test_audio_path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    
    print(f"Signal shape: {x.shape}")
    print(f"IR shape: {ir.shape}")
    
    # Apply reverb
    y = convolve_fast(x, ir, wet_mix=0.2)
    
    # Save output
    output_path = "reverb_output.wav"
    sf.write(output_path, y, sr_x)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
