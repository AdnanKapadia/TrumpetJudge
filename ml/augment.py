"""
Blind Data Augmentation for TrumpetJudge Training.

Implements 5 types of augmentation to prevent overfitting and improve generalization:

1. Loudness/Gain - Random gain changes to prevent "louder = better" shortcuts
2. Time Shift - Random shifts to make model alignment-invariant  
3. Additive Noise - White/pink/brown noise for recording condition robustness
4. Convolution Reverb - Room IRs to prevent room overfitting
5. SpecAugment - Time/frequency masking for spectral robustness

All augmentations are applied in waveform domain except SpecAugment which
operates on mel spectrograms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Callable
from pathlib import Path
import random
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 1. LOUDNESS / GAIN AUGMENTATION
# =============================================================================

def apply_gain_db(
    waveform: Union[np.ndarray, torch.Tensor],
    gain_db: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply gain in decibels to waveform.
    
    Args:
        waveform: Audio waveform (numpy or torch)
        gain_db: Gain to apply in dB (positive = louder, negative = quieter)
        
    Returns:
        Gained waveform (same type as input)
    """
    gain_linear = 10 ** (gain_db / 20.0)
    return waveform * gain_linear


def random_gain(
    waveform: Union[np.ndarray, torch.Tensor],
    min_db: float = -12.0,
    max_db: float = 12.0,
    p: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply random gain augmentation.
    
    Prevents "louder = better" shortcuts by randomly scaling volume.
    
    Args:
        waveform: Audio waveform
        min_db: Minimum gain in dB (default: -12)
        max_db: Maximum gain in dB (default: +12)
        p: Probability of applying augmentation
        
    Returns:
        Augmented waveform
    """
    if random.random() > p:
        return waveform
    
    gain_db = random.uniform(min_db, max_db)
    return apply_gain_db(waveform, gain_db)


def normalize_peak(
    waveform: Union[np.ndarray, torch.Tensor],
    target_peak: float = 0.95,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize waveform to target peak level.
    
    Args:
        waveform: Audio waveform
        target_peak: Target peak level (0.0 to 1.0)
        
    Returns:
        Normalized waveform
    """
    if isinstance(waveform, torch.Tensor):
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform * (target_peak / max_val)
    else:
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform * (target_peak / max_val)
    return waveform


# =============================================================================
# 2. TIME SHIFT AUGMENTATION
# =============================================================================

def time_shift(
    waveform: Union[np.ndarray, torch.Tensor],
    shift_samples: int,
    mode: str = "zero",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shift waveform in time.
    
    Args:
        waveform: Audio waveform (1D)
        shift_samples: Number of samples to shift (positive = delay, negative = advance)
        mode: Padding mode - "zero" or "reflect"
        
    Returns:
        Shifted waveform (same length as input)
    """
    is_torch = isinstance(waveform, torch.Tensor)
    
    if shift_samples == 0:
        return waveform
    
    length = len(waveform)
    
    if is_torch:
        if shift_samples > 0:
            # Delay: shift right, pad left
            if mode == "zero":
                padding = torch.zeros(shift_samples, dtype=waveform.dtype, device=waveform.device)
            else:  # reflect
                padding = waveform[:shift_samples].flip(0)
            shifted = torch.cat([padding, waveform[:-shift_samples]])
        else:
            # Advance: shift left, pad right
            shift_samples = abs(shift_samples)
            if mode == "zero":
                padding = torch.zeros(shift_samples, dtype=waveform.dtype, device=waveform.device)
            else:  # reflect
                padding = waveform[-shift_samples:].flip(0)
            shifted = torch.cat([waveform[shift_samples:], padding])
    else:
        if shift_samples > 0:
            # Delay: shift right, pad left
            if mode == "zero":
                padding = np.zeros(shift_samples, dtype=waveform.dtype)
            else:  # reflect
                padding = waveform[:shift_samples][::-1]
            shifted = np.concatenate([padding, waveform[:-shift_samples]])
        else:
            # Advance: shift left, pad right
            shift_samples = abs(shift_samples)
            if mode == "zero":
                padding = np.zeros(shift_samples, dtype=waveform.dtype)
            else:  # reflect
                padding = waveform[-shift_samples:][::-1]
            shifted = np.concatenate([waveform[shift_samples:], padding])
    
    return shifted


def random_time_shift(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 32000,
    min_ms: float = -50.0,
    max_ms: float = 50.0,
    mode: str = "zero",
    p: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply random time shift augmentation.
    
    Makes model invariant to absolute alignment.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        min_ms: Minimum shift in milliseconds (negative = advance)
        max_ms: Maximum shift in milliseconds (positive = delay)
        mode: Padding mode - "zero" or "reflect"
        p: Probability of applying augmentation
        
    Returns:
        Shifted waveform
    """
    if random.random() > p:
        return waveform
    
    shift_ms = random.uniform(min_ms, max_ms)
    shift_samples = int(shift_ms * sample_rate / 1000)
    
    return time_shift(waveform, shift_samples, mode=mode)


# =============================================================================
# 3. ADDITIVE NOISE AUGMENTATION  
# =============================================================================

def generate_white_noise(length: int, dtype=np.float32) -> np.ndarray:
    """Generate white noise."""
    return np.random.randn(length).astype(dtype)


def generate_pink_noise(length: int, dtype=np.float32) -> np.ndarray:
    """
    Generate pink noise (1/f noise) using Voss-McCartney algorithm.
    
    Pink noise has equal energy per octave, more natural sounding than white.
    """
    # Number of random sources
    num_sources = 16
    
    # Initialize
    max_key = num_sources - 1
    key = 0
    white_values = np.zeros(num_sources, dtype=dtype)
    
    samples = np.zeros(length, dtype=dtype)
    
    for i in range(length):
        last_key = key
        key += 1
        if key > max_key:
            key = 0
        
        # XOR to find changed bits
        diff = last_key ^ key
        
        # Update changed sources
        for j in range(num_sources):
            if diff & (1 << j):
                white_values[j] = np.random.randn()
        
        samples[i] = white_values.sum()
    
    # Normalize
    samples = samples / (np.abs(samples).max() + 1e-8)
    return samples


def generate_brown_noise(length: int, dtype=np.float32) -> np.ndarray:
    """
    Generate brown/red noise (1/f² noise) using integration.
    
    Brown noise has more bass emphasis than pink noise.
    """
    white = np.random.randn(length).astype(dtype)
    brown = np.cumsum(white)
    
    # High-pass filter to prevent DC drift
    brown = brown - np.mean(brown)
    
    # Normalize
    brown = brown / (np.abs(brown).max() + 1e-8)
    return brown


def add_noise(
    waveform: Union[np.ndarray, torch.Tensor],
    noise: Union[np.ndarray, torch.Tensor],
    snr_db: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add noise to waveform at specified SNR.
    
    Args:
        waveform: Clean audio waveform
        noise: Noise signal (will be scaled to match SNR)
        snr_db: Target signal-to-noise ratio in dB
        
    Returns:
        Noisy waveform
    """
    is_torch = isinstance(waveform, torch.Tensor)
    
    # Ensure noise is same length as waveform
    if len(noise) < len(waveform):
        # Tile noise to match length
        repeats = int(np.ceil(len(waveform) / len(noise)))
        if is_torch:
            noise = noise.repeat(repeats)[:len(waveform)]
        else:
            noise = np.tile(noise, repeats)[:len(waveform)]
    elif len(noise) > len(waveform):
        noise = noise[:len(waveform)]
    
    # Convert to numpy for calculation
    if is_torch:
        wave_np = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
        noise_np = noise.cpu().numpy() if isinstance(noise, torch.Tensor) and noise.is_cuda else (
            noise.numpy() if isinstance(noise, torch.Tensor) else noise
        )
    else:
        wave_np = waveform
        noise_np = noise if isinstance(noise, np.ndarray) else noise.numpy()
    
    # Calculate signal and noise power
    signal_power = np.mean(wave_np ** 2)
    noise_power = np.mean(noise_np ** 2)
    
    if noise_power == 0 or signal_power == 0:
        return waveform
    
    # Calculate scaling factor for noise
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    
    # Scale and add noise
    if is_torch:
        if isinstance(noise, np.ndarray):
            noise = torch.from_numpy(noise).to(waveform.device)
        noisy = waveform + scale * noise
    else:
        if isinstance(noise, torch.Tensor):
            noise = noise.numpy()
        noisy = waveform + scale * noise
    
    return noisy


def random_noise(
    waveform: Union[np.ndarray, torch.Tensor],
    noise_types: List[str] = ["white", "pink", "brown"],
    min_snr_db: float = 20.0,
    max_snr_db: float = 35.0,
    p: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply random additive noise augmentation.
    
    Improves robustness to recording conditions.
    
    Args:
        waveform: Audio waveform
        noise_types: Types of noise to choose from ("white", "pink", "brown")
        min_snr_db: Minimum SNR in dB (lower = more noise)
        max_snr_db: Maximum SNR in dB (higher = less noise)
        p: Probability of applying augmentation
        
    Returns:
        Noisy waveform
    """
    if random.random() > p:
        return waveform
    
    # Select random noise type
    noise_type = random.choice(noise_types)
    length = len(waveform)
    
    # Generate noise
    if noise_type == "white":
        noise = generate_white_noise(length)
    elif noise_type == "pink":
        noise = generate_pink_noise(length)
    elif noise_type == "brown":
        noise = generate_brown_noise(length)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Convert to torch if needed
    if isinstance(waveform, torch.Tensor):
        noise = torch.from_numpy(noise).to(waveform.dtype).to(waveform.device)
    
    # Random SNR
    snr_db = random.uniform(min_snr_db, max_snr_db)
    
    return add_noise(waveform, noise, snr_db)


# =============================================================================
# 4. CONVOLUTION REVERB AUGMENTATION
# =============================================================================

# Cache loaded IRs to avoid repeated disk reads
_IR_CACHE = {}


def load_ir_cached(ir_path: Path, sample_rate: int = 32000) -> np.ndarray:
    """Load and cache impulse response."""
    cache_key = (str(ir_path), sample_rate)
    
    if cache_key not in _IR_CACHE:
        from dsp.conv.conv_fast import load_impulse_response
        ir, _ = load_impulse_response(ir_path, target_sr=sample_rate)
        _IR_CACHE[cache_key] = ir
    
    return _IR_CACHE[cache_key]


def get_all_ir_paths(ir_dir: Optional[Path] = None) -> List[Path]:
    """Get list of all IR file paths."""
    if ir_dir is None:
        ir_dir = Path(__file__).parent.parent / "data" / "impulse_response"
    
    ir_dir = Path(ir_dir)
    return list(ir_dir.glob("*.wav"))


def random_reverb(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 32000,
    min_wet: float = 0.05,
    max_wet: float = 0.20,
    ir_dir: Optional[Path] = None,
    p: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply random convolution reverb augmentation.
    
    Removes room overfitting by adding various room acoustics.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        min_wet: Minimum wet mix (0.0 to 1.0)
        max_wet: Maximum wet mix (0.0 to 1.0)
        ir_dir: Directory containing impulse response files
        p: Probability of applying augmentation
        
    Returns:
        Reverberant waveform
    """
    if random.random() > p:
        return waveform
    
    from dsp.conv.conv_fast import convolve_fast
    
    # Get available IRs
    ir_paths = get_all_ir_paths(ir_dir)
    if not ir_paths:
        return waveform  # No IRs available
    
    # Select random IR
    ir_path = random.choice(ir_paths)
    ir = load_ir_cached(ir_path, sample_rate)
    
    # Random wet mix
    wet_mix = random.uniform(min_wet, max_wet)
    
    # Convert to numpy for convolution
    is_torch = isinstance(waveform, torch.Tensor)
    if is_torch:
        device = waveform.device
        dtype = waveform.dtype
        wave_np = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
    else:
        wave_np = waveform
    
    # Apply reverb
    reverbed = convolve_fast(wave_np, ir, wet_mix=wet_mix, normalize=True)
    
    # Convert back to torch if needed
    if is_torch:
        reverbed = torch.from_numpy(reverbed).to(dtype).to(device)
    
    return reverbed


def preload_all_irs(ir_dir: Optional[Path] = None, sample_rate: int = 32000):
    """
    Preload all impulse responses into cache.
    
    Call this before training to avoid IO during training.
    """
    ir_paths = get_all_ir_paths(ir_dir)
    print(f"Preloading {len(ir_paths)} impulse responses...")
    for ir_path in ir_paths:
        load_ir_cached(ir_path, sample_rate)
    print("  Done.")


# =============================================================================
# 5. SPECAUGMENT (Spectrogram-domain augmentation)
# =============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment augmentation for mel spectrograms.
    
    Applies time and frequency masking to improve robustness to local
    spectral variation.
    
    Paper: "SpecAugment: A Simple Data Augmentation Method for ASR"
    https://arxiv.org/abs/1904.08779
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 1.0,
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum width of frequency masks (F)
            time_mask_param: Maximum width of time masks (T)
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            p: Probability of applying augmentation
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Mel spectrogram of shape (batch, freq, time) or (freq, time)
            
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.p:
            return spectrogram
        
        # Handle batched or unbatched input
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            was_unbatched = True
        else:
            was_unbatched = False
        
        batch_size, num_freq, num_time = spectrogram.shape
        augmented = spectrogram.clone()
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, num_freq - 1))
            f0 = random.randint(0, num_freq - f)
            augmented[:, f0:f0 + f, :] = 0
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, num_time - 1))
            t0 = random.randint(0, num_time - t)
            augmented[:, :, t0:t0 + t] = 0
        
        if was_unbatched:
            augmented = augmented.squeeze(0)
        
        return augmented


def apply_spec_augment(
    spectrogram: torch.Tensor,
    freq_mask_param: int = 10,
    time_mask_param: int = 20,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    p: float = 1.0,
) -> torch.Tensor:
    """
    Functional version of SpecAugment.
    
    Args:
        spectrogram: Mel spectrogram tensor
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        p: Probability of applying augmentation
        
    Returns:
        Augmented spectrogram
    """
    augmenter = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        p=p,
    )
    return augmenter(spectrogram)


# =============================================================================
# COMBINED AUGMENTATION PIPELINE
# =============================================================================

class WaveformAugmentPipeline:
    """
    Combined waveform augmentation pipeline.
    
    Applies augmentations in sequence with configurable probabilities.
    Order: Gain → Time Shift → Noise → Reverb
    
    Note: SpecAugment is NOT included here as it operates on spectrograms,
    not waveforms. Apply it separately after mel spectrogram extraction.
    """
    
    def __init__(
        self,
        # Gain settings
        gain_p: float = 0.8,
        gain_min_db: float = -12.0,
        gain_max_db: float = 12.0,
        # Time shift settings
        shift_p: float = 0.8,
        shift_min_ms: float = -50.0,
        shift_max_ms: float = 50.0,
        shift_mode: str = "zero",
        # Noise settings
        noise_p: float = 0.5,
        noise_types: List[str] = None,
        noise_min_snr: float = 20.0,
        noise_max_snr: float = 35.0,
        # Reverb settings
        reverb_p: float = 0.5,
        reverb_min_wet: float = 0.05,
        reverb_max_wet: float = 0.20,
        reverb_ir_dir: Optional[Path] = None,
        # General
        sample_rate: int = 32000,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            gain_p: Probability of gain augmentation
            gain_min_db: Minimum gain in dB
            gain_max_db: Maximum gain in dB
            shift_p: Probability of time shift
            shift_min_ms: Minimum shift in ms
            shift_max_ms: Maximum shift in ms
            shift_mode: Shift padding mode ("zero" or "reflect")
            noise_p: Probability of additive noise
            noise_types: Types of noise to use
            noise_min_snr: Minimum SNR in dB
            noise_max_snr: Maximum SNR in dB
            reverb_p: Probability of reverb
            reverb_min_wet: Minimum wet mix
            reverb_max_wet: Maximum wet mix
            reverb_ir_dir: Directory for impulse responses
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
        # Gain
        self.gain_p = gain_p
        self.gain_min_db = gain_min_db
        self.gain_max_db = gain_max_db
        
        # Time shift
        self.shift_p = shift_p
        self.shift_min_ms = shift_min_ms
        self.shift_max_ms = shift_max_ms
        self.shift_mode = shift_mode
        
        # Noise
        self.noise_p = noise_p
        self.noise_types = noise_types or ["white", "pink", "brown"]
        self.noise_min_snr = noise_min_snr
        self.noise_max_snr = noise_max_snr
        
        # Reverb
        self.reverb_p = reverb_p
        self.reverb_min_wet = reverb_min_wet
        self.reverb_max_wet = reverb_max_wet
        self.reverb_ir_dir = reverb_ir_dir
    
    def __call__(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply augmentation pipeline to waveform.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Augmented waveform
        """
        # 1. Gain augmentation
        waveform = random_gain(
            waveform,
            min_db=self.gain_min_db,
            max_db=self.gain_max_db,
            p=self.gain_p,
        )
        
        # 2. Time shift
        waveform = random_time_shift(
            waveform,
            sample_rate=self.sample_rate,
            min_ms=self.shift_min_ms,
            max_ms=self.shift_max_ms,
            mode=self.shift_mode,
            p=self.shift_p,
        )
        
        # 3. Additive noise
        waveform = random_noise(
            waveform,
            noise_types=self.noise_types,
            min_snr_db=self.noise_min_snr,
            max_snr_db=self.noise_max_snr,
            p=self.noise_p,
        )
        
        # 4. Convolution reverb
        waveform = random_reverb(
            waveform,
            sample_rate=self.sample_rate,
            min_wet=self.reverb_min_wet,
            max_wet=self.reverb_max_wet,
            ir_dir=self.reverb_ir_dir,
            p=self.reverb_p,
        )
        
        return waveform


def create_train_augment_pipeline(
    sample_rate: int = 32000,
    ir_dir: Optional[Path] = None,
) -> WaveformAugmentPipeline:
    """
    Create the recommended augmentation pipeline for training.
    
    Uses conservative settings that are safe for music/performance audio.
    
    Args:
        sample_rate: Audio sample rate
        ir_dir: Directory for impulse responses
        
    Returns:
        Configured WaveformAugmentPipeline
    """
    return WaveformAugmentPipeline(
        # Gain: HIGH value, very safe
        gain_p=0.8,
        gain_min_db=-12.0,
        gain_max_db=12.0,
        # Time shift: HIGH value, safe
        shift_p=0.8,
        shift_min_ms=-50.0,
        shift_max_ms=50.0,
        shift_mode="zero",
        # Noise: MODERATE value, keep light (high SNR)
        noise_p=0.5,
        noise_types=["white", "pink", "brown"],
        noise_min_snr=20.0,
        noise_max_snr=35.0,
        # Reverb: MODERATE-HIGH value, conservative wet mix
        reverb_p=0.5,
        reverb_min_wet=0.05,
        reverb_max_wet=0.20,
        reverb_ir_dir=ir_dir,
        sample_rate=sample_rate,
    )


def create_spec_augment(
    freq_mask_param: int = 10,
    time_mask_param: int = 20,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    p: float = 0.8,
) -> SpecAugment:
    """
    Create SpecAugment module with conservative settings.
    
    Args:
        freq_mask_param: Max frequency mask width (default: 10 bins)
        time_mask_param: Max time mask width (default: 20 frames)
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        p: Probability of applying
        
    Returns:
        Configured SpecAugment module
    """
    return SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        p=p,
    )


# =============================================================================
# TESTING
# =============================================================================

def _test_augmentations():
    """Test all augmentation functions."""
    print("Testing augmentations...")
    
    # Create test waveform (1 second at 32kHz)
    sample_rate = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    waveform = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    print(f"  Input shape: {waveform.shape}")
    print(f"  Input range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Test 1: Gain
    print("\n1. Testing gain augmentation...")
    gained = random_gain(waveform, min_db=-6, max_db=6, p=1.0)
    print(f"   Output range: [{gained.min():.3f}, {gained.max():.3f}]")
    
    # Test 2: Time shift
    print("\n2. Testing time shift...")
    shifted = random_time_shift(waveform, sample_rate=sample_rate, min_ms=-20, max_ms=20, p=1.0)
    print(f"   Output shape: {shifted.shape}")
    
    # Test 3: Noise
    print("\n3. Testing additive noise...")
    noisy = random_noise(waveform, min_snr_db=20, max_snr_db=30, p=1.0)
    print(f"   Output range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # Test 4: Reverb (skip if no IRs)
    print("\n4. Testing convolution reverb...")
    try:
        reverbed = random_reverb(waveform, sample_rate=sample_rate, p=1.0)
        print(f"   Output shape: {reverbed.shape}")
    except FileNotFoundError:
        print("   Skipped (no IR files found)")
    
    # Test 5: SpecAugment
    print("\n5. Testing SpecAugment...")
    # Create fake spectrogram (64 mel bins, 100 time frames)
    spec = torch.randn(64, 100)
    augmented_spec = apply_spec_augment(spec, p=1.0)
    zeros = (augmented_spec == 0).sum().item()
    print(f"   Masked elements: {zeros} / {spec.numel()}")
    
    # Test 6: Full pipeline
    print("\n6. Testing full pipeline...")
    pipeline = create_train_augment_pipeline(sample_rate=sample_rate)
    try:
        augmented = pipeline(waveform)
        print(f"   Output shape: {augmented.shape}")
        print(f"   Output range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    except FileNotFoundError:
        print("   Partially skipped (no IR files)")
    
    # Test with torch tensor
    print("\n7. Testing with torch tensors...")
    waveform_torch = torch.from_numpy(waveform)
    gained_torch = random_gain(waveform_torch, min_db=-6, max_db=6, p=1.0)
    print(f"   Torch output type: {type(gained_torch)}")
    
    print("\n✓ All augmentation tests passed!")


if __name__ == "__main__":
    _test_augmentations()

