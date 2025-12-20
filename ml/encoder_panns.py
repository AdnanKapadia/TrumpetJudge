"""
PANNs CNN14 Pretrained Audio Encoder

This module wraps the pretrained PANNs (Pretrained Audio Neural Networks) CNN14 model
to extract audio embeddings from trumpet recordings. The encoder is frozen during
training - only the downstream regression head learns.

PANNs CNN14 reference:
    Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
    https://github.com/qiuqiangkong/audioset_tagging_cnn

Install: pip install panns-inference
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple

# Try to import panns_inference, provide helpful error if not installed
try:
    from panns_inference import AudioTagging
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False


class PANNsEncoder(nn.Module):
    """
    Wrapper around PANNs CNN14 for extracting audio embeddings.
    
    The model expects audio at 32kHz sample rate and outputs a 2048-dim embedding.
    This wrapper handles:
        - Resampling to 32kHz if needed
        - Padding/trimming to fixed duration
        - Extracting the embedding layer (before final classification)
    
    Attributes:
        sample_rate (int): Expected sample rate (32000 Hz for PANNs)
        embedding_dim (int): Output embedding dimension (2048 for CNN14)
        duration (float): Fixed audio duration in seconds
    """
    
    SAMPLE_RATE = 32000  # PANNs expects 32kHz
    EMBEDDING_DIM = 2048  # CNN14 embedding dimension
    
    def __init__(
        self,
        duration: float = 15.0,
        device: Optional[str] = None,
    ):
        """
        Initialize the PANNs encoder.
        
        Args:
            duration: Fixed audio duration in seconds. Audio will be padded/trimmed to this length.
            device: Device to load model on. If None, uses CUDA if available.
        """
        super().__init__()
        
        if not PANNS_AVAILABLE:
            raise ImportError(
                "panns_inference is required but not installed.\n"
                "Install with: pip install panns-inference"
            )
        
        self.duration = duration
        
        # Determine device - check CUDA compatibility
        if device:
            self.device = device
        elif torch.cuda.is_available():
            # Check GPU compute capability - need at least 7.0 for modern PyTorch
            try:
                capability = torch.cuda.get_device_capability(0)
                if capability[0] >= 7:
                    self.device = "cuda"
                else:
                    print(f"Warning: GPU compute capability {capability[0]}.{capability[1]} < 7.0. Using CPU.")
                    self.device = "cpu"
            except Exception:
                print("Warning: CUDA check failed. Using CPU.")
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        # Number of samples for fixed duration
        self.num_samples = int(self.SAMPLE_RATE * self.duration)
        
        # Load pretrained CNN14 using panns_inference
        # This downloads weights automatically on first use (~300MB)
        self.model = AudioTagging(
            checkpoint_path=None,  # Uses default CNN14 weights
            device=self.device,
        )
        
        # Freeze all parameters - we don't train the encoder
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        self.model.model.eval()
    
    def preprocess_audio(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: int,
    ) -> torch.Tensor:
        """
        Preprocess audio for the encoder.
        
        Steps:
            1. Convert to mono if stereo
            2. Resample to 32kHz
            3. Pad or trim to fixed duration
        
        Args:
            waveform: Audio tensor of shape (channels, samples) or (samples,)
            orig_sample_rate: Original sample rate of the audio
            
        Returns:
            Preprocessed audio tensor of shape (1, num_samples)
        """
        # Ensure 2D: (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 32kHz if needed
        if orig_sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        
        # Pad or trim to fixed length
        current_samples = waveform.shape[1]
        
        if current_samples < self.num_samples:
            # Pad with zeros at the end
            padding = self.num_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > self.num_samples:
            # Trim to fixed length (from the start)
            waveform = waveform[:, :self.num_samples]
        
        return waveform
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file from disk.
        
        Args:
            audio_path: Path to the audio file (.wav, .mp3, etc.)
            
        Returns:
            Tuple of (waveform tensor, sample rate)
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate
    
    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from preprocessed audio.
        
        Args:
            waveform: Preprocessed audio tensor of shape (batch, samples)
                      Audio should already be at 32kHz and fixed duration.
                      
        Returns:
            Embedding tensor of shape (batch, 2048)
        """
        # Ensure on correct device and numpy format for panns_inference
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform
        
        # panns_inference expects (batch, samples) as numpy array
        # It returns (clipwise_output, embedding)
        _, embedding = self.model.inference(waveform_np)
        
        # Convert back to tensor
        embedding = torch.from_numpy(embedding).to(self.device)
        
        return embedding
    
    @torch.no_grad()
    def encode_file(self, audio_path: str) -> torch.Tensor:
        """
        Convenience method: load audio file and extract embedding.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding tensor of shape (1, 2048)
        """
        # Load audio
        waveform, sample_rate = self.load_audio(audio_path)
        
        # Preprocess
        waveform = self.preprocess_audio(waveform, sample_rate)
        
        # Forward pass (waveform is already (1, samples) after preprocess)
        embedding = self.forward(waveform)
        
        return embedding
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM


def test_encoder():
    """Quick test to verify encoder works."""
    print("Initializing PANNs encoder...")
    print("(First run downloads ~300MB model weights)\n")
    
    encoder = PANNsEncoder(duration=15.0)
    print(f"  Device: {encoder.device}")
    print(f"  Sample rate: {encoder.SAMPLE_RATE} Hz")
    print(f"  Duration: {encoder.duration} s")
    print(f"  Embedding dim: {encoder.embedding_dim}")
    
    # Test with dummy audio
    print("\nTesting with dummy audio...")
    dummy_waveform = torch.randn(1, encoder.num_samples)
    embedding = encoder(dummy_waveform)
    print(f"  Input shape: {dummy_waveform.shape}")
    print(f"  Output shape: {embedding.shape}")
    
    # Verify output shape
    assert embedding.shape == (1, 2048), f"Expected (1, 2048), got {embedding.shape}"
    print("\n✓ Encoder test passed!")
    
    return encoder


if __name__ == "__main__":
    test_encoder()
