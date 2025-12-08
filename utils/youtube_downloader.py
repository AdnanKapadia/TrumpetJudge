"""
YouTube to WAV Downloader for TrumpetJudge

Downloads trumpet audition videos from YouTube, extracts a random chunk,
and prepares them for labeling.

Usage:
    # Single video
    python utils/youtube_downloader.py --url "https://youtube.com/watch?v=..."
    
    # Multiple videos from a text file (one URL per line)
    python utils/youtube_downloader.py --url_file urls.txt
    
    # With custom chunk duration
    python utils/youtube_downloader.py --url_file urls.txt --chunk_duration 20

Output:
    - WAV files (random chunks) in data/audio/
    - CSV template for labeling in data/to_label.csv
"""

import os
import sys
import argparse
import subprocess
import csv
import random
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Set

import torch
import soundfile as sf
import numpy as np


def url_to_id(url: str) -> str:
    """
    Generate a unique 8-character ID from a YouTube URL.
    Same URL always produces the same ID.
    """
    # Normalize URL (extract video ID if possible)
    url = url.strip()
    
    # Create hash and take first 8 chars
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return url_hash


def load_existing_urls(csv_path: str) -> Set[str]:
    """
    Load existing URLs from a CSV file to check for duplicates.
    
    Returns:
        Set of URLs already in the CSV
    """
    existing_urls = set()
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'url' in row and row['url']:
                        existing_urls.add(row['url'].strip())
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    
    return existing_urls


def load_existing_ids(csv_path: str) -> Set[str]:
    """
    Load existing IDs from a CSV file.
    
    Returns:
        Set of IDs already in the CSV
    """
    existing_ids = set()
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'id' in row and row['id']:
                        existing_ids.add(row['id'].strip())
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    
    return existing_ids


def check_yt_dlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_all_chunks(
    input_path: str,
    output_dir: str,
    base_id: str,
    chunk_duration: float = 20.0,
    target_sr: int = 32000,
) -> List[Tuple[str, str, float, float]]:
    """
    Extract ALL non-overlapping chunks from an audio file.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save chunks
        base_id: Base ID for naming (e.g., "a1b2c3d4")
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sample rate (32000 for PANNs)
        
    Returns:
        List of (chunk_id, output_path, start_time, end_time) tuples
    """
    results = []
    
    try:
        # Load audio using soundfile
        audio_data, sr = sf.read(input_path)
        
        # Convert to torch tensor and ensure correct shape (channels, samples)
        if audio_data.ndim == 1:
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        else:
            # Multi-channel: (samples, channels) -> (channels, samples)
            waveform = torch.from_numpy(audio_data.T).float()
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed (simple linear interpolation)
        if sr != target_sr:
            num_samples = waveform.shape[1]
            new_num_samples = int(num_samples * target_sr / sr)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_num_samples, mode='linear', align_corners=False
            ).squeeze(0)
            sr = target_sr
        
        # Calculate samples
        total_samples = waveform.shape[1]
        chunk_samples = int(chunk_duration * sr)
        total_duration = total_samples / sr
        
        # Calculate number of full chunks we can get
        num_chunks = total_samples // chunk_samples
        
        if num_chunks == 0:
            # Audio shorter than chunk duration - pad and use as single chunk
            padding = chunk_samples - total_samples
            padded = torch.nn.functional.pad(waveform, (0, padding))
            
            chunk_id = f"{base_id}_0"
            output_path = os.path.join(output_dir, f"{chunk_id}.wav")
            sf.write(output_path, padded.squeeze(0).numpy(), sr)
            
            results.append((chunk_id, output_path, 0.0, total_duration))
        else:
            # Extract each chunk
            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = start_sample + chunk_samples
                
                chunk_waveform = waveform[:, start_sample:end_sample]
                
                chunk_id = f"{base_id}_{i}"
                output_path = os.path.join(output_dir, f"{chunk_id}.wav")
                sf.write(output_path, chunk_waveform.squeeze(0).numpy(), sr)
                
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                results.append((chunk_id, output_path, start_time, end_time))
        
        return results
        
    except Exception as e:
        print(f"  Error extracting chunks: {e}")
        return []


def download_youtube_audio(
    url: str,
    output_dir: str,
    base_id: str,
    chunk_duration: float = 20.0,
) -> List[Tuple[str, str, float, float]]:
    """
    Download audio from YouTube video, extract ALL chunks as WAV files.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the WAV files
        base_id: Base ID for naming chunks (e.g., "a1b2c3d4")
        chunk_duration: Duration of each chunk in seconds
    
    Returns:
        List of (chunk_id, path, start_time, end_time) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Download to temporary file first
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_template = os.path.join(tmpdir, "temp.%(ext)s")
        
        try:
            # Run yt-dlp to download full audio
            result = subprocess.run([
                "yt-dlp",
                "-x",                          # Extract audio only
                "--audio-format", "wav",       # Convert to WAV
                "--audio-quality", "0",        # Best quality
                "-o", temp_template,           # Output path template
                "--no-playlist",               # Don't download playlists
                url
            ], capture_output=True, text=True, check=True)
            
            # Find the downloaded file
            temp_wav = None
            for f in os.listdir(tmpdir):
                if f.endswith('.wav'):
                    temp_wav = os.path.join(tmpdir, f)
                    break
            
            if not temp_wav or not os.path.exists(temp_wav):
                print(f"  Error: Could not find downloaded file")
                return []
            
            # Get duration for logging
            info = sf.info(temp_wav)
            total_duration = info.duration
            num_chunks = int(total_duration // chunk_duration)
            if num_chunks == 0:
                num_chunks = 1
            print(f"  Video duration: {total_duration:.1f}s → {num_chunks} chunks")
            
            # Extract all chunks
            chunks = extract_all_chunks(
                temp_wav, output_dir, base_id, chunk_duration
            )
            
            return chunks
            
        except subprocess.CalledProcessError as e:
            print(f"  Error downloading {url}: {e.stderr}")
            return []


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:100]  # Limit length


def download_batch(
    urls: List[str],
    output_dir: str = "data/audio",
    csv_output: str = "data/to_label.csv",
    chunk_duration: float = 20.0,
) -> List[dict]:
    """
    Download multiple YouTube videos, extract ALL chunks, and create CSV for labeling.
    
    Each video is split into multiple non-overlapping chunks.
    Skips URLs that already exist in the CSV. Uses URL hash + chunk index for unique IDs.
    
    Args:
        urls: List of YouTube URLs
        output_dir: Directory to save WAV files
        csv_output: Path for the labeling CSV template
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of dicts with download results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing data to check for duplicates
    existing_urls = load_existing_urls(csv_output)
    existing_ids = load_existing_ids(csv_output)
    
    results = []
    successful = []
    skipped = 0
    total_chunks = 0
    
    print(f"\nProcessing {len(urls)} URLs...")
    print(f"Extracting ALL {chunk_duration}s chunks from each video")
    if existing_urls:
        print(f"Found {len(existing_urls)} existing URLs in {csv_output}")
    print("=" * 60)
    
    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or url.startswith('#'):
            continue
        
        # Generate base ID from URL
        base_id = url_to_id(url)
        
        # Check for duplicate URL
        if url in existing_urls:
            print(f"\n[{i}/{len(urls)}] SKIP (already processed): {url[:50]}...")
            skipped += 1
            continue
            
        print(f"\n[{i}/{len(urls)}] {url[:60]}...")
        print(f"  Base ID: {base_id}")
        
        # Download and extract ALL chunks
        chunks = download_youtube_audio(
            url, output_dir, base_id=base_id, chunk_duration=chunk_duration
        )
        
        if chunks:
            for chunk_id, output_path, start_time, end_time in chunks:
                # Skip if this specific chunk ID already exists
                if chunk_id in existing_ids:
                    print(f"  SKIP chunk {chunk_id} (already exists)")
                    continue
                
                # Get relative path for CSV
                rel_path = os.path.relpath(output_path, start=os.path.dirname(csv_output))
                
                result = {
                    "id": chunk_id,
                    "video_id": base_id,
                    "path": rel_path,
                    "url": url,
                    "chunk_start": f"{start_time:.1f}",
                    "chunk_end": f"{end_time:.1f}",
                    "overall": "",
                    "intonation": "",
                    "tone": "",
                    "timing": "",
                    "technique": "",
                }
                results.append(result)
                successful.append(result)
                print(f"  ✓ Chunk {chunk_id}: {start_time:.1f}s - {end_time:.1f}s")
            
            total_chunks += len(chunks)
        else:
            print(f"  ✗ Failed to download")
            results.append({
                "video_id": base_id,
                "url": url,
                "error": "Download failed",
            })
    
    # Write/append to CSV for labeling
    if successful:
        print(f"\n" + "=" * 60)
        
        fieldnames = [
            "id", "video_id", "path", "overall", "intonation", "tone", "timing", "technique", 
            "chunk_start", "chunk_end", "url"
        ]
        
        # Check if file exists and has content
        file_exists = os.path.exists(csv_output) and os.path.getsize(csv_output) > 0
        
        if file_exists:
            # Append to existing file
            print(f"Appending to existing CSV: {csv_output}")
            with open(csv_output, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(successful)
            print(f"  ✓ Added {len(successful)} new chunks")
        else:
            # Create new file with header
            print(f"Creating new CSV: {csv_output}")
            with open(csv_output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(successful)
            print(f"  ✓ Created CSV with {len(successful)} chunks")
        
        print(f"\nNext steps:")
        print(f"  1. Open {csv_output}")
        print(f"  2. Listen to each {chunk_duration}s audio chunk")
        print(f"  3. Fill in scores (1-5) for each category")
        print(f"  4. Copy rows to data/train.csv and data/val.csv")
        print(f"  ⚠️  Keep all chunks from same video_id in the SAME split!")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Videos processed: {len(urls) - skipped}")
    print(f"  Total chunks created: {len(successful)}")
    print(f"  Videos skipped (duplicates): {skipped}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio for TrumpetJudge training data"
    )
    parser.add_argument("--url", type=str, help="Single YouTube URL to download")
    parser.add_argument("--url_file", type=str, help="Text file with URLs (one per line)")
    parser.add_argument("--output_dir", type=str, default="data/audio",
                        help="Output directory for WAV files")
    parser.add_argument("--csv_output", type=str, default="data/to_label.csv",
                        help="Output CSV file for labeling")
    parser.add_argument("--chunk_duration", type=float, default=20.0,
                        help="Duration of random chunk to extract (seconds)")
    
    args = parser.parse_args()
    
    # Check yt-dlp installation
    if not check_yt_dlp():
        print("Error: yt-dlp is not installed.")
        print("Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Collect URLs
    urls = []
    
    if args.url:
        urls.append(args.url)
    
    if args.url_file:
        with open(args.url_file, 'r') as f:
            urls.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
    
    if not urls:
        print("Error: No URLs provided.")
        print("Use --url for a single URL or --url_file for multiple URLs")
        sys.exit(1)
    
    # Download
    download_batch(urls, args.output_dir, args.csv_output, args.chunk_duration)


if __name__ == "__main__":
    main()

