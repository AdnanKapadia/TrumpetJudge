import numpy as np
import librosa
from scipy.ndimage import median_filter


def segment_notes_from_pitch(times, f0_midi, voiced_flag, smooth=True):
    """
    Segments notes using pitch stability and rate of change.
    Handles slurred notes where there is no strict onset or gap of silence.
    
    Parameters
    ----------
    times : np.ndarray
        Time (seconds) of each f0 frame.
    f0_midi : np.ndarray
        MIDI note numbers for each frame (nan where unvoiced).
    voiced_flag : np.ndarray
        Boolean array indicating voiced frames.
    smooth : bool
        Whether to apply median filtering to smooth pitch before segmentation.
        
    Returns
    -------
    segments : list of tuple
        List of (start_t, end_t, median_midi) tuples for each detected note segment.
    """
    valid = voiced_flag & np.isfinite(f0_midi)
    f = f0_midi.copy()
    f[~valid] = np.nan
    
    if smooth:
        f = median_filter(f, size=5)
    
    df = np.abs(np.diff(f))
    dt = np.diff(times)
    
    # Avoid division by zero
    rate = np.zeros_like(df)
    nonzero_dt = dt > 0
    rate[nonzero_dt] = df[nonzero_dt] / dt[nonzero_dt]
    
    # Threshold: e.g. sustained > 0.5 semitone difference over ≥40 ms
    # Detect jumps: large pitch change (>1 semitone) with high rate of change (>10 semitones/sec)
    jump_idx = np.where((df > 1.0) & (rate > 10))[0]
    
    # Add boundaries at start and end
    boundaries = np.concatenate(([0], jump_idx + 1, [len(f)]))
    boundaries = np.unique(boundaries)  # Remove duplicates
    
    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Handle edge case
        if start_idx >= len(times) or end_idx > len(times):
            continue
            
        start = times[start_idx]
        # End time is the last frame in the segment (end_idx is exclusive)
        end_idx_time = min(end_idx - 1, len(times) - 1)
        end = times[end_idx_time]
        
        # Get pitch values for this segment
        seg_pitch = f[start_idx:end_idx]
        seg_pitch = seg_pitch[np.isfinite(seg_pitch)]
        
        if len(seg_pitch) < 3:
            continue
        
        median_midi = np.median(seg_pitch)
        segments.append((start, end, median_midi))
    
    return segments


def analyze_tuning(
    wav_path: str,
    sr: int = 22050,
    fmin: str = "C3",
    fmax: str = "C6",
    min_note_duration: float = 0.08,  # seconds
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Analyze tuning of a (monophonic) trumpet recording.

    Parameters
    ----------
    wav_path : str
        Path to audio file.
    sr : int
        Target sample rate for analysis.
    fmin, fmax : str
        Pitch range for pyin, in note name strings.
    min_note_duration : float
        Minimum length (in seconds) for a segment to be treated as a note.

    Returns
    -------
    note_stats : list of dict
        One dict per detected note with tuning statistics.
        Keys:
          - note_index (int)
          - start_time, end_time (float, seconds)
          - note_midi (int)
          - note_name (str)
          - mean_cents_error (float)
          - std_cents_error (float)
          - max_abs_cents_error (float)
          - num_frames (int)
          - voiced_fraction (float)
    times : np.ndarray
        Time (seconds) of each f0 frame.
    cents_error : np.ndarray
        Frame-wise cents error (nan where unvoiced).
    """
    # 1. Load and trim audio
    y, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)

    # 2. Estimate f0 with pyin
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr,
    )
    times = librosa.times_like(f0, sr=sr)

    # 3. Convert to MIDI and cents error for voiced frames
    midi = np.full_like(f0, np.nan, dtype=float)
    cents_error = np.full_like(f0, np.nan, dtype=float)

    # Only for voiced frames
    voiced_idx = voiced_flag == True
    midi[voiced_idx] = 69 + 12 * np.log2(f0[voiced_idx] / 440.0)

    nearest_midi = np.round(midi)
    # cents = 100 * (actual_midi - nearest_integer_midi)
    cents_error[voiced_idx] = 100.0 * (midi[voiced_idx] - nearest_midi[voiced_idx])

    # 4. Segment notes using pitch-based segmentation (handles slurs better than onset detection)
    pitch_segments = segment_notes_from_pitch(times, midi, voiced_idx, smooth=True)
    
    # Fallback to onset detection if pitch segmentation finds too few segments
    if len(pitch_segments) < 2:
        # Use onset detection as fallback
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames", backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Ensure first onset at time 0 and last boundary at end
        if len(onset_times) == 0 or onset_times[0] > 0.05:
            onset_times = np.concatenate(([0.0], onset_times))
        if onset_times[-1] < times[-1] - min_note_duration:
            onset_times = np.concatenate((onset_times, [times[-1]]))
        
        # Convert onset times to segments format
        pitch_segments = [(start_t, end_t, np.nan) 
                         for start_t, end_t in zip(onset_times[:-1], onset_times[1:])]

    note_stats = []
    note_index = 0

    # 5. Build note segments from pitch-based segmentation
    for start_t, end_t, seg_median_midi in pitch_segments:
        # Skip segments that are too short
        if end_t - start_t < min_note_duration:
            continue

        # Frames in this segment
        seg_mask = (times >= start_t) & (times < end_t)
        seg_voiced = seg_mask & voiced_idx

        if np.sum(seg_mask) == 0:
            continue

        if np.sum(seg_voiced) < 3:
            # Very few voiced frames; likely noise or rest
            continue

        # MIDI in this segment (only voiced frames)
        seg_midi = midi[seg_voiced]
        seg_cents = cents_error[seg_voiced]

        # Representative pitch: use median from segmentation if available, otherwise compute
        if np.isfinite(seg_median_midi):
            rep_midi = float(seg_median_midi)
        else:
            rep_midi = float(np.nanmedian(seg_midi))
        nearest_rep_midi = int(np.round(rep_midi))
        note_name = librosa.midi_to_note(nearest_rep_midi)

        mean_cents = float(np.nanmean(seg_cents))
        std_cents = float(np.nanstd(seg_cents))
        max_abs_cents = float(np.nanmax(np.abs(seg_cents)))

        num_frames = int(np.sum(seg_mask))
        voiced_fraction = float(np.sum(seg_voiced) / num_frames)

        note_stats.append(
            {
                "note_index": note_index,
                "start_time": float(start_t),
                "end_time": float(end_t),
                "note_midi": nearest_rep_midi,
                "note_name": note_name,
                "mean_cents_error": mean_cents,
                "std_cents_error": std_cents,
                "max_abs_cents_error": max_abs_cents,
                "num_frames": num_frames,
                "voiced_fraction": voiced_fraction,
            }
        )
        note_index += 1

    return note_stats, times, cents_error
