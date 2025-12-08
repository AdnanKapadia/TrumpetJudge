import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter
from sheet_music import create_sheet_music

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_DIR / "data" / "audio" / "flex.wav"
SAVE_PATH = PROJECT_DIR / "data" / "plots" / "pitch_contour.png"
SAVE_PATH_MIDI = PROJECT_DIR / "data" / "plots" / "pitch_contour_midi.png"
SAVE_PATH_SCATTER = PROJECT_DIR / "data" / "plots" / "midi_vs_note_scatter.png"
SAVE_PATH_LOG = PROJECT_DIR / "data" / "plots" / "tuning_analysis_log.txt"
SHEET_MUSIC_DIR = PROJECT_DIR / "data" / "sheet_music"

print("=" * 50)
print("TUNING ANALYSIS")
print("=" * 50)

# --- Load audio ---
print(f"\n[1/6] Loading audio: {AUDIO_PATH.name}...")
y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
duration = len(y) / sr
print(f"       Loaded {duration:.2f}s of audio at {sr} Hz")

# --- Estimate pitch (Hz) using pYIN ---
print(f"\n[2/6] Estimating pitch with pYIN (this may take a while)...")
f0_hz, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7")
)
print(f"       Done! Extracted {len(f0_hz)} frames")

# --- Convert frame indices to time (seconds) ---
times = librosa.times_like(f0_hz, sr=sr)

# --- Replace unvoiced frames with NaN for clarity ---
# f0[~voiced_flag] = np.nan

print(f"\n[3/6] Processing pitch data...")
# Convert to MIDI (continuous) – NaNs stay NaN
f0_midi = librosa.hz_to_midi(f0_hz)

# Light median filter to kill frame-to-frame jitter
# (this doesn't erase real sliding/vibrato, just 1-frame noise)
f0_midi_smooth = f0_midi.copy()
nan_mask = np.isnan(f0_midi_smooth)
f0_midi_smooth[nan_mask] = 0
f0_midi_smooth = median_filter(f0_midi_smooth, size=3)
f0_midi_smooth[nan_mask] = np.nan

# Simple energy-based mask to remove very quiet frames:
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
times_rms = librosa.times_like(rms, sr=sr, hop_length=512)

# Align RMS to f0 (assuming same hop_length=512 in pyin)
valid = (~np.isnan(f0_midi_smooth)) & (rms > 0.01)   # tweak threshold
voiced_frames = np.sum(valid)
print(f"       {voiced_frames} voiced frames detected ({100*voiced_frames/len(valid):.1f}%)")

print(f"\n[4/6] Segmenting notes...")
notes = []
current_start = None
current_midi_vals = []

# Parameters for note segmentation
MIN_FRAMES_FOR_MEDIAN = 5
RECENT_WINDOW = 20  # Compare against recent frames, not entire note
NOTE_CHANGE_THRESHOLD = 0.35  # Semitones - lower threshold to catch gradual transitions

for i, (t, m, is_valid) in enumerate(zip(times, f0_midi_smooth, valid)):
    if is_valid:
        if current_start is None:
            current_start = t
            current_midi_vals = [m]
        else:
            # Use recent window for comparison (handles gradual transitions better)
            # Compare against median of recent frames, not entire note
            if len(current_midi_vals) >= MIN_FRAMES_FOR_MEDIAN:
                # Use recent window (last N frames) or all frames if note is shorter
                recent_vals = current_midi_vals[-min(RECENT_WINDOW, len(current_midi_vals)):]
                recent_median = np.nanmedian(recent_vals)
                
                # Detect new note if pitch is significantly different from recent median
                # Lower threshold (0.35 semitones) to catch gradual transitions
                if abs(m - recent_median) > NOTE_CHANGE_THRESHOLD:
                    # Check if change is sustained (not just a brief spike)
                    lookahead_frames = min(3, len(times) - i - 1)
                    if lookahead_frames > 0:
                        future_indices = range(i, min(i + lookahead_frames, len(f0_midi_smooth)))
                        future_vals = [f0_midi_smooth[j] for j in future_indices if not np.isnan(f0_midi_smooth[j])]
                        if len(future_vals) > 0:
                            future_median = np.nanmedian(future_vals)
                            # If future frames also differ from recent median, it's a real change
                            if abs(future_median - recent_median) > NOTE_CHANGE_THRESHOLD:
                                # Close previous note
                                end_time = times[i-1]
                                notes.append({
                                    "start": current_start,
                                    "end": end_time,
                                    "midi_series": np.array(current_midi_vals),
                                    "time_series": times[(times >= current_start) & (times <= end_time)],
                                })
                                # Start new note
                                current_start = t
                                current_midi_vals = [m]
                            else:
                                current_midi_vals.append(m)
                        else:
                            current_midi_vals.append(m)
                    else:
                        # Can't look ahead, use simple threshold
                        end_time = times[i-1]
                        notes.append({
                            "start": current_start,
                            "end": end_time,
                            "midi_series": np.array(current_midi_vals),
                            "time_series": times[(times >= current_start) & (times <= end_time)],
                        })
                        current_start = t
                        current_midi_vals = [m]
                else:
                    current_midi_vals.append(m)
            else:
                # Not enough frames yet, just compare to last value
                if len(current_midi_vals) > 0 and abs(m - current_midi_vals[-1]) > 0.5:
                    # Big jump even early on, probably a new note
                    end_time = times[i-1]
                    notes.append({
                        "start": current_start,
                        "end": end_time,
                        "midi_series": np.array(current_midi_vals),
                        "time_series": times[(times >= current_start) & (times <= end_time)],
                    })
                    current_start = t
                    current_midi_vals = [m]
                else:
                    current_midi_vals.append(m)
    else:
        # gap/silence: end current note if any
        if current_start is not None and current_midi_vals:
            end_time = times[i-1]
            notes.append({
                "start": current_start,
                "end": end_time,
                "midi_series": np.array(current_midi_vals),
                "time_series": times[(times >= current_start) & (times <= end_time)],
            })
        current_start = None
        current_midi_vals = []

# handle last note
if current_start is not None and current_midi_vals:
    notes.append({
        "start": current_start,
        "end": times[-1],
        "midi_series": np.array(current_midi_vals),
        "time_series": times[(times >= current_start) & (times <= times[-1])],
    })


def analyze_note(note):
    midi_series = note["midi_series"]
    n = len(midi_series)
    if n < 5:
        return None  # too short

    # Use the central 60% of frames
    i0 = int(0.2 * n)
    i1 = int(0.8 * n)
    core = midi_series[i0:i1]

    midi_median = np.nanmedian(core)
    midi_target = round(midi_median)
    
    # Check cents error - if very large, assign to the closer note
    # A -50 cent C3 is really a B2 played slightly sharp
    # A +50 cent B2 is really a C3 played slightly flat
    initial_cents = 100 * (midi_median - midi_target)
    if initial_cents > 40:
        # Very sharp - this is really the next note played flat
        midi_target = midi_target + 1
    elif initial_cents < -40:
        # Very flat - this is really the previous note played sharp
        midi_target = midi_target - 1
    
    note_name = librosa.midi_to_note(midi_target)

    # cents error over time (relative to corrected target)
    cents_series = 100 * (core - midi_target)

    mean_cents = float(np.nanmean(cents_series))
    std_cents = float(np.nanstd(cents_series))
    p2p_cents = float(np.nanmax(cents_series) - np.nanmin(cents_series))

    return {
        "note_name": note_name,
        "midi_target": midi_target,
        "mean_cents": mean_cents,    # how sharp/flat overall
        "std_cents": std_cents,      # steadiness
        "p2p_cents": p2p_cents,      # total wobble
    }

print(f"       Found {len(notes)} raw note segments")

print(f"\n[5/6] Analyzing notes...")
note_reports = []
note_indices = []
filtered_count = 0
for i, n in enumerate(notes):
    report = analyze_note(n)
    if report is not None:
        note_reports.append(report)
        note_indices.append(i)
    else:
        filtered_count += 1

print(f"       {len(note_reports)} notes analyzed ({filtered_count} filtered out as too short)")
print()
print("-" * 50)
print("NOTE ANALYSIS RESULTS")
print("-" * 50)
for i, r in enumerate(note_reports):
    print(
        f"  Note {i+1:2d}: {r['note_name']:4s}  "
        f"mean {r['mean_cents']:+6.1f}¢  "
        f"std {r['std_cents']:5.1f}¢  "
        f"p2p {r['p2p_cents']:5.1f}¢"
    )
print("-" * 50)


print(f"\n[6/6] Saving outputs...")

# Pitch contour (Hz) plot
print(f"       Saving Hz plot...")
plt.figure()
plt.plot(times, f0_hz)
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.title("Pitch contour")
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
plt.close()

# Pitch contour (MIDI) plot
print(f"       Saving MIDI plot...")
plt.figure()
plt.plot(times, f0_midi_smooth)
plt.xlabel("Time (s)")
plt.ylabel("Pitch (MIDI)")
plt.title("Pitch contour")
plt.savefig(SAVE_PATH_MIDI, dpi=150, bbox_inches='tight')
plt.close()

# Scatter plot: MIDI value vs note number
if note_reports:
    print(f"       Saving scatter plot...")
    note_numbers = [i + 1 for i in range(len(note_reports))]
    midi_values = [r['midi_target'] for r in note_reports]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(note_numbers, midi_values, alpha=0.6, s=100)
    plt.xlabel("Note #")
    plt.ylabel("MIDI Value")
    plt.title("MIDI Value vs Note Number")
    plt.grid(True, alpha=0.3)
    plt.xticks(note_numbers)
    plt.savefig(SAVE_PATH_SCATTER, dpi=150, bbox_inches='tight')
    plt.close()

# Save log file with all note analysis information
print(f"       Saving log file...")
with open(SAVE_PATH_LOG, 'w') as f:
    f.write("Tuning Analysis Log\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Audio file: {AUDIO_PATH}\n")
    f.write(f"Sample rate: {sr} Hz\n")
    f.write(f"Total notes detected: {len(note_reports)}\n\n")
    f.write("-" * 60 + "\n\n")
    
    for i, (r, note_idx) in enumerate(zip(note_reports, note_indices)):
        note = notes[note_idx]
        f.write(f"Note {i+1}:\n")
        f.write(f"  Note name: {r['note_name']}\n")
        f.write(f"  MIDI target: {r['midi_target']}\n")
        f.write(f"  Mean cents error: {r['mean_cents']:+.2f} cents\n")
        f.write(f"  Standard deviation: {r['std_cents']:.2f} cents\n")
        f.write(f"  Peak-to-peak variation: {r['p2p_cents']:.2f} cents\n")
        f.write(f"  Start time: {note['start']:.3f} s\n")
        f.write(f"  End time: {note['end']:.3f} s\n")
        f.write(f"  Duration: {note['end'] - note['start']:.3f} s\n")
        f.write("\n")
    
    f.write("=" * 60 + "\n")
    f.write("Summary Statistics\n")
    f.write("-" * 60 + "\n")
    if note_reports:
        mean_cents_list = [r['mean_cents'] for r in note_reports]
        std_cents_list = [r['std_cents'] for r in note_reports]
        p2p_cents_list = [r['p2p_cents'] for r in note_reports]
        
        f.write(f"Average mean cents error: {np.mean(mean_cents_list):+.2f} cents\n")
        f.write(f"Average std deviation: {np.mean(std_cents_list):.2f} cents\n")
        f.write(f"Average peak-to-peak: {np.mean(p2p_cents_list):.2f} cents\n")
        f.write(f"Max mean cents error: {np.max(np.abs(mean_cents_list)):.2f} cents\n")
        f.write(f"Max std deviation: {np.max(std_cents_list):.2f} cents\n")
        f.write(f"Max peak-to-peak: {np.max(p2p_cents_list):.2f} cents\n")

# Generate sheet music from detected notes
print(f"       Generating sheet music...")
if note_reports and note_indices:
    # Prepare note data for sheet music generation
    sheet_music_data = []
    for r, note_idx in zip(note_reports, note_indices):
        n = notes[note_idx]
        sheet_music_data.append({
            "midi_target": r['midi_target'],
            "start": n['start'],
            "end": n['end'],
            "note_name": r['note_name'],
        })
    
    # Generate the sheet music files
    xml_path, midi_path = create_sheet_music(
        sheet_music_data, 
        output_dir=SHEET_MUSIC_DIR,
        filename="transcription"
    )
    print(f"       Sheet music saved!")

print()
print("=" * 50)
print("COMPLETE!")
print("=" * 50)
print(f"  Hz plot:      {SAVE_PATH.name}")
print(f"  MIDI plot:    {SAVE_PATH_MIDI.name}")
print(f"  Scatter plot: {SAVE_PATH_SCATTER.name}")
print(f"  Log file:     {SAVE_PATH_LOG.name}")
if note_reports and note_indices:
    print(f"  Sheet music:  {xml_path.name}")
    print(f"  MIDI file:    {midi_path.name}")
print("=" * 50)

