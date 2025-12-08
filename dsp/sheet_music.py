from music21 import stream, note, meter, tempo, duration
from pathlib import Path


# Valid quarter note lengths that MusicXML can express
VALID_DURATIONS = [
    4.0,    # whole
    3.0,    # dotted half
    2.0,    # half
    1.5,    # dotted quarter
    1.0,    # quarter
    0.75,   # dotted eighth
    0.5,    # eighth
    0.375,  # dotted sixteenth
    0.25,   # sixteenth
    0.125,  # thirty-second
]


def quantize_duration(quarter_length):
    """Snap a duration to the nearest valid musical duration."""
    # Clamp to reasonable range
    quarter_length = max(0.125, min(quarter_length, 4.0))
    
    # Find the closest valid duration
    closest = min(VALID_DURATIONS, key=lambda x: abs(x - quarter_length))
    return closest


def create_sheet_music(note_data, output_dir=None, filename="transcription"):
    """
    Create sheet music from analyzed note data.
    
    Args:
        note_data: List of dicts with keys:
            - midi_target: MIDI pitch number
            - start: Start time in seconds
            - end: End time in seconds
            - note_name: Note name (e.g., "C4")
        output_dir: Directory to save files (defaults to data/sheet_music/)
        filename: Base filename without extension
    
    Returns:
        Path to the saved MusicXML file
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "sheet_music"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a Stream
    s = stream.Stream()
    
    # Add tempo marking (quarter = 60, so 1 beat = 1 second for easy duration mapping)
    s.append(tempo.MetronomeMark(number=60))
    
    # Add time signature
    s.append(meter.TimeSignature('4/4'))
    
    # Add notes as quarter notes (ignoring actual timing)
    for nd in note_data:
        n = note.Note(nd['midi_target'], quarterLength=1.0)
        s.append(n)
    
    # Save to MusicXML (opens in MuseScore)
    xml_path = output_dir / f"{filename}.musicxml"
    s.write('musicxml', fp=str(xml_path))
    
    # Also save MIDI for playback comparison
    midi_path = output_dir / f"{filename}.mid"
    s.write('midi', fp=str(midi_path))
    
    return xml_path, midi_path


if __name__ == "__main__":
    # Example usage with a C major scale
    example_notes = [
        {"midi_target": 60, "start": 0.0, "end": 1.0, "note_name": "C4"},
        {"midi_target": 62, "start": 1.0, "end": 2.0, "note_name": "D4"},
        {"midi_target": 64, "start": 2.0, "end": 3.0, "note_name": "E4"},
        {"midi_target": 65, "start": 3.0, "end": 4.0, "note_name": "F4"},
        {"midi_target": 67, "start": 4.0, "end": 5.0, "note_name": "G4"},
        {"midi_target": 69, "start": 5.0, "end": 6.0, "note_name": "A4"},
        {"midi_target": 71, "start": 6.0, "end": 7.0, "note_name": "B4"},
        {"midi_target": 72, "start": 7.0, "end": 8.0, "note_name": "C5"},
    ]
    
    xml_path, midi_path = create_sheet_music(example_notes, filename="example_scale")
    print(f"Saved: {xml_path}")
    print(f"Saved: {midi_path}")
