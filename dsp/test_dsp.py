"""
Test script for DSP analysis functions.
Runs tuning analysis on an audio file and displays plots.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.tuning_analysis import analyze_tuning
from dsp.plots import plot_note_tuning


def plot_frame_wise_tuning(times, cents_error, note_stats=None, save_path=None):
    """
    Plot frame-wise cents error over time.
    
    Parameters
    ----------
    times : np.ndarray
        Time stamps for each frame
    cents_error : np.ndarray
        Frame-wise cents error (nan where unvoiced)
    note_stats : list of dict, optional
        Note statistics to overlay note boundaries
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot frame-wise cents error
    valid_mask = np.isfinite(cents_error)
    plt.plot(times[valid_mask], cents_error[valid_mask], alpha=0.6, linewidth=0.5, label='Frame-wise cents error')
    
    # Overlay note boundaries if provided
    if note_stats:
        for note in note_stats:
            start_t = note['start_time']
            end_t = note['end_time']
            mean_cents = note['mean_cents_error']
            note_name = note['note_name']
            
            # Draw vertical lines for note boundaries
            plt.axvline(start_t, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            plt.axvline(end_t, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            
            # Label note with mean cents error
            mid_time = (start_t + end_t) / 2
            plt.text(mid_time, mean_cents, note_name, 
                    ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    plt.axhline(0, color='k', linestyle='-', linewidth=1, label='In tune')
    plt.axhline(25, color='gray', linestyle=':', alpha=0.7, label='±25 cents')
    plt.axhline(-25, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cents Error (sharp + / flat −)')
    plt.title('Frame-wise Tuning Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test DSP analysis on an audio file')
    parser.add_argument('audio_file', type=str, help='Path to audio file (WAV)')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate (default: 22050)')
    parser.add_argument('--fmin', type=str, default='C3', help='Minimum pitch (default: C3)')
    parser.add_argument('--fmax', type=str, default='C6', help='Maximum pitch (default: C6)')
    parser.add_argument('--min-duration', type=float, default=0.08, 
                       help='Minimum note duration in seconds (default: 0.08)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as audio file)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    print(f"Analyzing: {args.audio_file}")
    print(f"Sample rate: {args.sr} Hz")
    print(f"Pitch range: {args.fmin} to {args.fmax}")
    print(f"Min note duration: {args.min_duration} seconds")
    print("\nRunning tuning analysis...")
    
    # Run analysis
    try:
        note_stats, times, cents_error = analyze_tuning(
            args.audio_file,
            sr=args.sr,
            fmin=args.fmin,
            fmax=args.fmax,
            min_note_duration=args.min_duration
        )
        
        print(f"\nAnalysis complete!")
        print(f"Detected {len(note_stats)} notes")
        
        # Print summary statistics
        if note_stats:
            mean_errors = [n['mean_cents_error'] for n in note_stats]
            std_errors = [n['std_cents_error'] for n in note_stats]
            
            print(f"\nTuning Statistics:")
            print(f"  Mean cents error: {np.mean(mean_errors):.2f} cents")
            print(f"  Std of mean errors: {np.std(mean_errors):.2f} cents")
            print(f"  Max absolute error: {np.max(np.abs(mean_errors)):.2f} cents")
            print(f"  Mean stability (std): {np.mean(std_errors):.2f} cents")
            
            # Show first few notes
            print(f"\nFirst 5 notes:")
            for note in note_stats[:5]:
                print(f"  {note['note_name']:4s} | "
                      f"Time: {note['start_time']:6.2f}-{note['end_time']:6.2f}s | "
                      f"Error: {note['mean_cents_error']:6.2f} cents | "
                      f"Stability: {note['std_cents_error']:5.2f} cents")
        
        # Generate plots
        print("\nGenerating plots...")
        
        # Determine output directory and filenames
        audio_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Save in same directory as audio file
            output_dir = os.path.dirname(os.path.abspath(args.audio_file))
        
        plot1_path = os.path.join(output_dir, f"{audio_basename}_note_tuning.png")
        plot2_path = os.path.join(output_dir, f"{audio_basename}_frame_wise_tuning.png")
        
        # Plot 1: Note-wise tuning (bar chart)
        plot_note_tuning(note_stats, save_path=plot1_path)
        
        # Plot 2: Frame-wise tuning over time
        plot_frame_wise_tuning(times, cents_error, note_stats, save_path=plot2_path)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

