"""
TrumpetJudge Demo App

Upload or record a trumpet performance and get instant AI feedback on:
- Overall Quality
- Intonation (pitch accuracy)
- Tone Quality
- Timing/Rhythm
- Technique

Usage:
    python demo/run_demo.py [--run RUN_NAME] [--share]
    
    Examples:
        python demo/run_demo.py --run run_20251218_052428
        python demo/run_demo.py --run run_20251218_052428 --share
        TRUMPETJUDGE_RUN=run_20251218_052428 python demo/run_demo.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
import numpy as np
import soundfile as sf
import tempfile
import librosa
import matplotlib.pyplot as plt  # still available if needed elsewhere
import plotly.graph_objects as go
import time
import random

from ml.encoder_panns import PANNsEncoder
from ml.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores, GatingHead


# Global model instances (loaded once)
encoder = None
head = None  # Can be RegressionHead or EnsembleModel
gating_head = None  # Optional gating head (valid / rejected)
device = None
selected_run = None  # Store the selected run/ensemble name
is_ensemble = False  # Track if we're using an ensemble
GATING_THRESHOLD = 0.5


def load_models(run_name=None, ensemble_path=None, gating_path=None, device_override=None):
    """Load encoder and regression head models.
    
    Args:
        run_name: Name of the run to load (e.g., 'run_20251218_052428'). 
                  If None, uses the latest run or the one specified via environment variable.
        ensemble_path: Path to ensemble checkpoint file. If provided, loads ensemble instead of single model.
        gating_path: Path to gating checkpoint (e.g., 'checkpoints_gating/best_gating.pt')
        device_override: Force a specific device (e.g., 'cpu')
    """
    global encoder, head, gating_head, device, selected_run, is_ensemble
    
    if encoder is not None:
        return True, f"Models already loaded from {selected_run}"
    
    try:
        # Determine device
        if device_override:
            device = device_override
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load encoder
        encoder = PANNsEncoder(duration=20.0, device=device)
        device = encoder.device
        
        # Load model (ensemble or single)
        if ensemble_path:
            # Load ensemble model
            from ml.ensemble import EnsembleModel
            
            ensemble_file = Path(ensemble_path)
            if not ensemble_file.exists():
                return False, f"Ensemble file not found: {ensemble_path}"
            
            head = EnsembleModel.load(str(ensemble_file), device=device)
            is_ensemble = True
            selected_run = f"ensemble ({ensemble_file.name})"
            model_msg = f"Ensemble loaded from {ensemble_path} ({head.num_models} models)"
        else:
            # Load single model from checkpoint directory
            checkpoints_dir = Path(__file__).parent.parent / "models" / "checkpoints"
            
            # Determine which run to use
            if run_name is None:
                # Check environment variable
                run_name = os.environ.get("TRUMPETJUDGE_RUN", None)
            
            if run_name:
                # Use specified run
                run_path = checkpoints_dir / run_name
                if not run_path.exists():
                    return False, f"Run '{run_name}' not found in checkpoints directory"
            else:
                # Find latest checkpoint
                runs = sorted(checkpoints_dir.glob("run_*"))
                if not runs:
                    return False, "No trained model found. Please train a model first with: python ml/train.py"
                run_path = runs[-1]
                run_name = run_path.name
            
            checkpoint_path = run_path / "best_model.pt"
            
            if not checkpoint_path.exists():
                return False, f"No best_model.pt found in {run_path}"
            
            # Load regression head
            head = RegressionHead(embedding_dim=encoder.embedding_dim)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            head.load_state_dict(checkpoint["head_state_dict"])
            head = head.to(device)
            head.eval()
            
            is_ensemble = False
            selected_run = run_name
            model_msg = f"Models loaded from {run_name}"
        
        # Optionally load gating head
        if gating_path:
            gating_file = Path(gating_path)
        else:
            gating_file = Path(__file__).parent.parent / "models" / "checkpoints" / "gating" / "best_gating.pt"
        
        if gating_file.exists():
            try:
                ckpt = torch.load(gating_file, map_location=device, weights_only=False)
                gating = GatingHead(embedding_dim=encoder.embedding_dim)
                gating.load_state_dict(ckpt["state_dict"])
                gating = gating.to(device)
                gating.eval()
                gating_head = gating
                model_msg += f" | Gating head loaded from {gating_file}"
            except Exception as ge:
                model_msg += f" | Warning: failed to load gating head from {gating_file} ({ge})"
        else:
            model_msg += f" | No gating head found at {gating_file} (skipping)"
        
        return True, model_msg
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error loading models: {str(e)}"


def process_audio(audio_input):
    """
    Process uploaded or recorded audio and return scores.
    
    Args:
        audio_input: Tuple of (sample_rate, audio_data) from Gradio
        
    Returns:
        Tuple of (scores_html, overall_score, intonation, tone, timing, technique)
    """
    start_time = time.time()
    
    if audio_input is None:
        return (
            "<div style='text-align: center; color: #888; padding: 40px;'>🎺 Upload or record audio to get started</div>",
            None, None, None, None, None, None, ""
        )
    
    # Load models if not already loaded
    success, message = load_models()
    if not success:
        return (
            f"<div style='text-align: center; color: #e74c3c; padding: 40px;'>❌ {message}</div>",
            None, None, None, None, None, None, ""
        )
    
    try:
        sample_rate, audio_data = audio_input
        
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 32kHz if needed
        if sample_rate != 32000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        # Process in 20-second chunks
        chunk_duration = 20.0
        chunk_samples = int(chunk_duration * 32000)
        
        all_scores = []
        all_uncertainties = []  # For ensemble
        gate_probs = []
        
        # Split into chunks
        for start in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[start:start + chunk_samples]
            
            # Skip if chunk is too short (less than 5 seconds)
            if len(chunk) < 5 * 32000:
                continue
            
            # Pad if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            # Run inference
            with torch.no_grad():
                waveform = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                embedding = encoder(waveform)
                
                # Gating prediction (valid / rejected) if available
                if gating_head is not None:
                    gate_p = gating_head(embedding).squeeze(-1).mean().item()
                    gate_probs.append(gate_p)
                
                if is_ensemble:
                    # Ensemble: get predictions with uncertainty
                    mean_pred, std_pred = head.predict_with_uncertainty(embedding)
                    scores = mean_pred.squeeze(0).cpu().numpy()
                    uncertainty = std_pred.squeeze(0).cpu().numpy()
                    all_uncertainties.append(uncertainty)
                else:
                    # Single model
                    prediction = head(embedding)
                    scores = unscale_scores(prediction).squeeze(0).cpu().numpy()
                
                all_scores.append(scores)
        
        if not all_scores:
            return (
                "<div style='text-align: center; color: #e67e22; padding: 40px;'>⚠️ Audio too short. Please provide at least 5 seconds of audio.</div>",
                None, None, None, None, None, None, ""
            )
        
        # Aggregate gating decision across chunks (if available)
        gate_prob = None
        gate_valid = None
        if gate_probs:
            gate_prob = float(np.mean(gate_probs))
            gate_valid = gate_prob >= GATING_THRESHOLD
        
        # Average scores across chunks
        avg_scores = np.mean(all_scores, axis=0)
        avg_uncertainty = np.mean(all_uncertainties, axis=0) if all_uncertainties else None
        
        # Build results
        overall = float(avg_scores[0])
        intonation = float(avg_scores[1])
        tone = float(avg_scores[2])
        timing = float(avg_scores[3])
        technique = float(avg_scores[4])
        
        # Create visual HTML display
        if gate_valid is False:
            reason = generate_rejection_reason()
            html = create_rejection_display(gate_prob, reason)
            # For rejected clips, hide numeric scores
            num_overall = num_inton = num_tone = num_timing = num_technique = None
        else:
            html = create_score_display(
                overall, intonation, tone, timing, technique, len(all_scores),
                uncertainty=avg_uncertainty
            )
            num_overall = overall
            num_inton = intonation
            num_tone = tone
            num_timing = timing
            num_technique = technique
        
        # Compute pitch contour plot (using full processed audio), with colors per note
        pitch_fig = None
        try:
            f0_hz, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=32000,
            )
            times = librosa.times_like(f0_hz, sr=32000)

            # Simple note segmentation based on pitch changes
            # Convert to MIDI for easier thresholds
            f0_midi = librosa.hz_to_midi(f0_hz)
            valid = ~np.isnan(f0_midi)

            note_segments = []
            current_indices = []
            NOTE_CHANGE_THRESHOLD = 0.5  # semitones

            for i, (m, is_valid) in enumerate(zip(f0_midi, valid)):
                if is_valid:
                    if not current_indices:
                        current_indices = [i]
                    else:
                        last_m = f0_midi[current_indices[-1]]
                        if np.isnan(last_m) or abs(m - last_m) <= NOTE_CHANGE_THRESHOLD:
                            current_indices.append(i)
                        else:
                            # Close current note if it has enough frames
                            if len(current_indices) >= 3:
                                note_segments.append(current_indices)
                            current_indices = [i]
                else:
                    if current_indices:
                        if len(current_indices) >= 3:
                            note_segments.append(current_indices)
                        current_indices = []

            # Close any remaining segment
            if current_indices and len(current_indices) >= 3:
                note_segments.append(current_indices)

            # Build Plotly figure for better Gradio integration
            fig = go.Figure()

            if note_segments:
                n_notes = len(note_segments)
                for idx, seg in enumerate(note_segments):
                    seg_times = times[seg]
                    seg_f0 = f0_midi[seg]
                    # rainbow-style color across notes
                    color = f"hsl({int(360 * idx / max(1, n_notes))}, 80%, 55%)"
                    fig.add_trace(
                        go.Scatter(
                            x=seg_times,
                            y=seg_f0,
                            mode="lines",
                            line=dict(width=3, color=color),
                            showlegend=False,
                        )
                    )
            else:
                # Fallback: single-color contour
                valid_idx = ~np.isnan(f0_midi)
                fig.add_trace(
                    go.Scatter(
                        x=times[valid_idx],
                        y=f0_midi[valid_idx],
                        mode="lines",
                        line=dict(width=2, color="#3498db"),
                        showlegend=False,
                    )
                )

            # Build Bb-major (concert) scale ticks, labeled as trumpet C-major (C=concert Bb)
            valid_midi = f0_midi[~np.isnan(f0_midi)]
            if valid_midi.size > 0:
                midi_min = int(np.floor(valid_midi.min()))
                midi_max = int(np.ceil(valid_midi.max()))
                bb_scale_pc = {10, 0, 2, 3, 5, 7, 9}  # Bb, C, D, Eb, F, G, A (concert)
                tick_vals = []
                tick_text = []
                for m in range(midi_min, midi_max + 1):
                    if (m % 12) in bb_scale_pc:
                        tick_vals.append(m)
                        # Label as written (transposed up a whole step)
                        tick_text.append(librosa.midi_to_note(m + 2))
                # Add a moving tracer as a vertical line in MIDI space
                y_min = midi_min - 1
                y_max = midi_max + 1

                # Initial cursor at t=0
                cursor_index = len(fig.data)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 0],
                        y=[y_min, y_max],
                        mode="lines",
                        line=dict(width=2, color="#ffffff"),
                        showlegend=False,
                        name="cursor",
                    )
                )

                # Build frames for simple left-to-right sweep
                total_time = float(times[-1]) if len(times) > 0 else 0.0
                n_frames = 60 if total_time > 0 else 1
                frame_times = np.linspace(0.0, total_time, n_frames)

                frames = []
                for ft in frame_times:
                    frames.append(
                        go.Frame(
                            data=[go.Scatter(x=[ft, ft], y=[y_min, y_max])],
                            traces=[cursor_index],
                            name=f"t={ft:.2f}s",
                        )
                    )
                fig.frames = frames
            else:
                tick_vals = None
                tick_text = None

            fig.update_layout(
                margin=dict(l=60, r=20, t=40, b=50),
                xaxis_title="Time (s)",
                yaxis_title="Trumpet pitch (C = concert Bb)",
                yaxis=dict(
                    tickmode="array" if tick_vals is not None else "auto",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                ),
                title="Pitch over time (Bb-major grid, trumpet labels)",
                template="plotly_dark",
                height=600,
            )

            pitch_fig = fig
        except Exception:
            pitch_fig = None
        
        # Status message
        duration_sec = len(audio_data) / 32000
        model_type = "ensemble" if is_ensemble else "model"
        gating_str = ""
        if gate_prob is not None:
            gating_str = f", gate={gate_prob:.2f} ({'valid' if gate_valid else 'rejected'})"
        elapsed = time.time() - start_time
        status = (
            f"✓ Analyzed {duration_sec:.1f}s of audio "
            f"in {elapsed:.2f}s "
            f"({len(all_scores)} chunk{'s' if len(all_scores) > 1 else ''}, {model_type}{gating_str})"
        )
        
        return html, num_overall, num_inton, num_tone, num_timing, num_technique, pitch_fig, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<div style='text-align: center; color: #e74c3c; padding: 40px;'>❌ Error processing audio: {str(e)}</div>",
            None, None, None, None, None, None, ""
        )


## NOTE: Previously had helper functions for a 'Play with tracer' feature here.
## That functionality (and its controls on the plot) has been removed for a simpler UI.

def create_score_display(overall, intonation, tone, timing, technique, num_chunks, uncertainty=None):
    """Create a beautiful HTML display for the scores.
    
    Args:
        overall, intonation, tone, timing, technique: mean scores
        num_chunks: how many audio chunks were analyzed
        uncertainty: optional array of std-devs per score (overall, intonation, tone, timing, technique)
    """
    
    def get_color(score):
        """Get color based on score (1-5)."""
        if score >= 4.0:
            return "#2ecc71"  # Green
        elif score >= 3.0:
            return "#f1c40f"  # Yellow
        elif score >= 2.0:
            return "#e67e22"  # Orange
        else:
            return "#e74c3c"  # Red
    
    def get_grade(score):
        """Get letter grade based on score."""
        if score >= 4.5:
            return "A+"
        elif score >= 4.0:
            return "A"
        elif score >= 3.5:
            return "B+"
        elif score >= 3.0:
            return "B"
        elif score >= 2.5:
            return "C+"
        elif score >= 2.0:
            return "C"
        elif score >= 1.5:
            return "D"
        else:
            return "F"
    
    def score_bar(name, score, description, std=None):
        color = get_color(score)
        width = (score / 5.0) * 100
        unc_html = ""
        if std is not None:
            unc_html = f"<span style='font-size: 0.75em; color: #7f8c8d; margin-left: 8px;'>±{std:.2f}</span>"
        return f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #ecf0f1;">{name}</span>
                <span style="font-weight: 700; color: {color}; font-size: 1.2em;">
                    {score:.2f}{unc_html}
                </span>
            </div>
            <div style="background: #34495e; border-radius: 10px; height: 16px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {color}, {color}dd); width: {width}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
            </div>
            <div style="font-size: 0.8em; color: #95a5a6; margin-top: 4px;">{description}</div>
        </div>
        """
    
    overall_color = get_color(overall)
    overall_grade = get_grade(overall)
    
    # Unpack uncertainty if provided (std for each score)
    overall_std = intonation_std = tone_std = timing_std = technique_std = None
    if uncertainty is not None and len(uncertainty) >= 5:
        overall_std = float(uncertainty[0])
        intonation_std = float(uncertainty[1])
        tone_std = float(uncertainty[2])
        timing_std = float(uncertainty[3])
        technique_std = float(uncertainty[4])
    
    html = f"""
    <div style="font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 30px; border-radius: 20px; color: #ecf0f1;">
        
        <!-- Overall Score Hero -->
        <div style="text-align: center; margin-bottom: 35px; padding: 30px; background: rgba(255,255,255,0.05); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 2px; color: #95a5a6; margin-bottom: 10px;">Overall Score</div>
            <div style="font-size: 4em; font-weight: 800; color: {overall_color}; line-height: 1;">
                {overall:.2f}{f' ±{overall_std:.2f}' if overall_std is not None else ''}
            </div>
            <div style="font-size: 1.5em; font-weight: 600; color: {overall_color}; margin-top: 5px;">Grade: {overall_grade}</div>
            <div style="font-size: 0.85em; color: #7f8c8d; margin-top: 10px;">out of 5.0 (ensemble average)</div>
        </div>
        
        <!-- Individual Scores -->
        <div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 16px;">
            <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 20px; color: #bdc3c7; text-transform: uppercase; letter-spacing: 1px;">
                Detailed Breakdown{f' · Uncertainty shown as ±1σ' if uncertainty is not None else ''}
            </div>
            
            {score_bar("Intonation", intonation, "Pitch accuracy and tuning", intonation_std)}
            {score_bar("Tone Quality", tone, "Warmth, clarity, and richness of sound", tone_std)}
            {score_bar("Timing", timing, "Rhythmic accuracy and steadiness", timing_std)}
            {score_bar("Technique", technique, "Articulation, dynamics, and control", technique_std)}
        </div>
    </div>
    """
    
    return html


def get_tips(intonation, tone, timing, technique):
    """Generate personalized tips based on lowest scores."""
    tips = []
    
    scores = [
        ("intonation", intonation, "Practice with a tuner and focus on long tones to improve pitch accuracy."),
        ("tone", tone, "Work on breath support and embouchure. Try buzzing exercises on the mouthpiece."),
        ("timing", timing, "Practice with a metronome regularly. Start slow and gradually increase tempo."),
        ("technique", technique, "Focus on articulation exercises and scales. Pay attention to dynamics."),
    ]
    
    # Sort by score (lowest first)
    scores.sort(key=lambda x: x[1])
    
    # Get tips for the two lowest scores
    for name, score, tip in scores[:2]:
        if score < 4.0:
            tips.append(f"• <strong>{name.title()}</strong>: {tip}")
    
    if not tips:
        tips.append("• Great job! Keep up the consistent practice to maintain your skills.")
    
    return "<br>".join(tips)


def generate_rejection_reason() -> str:
    """Return an example rejection reason to show when gating flags invalid audio."""
    reasons = [
        "The recording does not sound like trumpet playing (e.g., speech or other sounds).",
        "Less than about half of the clip contains actual trumpet playing, so it is hard to judge.",
        "The audio quality is too poor (noise, clipping, or very low level) to reliably score the performance.",
    ]
    return random.choice(reasons)


def create_rejection_display(gate_prob: float, reason: str) -> str:
    """HTML block shown when the gating model flags the clip as rejected."""
    prob_pct = gate_prob * 100.0 if gate_prob is not None else 0.0
    return f"""
    <div style="font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #2c3e50 0%, #8e44ad 100%); padding: 30px; border-radius: 20px; color: #ecf0f1;">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 2px; color: #bdc3c7; margin-bottom: 8px;">
                Gating Decision
            </div>
            <div style="font-size: 2.2em; font-weight: 800; color: #e74c3c;">
                ❌ Performance Rejected
            </div>
            <div style="font-size: 0.95em; color: #ecf0f1; margin-top: 8px;">
                The gating model estimates only {prob_pct:.1f}% probability that this audio is a valid trumpet performance.
            </div>
        </div>
        <div style="margin-top: 20px; padding: 16px; background: rgba(0,0,0,0.25); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 1em; font-weight: 600; margin-bottom: 6px; color: #f1c40f;">
                Example reason:
            </div>
            <div style="font-size: 0.95em; color: #ecf0f1;">
                {reason}
            </div>
            <div style="font-size: 0.8em; color: #bdc3c7; margin-top: 10px;">
                If this actually was trumpet playing, try re-recording with at least 50% of the clip as clear playing,
                good mic placement, and minimal background noise.
            </div>
        </div>
    </div>
    """


# Build the Gradio app
with gr.Blocks() as app:
    
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, rgba(155, 89, 182, 0.15) 0%, rgba(52, 152, 219, 0.15) 100%); border-radius: 20px; margin-bottom: 20px;">
        <h1 style="font-size: 2.5em; font-weight: 800; background: linear-gradient(135deg, #9b59b6, #3498db); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 10px 0;">🎺 TrumpetJudge</h1>
        <p style="color: #666; font-size: 1.1em; margin: 0;">AI-powered feedback for trumpet performances</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Your Performance")
            audio_input = gr.Audio(
                label="Upload or Record",
                sources=["upload", "microphone"],
                type="numpy"
            )
            
            analyze_btn = gr.Button(
                "🎯 Analyze Performance",
                variant="primary",
                size="lg"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=False,
                container=False
            )
            
            gr.Markdown("""
            ---
            **How it works:**
            1. Upload a recording or use your microphone
            2. Click "Analyze Performance"
            3. Get instant AI feedback on 5 dimensions
            
            **Best results:** Use clear recordings, 10-60 seconds, minimal background noise.
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Your Scores")
            scores_display = gr.HTML(
                value="<div style='text-align: center; color: #888; padding: 80px 40px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);'>🎺 Upload or record audio to get your scores</div>"
            )
            
            pitch_plot = gr.Plot(label="Pitch over time (Hz)")
            
            # Hidden number outputs for potential API use
            with gr.Row(visible=False):
                overall_score = gr.Number(label="Overall")
                intonation_score = gr.Number(label="Intonation")
                tone_score = gr.Number(label="Tone")
                timing_score = gr.Number(label="Timing")
                technique_score = gr.Number(label="Technique")
    
    # Event handlers
    analyze_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[
            scores_display,
            overall_score,
            intonation_score,
            tone_score,
            timing_score,
            technique_score,
            pitch_plot,
            status_text
        ]
    )

    # Also analyze on audio upload
    audio_input.change(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[
            scores_display,
            overall_score,
            intonation_score,
            tone_score,
            timing_score,
            technique_score,
            pitch_plot,
            status_text
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrumpetJudge Demo App")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Name of the run to use (e.g., 'run_20251218_052428'). If not specified, uses latest run or TRUMPETJUDGE_RUN env var."
    )
    parser.add_argument(
        "    --ensemble",
        type=str,
        default=None,
        help="Path to an ensemble checkpoint (e.g., 'models/weights/best_ensemble.pt'). If provided, overrides --run."
    )
    parser.add_argument(
        "--gating",
        type=str,
        default=None,
        help="Path to gating checkpoint (e.g., 'models/checkpoints/gating/best_gating.pt'). If not provided, tries default path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (e.g., 'cpu' or 'cuda'). Defaults to auto-detect."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎺 TrumpetJudge Demo")
    print("=" * 60)
    
    # Pre-load models
    print("\nLoading models...")
    success, message = load_models(
        run_name=args.run,
        ensemble_path=args.ensemble,
        gating_path=args.gating,
        device_override=args.device,
    )
    print(f"  {message}")
    
    if success:
        print("\n✓ Ready! Launching app...")
        app.launch(share=args.share)
    else:
        print(f"\n⚠️ {message}")
        print("Launching app anyway (will show error to users)...")
        app.launch(share=args.share)

