"""
TrumpetJudge Demo App

Upload or record a trumpet performance and get instant AI feedback on:
- Overall Quality
- Intonation (pitch accuracy)
- Tone Quality
- Timing/Rhythm
- Technique

Usage:
    python demo/run_demo.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
import numpy as np
import soundfile as sf
import tempfile

from models.encoder_panns import PANNsEncoder
from models.head_regressor import RegressionHead, SCORE_NAMES, unscale_scores


# Global model instances (loaded once)
encoder = None
head = None
device = None


def load_models():
    """Load encoder and regression head models."""
    global encoder, head, device
    
    if encoder is not None:
        return True, "Models already loaded"
    
    try:
        # Find latest checkpoint
        checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
        runs = sorted(checkpoints_dir.glob("run_*"))
        
        if not runs:
            return False, "No trained model found. Please train a model first with: python ml/train.py"
        
        latest_run = runs[-1]
        checkpoint_path = latest_run / "best_model.pt"
        
        if not checkpoint_path.exists():
            return False, f"No best_model.pt found in {latest_run}"
        
        # Load encoder
        encoder = PANNsEncoder(duration=20.0, device=None)
        device = encoder.device
        
        # Load regression head
        head = RegressionHead(embedding_dim=encoder.embedding_dim)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        head.load_state_dict(checkpoint["head_state_dict"])
        head = head.to(device)
        head.eval()
        
        return True, f"Models loaded from {latest_run.name}"
    
    except Exception as e:
        return False, f"Error loading models: {str(e)}"


def process_audio(audio_input):
    """
    Process uploaded or recorded audio and return scores.
    
    Args:
        audio_input: Tuple of (sample_rate, audio_data) from Gradio
        
    Returns:
        Tuple of (scores_html, overall_score, intonation, tone, timing, technique)
    """
    if audio_input is None:
        return (
            "<div style='text-align: center; color: #888; padding: 40px;'>🎺 Upload or record audio to get started</div>",
            None, None, None, None, None, ""
        )
    
    # Load models if not already loaded
    success, message = load_models()
    if not success:
        return (
            f"<div style='text-align: center; color: #e74c3c; padding: 40px;'>❌ {message}</div>",
            None, None, None, None, None, ""
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
                prediction = head(embedding)
                scores = unscale_scores(prediction).squeeze(0).cpu().numpy()
                all_scores.append(scores)
        
        if not all_scores:
            return (
                "<div style='text-align: center; color: #e67e22; padding: 40px;'>⚠️ Audio too short. Please provide at least 5 seconds of audio.</div>",
                None, None, None, None, None, ""
            )
        
        # Average scores across chunks
        avg_scores = np.mean(all_scores, axis=0)
        
        # Build results
        overall = float(avg_scores[0])
        intonation = float(avg_scores[1])
        tone = float(avg_scores[2])
        timing = float(avg_scores[3])
        technique = float(avg_scores[4])
        
        # Create visual HTML display
        html = create_score_display(overall, intonation, tone, timing, technique, len(all_scores))
        
        # Status message
        duration_sec = len(audio_data) / 32000
        status = f"✓ Analyzed {duration_sec:.1f}s of audio ({len(all_scores)} chunk{'s' if len(all_scores) > 1 else ''})"
        
        return html, overall, intonation, tone, timing, technique, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<div style='text-align: center; color: #e74c3c; padding: 40px;'>❌ Error processing audio: {str(e)}</div>",
            None, None, None, None, None, ""
        )


def create_score_display(overall, intonation, tone, timing, technique, num_chunks):
    """Create a beautiful HTML display for the scores."""
    
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
    
    def score_bar(name, score, description):
        color = get_color(score)
        width = (score / 5.0) * 100
        return f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #ecf0f1;">{name}</span>
                <span style="font-weight: 700; color: {color}; font-size: 1.2em;">{score:.2f}</span>
            </div>
            <div style="background: #34495e; border-radius: 10px; height: 16px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {color}, {color}dd); width: {width}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
            </div>
            <div style="font-size: 0.8em; color: #95a5a6; margin-top: 4px;">{description}</div>
        </div>
        """
    
    overall_color = get_color(overall)
    overall_grade = get_grade(overall)
    
    html = f"""
    <div style="font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 30px; border-radius: 20px; color: #ecf0f1;">
        
        <!-- Overall Score Hero -->
        <div style="text-align: center; margin-bottom: 35px; padding: 30px; background: rgba(255,255,255,0.05); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 2px; color: #95a5a6; margin-bottom: 10px;">Overall Score</div>
            <div style="font-size: 4em; font-weight: 800; color: {overall_color}; line-height: 1;">{overall:.2f}</div>
            <div style="font-size: 1.5em; font-weight: 600; color: {overall_color}; margin-top: 5px;">Grade: {overall_grade}</div>
            <div style="font-size: 0.85em; color: #7f8c8d; margin-top: 10px;">out of 5.0</div>
        </div>
        
        <!-- Individual Scores -->
        <div style="background: rgba(0,0,0,0.2); padding: 25px; border-radius: 16px;">
            <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 20px; color: #bdc3c7; text-transform: uppercase; letter-spacing: 1px;">Detailed Breakdown</div>
            
            {score_bar("🎯 Intonation", intonation, "Pitch accuracy and tuning")}
            {score_bar("🎵 Tone Quality", tone, "Warmth, clarity, and richness of sound")}
            {score_bar("⏱️ Timing", timing, "Rhythmic accuracy and steadiness")}
            {score_bar("🎼 Technique", technique, "Articulation, dynamics, and control")}
        </div>
        
        <!-- Tips -->
        <div style="margin-top: 25px; padding: 20px; background: rgba(52, 152, 219, 0.1); border-radius: 12px; border-left: 4px solid #3498db;">
            <div style="font-weight: 600; color: #3498db; margin-bottom: 8px;">💡 Tips for Improvement</div>
            <div style="font-size: 0.9em; color: #95a5a6; line-height: 1.6;">
                {get_tips(intonation, tone, timing, technique)}
            </div>
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
            status_text
        ]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🎺 TrumpetJudge Demo")
    print("=" * 60)
    
    # Pre-load models
    print("\nLoading models...")
    success, message = load_models()
    print(f"  {message}")
    
    if success:
        print("\n✓ Ready! Launching app...")
        app.launch(share=True)
    else:
        print(f"\n⚠️ {message}")
        print("Launching app anyway (will show error to users)...")
        app.launch(share=True)

