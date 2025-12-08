"""
Trumpet Performance Labeling UI

A simple Gradio interface for labeling trumpet audio clips.
Labelers rate clips on 5 dimensions: overall, intonation, tone, timing, technique.
Each clip can also be rejected if it's not suitable for labeling.
"""

import gradio as gr
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TO_LABEL_CSV = DATA_DIR / "to_label.csv"
LABELS_DIR = DATA_DIR / "labels"

# Ensure labels directory exists
LABELS_DIR.mkdir(exist_ok=True)

# Score categories
SCORE_CATEGORIES = ["overall", "intonation", "tone", "timing", "technique"]


class LabelingSession:
    """Manages a labeling session for a user."""
    
    def __init__(self):
        self.user_id = None
        self.output_file = None
        self.remaining_ids = []
        self.current_id = None
        self.samples_df = None
        self.labeled_count = 0
        self.total_count = 0
        
    def start_session(self, user_id: str) -> tuple:
        """Initialize a new labeling session."""
        if not user_id or not user_id.strip():
            return None, "Please enter a valid User ID"
        
        self.user_id = user_id.strip()
        self.output_file = LABELS_DIR / f"labels_{self.user_id}.csv"
        
        # Load samples to label
        self.samples_df = pd.read_csv(TO_LABEL_CSV)
        all_ids = set(self.samples_df["id"].tolist())
        self.total_count = len(all_ids)
        
        # Check what's already labeled by this user
        already_labeled = set()
        if self.output_file.exists():
            existing_df = pd.read_csv(self.output_file)
            already_labeled = set(existing_df["sample_id"].tolist())
            self.labeled_count = len(already_labeled)
        
        # Get remaining samples (randomized)
        self.remaining_ids = list(all_ids - already_labeled)
        random.shuffle(self.remaining_ids)
        
        remaining = len(self.remaining_ids)
        
        if remaining == 0:
            return None, f"🎉 All {self.total_count} samples already labeled! Great work!"
        
        return True, f"Welcome {self.user_id}! {remaining}/{self.total_count} samples remaining."
    
    def get_next_sample(self) -> tuple:
        """Get the next sample to label."""
        if not self.remaining_ids:
            return None, None, f"Done! {self.labeled_count}/{self.total_count}"
        
        self.current_id = self.remaining_ids[0]
        audio_path = AUDIO_DIR / f"{self.current_id}.wav"
        
        if not audio_path.exists():
            # Skip missing files
            self.remaining_ids.pop(0)
            return self.get_next_sample()
        
        progress = f"{self.labeled_count}/{self.total_count}"
        return str(audio_path), self.current_id, progress
    
    def save_label(self, overall, intonation, tone, timing, technique, rejected=False) -> str:
        """Save a label to the CSV file."""
        if self.current_id is None:
            return "No sample loaded!"
        
        # Create label record
        record = {
            "sample_id": self.current_id,
            "user_id": self.user_id,
            "overall": None if rejected else overall,
            "intonation": None if rejected else intonation,
            "tone": None if rejected else tone,
            "timing": None if rejected else timing,
            "technique": None if rejected else technique,
            "rejected": rejected,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to CSV
        record_df = pd.DataFrame([record])
        if self.output_file.exists():
            record_df.to_csv(self.output_file, mode='a', header=False, index=False)
        else:
            record_df.to_csv(self.output_file, index=False)
        
        # Update state
        self.remaining_ids.pop(0)
        self.labeled_count += 1
        self.current_id = None
        
        action = "rejected" if rejected else "saved"
        return f"✓ {action}"


# Global session
session = LabelingSession()


def get_initial_button_styles():
    """Return all buttons as secondary (unselected)."""
    return [gr.update(variant="secondary") for _ in range(25)]  # 5 categories * 5 buttons


def get_button_styles_for_scores(scores):
    """Return button styles based on current scores."""
    styles = []
    for cat in SCORE_CATEGORIES:
        selected = scores.get(cat)
        for i in range(1, 6):
            if selected == i:
                styles.append(gr.update(variant="primary"))
            else:
                styles.append(gr.update(variant="secondary"))
    return styles


# Build the UI
with gr.Blocks(title="🎺 Trumpet Judge - Labeling Tool") as app:
    
    # State for current scores
    scores_state = gr.State({cat: None for cat in SCORE_CATEGORIES})
    
    gr.Markdown("# 🎺 Trumpet Judge")
    gr.Markdown("*Rate trumpet performances on 5 dimensions*")
    
    status_display = gr.Textbox(
        label="Status", 
        interactive=False
    )
    
    # Login section
    with gr.Group(visible=True) as login_section:
        gr.Markdown("### Enter your User ID to begin")
        with gr.Row():
            user_id_input = gr.Textbox(
                label="User ID",
                placeholder="e.g., john_doe",
                scale=3
            )
            start_btn = gr.Button("Start Labeling", variant="primary", scale=1)
    
    # Labeling section
    with gr.Group(visible=False) as labeling_section:
        with gr.Row():
            with gr.Column(scale=2):
                sample_id_display = gr.Textbox(
                    label="Sample ID", 
                    interactive=False
                )
                audio_player = gr.Audio(
                    label="Listen to the performance",
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )
                progress_display = gr.Textbox(
                    label="Progress",
                    interactive=False
                )
            
            with gr.Column(scale=3):
                gr.Markdown("### Rate the performance (1-5)")
                
                # Score display
                score_display = gr.Markdown("Overall: - | Intonation: - | Tone: - | Timing: - | Technique: -")
                
                # Store all buttons in a flat list for easy updating
                all_buttons = []
                
                # Overall
                gr.Markdown("**Overall Quality** - General impression")
                with gr.Row():
                    for i in range(1, 6):
                        btn = gr.Button(str(i), scale=1, min_width=50, variant="secondary")
                        all_buttons.append(btn)
                
                # Intonation
                gr.Markdown("**Intonation** - Pitch accuracy and tuning")
                with gr.Row():
                    for i in range(1, 6):
                        btn = gr.Button(str(i), scale=1, min_width=50, variant="secondary")
                        all_buttons.append(btn)
                
                # Tone
                gr.Markdown("**Tone Quality** - Warmth, clarity, richness")
                with gr.Row():
                    for i in range(1, 6):
                        btn = gr.Button(str(i), scale=1, min_width=50, variant="secondary")
                        all_buttons.append(btn)
                
                # Timing
                gr.Markdown("**Timing/Rhythm** - Rhythmic accuracy")
                with gr.Row():
                    for i in range(1, 6):
                        btn = gr.Button(str(i), scale=1, min_width=50, variant="secondary")
                        all_buttons.append(btn)
                
                # Technique
                gr.Markdown("**Technique** - Articulation, dynamics, control")
                with gr.Row():
                    for i in range(1, 6):
                        btn = gr.Button(str(i), scale=1, min_width=50, variant="secondary")
                        all_buttons.append(btn)
                
                gr.Markdown("---")
                
                with gr.Row():
                    reject_btn = gr.Button(
                        "❌ Reject (Invalid)", 
                        variant="secondary",
                        scale=1
                    )
                    submit_btn = gr.Button(
                        "✓ Submit & Next", 
                        variant="primary",
                        scale=2
                    )
    
    # Helper to format score display
    def format_score_display(scores):
        def fmt(v):
            return str(v) if v is not None else "-"
        return f"Overall: {fmt(scores.get('overall'))} | Intonation: {fmt(scores.get('intonation'))} | Tone: {fmt(scores.get('tone'))} | Timing: {fmt(scores.get('timing'))} | Technique: {fmt(scores.get('technique'))}"
    
    # Create score handler for each button
    def make_score_handler(category_idx, value):
        category = SCORE_CATEGORIES[category_idx]
        def handler(scores):
            scores = dict(scores)
            scores[category] = value
            # Return: new scores, display text, then 25 button updates
            button_styles = get_button_styles_for_scores(scores)
            return [scores, format_score_display(scores)] + button_styles
        return handler
    
    # Wire up all score buttons
    for idx, btn in enumerate(all_buttons):
        category_idx = idx // 5  # 0-4 for each category
        value = (idx % 5) + 1    # 1-5
        btn.click(
            fn=make_score_handler(category_idx, value),
            inputs=[scores_state],
            outputs=[scores_state, score_display] + all_buttons
        )
    
    # Start labeling handler
    def start_labeling(user_id):
        success, message = session.start_session(user_id)
        
        if not success:
            return [
                gr.update(visible=True),   # login still visible
                gr.update(visible=False),  # labeling hidden
                message,                   # status
                None,                      # audio
                "",                        # sample_id
                ""                         # progress
            ]
        
        audio_path, sample_id, progress = session.get_next_sample()
        
        if audio_path is None:
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                "No samples available",
                None,
                "",
                ""
            ]
        
        return [
            gr.update(visible=False),  # hide login
            gr.update(visible=True),   # show labeling
            message,                   # status
            audio_path,                # audio
            sample_id,                 # sample_id
            progress                   # progress
        ]
    
    # Submit handler
    def submit_scores(scores):
        # Validate all scores provided
        if any(scores.get(cat) is None for cat in SCORE_CATEGORIES):
            missing = [cat for cat in SCORE_CATEGORIES if scores.get(cat) is None]
            # Return without changing anything except status
            return [
                gr.update(),  # audio
                gr.update(),  # sample_id
                gr.update(),  # progress
                f"⚠️ Rate: {', '.join(missing)}",
                scores,       # scores unchanged
                gr.update()   # score_display unchanged
            ] + [gr.update() for _ in range(25)]  # buttons unchanged
        
        result = session.save_label(
            scores["overall"],
            scores["intonation"],
            scores["tone"],
            scores["timing"],
            scores["technique"],
            rejected=False
        )
        
        # Get next sample
        audio_path, sample_id, progress = session.get_next_sample()
        
        # Reset scores
        new_scores = {cat: None for cat in SCORE_CATEGORIES}
        
        if audio_path is None:
            return [
                None,
                "🎉 All done!",
                progress,
                result,
                new_scores,
                format_score_display(new_scores)
            ] + get_initial_button_styles()
        
        return [
            audio_path,
            sample_id,
            progress,
            result,
            new_scores,
            format_score_display(new_scores)
        ] + get_initial_button_styles()
    
    # Reject handler
    def reject_sample(scores):
        result = session.save_label(None, None, None, None, None, rejected=True)
        
        # Get next sample
        audio_path, sample_id, progress = session.get_next_sample()
        
        # Reset scores
        new_scores = {cat: None for cat in SCORE_CATEGORIES}
        
        if audio_path is None:
            return [
                None,
                "🎉 All done!",
                progress,
                result,
                new_scores,
                format_score_display(new_scores)
            ] + get_initial_button_styles()
        
        return [
            audio_path,
            sample_id,
            progress,
            result,
            new_scores,
            format_score_display(new_scores)
        ] + get_initial_button_styles()
    
    # Event handlers
    start_btn.click(
        fn=start_labeling,
        inputs=[user_id_input],
        outputs=[
            login_section,
            labeling_section,
            status_display,
            audio_player,
            sample_id_display,
            progress_display
        ]
    )
    
    user_id_input.submit(
        fn=start_labeling,
        inputs=[user_id_input],
        outputs=[
            login_section,
            labeling_section,
            status_display,
            audio_player,
            sample_id_display,
            progress_display
        ]
    )
    
    submit_btn.click(
        fn=submit_scores,
        inputs=[scores_state],
        outputs=[
            audio_player,
            sample_id_display,
            progress_display,
            status_display,
            scores_state,
            score_display
        ] + all_buttons
    )
    
    reject_btn.click(
        fn=reject_sample,
        inputs=[scores_state],
        outputs=[
            audio_player,
            sample_id_display,
            progress_display,
            status_display,
            scores_state,
            score_display
        ] + all_buttons
    )


if __name__ == "__main__":
    app.launch(share=True)  # Creates a public URL for remote labelers
