# 🎺 TrumpetJudge

AI-powered feedback for trumpet performances. Upload or record a trumpet clip and get instant scores on 5 dimensions of playing quality.

## Features

- **Instant AI Feedback** - Get scores in seconds, not days
- **5 Scoring Dimensions**:
  - 🎯 **Intonation** - Pitch accuracy and tuning
  - 🎵 **Tone Quality** - Warmth, clarity, and richness
  - ⏱️ **Timing** - Rhythmic accuracy and steadiness
  - 🎼 **Technique** - Articulation, dynamics, and control
  - ⭐ **Overall** - General performance quality
- **Web Interface** - Upload files or record directly from your browser
- **Personalized Tips** - Get improvement suggestions based on your weakest areas

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Demo App

```bash
python demo/run_demo.py
```

This launches a web interface where you can upload or record trumpet audio and get instant feedback.

## Project Structure

```
TrumpetJudge/
├── demo/
│   └── run_demo.py        # 🎺 Main demo app - upload/record and get scores
├── ml/
│   ├── train.py           # Train the regression model
│   ├── predict.py         # Run inference on audio files
│   ├── dataset.py         # PyTorch dataset for trumpet audio
│   ├── prepare_data.py    # Prepare train/val splits from labels
│   └── eval.py            # Model evaluation
├── models/
│   ├── encoder_panns.py   # PANNs CNN14 audio encoder (frozen)
│   └── head_regressor.py  # Trainable regression head
├── label/
│   └── app.py             # Gradio UI for human labelers
├── dsp/
│   ├── tuning_analysis.py # Pitch/intonation analysis
│   ├── rhythm_analysis.py # Timing analysis
│   ├── sheet_music.py     # Generate sheet music from audio
│   └── plots.py           # Visualization utilities
├── data/
│   ├── audio/             # Audio chunks for training
│   ├── labels/            # Human labels (per-labeler CSVs)
│   ├── prepared/          # Prepared train/val splits
│   └── check_audio/       # Test audio files
├── checkpoints/           # Saved model weights
└── requirements.txt
```

## How It Works

### Architecture

```
Audio (WAV) → PANNs CNN14 Encoder → 2048-dim embedding → Regression Head → 5 scores
              (frozen, pretrained)                      (trained on labels)
```

1. **PANNs Encoder**: Pretrained audio neural network (CNN14) extracts rich audio features. This is frozen during training.
2. **Regression Head**: Small MLP trained on human-labeled trumpet performances to predict 5 quality scores.

### Training Pipeline

1. **Collect Data**: Download trumpet performances from YouTube
2. **Chunk Audio**: Split into 20-second clips
3. **Label**: Human raters score each clip (1-5) on 5 dimensions
4. **Train**: Fine-tune regression head on labeled data
5. **Evaluate**: Check MAE on held-out validation set

## Usage

### Training a Model

```bash
# Prepare data from labels
python ml/prepare_data.py --labels data/labels/labels_yourname.csv

# Train the model
python ml/train.py --train_csv data/prepared/train.csv --val_csv data/prepared/val.csv
```

### Running Inference

```bash
# CLI inference
python ml/predict.py --audio path/to/trumpet.wav

# Or use the web demo
python demo/run_demo.py
```

### Labeling Data

```bash
# Launch the labeling UI
python label/app.py
```

This opens a web interface where labelers can:
- Listen to audio clips
- Rate on 5 dimensions (1-5 scale)
- Reject invalid clips
- Track progress

## Scoring Guide

| Score | Description |
|-------|-------------|
| 5 | Excellent - Professional quality |
| 4 | Good - Minor issues, mostly solid |
| 3 | Average - Noticeable problems but acceptable |
| 2 | Below Average - Significant issues |
| 1 | Poor - Major problems throughout |

### Dimension Definitions

- **Overall**: Your gut feeling about the performance quality
- **Intonation**: Is it in tune? Are intervals accurate?
- **Tone**: Does it sound good? Warm, clear, resonant?
- **Timing**: Is the rhythm steady? Are rhythms accurate?
- **Technique**: Clean articulation? Good dynamics? Control?

## Requirements

- Python 3.8+
- PyTorch 2.0+
- ~500MB disk space for PANNs model weights (downloaded automatically)
- GPU recommended but not required

## Development

### Adding More Training Data

1. Add YouTube URLs to `data/youtube_urls.csv`
2. Download and chunk audio (creates entries in `data/to_label.csv`)
3. Run labeling app and rate clips
4. Prepare data and retrain

### DSP Analysis (Experimental)

```bash
# Analyze tuning/pitch of an audio file
python dsp/tuning_analysis.py
```

Generates pitch contour plots, note-by-note tuning analysis, and sheet music transcription.

## License

MIT

## Acknowledgments

- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) - Pretrained audio neural networks
- [Gradio](https://gradio.app) - Web UI framework
- [librosa](https://librosa.org) - Audio analysis
