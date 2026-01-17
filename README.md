# SPOT SoccerNet Ball Action Spotting

A complete pipeline for training, evaluating, and visualizing **E2E-SPOT** (End-to-End SPOTting) models on the **SoccerNet-Ball** dataset for temporal action spotting in soccer videos.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.11+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project provides:
- **Frame Extraction**: Extract frames from SoccerNet-Ball videos at configurable FPS
- **Annotation Parsing**: Parse SoccerNet-Ball labels into a unified format
- **Training**: End-to-end training with ResNet/RegNet backbones and GRU temporal heads
- **Inference**: Run predictions on single videos with visualization
- **Web Application**: Flask-based UI for uploading videos and viewing results

### Supported Action Classes (12 total)
- Ball Player Block
- Cross
- Drive
- Free Kick
- Goal
- Header
- High Pass
- Out
- Pass
- Player Successful Tackle
- Shot

## Project Structure

```
SPOT-SoccerNet-Ball/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── frames_as_jpg_soccernet_ball.py   # Frame extraction script
├── parse_soccernet_ball.py           # Label parsing script
├── train_e2e.py                      # Training script
├── eval_soccernet_ball.py            # Evaluation script
├── inference_spot.py                 # Inference & visualization script
├── dataset/                          # Dataset loading modules
│   ├── __init__.py
│   ├── frame.py                      # Frame dataset class
│   ├── feature.py                    # Feature dataset class
│   └── transform.py                  # Data augmentation transforms
├── model/                            # Model architecture modules
│   ├── __init__.py
│   ├── common.py                     # Base model classes
│   ├── modules.py                    # Temporal heads (GRU, TCN, ASFormer)
│   ├── shift.py                      # Temporal shift modules (TSM, GSM)
│   └── impl/                         # Implementation details
│       ├── asformer.py
│       ├── tsm.py
│       ├── gsm.py
│       └── ...
├── util/                             # Utility modules
│   ├── __init__.py
│   ├── dataset.py                    # Dataset utilities
│   ├── eval.py                       # Evaluation utilities
│   ├── io.py                         # I/O utilities
│   ├── score.py                      # Scoring utilities
│   └── video.py                      # Video utilities
├── app/                              # Flask web application
│   ├── app.py                        # Main Flask application
│   ├── templates/                    # HTML templates
│   │   ├── index.html               # Upload page
│   │   ├── processing.html          # Processing status page
│   │   └── result.html              # Results display page
│   ├── uploads/                      # Uploaded videos (created at runtime)
│   ├── frames/                       # Extracted frames (created at runtime)
│   └── outputs/                      # Output videos (created at runtime)
└── data/                             # Data directory (not included)
    └── soccernet_ball/               # Parsed labels (created by parse script)
        ├── class.txt
        ├── train.json
        ├── val.json
        ├── test.json
        └── challenge.json
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SPOT-SoccerNet-Ball.git
cd SPOT-SoccerNet-Ball
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SoccerNet-Ball Dataset

You need to download the SoccerNet-Ball dataset from the [SoccerNet website](https://www.soccer-net.org/).

```bash
# Using SoccerNet pip package
pip install SoccerNet

# Download videos and labels
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/soccernet")
mySoccerNetDownloader.downloadDataTask(task="spotting-ball", split=["train", "valid", "test", "challenge"])
```

## Quick Start

### Step 1: Extract Frames from Videos

Extract frames from downloaded videos at 25 FPS (or your preferred rate):

```bash
python frames_as_jpg_soccernet_ball.py /path/to/videos -o /path/to/output/frames --sample_fps 25 -j 4
```

**Arguments:**
- `video_dir`: Path to SoccerNet-Ball video directory
- `-o, --out_dir`: Output directory for extracted frames
- `--sample_fps`: Target frames per second (default: 2)
- `-j, --num_workers`: Number of parallel workers (default: CPU_count/4)

### Step 2: Parse Annotations

Parse the SoccerNet-Ball labels into the training format:

```bash
python parse_soccernet_ball.py /path/to/labels /path/to/frames -o data/soccernet_ball
```

**Arguments:**
- `label_dir`: Path to SoccerNet-Ball labels directory
- `frame_dir`: Path to extracted frames
- `-o, --out_dir`: Output directory for parsed labels

This creates:
- `class.txt`: List of action classes
- `train.json`, `val.json`, `test.json`, `challenge.json`: Parsed labels per split

### Step 3: Train the Model

Train the E2E-SPOT model:

```bash
python train_e2e.py soccernet_ball /path/to/frames \
    -m rn18 \
    -t gru \
    --clip_len 100 \
    --batch_size 8 \
    --num_epochs 50 \
    -lr 0.001 \
    -s checkpoints/soccernet_ball_model
```

**Key Arguments:**
- `dataset`: Dataset name (`soccernet_ball`)
- `frame_dir`: Path to extracted frames
- `-m, --feature_arch`: CNN backbone (`rn18`, `rn50`, `rny002`, `rny008`, with optional `_tsm` or `_gsm` suffix)
- `-t, --temporal_arch`: Temporal head (`gru`, `deeper_gru`, `mstcn`, `asformer`)
- `--clip_len`: Number of frames per clip (default: 100)
- `--batch_size`: Training batch size (default: 8)
- `--num_epochs`: Number of training epochs (default: 50)
- `-lr`: Learning rate (default: 0.001)
- `-s, --save_dir`: Directory to save checkpoints
- `--resume`: Resume training from checkpoint
- `--mixup`: Enable mixup augmentation (default: True)
- `--dilate_len`: Label dilation length (default: 0)
- `-mgpu`: Enable multi-GPU training

### Step 4: Evaluate the Model

```bash
python eval_soccernet_ball.py soccernet_ball /path/to/frames checkpoints/soccernet_ball_model
```

### Step 5: Run Inference on a Single Video

```bash
python inference_spot.py \
    --model_dir checkpoints/soccernet_ball_model \
    --frame_dir /path/to/frames \
    --video_name "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City/1" \
    --output_dir output \
    --visualize \
    --create_video \
    --score_threshold 0.01
```

**Arguments:**
- `--model_dir`: Path to trained model checkpoint directory
- `--frame_dir`: Path to extracted frames
- `--video_name`: Video/match identifier
- `--output_dir`: Output directory for predictions and visualizations
- `--visualize`: Generate visualization frames
- `--create_video`: Create annotated output video
- `--score_threshold`: Detection confidence threshold (default: 0.3, use 0.01 for low-confidence models)
- `--max_frames`: Limit number of frames to process (for testing)
- `--nms_window`: NMS window size in frames (default: 25)

## Web Application

### Running the Flask App

```bash
cd app
python app.py
```

Then open your browser and navigate to: **http://localhost:5000**

### Using the Web App

1. **Upload Page**: Drag and drop or select a video file (.mp4, .mkv, .avi, .mov)
2. **Processing Page**: Watch the progress as frames are extracted and analyzed
3. **Results Page**: View the annotated video with detected actions
   - Download the processed video
   - Toggle between light/dark themes

### Configuration

Before running the app, update the model path in `app/app.py`:

```python
DEFAULT_MODEL_DIR = r"path/to/your/checkpoints/soccernet_ball_model"
```

## Model Architecture

### Feature Backbone Options
| Architecture | Description | Parameters |
|-------------|-------------|------------|
| `rn18` | ResNet-18 | 11.7M |
| `rn18_tsm` | ResNet-18 + Temporal Shift Module | 11.7M |
| `rn18_gsm` | ResNet-18 + Gated Shift Module | 11.8M |
| `rn50` | ResNet-50 | 25.6M |
| `rny002` | RegNet-Y 200MF | 3.2M |
| `rny008` | RegNet-Y 800MF | 6.3M |

### Temporal Head Options
| Architecture | Description |
|-------------|-------------|
| `gru` | Bidirectional GRU (default) |
| `deeper_gru` | Multi-layer GRU |
| `mstcn` | Multi-Stage TCN |
| `asformer` | ASFormer Transformer |

## Output Format

### Predictions JSON
```json
{
    "video": "league/season/game/half",
    "fps": 25.0,
    "events": [
        {
            "frame": 1234,
            "label": "PASS",
            "score": 0.85
        }
    ]
}
```

### Visualization
- Action detections are overlaid on video frames
- Slow-motion effect is applied to frames with detected actions
- Color-coded labels for each action class

## Troubleshooting

### Low Confidence Scores
The model may output very low scores (0.01-0.05) on some data. Try:
- Lowering `--score_threshold` to 0.01
- Ensuring frames are extracted at the same FPS used during training
- Using the correct checkpoint epoch

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--clip_len`
- Use a smaller backbone (`rn18` instead of `rn50`)

### Missing Frames
- Ensure all frames are extracted with `frames_as_jpg_soccernet_ball.py`
- Check that `fps.txt` exists in each video frame directory

## References

- [E2E-Spot Paper](https://arxiv.org/abs/2207.10213)
- [SoccerNet](https://www.soccer-net.org/)
- [Original E2E-Spot Repository](https://github.com/jhong93/spot)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original E2E-SPOT implementation by Jongseo Hong
- SoccerNet dataset by Silvio Giancola et al.
