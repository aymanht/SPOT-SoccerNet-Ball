#!/usr/bin/env python3
"""
Flask Web Application for SPOT Action Spotting

This app allows users to:
1. Upload a video file
2. Extract frames from the video
3. Run SPOT model predictions
4. View the results with annotated video
"""

import os
import sys
import uuid
import shutil
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Add paths - project root is parent of app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_DIR)

from model.common import BaseRGBModel
from model.shift import make_temporal_shift
from model.modules import GRUPrediction, FCPrediction, TCNPrediction, ASFormerPrediction
from dataset.frame import ActionSpotVideoDataset
from util.io import load_json, store_json
from util.dataset import load_classes
from util.eval import non_maximum_supression

# Configuration
UPLOAD_FOLDER = os.path.join(APP_DIR, 'uploads')
FRAMES_FOLDER = os.path.join(APP_DIR, 'frames')
OUTPUT_FOLDER = os.path.join(APP_DIR, 'outputs')
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov'}

# Frame extraction settings
TARGET_HEIGHT = 224
TARGET_WIDTH = 398

# Model settings
MAX_GRU_HIDDEN_DIM = 768
INFERENCE_BATCH_SIZE = 4

# Default model path - update this to your checkpoint location
DEFAULT_MODEL_DIR = r"C:\Users\ADMIN\Downloads\checkpoints\checkpoints\soccernet_ball_model"

# Class colors for visualization
CLASS_COLORS = {
    'BALL PLAYER BLOCK': (255, 0, 0),
    'CROSS': (0, 255, 0),
    'DRIVE': (0, 0, 255),
    'FREE KICK': (255, 255, 0),
    'GOAL': (255, 0, 255),
    'HEADER': (0, 255, 255),
    'HIGH PASS': (255, 128, 0),
    'OUT': (128, 0, 255),
    'PASS': (0, 128, 255),
    'PLAYER SUCCESSFUL TACKLE': (128, 255, 0),
    'SHOT': (255, 64, 64),
}

app = Flask(__name__)
app.secret_key = 'spot-action-spotting-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_frames(video_path, output_dir, sample_fps=25, max_frames=None):
    """Extract frames from video at specified FPS"""
    os.makedirs(output_dir, exist_ok=True)
    
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate stride for target FPS
    stride = max(1, int(fps / sample_fps))
    effective_fps = fps / stride
    
    print(f"Video FPS: {fps}, Stride: {stride}, Effective FPS: {effective_fps}")
    
    out_frame_num = 0
    i = 0
    
    while True:
        ret, frame = vc.read()
        if not ret:
            break
            
        if i % stride == 0:
            # Resize to target dimensions
            if frame.shape[0] != TARGET_HEIGHT or frame.shape[1] != TARGET_WIDTH:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            frame_path = os.path.join(output_dir, f'{out_frame_num:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            out_frame_num += 1
            
            if max_frames and out_frame_num >= max_frames:
                break
        i += 1
    
    vc.release()
    
    # Save FPS info
    with open(os.path.join(output_dir, 'fps.txt'), 'w') as fp:
        fp.write(str(effective_fps))
    
    return out_frame_num, effective_fps


class E2EModel(BaseRGBModel):
    """E2E-SPOT Model wrapper for inference"""
    
    class Impl(torch.nn.Module):
        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, modality):
            super().__init__()
            import torchvision
            import timm
            import torch.nn as nn
            import torch.nn.functional as F
            
            is_rgb = modality == 'rgb'
            in_channels = {'flow': 2, 'bw': 1, 'rgb': 3}[modality]

            if feature_arch.startswith(('rn18', 'rn50')):
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                features = getattr(torchvision.models, resnet_name)(pretrained=is_rgb)
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()

                if not is_rgb:
                    features.conv1 = nn.Conv2d(
                        in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                        padding=(3, 3), bias=False)

            elif feature_arch.startswith(('rny002', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem.conv = nn.Conv2d(
                        in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                        padding=(1, 1), bias=False)

            elif 'convnextt' in feature_arch:
                features = timm.create_model('convnext_tiny', pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
            else:
                raise NotImplementedError(feature_arch)

            self._require_clip_len = -1
            if feature_arch.endswith('_tsm'):
                make_temporal_shift(features, clip_len, is_gsm=False)
                self._require_clip_len = clip_len
            elif feature_arch.endswith('_gsm'):
                make_temporal_shift(features, clip_len, is_gsm=True)
                self._require_clip_len = clip_len

            self._features = features
            self._feat_dim = feat_dim

            if 'gru' in temporal_arch:
                hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM)
                if temporal_arch in ('gru', 'deeper_gru'):
                    self._pred_fine = GRUPrediction(
                        feat_dim, num_classes, hidden_dim,
                        num_layers=3 if temporal_arch[0] == 'd' else 1)
                else:
                    raise NotImplementedError(temporal_arch)
            elif temporal_arch == 'mstcn':
                self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == 'asformer':
                self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == '':
                self._pred_fine = FCPrediction(feat_dim, num_classes)
            else:
                raise NotImplementedError(temporal_arch)

        def forward(self, x):
            import torch.nn.functional as F
            batch_size, true_clip_len, channels, height, width = x.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                assert true_clip_len <= self._require_clip_len
                if true_clip_len < self._require_clip_len:
                    x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._feat_dim)

            if true_clip_len != clip_len:
                im_feat = im_feat[:, :true_clip_len, :]

            return self._pred_fine(im_feat)

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
                 modality, device='cuda', multi_gpu=False):
        self.device = device
        self._multi_gpu = multi_gpu
        self._model = E2EModel.Impl(num_classes, feature_arch, temporal_arch, clip_len, modality)

        if multi_gpu:
            self._model = torch.nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def predict(self, seq, use_amp=True):
        from contextlib import nullcontext
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp and self.device == 'cuda' else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


# Global model instance (loaded once)
model = None
classes = None
config = None


def load_model(model_dir=DEFAULT_MODEL_DIR):
    """Load the SPOT model"""
    global model, classes, config
    
    if model is not None:
        return model, classes, config
    
    config_path = os.path.join(model_dir, 'config.json')
    config = load_json(config_path)
    
    # Load classes
    data_dir = os.path.join(SPOT_DIR, 'data', config['dataset'])
    classes = load_classes(os.path.join(data_dir, 'class.txt'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    
    model = E2EModel(
        len(classes) + 1,
        config['feature_arch'],
        config['temporal_arch'],
        clip_len=config['clip_len'],
        modality=config['modality'],
        device=device,
        multi_gpu=False
    )
    
    # Find best checkpoint
    import re
    regex = re.compile(r'checkpoint_(\d+)\.pt')
    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    
    checkpoint_path = os.path.join(model_dir, f'checkpoint_{last_epoch:03d}.pt')
    model.load(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
    
    return model, classes, config


def run_inference(model, classes, config, frame_dir, num_frames, fps):
    """Run SPOT inference on extracted frames"""
    from torch.utils.data import DataLoader
    
    # Create temporary dataset JSON
    temp_json = [{
        'video': 'video',
        'num_frames': num_frames,
        'fps': fps,
        'width': TARGET_WIDTH,
        'height': TARGET_HEIGHT,
        'events': [],
        'num_events': 0
    }]
    
    temp_json_path = os.path.join(frame_dir, 'dataset.json')
    store_json(temp_json_path, temp_json)
    
    # Create dataset - frame_dir should be parent of 'video' folder
    parent_dir = os.path.dirname(frame_dir)
    video_name = os.path.basename(frame_dir)
    
    # Rename for dataset compatibility
    temp_video_dir = os.path.join(parent_dir, 'video')
    if os.path.exists(temp_video_dir) and temp_video_dir != frame_dir:
        shutil.rmtree(temp_video_dir)
    if frame_dir != temp_video_dir:
        os.rename(frame_dir, temp_video_dir)
        frame_dir = temp_video_dir
    
    # Update JSON with correct video name
    temp_json[0]['video'] = 'video'
    store_json(temp_json_path, temp_json)
    
    dataset = ActionSpotVideoDataset(
        classes,
        temp_json_path,
        parent_dir,
        config['modality'],
        config['clip_len'],
        overlap_len=config['clip_len'] // 2,
        crop_dim=config['crop_dim']
    )
    
    classes_inv = {v: k for k, v in classes.items()}
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))
    
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    
    for clip in tqdm(DataLoader(dataset, num_workers=0, pin_memory=True, batch_size=batch_size)):
        if batch_size > 1:
            _, batch_pred_scores = model.predict(clip['frame'])
            
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1
        else:
            scores, support = pred_dict[clip['video'][0]]
            start = clip['start'][0].item()
            _, pred_scores = model.predict(clip['frame'][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, :end - start, :]
            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]
    
    # Process predictions
    pred_events = []
    for video, (scores, support) in sorted(pred_dict.items()):
        support[support == 0] = 1
        scores /= support[:, None]
        
        events = []
        for i in range(scores.shape[0]):
            for j in classes_inv:
                if scores[i, j] >= 0.01:  # Low threshold
                    events.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': float(scores[i, j])
                    })
        
        pred_events.append({
            'video': video,
            'events': events,
            'fps': fps,
            'num_events': len(events)
        })
    
    return pred_events, frame_dir


def visualize_frame(frame_path, events_at_frame, output_path, frame_num, total_frames):
    """Draw event annotations on a single frame"""
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
        big_font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
        small_font = font
        big_font = font
    
    # Draw frame info
    info_text = f"Frame: {frame_num}/{total_frames}"
    draw.rectangle([(0, 0), (180, 30)], fill=(0, 0, 0))
    draw.text((5, 5), info_text, fill=(255, 255, 255), font=small_font)
    
    if events_at_frame:
        draw.rectangle([(0, 30), (img_width, 70)], fill=(0, 100, 0))
        draw.text((10, 35), "ACTION DETECTED!", fill=(255, 255, 0), font=font)
        
        y_offset = 75
        for event in events_at_frame:
            label = event['label']
            score = event['score']
            color = CLASS_COLORS.get(label, (255, 255, 255))
            
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((10, y_offset), text, font=big_font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], 
                          fill=(0, 0, 0))
            draw.text((10, y_offset), text, fill=color, font=big_font)
            y_offset += 40
    else:
        draw.rectangle([(0, 30), (120, 50)], fill=(50, 50, 50))
        draw.text((5, 32), "No action", fill=(150, 150, 150), font=small_font)
    
    img.save(output_path)


def create_output_video(frame_dir, pred_events, output_path, fps=25, nms_window=25, score_threshold=0.01):
    """Create output video with annotations and slow motion on events"""
    
    # Apply NMS
    if nms_window > 0:
        pred_events = non_maximum_supression(pred_events, nms_window)
    
    # Create frame-to-events mapping
    frame_events = {}
    for video_pred in pred_events:
        for event in video_pred['events']:
            if event['score'] >= score_threshold:
                frame = event['frame']
                if frame not in frame_events:
                    frame_events[frame] = []
                frame_events[frame].append(event)
    
    # Count frames
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') and f != 'fps.txt'])
    total_frames = len(frame_files)
    
    if total_frames == 0:
        return None
    
    # Create visualized frames
    vis_dir = os.path.join(os.path.dirname(output_path), 'vis_frames')
    os.makedirs(vis_dir, exist_ok=True)
    
    for frame_num in tqdm(range(total_frames), desc="Creating visualizations"):
        frame_path = os.path.join(frame_dir, f'{frame_num:06d}.jpg')
        if os.path.exists(frame_path):
            events_at_frame = frame_events.get(frame_num, [])
            output_frame_path = os.path.join(vis_dir, f'{frame_num:06d}.jpg')
            visualize_frame(frame_path, events_at_frame, output_frame_path, frame_num, total_frames)
    
    # Create video with slow motion
    first_frame = os.path.join(vis_dir, '000000.jpg')
    img = cv2.imread(first_frame)
    height, width = img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Slow motion settings
    slow_motion_repeats = 15
    context_before = 10
    context_after = 10
    
    slow_frames = set()
    for ef in frame_events.keys():
        for f in range(max(0, ef - context_before), min(total_frames, ef + context_after + 1)):
            slow_frames.add(f)
    
    for frame_num in tqdm(range(total_frames), desc="Creating video"):
        frame_path = os.path.join(vis_dir, f'{frame_num:06d}.jpg')
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            
            if frame_num in slow_frames:
                repeat_count = slow_motion_repeats if frame_num in frame_events else 3
                for _ in range(repeat_count):
                    out.write(img)
            else:
                out.write(img)
    
    out.release()
    
    # Cleanup vis_frames
    shutil.rmtree(vis_dir)
    
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file provided')
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    if file.filename == '':
        flash('No video selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())[:8]
        
        # Save uploaded video
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_{filename}')
        file.save(video_path)
        
        # Get parameters
        max_frames = request.form.get('max_frames', type=int) or None
        score_threshold = request.form.get('score_threshold', type=float) or 0.01
        
        flash(f'Video uploaded successfully! Job ID: {job_id}')
        return redirect(url_for('process', job_id=job_id, filename=f'{job_id}_{filename}', 
                               max_frames=max_frames or 0, score_threshold=score_threshold))
    
    flash('Invalid file type. Allowed: mp4, mkv, avi, mov')
    return redirect(url_for('index'))


@app.route('/process/<job_id>/<filename>')
def process(job_id, filename):
    max_frames = request.args.get('max_frames', type=int) or None
    if max_frames == 0:
        max_frames = None
    score_threshold = request.args.get('score_threshold', type=float) or 0.01
    
    return render_template('processing.html', job_id=job_id, filename=filename,
                          max_frames=max_frames, score_threshold=score_threshold)


@app.route('/api/process/<job_id>/<filename>', methods=['POST'])
def api_process(job_id, filename):
    """API endpoint to process the video"""
    try:
        max_frames = request.json.get('max_frames') or None
        score_threshold = request.json.get('score_threshold', 0.01)
        
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Step 1: Extract frames
        frame_dir = os.path.join(FRAMES_FOLDER, job_id, 'video')
        os.makedirs(frame_dir, exist_ok=True)
        
        num_frames, fps = extract_frames(video_path, frame_dir, sample_fps=25, max_frames=max_frames)
        
        if num_frames == 0:
            return jsonify({'error': 'Failed to extract frames'}), 500
        
        # Step 2: Load model and run inference
        model, classes, config = load_model()
        pred_events, frame_dir = run_inference(model, classes, config, frame_dir, num_frames, fps)
        
        # Step 3: Create output video
        output_video_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_output.mp4')
        create_output_video(frame_dir, pred_events, output_video_path, fps=25, 
                           nms_window=25, score_threshold=score_threshold)
        
        # Count detections
        total_events = sum(len([e for e in vp['events'] if e['score'] >= score_threshold]) 
                          for vp in pred_events)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'num_frames': num_frames,
            'total_events': total_events,
            'video_url': url_for('get_result', job_id=job_id)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/result/<job_id>')
def get_result(job_id):
    """Serve the output video"""
    video_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_output.mp4')
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    return "Video not found", 404


@app.route('/view/<job_id>')
def view_result(job_id):
    """View the result page"""
    return render_template('result.html', job_id=job_id)


if __name__ == '__main__':
    print("Loading SPOT model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
