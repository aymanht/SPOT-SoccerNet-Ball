#!/usr/bin/env python3
"""
Inference and Visualization script for E2E-SPOT on a single video/match.

This script:
1. Loads a trained SPOT checkpoint
2. Runs inference on frames from a single match
3. Saves predictions to JSON
4. Optionally visualizes results by drawing annotations on frames

Usage:
    python inference_spot.py --model_dir <checkpoint_dir> --frame_dir <frames_path> --output_dir <output>

Example for SoccerNet Ball:
    python inference_spot.py --model_dir <model_dir> --frame_dir <frame_dir> --video_name <video> --output_dir <output> --visualize
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add project directory to path for module imports
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from model.common import BaseRGBModel
from model.shift import make_temporal_shift
from model.modules import GRUPrediction, FCPrediction, TCNPrediction, ASFormerPrediction
from dataset.frame import ActionSpotVideoDataset
from util.io import load_json, store_json, store_gz_json
from util.dataset import load_classes
from util.eval import non_maximum_supression


# Constants
INFERENCE_BATCH_SIZE = 4
MAX_GRU_HIDDEN_DIM = 768

# Class colors for visualization
CLASS_COLORS = {
    'BALL PLAYER BLOCK': (255, 0, 0),       # Red
    'CROSS': (0, 255, 0),                    # Green
    'DRIVE': (0, 0, 255),                    # Blue
    'FREE KICK': (255, 255, 0),              # Yellow
    'GOAL': (255, 0, 255),                   # Magenta
    'HEADER': (0, 255, 255),                 # Cyan
    'HIGH PASS': (255, 128, 0),              # Orange
    'OUT': (128, 0, 255),                    # Purple
    'PASS': (0, 128, 255),                   # Light Blue
    'PLAYER SUCCESSFUL TACKLE': (128, 255, 0), # Lime
    'SHOT': (255, 64, 64),                   # Light Red
}


def get_args():
    parser = argparse.ArgumentParser(description='SPOT Inference and Visualization')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the model checkpoint directory')
    parser.add_argument('--frame_dir', type=str, required=True,
                        help='Base path to extracted frames')
    parser.add_argument('--video_name', type=str, required=True,
                        help='Video/match name (e.g., england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City/1)')
    parser.add_argument('--output_dir', type=str, default='./spot_output',
                        help='Output directory for predictions and visualizations')
    parser.add_argument('--dataset', type=str, default='soccernet_ball',
                        help='Dataset name (default: soccernet_ball)')
    parser.add_argument('--nms_window', type=int, default=25,
                        help='NMS window size (frames)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Score threshold for detections')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization frames')
    parser.add_argument('--vis_fps', type=int, default=25,
                        help='FPS for output visualization video')
    parser.add_argument('--create_video', action='store_true',
                        help='Create a video from visualized frames')
    parser.add_argument('--max_vis_frames', type=int, default=None,
                        help='Maximum frames to visualize (None = all)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch checkpoint to use (default: best or last)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process for inference (None = all). Use this to test on a subset of the video.')
    return parser.parse_args()


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

            # Add Temporal Shift Modules
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
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


def get_best_epoch(model_dir, key='val_mAP'):
    """Get best epoch from training history"""
    loss_file = os.path.join(model_dir, 'loss.json')
    if os.path.exists(loss_file):
        data = load_json(loss_file)
        if data and data[0].get(key, 0) > 0:
            best = max(data, key=lambda x: x.get(key, 0))
            return best['epoch']
        # Fallback to best validation loss
        best = min(data, key=lambda x: x.get('val', float('inf')))
        return best['epoch']
    return None


def get_last_epoch(model_dir):
    """Get the last available checkpoint epoch"""
    import re
    regex = re.compile(r'checkpoint_(\d+)\.pt')
    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    return last_epoch if last_epoch >= 0 else None


def create_single_video_dataset_json(video_name, frame_dir, fps=25.0, max_frames=None):
    """Create a minimal dataset JSON for a single video"""
    video_frame_dir = os.path.join(frame_dir, video_name)
    
    # Count frames
    num_frames = 0
    for f in os.listdir(video_frame_dir):
        if f.endswith('.jpg'):
            num_frames += 1
    
    # Limit frames if max_frames is specified
    if max_frames is not None and max_frames < num_frames:
        print(f"Limiting inference to first {max_frames} frames (out of {num_frames})")
        num_frames = max_frames
    
    # Try to get FPS from fps.txt if available
    fps_file = os.path.join(video_frame_dir, 'fps.txt')
    if os.path.exists(fps_file):
        with open(fps_file) as fp:
            fps = float(fp.read().strip())
    
    # Get image dimensions
    sample_img = os.path.join(video_frame_dir, '000000.jpg')
    if os.path.exists(sample_img):
        img = Image.open(sample_img)
        width, height = img.size
    else:
        width, height = 398, 224  # Default
    
    return [{
        'video': video_name,
        'num_frames': num_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'events': [],
        'num_events': 0
    }]


def run_inference(model, dataset, classes):
    """Run inference on the dataset and collect predictions"""
    from torch.utils.data import DataLoader
    
    classes_inv = {v: k for k, v in classes.items()}
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))
    
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    
    print(f"Running inference with batch size {batch_size}...")
    for clip in tqdm(DataLoader(
            dataset, num_workers=0, pin_memory=True, batch_size=batch_size
    )):
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
    
    return pred_dict, classes_inv


def process_predictions(pred_dict, classes_inv, fps_dict, high_recall_threshold=0.01):
    """Process raw predictions into event lists"""
    pred_events = []
    pred_events_high_recall = []
    
    for video, (scores, support) in sorted(pred_dict.items()):
        support[support == 0] = 1  # Avoid division by zero
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        
        events = []
        events_high_recall = []
        
        for i in range(pred.shape[0]):
            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': float(scores[i, pred[i]])
                })
            
            for j in classes_inv:
                if scores[i, j] >= high_recall_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': float(scores[i, j])
                    })
        
        fps = fps_dict.get(video, 25.0)
        pred_events.append({
            'video': video,
            'events': events,
            'fps': fps,
            'num_events': len(events)
        })
        pred_events_high_recall.append({
            'video': video,
            'events': events_high_recall,
            'fps': fps,
            'num_events': len(events_high_recall)
        })
    
    return pred_events, pred_events_high_recall


def visualize_frame(frame_path, events_at_frame, output_path, frame_num, total_frames):
    """Draw event annotations on a single frame"""
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
        big_font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
        small_font = font
        big_font = font
    
    # Draw frame info at top-left
    info_text = f"Frame: {frame_num}/{total_frames}"
    draw.rectangle([(0, 0), (180, 30)], fill=(0, 0, 0))
    draw.text((5, 5), info_text, fill=(255, 255, 255), font=small_font)
    
    # Draw events if any
    if events_at_frame:
        # Draw a highlight banner at the top
        draw.rectangle([(0, 30), (img_width, 70)], fill=(0, 100, 0))
        draw.text((10, 35), "ACTION DETECTED!", fill=(255, 255, 0), font=font)
        
        # Draw each event
        y_offset = 75
        for event in events_at_frame:
            label = event['label']
            score = event['score']
            color = CLASS_COLORS.get(label, (255, 255, 255))
            
            text = f"{label}: {score:.2f}"
            # Draw background rectangle
            text_bbox = draw.textbbox((10, y_offset), text, font=big_font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-3, text_bbox[2]+5, text_bbox[3]+3], 
                          fill=(0, 0, 0))
            draw.text((10, y_offset), text, fill=color, font=big_font)
            y_offset += 40
    else:
        # No event - show "No action" in smaller text
        draw.rectangle([(0, 30), (120, 50)], fill=(50, 50, 50))
        draw.text((5, 32), "No action", fill=(150, 150, 150), font=small_font)
    
    img.save(output_path)
    return img


def create_event_timeline(pred_events, nms_window, score_threshold):
    """Create a frame-to-events mapping for visualization"""
    # Apply NMS
    if nms_window > 0:
        pred_events = non_maximum_supression(pred_events, nms_window)
    
    frame_events = {}
    for video_pred in pred_events:
        for event in video_pred['events']:
            if event['score'] >= score_threshold:
                frame = event['frame']
                if frame not in frame_events:
                    frame_events[frame] = []
                frame_events[frame].append(event)
    
    return frame_events, pred_events


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SPOT Inference and Visualization")
    print("=" * 60)
    
    # Load model config
    config_path = os.path.join(args.model_dir, 'config.json')
    config = load_json(config_path)
    print(f"\nModel Configuration:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Feature arch: {config['feature_arch']}")
    print(f"  Temporal arch: {config['temporal_arch']}")
    print(f"  Clip length: {config['clip_len']}")
    print(f"  Modality: {config['modality']}")
    
    # Determine checkpoint epoch
    if args.epoch is not None:
        epoch = args.epoch
    else:
        epoch = get_best_epoch(args.model_dir)
        if epoch is None:
            epoch = get_last_epoch(args.model_dir)
    
    print(f"\nUsing checkpoint epoch: {epoch}")
    
    # Load classes
    data_dir = os.path.join(SPOT_DIR, 'data', args.dataset)
    classes = load_classes(os.path.join(data_dir, 'class.txt'))
    print(f"\nClasses ({len(classes)}): {list(classes.keys())}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = E2EModel(
        len(classes) + 1,
        config['feature_arch'],
        config['temporal_arch'],
        clip_len=config['clip_len'],
        modality=config['modality'],
        device=device,
        multi_gpu=False
    )
    
    checkpoint_path = os.path.join(args.model_dir, f'checkpoint_{epoch:03d}.pt')
    model.load(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
    
    # Create temporary dataset JSON for single video
    print(f"\nProcessing video: {args.video_name}")
    temp_json = create_single_video_dataset_json(args.video_name, args.frame_dir, max_frames=args.max_frames)
    temp_json_path = os.path.join(args.output_dir, 'temp_dataset.json')
    store_json(temp_json_path, temp_json)
    
    fps_dict = {args.video_name: temp_json[0]['fps']}
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = ActionSpotVideoDataset(
        classes,
        temp_json_path,
        args.frame_dir,
        config['modality'],
        config['clip_len'],
        overlap_len=config['clip_len'] // 2,
        crop_dim=config['crop_dim']
    )
    
    # Run inference
    print("\nRunning inference...")
    pred_dict, classes_inv = run_inference(model, dataset, classes)
    
    # Process predictions
    pred_events, pred_events_high_recall = process_predictions(
        pred_dict, classes_inv, fps_dict
    )
    
    # Apply NMS and threshold
    frame_events, filtered_events = create_event_timeline(
        pred_events_high_recall, args.nms_window, args.score_threshold
    )
    
    # Save predictions
    output_pred_path = os.path.join(args.output_dir, 'predictions.json')
    output_pred_hr_path = os.path.join(args.output_dir, 'predictions_high_recall.json')
    store_json(output_pred_path, filtered_events, pretty=True)
    store_json(output_pred_hr_path, pred_events_high_recall, pretty=True)
    print(f"\nSaved predictions to: {output_pred_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    
    event_counts = {}
    for video_pred in filtered_events:
        for event in video_pred['events']:
            label = event['label']
            event_counts[label] = event_counts.get(label, 0) + 1
    
    print(f"\nTotal detections (score >= {args.score_threshold}, NMS window = {args.nms_window}):")
    for label, count in sorted(event_counts.items()):
        print(f"  {label}: {count}")
    print(f"\nTotal: {sum(event_counts.values())} events")
    
    # Visualization
    if args.visualize:
        print("\n" + "=" * 60)
        print("Generating Visualizations")
        print("=" * 60)
        
        vis_dir = os.path.join(args.output_dir, 'visualized_frames')
        os.makedirs(vis_dir, exist_ok=True)
        
        video_frame_dir = os.path.join(args.frame_dir, args.video_name)
        total_frames = temp_json[0]['num_frames']
        
        if args.max_vis_frames:
            total_frames = min(total_frames, args.max_vis_frames)
        
        print(f"Total frames available: {total_frames}")
        
        # Visualize ALL frames (continuous video)
        frames_to_visualize = list(range(total_frames))
        print(f"Visualizing all {len(frames_to_visualize)} frames...")
        
        for frame_num in tqdm(frames_to_visualize):
            frame_path = os.path.join(video_frame_dir, f'{frame_num:06d}.jpg')
            if os.path.exists(frame_path):
                events_at_frame = frame_events.get(frame_num, [])
                output_path = os.path.join(vis_dir, f'{frame_num:06d}.jpg')
                visualize_frame(frame_path, events_at_frame, output_path, frame_num, total_frames)
        
        print(f"\nVisualized frames saved to: {vis_dir}")
        
        # Create video if requested
        if args.create_video:
            print("\nCreating output video with slow-motion on events...")
            video_path = os.path.join(args.output_dir, 'visualization.mp4')
            
            # Get frame size from first visualized frame
            first_frame = os.path.join(vis_dir, f'{frames_to_visualize[0]:06d}.jpg')
            img = cv2.imread(first_frame)
            height, width = img.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, args.vis_fps, (width, height))
            
            # Settings for slow-motion effect
            slow_motion_repeats = 15  # Repeat event frames 15 times (slows to ~1.7 fps for events)
            context_before = 10  # Start slowing down 10 frames before event
            context_after = 10   # Keep slow 10 frames after event
            
            # Find all frames that should be slowed (events + context)
            slow_frames = set()
            for ef in frame_events.keys():
                for f in range(max(0, ef - context_before), min(total_frames, ef + context_after + 1)):
                    slow_frames.add(f)
            
            print(f"Normal frames: {total_frames - len(slow_frames)}, Slow-motion frames: {len(slow_frames)}")
            
            for frame_num in tqdm(frames_to_visualize):
                frame_path = os.path.join(vis_dir, f'{frame_num:06d}.jpg')
                if os.path.exists(frame_path):
                    img = cv2.imread(frame_path)
                    
                    if frame_num in slow_frames:
                        # Slow motion: repeat the frame multiple times
                        repeat_count = slow_motion_repeats if frame_num in frame_events else 3
                        for _ in range(repeat_count):
                            out.write(img)
                    else:
                        # Normal speed
                        out.write(img)
            
            out.release()
            print(f"Video saved to: {video_path}")
    
    # Clean up temp file
    os.remove(temp_json_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
