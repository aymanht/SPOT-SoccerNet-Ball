#!/usr/bin/env python3
"""Ensemble prediction utilities"""

import numpy as np
from collections import defaultdict
from util.io import load_gz_json


def ensemble(dataset_name, score_list, fps_dict=None, weights=None):
    """
    Ensemble multiple score predictions.
    
    Args:
        dataset_name: Name of the dataset
        score_list: List of score dictionaries from different models/epochs
        fps_dict: Dictionary mapping video names to fps values
        weights: Optional weights for each model's predictions
    
    Returns:
        tuple: (ensemble_scores, ensemble_predictions)
    """
    if weights is None:
        weights = [1.0] * len(score_list)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Aggregate scores
    video_scores = defaultdict(lambda: defaultdict(list))
    
    for score_dict, weight in zip(score_list, weights):
        for video_data in score_dict:
            video = video_data['video']
            for frame_idx, scores in enumerate(video_data['scores']):
                for class_idx, score in enumerate(scores):
                    video_scores[video][frame_idx].append((score * weight, class_idx))
    
    # Build predictions
    predictions = []
    ensemble_scores = {}
    
    for video, frame_scores in sorted(video_scores.items()):
        fps = fps_dict.get(video, 25) if fps_dict else 25
        events = []
        video_ensemble_scores = []
        
        for frame_idx in sorted(frame_scores.keys()):
            # Aggregate scores by class
            class_scores = defaultdict(float)
            for score, class_idx in frame_scores[frame_idx]:
                class_scores[class_idx] += score
            
            # Find best class
            best_class = max(class_scores.keys(), key=lambda k: class_scores[k])
            best_score = class_scores[best_class]
            
            video_ensemble_scores.append(list(class_scores.values()))
            
            if best_class != 0 and best_score > 0.01:  # Not background
                events.append({
                    'label': str(best_class),
                    'frame': frame_idx,
                    'score': best_score
                })
        
        predictions.append({
            'video': video,
            'fps': fps,
            'events': events
        })
        ensemble_scores[video] = video_ensemble_scores
    
    return ensemble_scores, predictions
