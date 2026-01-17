#!/usr/bin/env python3
"""Helper functions for evaluation"""

import os
import glob
import re


def get_pred_file(model_dir, split):
    """
    Get the best prediction file from a model directory.
    
    Args:
        model_dir: Path to the model checkpoint directory
        split: Data split ('train', 'val', 'test', 'challenge')
    
    Returns:
        tuple: (pred_file_path, epoch_number)
    """
    # Look for prediction files matching the pattern pred-{split}.{epoch}.json
    pattern = os.path.join(model_dir, f'pred-{split}.*.json')
    pred_files = glob.glob(pattern)
    
    # Filter out score files and recall files (keep only main pred files)
    pred_files = [f for f in pred_files if not f.endswith('.score.json') 
                  and not f.endswith('.recall.json')
                  and not f.endswith('.json.gz')]
    
    if not pred_files:
        # Try with .gz extension
        pattern = os.path.join(model_dir, f'pred-{split}.*.recall.json.gz')
        pred_files = glob.glob(pattern)
        if not pred_files:
            raise FileNotFoundError(
                f"No prediction files found for split '{split}' in {model_dir}")
    
    # Extract epoch numbers and find the latest
    epoch_pattern = re.compile(rf'pred-{split}\.(\d+)\.(?:json|recall\.json\.gz)$')
    epochs = []
    for f in pred_files:
        match = epoch_pattern.search(f)
        if match:
            epochs.append((int(match.group(1)), f))
    
    if not epochs:
        raise ValueError(f"Could not parse epoch from prediction files: {pred_files}")
    
    # Sort by epoch and get the latest
    epochs.sort(key=lambda x: x[0], reverse=True)
    best_epoch, best_file = epochs[0]
    
    # Return the recall file if that's what we found, otherwise the json file
    recall_file = os.path.join(model_dir, f'pred-{split}.{best_epoch}.recall.json.gz')
    if os.path.exists(recall_file):
        return recall_file, best_epoch
    
    return best_file, best_epoch
