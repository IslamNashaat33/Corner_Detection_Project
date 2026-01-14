"""
Data loading utilities for stroke and label files.
"""
import os
import numpy as np
from typing import Tuple, List


def load_stroke(filepath: str) -> np.ndarray:
    """Load a single stroke file."""
    return np.load(filepath)


def load_label(filepath: str) -> np.ndarray:
    """Load a single label file."""
    return np.load(filepath)


def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load all strokes and labels from a dataset directory.
    
    Args:
        data_dir: Path to directory containing 'strokes' and 'labels' subdirs
        
    Returns:
        Tuple of (list of stroke arrays, list of label arrays)
    """
    strokes_dir = os.path.join(data_dir, "strokes")
    labels_dir = os.path.join(data_dir, "labels")
    
    stroke_files = sorted(os.listdir(strokes_dir))
    
    strokes = []
    labels = []
    
    for filename in stroke_files:
        stroke_path = os.path.join(strokes_dir, filename)
        label_path = os.path.join(labels_dir, filename)
        
        if os.path.exists(label_path):
            strokes.append(load_stroke(stroke_path))
            labels.append(load_label(label_path))
    
    return strokes, labels


def get_dataset_info(strokes: List[np.ndarray], labels: List[np.ndarray]) -> dict:
    """
    Get summary statistics about the dataset.
    
    Args:
        strokes: List of stroke arrays
        labels: List of label arrays
        
    Returns:
        Dictionary with dataset statistics
    """
    total_points = sum(len(s) for s in strokes)
    total_corners = sum(np.sum(l) for l in labels)
    stroke_lengths = [len(s) for s in strokes]
    
    return {
        "num_strokes": len(strokes),
        "total_points": total_points,
        "total_corners": int(total_corners),
        "corner_ratio": total_corners / total_points if total_points > 0 else 0,
        "avg_stroke_length": np.mean(stroke_lengths),
        "min_stroke_length": min(stroke_lengths),
        "max_stroke_length": max(stroke_lengths),
    }
