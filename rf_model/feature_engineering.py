"""
Feature engineering for corner detection.

Extracts geometric features from stroke points to identify corners/angles.
"""
import numpy as np
from typing import List, Tuple


def compute_direction_angles(stroke: np.ndarray) -> np.ndarray:
    """
    Compute direction angle at each point.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        
    Returns:
        (N,) array of direction angles in radians
    """
    n = len(stroke)
    angles = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            dx = stroke[1, 0] - stroke[0, 0]
            dy = stroke[1, 1] - stroke[0, 1]
        elif i == n - 1:
            dx = stroke[-1, 0] - stroke[-2, 0]
            dy = stroke[-1, 1] - stroke[-2, 1]
        else:
            dx = stroke[i + 1, 0] - stroke[i - 1, 0]
            dy = stroke[i + 1, 1] - stroke[i - 1, 1]
        
        angles[i] = np.arctan2(dy, dx)
    
    return angles


def compute_curvature(stroke: np.ndarray) -> np.ndarray:
    """
    Compute curvature at each point using the angle between consecutive segments.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        
    Returns:
        (N,) array of curvature values
    """
    n = len(stroke)
    curvature = np.zeros(n)
    
    for i in range(1, n - 1):
        v1 = stroke[i] - stroke[i - 1]
        v2 = stroke[i + 1] - stroke[i]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-8 and norm2 > 1e-8:
            v1_norm = v1 / norm1
            v2_norm = v2 / norm2
            
            cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
            dot = np.dot(v1_norm, v2_norm)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            curvature[i] = angle * np.sign(cross)
    
    return curvature


def compute_angle_change(stroke: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Compute the change in direction angle over a window.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        window: Size of the window for computing angle change
        
    Returns:
        (N,) array of angle change values
    """
    n = len(stroke)
    angle_change = np.zeros(n)
    half_window = window // 2
    
    for i in range(n):
        left = max(0, i - half_window)
        right = min(n - 1, i + half_window)
        
        if right > left:
            v1 = stroke[i] - stroke[left]
            v2 = stroke[right] - stroke[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle_change[i] = np.arccos(dot)
    
    return angle_change


def compute_local_distances(stroke: np.ndarray) -> np.ndarray:
    """
    Compute distance to previous and next points.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        
    Returns:
        (N, 2) array of [dist_to_prev, dist_to_next]
    """
    n = len(stroke)
    distances = np.zeros((n, 2))
    
    for i in range(n):
        if i > 0:
            distances[i, 0] = np.linalg.norm(stroke[i] - stroke[i - 1])
        if i < n - 1:
            distances[i, 1] = np.linalg.norm(stroke[i + 1] - stroke[i])
    
    return distances


def compute_speed_variation(stroke: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Compute local speed variation (std of segment lengths in a window).
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        window: Window size for computing variation
        
    Returns:
        (N,) array of speed variation values
    """
    n = len(stroke)
    
    segments = np.zeros(n)
    for i in range(1, n):
        segments[i] = np.linalg.norm(stroke[i] - stroke[i - 1])
    
    speed_var = np.zeros(n)
    half_window = window // 2
    
    for i in range(n):
        left = max(0, i - half_window)
        right = min(n, i + half_window + 1)
        if right > left:
            speed_var[i] = np.std(segments[left:right])
    
    return speed_var


def compute_position_features(stroke: np.ndarray) -> np.ndarray:
    """
    Compute normalized position features.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        
    Returns:
        (N, 3) array of [normalized_x, normalized_y, relative_position]
    """
    n = len(stroke)
    
    min_coords = stroke.min(axis=0)
    max_coords = stroke.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords < 1e-8] = 1.0
    
    normalized = (stroke - min_coords) / range_coords
    relative_pos = np.linspace(0, 1, n).reshape(-1, 1)
    
    return np.hstack([normalized, relative_pos])


def compute_window_features(stroke: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute statistical features over a local window.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        window: Window size
        
    Returns:
        (N, 4) array of [mean_x, mean_y, std_x, std_y] relative to center
    """
    n = len(stroke)
    features = np.zeros((n, 4))
    half_window = window // 2
    
    for i in range(n):
        left = max(0, i - half_window)
        right = min(n, i + half_window + 1)
        
        local_points = stroke[left:right]
        center = stroke[i]
        relative = local_points - center
        
        features[i, 0] = np.mean(relative[:, 0])
        features[i, 1] = np.mean(relative[:, 1])
        features[i, 2] = np.std(relative[:, 0])
        features[i, 3] = np.std(relative[:, 1])
    
    return features


def extract_features(stroke: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Extract all features for a single stroke.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        window_size: Window size for local features
        
    Returns:
        (N, num_features) array of features for each point
    """
    direction_angles = compute_direction_angles(stroke)
    curvature = compute_curvature(stroke)
    angle_change = compute_angle_change(stroke, window_size)
    local_distances = compute_local_distances(stroke)
    speed_var = compute_speed_variation(stroke, window_size)
    position_features = compute_position_features(stroke)
    window_features = compute_window_features(stroke, window_size)
    
    features = np.column_stack([
        direction_angles,           # 1 feature
        np.abs(curvature),          # 1 feature (absolute curvature)
        curvature,                  # 1 feature (signed curvature)
        angle_change,               # 1 feature
        local_distances,            # 2 features
        speed_var,                  # 1 feature
        position_features,          # 3 features
        window_features,            # 4 features
    ])
    
    return features


def extract_features_batch(
    strokes: List[np.ndarray],
    labels: List[np.ndarray],
    window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for a batch of strokes.
    
    Args:
        strokes: List of stroke arrays
        labels: List of label arrays
        window_size: Window size for local features
        
    Returns:
        Tuple of (features array, labels array)
    """
    all_features = []
    all_labels = []
    
    for stroke, label in zip(strokes, labels):
        features = extract_features(stroke, window_size)
        all_features.append(features)
        all_labels.append(label)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y


# Feature names for interpretability
FEATURE_NAMES = [
    "direction_angle",
    "abs_curvature",
    "signed_curvature",
    "angle_change",
    "dist_to_prev",
    "dist_to_next",
    "speed_variation",
    "normalized_x",
    "normalized_y",
    "relative_position",
    "window_mean_x",
    "window_mean_y",
    "window_std_x",
    "window_std_y",
]
