"""
Prediction utilities for corner detection on new strokes.
"""
import os
import sys
import numpy as np
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model import config
from ml_model.feature_engineering import extract_features
from ml_model.model import CornerDetectionModel


def load_trained_model() -> CornerDetectionModel:
    """Load the trained model from disk."""
    model_path = os.path.join(config.MODEL_DIR, "decision_tree_model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train.py first."
        )
    
    model = CornerDetectionModel()
    model.load(model_path)
    return model


def predict_corners(
    stroke: np.ndarray,
    model: CornerDetectionModel = None,
    return_probabilities: bool = False,
) -> Tuple[np.ndarray, List[int], int]:
    """
    Predict corners in a single stroke.
    
    Args:
        stroke: (N, 2) array of (x, y) coordinates
        model: Trained model (loads default if None)
        return_probabilities: Whether to return corner probabilities
        
    Returns:
        Tuple of:
            - predictions: (N,) binary array of corner predictions
            - corner_indices: List of indices where corners were detected
            - num_corners: Number of corners detected
    """
    if model is None:
        model = load_trained_model()
    
    # Extract features
    features = extract_features(stroke, config.WINDOW_SIZE)
    
    # Predict
    predictions = model.predict(features)
    corner_indices = np.where(predictions == 1)[0].tolist()
    num_corners = len(corner_indices)
    
    if return_probabilities:
        proba = model.predict_proba(features)[:, 1]
        return predictions, corner_indices, num_corners, proba
    
    return predictions, corner_indices, num_corners


def predict_corners_batch(
    strokes: List[np.ndarray],
    model: CornerDetectionModel = None,
) -> List[Tuple[np.ndarray, List[int], int]]:
    """
    Predict corners in multiple strokes.
    
    Args:
        strokes: List of stroke arrays
        model: Trained model (loads default if None)
        
    Returns:
        List of (predictions, corner_indices, num_corners) tuples
    """
    if model is None:
        model = load_trained_model()
    
    results = []
    for stroke in strokes:
        result = predict_corners(stroke, model)
        results.append(result)
    
    return results


def main():
    """Demo prediction on a sample stroke."""
    from ml_model.data_loader import load_dataset
    
    print("=" * 60)
    print("Corner Detection Demo")
    print("=" * 60)
    
    # Load a few test strokes
    print("\nLoading test strokes...")
    test_strokes, test_labels = load_dataset(config.TEST_DIR)
    
    # Load model
    print("Loading model...")
    model = load_trained_model()
    
    # Predict on first 5 strokes
    print("\nPredictions on sample strokes:")
    print("-" * 40)
    
    for i in range(min(5, len(test_strokes))):
        stroke = test_strokes[i]
        true_label = test_labels[i]
        
        predictions, corner_indices, num_corners = predict_corners(stroke, model)
        true_corners = np.where(true_label == 1)[0].tolist()
        
        print(f"\nStroke {i + 1}:")
        print(f"  Length: {len(stroke)} points")
        print(f"  Predicted corners: {num_corners} at indices {corner_indices}")
        print(f"  True corners:      {len(true_corners)} at indices {true_corners}")
        
        # Check if predictions match
        if set(corner_indices) == set(true_corners):
            print("  Status: ✓ Perfect match!")
        else:
            print("  Status: Partial match")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
