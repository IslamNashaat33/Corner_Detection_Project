"""
Evaluation script for the corner detection model on test data.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DT_model import config
from DT_model.data_loader import load_dataset, get_dataset_info
from DT_model.feature_engineering import extract_features_batch, FEATURE_NAMES
from DT_model.model import CornerDetectionModel


def main():
    print("=" * 60)
    print("Corner Detection Model Evaluation (Test Set)")
    print("=" * 60)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    test_strokes, test_labels = load_dataset(config.TEST_DIR)
    test_info = get_dataset_info(test_strokes, test_labels)
    print(f"  Loaded {test_info['num_strokes']} strokes")
    print(f"  Total points: {test_info['total_points']}")
    print(f"  Total corners: {test_info['total_corners']}")
    print(f"  Corner ratio: {test_info['corner_ratio']:.4f}")
    
    # Extract features
    print("\n[2/3] Extracting features...")
    X_test, y_test = extract_features_batch(
        test_strokes, test_labels, config.WINDOW_SIZE
    )
    print(f"  Test features shape: {X_test.shape}")
    
    # Load model
    print("\n[3/3] Loading and evaluating model...")
    model_path = os.path.join(config.MODEL_DIR, "decision_tree_model.joblib")
    
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        print("  Please run train.py first to train the model.")
        return
    
    model = CornerDetectionModel()
    model.load(model_path)
    print(f"  Model loaded from: {model_path}")
    
    # Evaluate
    print("\n--- Test Set Performance ---")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = test_metrics['confusion_matrix']
    print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
    
    print("\n--- Classification Report ---")
    print(test_metrics['classification_report'])
    
    # Feature importance
    print("--- Feature Importance ---")
    importance = model.get_feature_importance(FEATURE_NAMES)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance:
        print(f"  {name:20s}: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
