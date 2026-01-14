"""
Training script for the Random Forest corner detection model.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rf_model import config
from rf_model.data_loader import load_dataset, get_dataset_info
from rf_model.feature_engineering import extract_features_batch, FEATURE_NAMES
from rf_model.feature_selection import FeatureSelector
from rf_model.model import RandomForestCornerDetector


def main():
    print("=" * 60)
    print("Random Forest Corner Detection Model Training")
    print("=" * 60)
    
    # Load training data
    print("\n[1/6] Loading training data...")
    train_strokes, train_labels = load_dataset(config.TRAIN_DIR)
    train_info = get_dataset_info(train_strokes, train_labels)
    print(f"  Loaded {train_info['num_strokes']} strokes")
    print(f"  Total points: {train_info['total_points']}")
    print(f"  Total corners: {train_info['total_corners']}")
    print(f"  Corner ratio: {train_info['corner_ratio']:.4f}")
    
    # Load validation data
    print("\n[2/6] Loading validation data...")
    val_strokes, val_labels = load_dataset(config.VAL_DIR)
    val_info = get_dataset_info(val_strokes, val_labels)
    print(f"  Loaded {val_info['num_strokes']} strokes")
    print(f"  Total points: {val_info['total_points']}")
    print(f"  Total corners: {val_info['total_corners']}")
    
    # Extract features
    print("\n[3/6] Extracting features...")
    X_train, y_train = extract_features_batch(
        train_strokes, train_labels, config.WINDOW_SIZE
    )
    X_val, y_val = extract_features_batch(
        val_strokes, val_labels, config.WINDOW_SIZE
    )
    print(f"  Training features shape: {X_train.shape}")
    print(f"  Validation features shape: {X_val.shape}")
    
    # Feature selection
    selected_indices = None
    selected_names = FEATURE_NAMES
    
    if config.FEATURE_SELECTION_ENABLED:
        print("\n[4/6] Performing feature selection...")
        print(f"  Threshold: {config.FEATURE_SELECTION_THRESHOLD}")
        print(f"  Max features: {config.FEATURE_SELECTION_MAX_FEATURES or 'unlimited'}")
        
        selector = FeatureSelector(
            threshold=config.FEATURE_SELECTION_THRESHOLD,
            max_features=config.FEATURE_SELECTION_MAX_FEATURES,
        )
        
        # Fit selector and transform training data
        X_train_selected = selector.fit_transform(X_train, y_train, FEATURE_NAMES)
        X_val_selected = selector.transform(X_val)
        
        selected_indices = selector.selected_indices
        selected_names = selector.selected_names
        
        print(f"\n  Selected {len(selected_names)} of {len(FEATURE_NAMES)} features:")
        for name in selected_names:
            print(f"    ✓ {name}")
        
        # Use selected features
        X_train = X_train_selected
        X_val = X_val_selected
        
        print(f"\n  New training features shape: {X_train.shape}")
        print(f"  New validation features shape: {X_val.shape}")
    else:
        print("\n[4/6] Feature selection: DISABLED")
    
    # Train model
    print("\n[5/6] Training Random Forest model...")
    print(f"  n_estimators: {config.N_ESTIMATORS}")
    print(f"  max_depth: {config.MAX_DEPTH}")
    print(f"  n_jobs: {config.N_JOBS}")
    
    model = RandomForestCornerDetector(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )
    
    # Store feature selection info in model
    if selected_indices is not None:
        model.set_selected_features(selected_indices, selected_names)
    
    model.train(X_train, y_train)
    print("  Training complete!")
    
    # Evaluate on training data
    print("\n[6/6] Evaluating model...")
    print("\n--- Training Set Performance ---")
    train_metrics = model.evaluate(X_train, y_train)
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1_score']:.4f}")
    
    # Evaluate on validation data
    print("\n--- Validation Set Performance ---")
    val_metrics = model.evaluate(X_val, y_val)
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1_score']:.4f}")
    
    print("\n--- Confusion Matrix (Validation) ---")
    cm = val_metrics['confusion_matrix']
    print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
    
    # Feature importance (using selected feature names)
    print("\n--- Feature Importance ---")
    importance = model.get_feature_importance(selected_names)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance:
        print(f"  {name:20s}: {score:.4f}")
    
    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, "random_forest_model.joblib")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
