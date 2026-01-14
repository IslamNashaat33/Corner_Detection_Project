"""
Random Forest model for corner detection.
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from typing import Dict, Any, Optional, List

from . import config


class RandomForestCornerDetector:
    """Random Forest classifier for detecting corners in strokes."""
    
    def __init__(
        self,
        n_estimators: int = config.N_ESTIMATORS,
        max_depth: int = config.MAX_DEPTH,
        min_samples_split: int = config.MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = config.MIN_SAMPLES_LEAF,
        random_state: int = config.RANDOM_STATE,
        n_jobs: int = config.N_JOBS,
        class_weight: Optional[str] = "balanced",
    ):
        """
        Initialize the Random Forest corner detection model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use (-1 for all)
            class_weight: Class weight strategy ('balanced' recommended due to imbalance)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )
        self.is_trained = False
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.selected_feature_names: Optional[List[str]] = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on feature matrix X and labels y.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,)
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def set_selected_features(
        self,
        indices: np.ndarray,
        names: List[str],
    ) -> None:
        """
        Set the selected feature indices and names after feature selection.
        
        Args:
            indices: Array of selected feature indices
            names: List of selected feature names
        """
        self.selected_feature_indices = indices
        self.selected_feature_names = names
    
    def apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection to input features if enabled.
        
        Args:
            X: Full feature matrix
            
        Returns:
            Selected features only (or full X if no selection)
        """
        if self.selected_feature_indices is not None:
            return X[:, self.selected_feature_indices]
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict corner labels for feature matrix X.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict corner probabilities for feature matrix X.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, zero_division=0),
        }
    
    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
    
    def get_oob_score(self) -> float:
        """
        Get out-of-bag score if available.
        
        Returns:
            OOB score or None if not available
        """
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
    def save(self, filepath: str) -> None:
        """
        Save the model and feature selection info to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            "model": self.model,
            "selected_feature_indices": self.selected_feature_indices,
            "selected_feature_names": self.selected_feature_names,
        }
        joblib.dump(save_data, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        save_data = joblib.load(filepath)
        
        # Handle both old format (just model) and new format (dict with metadata)
        if isinstance(save_data, dict):
            self.model = save_data["model"]
            self.selected_feature_indices = save_data.get("selected_feature_indices")
            self.selected_feature_names = save_data.get("selected_feature_names")
        else:
            # Old format: just the model
            self.model = save_data
            self.selected_feature_indices = None
            self.selected_feature_names = None
        
        self.is_trained = True
