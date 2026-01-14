"""
Decision Tree model for corner detection.
"""
import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from typing import Dict, Any, Optional

from . import config


class CornerDetectionModel:
    """Decision Tree classifier for detecting corners in strokes."""
    
    def __init__(
        self,
        max_depth: int = config.MAX_DEPTH,
        min_samples_split: int = config.MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = config.MIN_SAMPLES_LEAF,
        random_state: int = config.RANDOM_STATE,
        class_weight: Optional[str] = "balanced",
    ):
        """
        Initialize the corner detection model.
        
        Args:
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            random_state: Random seed for reproducibility
            class_weight: Class weight strategy ('balanced' recommended due to imbalance)
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on feature matrix X and labels y.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,)
        """
        self.model.fit(X, y)
        self.is_trained = True
    
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
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
