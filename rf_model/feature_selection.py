"""
Feature selection utilities for corner detection models.
"""
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Optional


class FeatureSelector:
    """Wrapper for feature selection based on model importances."""
    
    def __init__(
        self,
        threshold: str = "mean",
        max_features: Optional[int] = None,
    ):
        """
        Initialize the feature selector.
        
        Args:
            threshold: Threshold for feature selection. Can be:
                - "mean": Select features with importance >= mean importance
                - "median": Select features with importance >= median importance
                - float: Select features with importance >= threshold
            max_features: Maximum number of features to select (optional)
        """
        self.threshold = threshold
        self.max_features = max_features
        self.selector = None
        self.selected_indices = None
        self.selected_names = None
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        estimator: Optional[RandomForestClassifier] = None,
    ) -> "FeatureSelector":
        """
        Fit the feature selector using a model's feature importances.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            estimator: Pre-fitted estimator with feature_importances_ attribute.
                       If None, a RandomForestClassifier is trained.
        
        Returns:
            self
        """
        if estimator is None:
            # Train a quick RF to get importances
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            estimator.fit(X, y)
        
        # Create selector
        self.selector = SelectFromModel(
            estimator,
            threshold=self.threshold,
            max_features=self.max_features,
            prefit=True,
        )
        
        # Get selected feature indices and names
        self.selected_indices = self.selector.get_support(indices=True)
        self.selected_names = [feature_names[i] for i in self.selected_indices]
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix to selected features only.
        
        Args:
            X: Feature matrix
            
        Returns:
            Reduced feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        return self.selector.transform(X)
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        estimator: Optional[RandomForestClassifier] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names, estimator)
        return self.transform(X)
    
    def get_feature_report(self, importances: np.ndarray, feature_names: List[str]) -> str:
        """
        Generate a report of selected vs excluded features.
        
        Args:
            importances: Feature importance scores
            feature_names: List of feature names
            
        Returns:
            Formatted report string
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted first")
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        lines = ["Feature Selection Report:", "-" * 40]
        
        for idx in sorted_idx:
            name = feature_names[idx]
            imp = importances[idx]
            status = "✓ SELECTED" if idx in self.selected_indices else "✗ excluded"
            lines.append(f"  {name:20s}: {imp:.4f}  {status}")
        
        lines.append("-" * 40)
        lines.append(f"Selected {len(self.selected_indices)} of {len(feature_names)} features")
        
        return "\n".join(lines)
    
    def get_state(self) -> dict:
        """Get selector state for serialization."""
        return {
            "threshold": self.threshold,
            "max_features": self.max_features,
            "selected_indices": self.selected_indices.tolist() if self.selected_indices is not None else None,
            "selected_names": self.selected_names,
        }
    
    def set_state(self, state: dict) -> None:
        """Restore selector state from serialization."""
        self.threshold = state["threshold"]
        self.max_features = state["max_features"]
        self.selected_indices = np.array(state["selected_indices"]) if state["selected_indices"] else None
        self.selected_names = state["selected_names"]
        self.is_fitted = self.selected_indices is not None
        # Note: selector object not restored, but we can transform using indices directly
    
    def transform_by_indices(self, X: np.ndarray) -> np.ndarray:
        """Transform using stored indices (for loaded models)."""
        if self.selected_indices is None:
            raise RuntimeError("No selected indices available")
        return X[:, self.selected_indices]
