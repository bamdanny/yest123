"""
Base Model Interface

All models must implement this interface for consistency.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.is_fitted = False
        self.feature_names: list = []
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Returns:
            Array of shape (n_samples, 2) with [prob_down, prob_up]
        """
        pass
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        return joblib.load(path)
