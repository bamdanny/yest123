"""
Ensemble Model - Combine multiple models for robustness

Why ensemble:
1. Reduces variance (less overfitting)
2. Different models capture different patterns
3. More stable predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models.
    
    Combines predictions by weighted averaging of probabilities.
    Weights are determined by validation performance.
    """
    
    def __init__(self, models: List[BaseModel] = None):
        super().__init__(name='ensemble')
        
        # Default: Try XGBoost + LightGBM, fall back to what's available
        if models is None:
            self.models = []
            
            try:
                from .xgboost_model import XGBoostModel
                self.models.append(XGBoostModel())
            except ImportError:
                logger.warning("XGBoost not available")
            
            try:
                from .lightgbm_model import LightGBMModel
                self.models.append(LightGBMModel())
            except ImportError:
                logger.warning("LightGBM not available")
            
            if not self.models:
                raise ImportError("No ML models available. Install xgboost or lightgbm.")
        else:
            self.models = models
        
        self.weights: List[float] = []
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'EnsembleModel':
        """Train all models in the ensemble."""
        
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        self.feature_names = list(X.columns)
        val_scores = []
        
        for model in self.models:
            logger.info(f"\nTraining {model.name}...")
            model.fit(X, y, X_val, y_val)
            
            # Calculate validation score for weighting
            if X_val is not None and y_val is not None:
                try:
                    proba = model.predict_proba(X_val)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, proba)
                    val_scores.append(score)
                    logger.info(f"  {model.name} validation AUC: {score:.4f}")
                except Exception as e:
                    logger.warning(f"  Could not calculate AUC: {e}")
                    val_scores.append(0.5)
            else:
                val_scores.append(1.0)
        
        # Weight by validation performance
        # Use softmax-style weighting to emphasize better models
        if all(s > 0 for s in val_scores):
            # Subtract 0.5 (random baseline) and exponentiate
            adjusted = [max(0.01, (s - 0.5) * 10) for s in val_scores]
            total = sum(adjusted)
            self.weights = [a / total for a in adjusted]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(f"\nEnsemble weights: {dict(zip([m.name for m in self.models], self.weights))}")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of model probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        probas = []
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            probas.append(proba * weight)
        
        return np.sum(probas, axis=0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Average feature importance across models."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        combined = {}
        for model, weight in zip(self.models, self.weights):
            importance = model.get_feature_importance()
            for feat, imp in importance.items():
                combined[feat] = combined.get(feat, 0) + imp * weight
        
        return combined
