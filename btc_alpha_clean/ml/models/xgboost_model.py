"""
XGBoost Model Implementation

XGBoost is excellent for tabular data because:
1. Handles missing values natively
2. Built-in regularization (prevents overfitting)
3. Feature importance built-in
4. Fast training
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier for direction prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        
        # Conservative defaults to prevent overfitting
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,              # Shallow trees = less overfitting
            'learning_rate': 0.05,       # Slow learning = more robust
            'n_estimators': 500,         # Will early stop before this
            'min_child_weight': 10,      # Require more samples per leaf
            'subsample': 0.8,            # Row sampling
            'colsample_bytree': 0.8,     # Column sampling
            'reg_alpha': 0.1,            # L1 regularization
            'reg_lambda': 1.0,           # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'verbosity': 0,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name='xgboost', params=default_params)
        self.model: Optional[xgb.XGBClassifier] = None
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'XGBoostModel':
        """Train the XGBoost model."""
        
        logger.info(f"Training XGBoost on {len(X)} samples, {len(X.columns)} features")
        
        self.feature_names = list(X.columns)
        
        # Extract early stopping param
        params = self.params.copy()
        early_stopping = params.pop('early_stopping_rounds', 50)
        
        self.model = xgb.XGBClassifier(**params)
        
        # Fit with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X.fillna(0), y,
                eval_set=[(X_val.fillna(0), y_val)],
                verbose=False
            )
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X.fillna(0), y, verbose=False)
        
        self.is_fitted = True
        self.params['early_stopping_rounds'] = early_stopping
        
        logger.info(f"XGBoost training complete")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X.fillna(0))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
