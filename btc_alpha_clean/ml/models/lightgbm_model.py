"""
LightGBM Model Implementation

LightGBM is often faster than XGBoost and can be more accurate.
Good to have both for ensemble.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier for direction prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        
        # Conservative defaults
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 16,            # Shallow = less overfitting
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name='lightgbm', params=default_params)
        self.model: Optional[lgb.LGBMClassifier] = None
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LightGBMModel':
        """Train the LightGBM model."""
        
        logger.info(f"Training LightGBM on {len(X)} samples, {len(X.columns)} features")
        
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        
        # Fit with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X.fillna(0), y,
                eval_set=[(X_val.fillna(0), y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            if hasattr(self.model, 'best_iteration_'):
                logger.info(f"Best iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X.fillna(0), y)
        
        self.is_fitted = True
        logger.info(f"LightGBM training complete")
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
