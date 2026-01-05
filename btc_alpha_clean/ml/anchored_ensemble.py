"""
Anchored Ensemble Model

Architecture:
  - Anchor Model: Uses only OOS-proven features (70% weight)
  - Refinement Model: Uses all features (30% weight)
  
This ensures we never lose the alpha that Phase 1 already validated.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)


class AnchoredEnsemble:
    """
    Two-stage ensemble that respects Phase 1 validation results.
    
    Stage 1 (Anchor): Simple model using ONLY proven features
    Stage 2 (Refinement): Complex model using ALL features
    
    Final prediction = anchor_weight * anchor_pred + (1 - anchor_weight) * refine_pred
    """
    
    def __init__(self,
                 anchor_weight: float = 0.7,
                 proven_features: Dict[str, float] = None,
                 min_oos_sharpe: float = 3.0,
                 config_path: str = None):
        """
        Args:
            anchor_weight: Weight for anchor model (0.0-1.0)
            proven_features: Dict of {feature_name: oos_sharpe}
            min_oos_sharpe: Minimum OOS Sharpe to be "proven"
            config_path: Path to phase1_rules.json
        """
        self.anchor_weight = anchor_weight
        self.min_oos_sharpe = min_oos_sharpe
        
        # Load proven features
        if proven_features:
            self.proven_features = proven_features
        elif config_path:
            self.proven_features = self._load_from_config(config_path)
        else:
            # Try default path
            default_path = Path(__file__).parent.parent / "config" / "phase1_rules.json"
            if default_path.exists():
                self.proven_features = self._load_from_config(str(default_path))
            else:
                self.proven_features = {}
        
        # Filter to features meeting threshold
        self.anchor_features = [
            f for f, sharpe in self.proven_features.items()
            if sharpe >= min_oos_sharpe
        ]
        
        # Models
        self.anchor_model = LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        )
        
        if HAS_LGBM:
            self.refine_model = LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=16,
                min_child_samples=20,
                class_weight='balanced',
                verbose=-1
            )
        else:
            self.refine_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05
            )
        
        # Scalers
        self.anchor_scaler = StandardScaler()
        self.refine_scaler = StandardScaler()
        
        # State
        self.available_features = []
        self.all_features = []
        self.fitted = False
    
    def _load_from_config(self, path: str) -> Dict[str, float]:
        """Load proven features from config file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return config.get('feature_weights', {})
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Fit both models.
        
        IMPORTANT: y must be BINARY (0/1), not continuous returns!
        """
        # Validate binary target
        unique = np.unique(y.dropna())
        if not set(unique).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Target must be binary (0/1), got values: {unique[:10]}. "
                "Use: y = (returns > 0).astype(int)"
            )
        
        logger.info("=" * 60)
        logger.info("TRAINING ANCHORED ENSEMBLE")
        logger.info("=" * 60)
        
        # Find available anchor features
        self.available_features = [f for f in self.anchor_features if f in X.columns]
        self.all_features = list(X.columns)
        
        logger.info(f"Anchor features: {len(self.available_features)}/{len(self.anchor_features)}")
        for f in self.available_features:
            sharpe = self.proven_features.get(f, 0)
            logger.info(f"  ✓ {f[:50]}: OOS Sharpe {sharpe:.2f}")
        
        if len(self.available_features) < 2:
            logger.warning("⚠️ Fewer than 2 anchor features available!")
        
        # Prepare data
        X_anchor = X[self.available_features].fillna(0)
        X_anchor_scaled = self.anchor_scaler.fit_transform(X_anchor)
        
        X_refine = X.fillna(0)
        X_refine_scaled = self.refine_scaler.fit_transform(X_refine)
        
        y_clean = y.fillna(0).astype(int).values
        
        # Train anchor
        logger.info("")
        logger.info("Training anchor model...")
        self.anchor_model.fit(X_anchor_scaled, y_clean)
        anchor_acc = np.mean(self.anchor_model.predict(X_anchor_scaled) == y_clean)
        logger.info(f"  Train accuracy: {anchor_acc:.1%}")
        
        # Train refinement
        logger.info("")
        logger.info("Training refinement model...")
        if X_val is not None and y_val is not None and HAS_LGBM:
            X_val_scaled = self.refine_scaler.transform(X_val.fillna(0))
            y_val_clean = y_val.fillna(0).astype(int).values
            self.refine_model.fit(
                X_refine_scaled, y_clean,
                eval_set=[(X_val_scaled, y_val_clean)]
            )
        else:
            self.refine_model.fit(X_refine_scaled, y_clean)
        
        refine_acc = np.mean(self.refine_model.predict(X_refine_scaled) == y_clean)
        logger.info(f"  Train accuracy: {refine_acc:.1%}")
        
        # Combined
        anchor_proba = self.anchor_model.predict_proba(X_anchor_scaled)[:, 1]
        refine_proba = self.refine_model.predict_proba(X_refine_scaled)[:, 1]
        combined = self.anchor_weight * anchor_proba + (1 - self.anchor_weight) * refine_proba
        combined_acc = np.mean((combined > 0.5).astype(int) == y_clean)
        
        logger.info("")
        logger.info(f"Combined train accuracy: {combined_acc:.1%}")
        logger.info(f"Weights: Anchor {self.anchor_weight:.0%}, Refine {1-self.anchor_weight:.0%}")
        
        self.fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.fitted:
            raise ValueError("Must fit() first")
        
        X_anchor = X[self.available_features].fillna(0)
        X_anchor_scaled = self.anchor_scaler.transform(X_anchor)
        
        X_refine = X.fillna(0)
        X_refine_scaled = self.refine_scaler.transform(X_refine)
        
        anchor_proba = self.anchor_model.predict_proba(X_anchor_scaled)[:, 1]
        refine_proba = self.refine_model.predict_proba(X_refine_scaled)[:, 1]
        
        combined = self.anchor_weight * anchor_proba + (1 - self.anchor_weight) * refine_proba
        
        return np.column_stack([1 - combined, combined])
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > threshold).astype(int)
    
    def get_signals(self, X: pd.DataFrame, 
                    long_threshold: float = 0.55,
                    short_threshold: float = 0.45) -> np.ndarray:
        """
        Get trading signals.
        
        Returns: 1 (long), -1 (short), 0 (no trade)
        """
        proba = self.predict_proba(X)[:, 1]
        
        signals = np.zeros(len(proba))
        signals[proba > long_threshold] = 1
        signals[proba < short_threshold] = -1
        
        return signals
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances from both models."""
        importances = []
        
        # Anchor model coefficients
        for i, feat in enumerate(self.available_features):
            importances.append({
                'feature': feat,
                'model': 'anchor',
                'importance': abs(self.anchor_model.coef_[0][i]),
                'oos_sharpe': self.proven_features.get(feat, 0)
            })
        
        # Refinement model importances
        if hasattr(self.refine_model, 'feature_importances_'):
            for i, feat in enumerate(self.all_features):
                importances.append({
                    'feature': feat,
                    'model': 'refine',
                    'importance': self.refine_model.feature_importances_[i],
                    'oos_sharpe': self.proven_features.get(feat, 0)
                })
        
        return pd.DataFrame(importances).sort_values('importance', ascending=False)
