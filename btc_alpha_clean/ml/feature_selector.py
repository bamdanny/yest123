"""
Feature Selection - Reduce features using Phase 1 insights + statistical methods

Key principle: Prioritize features that PASSED OOS validation in Phase 1.

Phase 1 Results:
- Tier 1 (OOS Sharpe > 5): oi_close_change_1h, liq_ratio, oi_close_accel
- Tier 2 (Sharpe 3-5): Various OI and funding features
- Tier 3 (Sharpe 1-3): Weak but valid signals
- Blacklist: VIX features and others that failed OOS
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import json
import logging
import warnings

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Multi-stage feature selection with Phase 1 priority weighting.
    
    Stage 1: Apply Phase 1 priority weights
    Stage 2: Remove low-variance features
    Stage 3: Remove highly correlated features (keep higher priority)
    Stage 4: Select top features by importance + priority
    """
    
    def __init__(
        self,
        max_features: int = 50,
        variance_threshold: float = 0.001,
        correlation_threshold: float = 0.85,
        config_path: str = None,
        priority_path: str = None
    ):
        self.max_features = max_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Load feature priority from Phase 1 results
        self.priority = None
        self.priority_weights = {
            'tier1': 3.0,
            'tier2': 2.0,
            'tier3': 1.0,
            'other': 0.5,
            'blacklist': 0.1
        }
        
        # Try to load priority config
        priority_paths = [
            priority_path,
            "config/feature_priority.json",
            Path(__file__).parent.parent / "config" / "feature_priority.json"
        ]
        
        for p in priority_paths:
            if p and Path(p).exists():
                with open(p) as f:
                    self.priority = json.load(f)
                self.priority_weights = self.priority.get('feature_weights', self.priority_weights)
                logger.info(f"Loaded feature priority from {p}")
                break
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature DataFrame (training data only!)
            y: Target Series
        """
        logger.info(f"Feature selection starting with {len(X.columns)} features")
        
        # Drop any columns that are all NaN
        valid_cols = X.columns[X.notna().any()].tolist()
        X = X[valid_cols]
        logger.info(f"After dropping all-NaN columns: {len(X.columns)} features")
        
        # Get priority scores from Phase 1 results
        priority_scores = self._get_priority_scores(X.columns.tolist())
        
        # Stage 1: Remove low variance
        features = self._remove_low_variance(X)
        logger.info(f"After variance filter: {len(features)} features")
        
        if len(features) == 0:
            logger.warning("No features passed variance filter! Using all features.")
            features = list(X.columns)
        
        # Stage 2: Remove highly correlated (keep higher priority)
        features = self._remove_correlated(X[features], priority_scores)
        logger.info(f"After correlation filter: {len(features)} features")
        
        # Stage 3: Select top by combined importance + priority
        features, importance = self._select_by_combined_score(X[features], y, priority_scores)
        logger.info(f"After importance selection: {len(features)} features")
        
        self.selected_features = features
        self.feature_importance = importance
        
        # Log selection by tier
        self._log_selection_by_tier()
        
        return self
    
    def _get_priority_scores(self, feature_names: List[str]) -> Dict[str, float]:
        """Assign priority scores based on Phase 1 OOS results."""
        scores = {}
        
        if self.priority is None:
            # No priority config - all features equal
            return {f: 1.0 for f in feature_names}
        
        tier1 = set(self.priority.get('tier1_features', {}).get('features', []))
        tier2 = set(self.priority.get('tier2_features', {}).get('features', []))
        tier3 = set(self.priority.get('tier3_features', {}).get('features', []))
        blacklist = set(self.priority.get('blacklist_features', {}).get('features', []))
        
        for feat in feature_names:
            if feat in tier1:
                scores[feat] = self.priority_weights['tier1']
            elif feat in tier2:
                scores[feat] = self.priority_weights['tier2']
            elif feat in tier3:
                scores[feat] = self.priority_weights['tier3']
            elif feat in blacklist:
                scores[feat] = self.priority_weights['blacklist']
            else:
                scores[feat] = self.priority_weights['other']
        
        return scores
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to selected subset."""
        if not self.selected_features:
            raise ValueError("Must call fit() first")
        
        # Handle case where some features might not exist
        available = [f for f in self.selected_features if f in X.columns]
        if len(available) < len(self.selected_features):
            missing = set(self.selected_features) - set(available)
            logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
        
        return X[available]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _remove_low_variance(self, X: pd.DataFrame) -> List[str]:
        """Remove features with variance below threshold."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Convert all columns to numeric (handles boolean columns)
            X_numeric = X.apply(pd.to_numeric, errors='coerce')
            
            # Fill NaN and compute range
            X_filled = X_numeric.fillna(0)
            
            # Handle any remaining non-numeric columns
            numeric_cols = X_filled.select_dtypes(include=[np.number]).columns
            X_filled = X_filled[numeric_cols]
            
            if len(X_filled.columns) == 0:
                logger.warning("No numeric columns found for variance filter")
                return list(X.columns)
            
            X_range = X_filled.max() - X_filled.min()
            
            # Avoid division by zero
            X_range = X_range.replace(0, 1e-10)
            X_norm = (X_filled - X_filled.min()) / X_range
            
            variances = X_norm.var()
        
        selected = variances[variances > self.variance_threshold].index.tolist()
        removed = len(X_filled.columns) - len(selected)
        
        if removed > 0:
            logger.info(f"  Removed {removed} low-variance features")
        
        return selected
    
    def _remove_correlated(self, X: pd.DataFrame, priority_scores: Dict[str, float]) -> List[str]:
        """Remove highly correlated features, keeping higher priority ones."""
        if len(X.columns) <= 1:
            return list(X.columns)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Convert to numeric to handle boolean columns
            X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            corr_matrix = X_numeric.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
            for corr_feat in correlated:
                # Keep the one with higher priority score
                if priority_scores.get(col, 0) >= priority_scores.get(corr_feat, 0):
                    to_drop.add(corr_feat)
                else:
                    to_drop.add(col)
        
        selected = [c for c in X.columns if c not in to_drop]
        
        if to_drop:
            logger.info(f"  Removed {len(to_drop)} correlated features (kept higher priority)")
        
        return selected
    
    def _select_by_combined_score(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        priority_scores: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select top features by combined importance + Phase 1 priority score."""
        
        # Convert to numeric to handle boolean columns
        X_filled = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Align y with X
        common_idx = X_filled.index.intersection(y.index)
        X_aligned = X_filled.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Convert y to int if it's boolean
        if y_aligned.dtype == bool:
            y_aligned = y_aligned.astype(int)
        
        if len(X_aligned.columns) <= self.max_features:
            # If we have fewer features than max, keep all
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_aligned, y_aligned)
            importance = dict(zip(X_aligned.columns, rf.feature_importances_))
            return list(X_aligned.columns), importance
        
        # Method 1: Mutual Information
        try:
            logger.info("  Calculating mutual information...")
            mi_scores = mutual_info_classif(X_aligned, y_aligned, random_state=42)
            mi_importance = dict(zip(X_aligned.columns, mi_scores))
        except Exception as e:
            logger.warning(f"Mutual information failed: {e}")
            mi_importance = {c: 1.0 for c in X_aligned.columns}
        
        # Method 2: Random Forest importance
        logger.info("  Training Random Forest for importance...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_aligned, y_aligned)
        rf_importance = dict(zip(X_aligned.columns, rf.feature_importances_))
        
        # Normalize all scores to 0-1
        mi_max = max(mi_importance.values()) or 1
        rf_max = max(rf_importance.values()) or 1
        priority_max = max(priority_scores.values()) or 1
        
        # Combined score: 40% MI + 40% RF + 20% Phase 1 priority
        combined = {}
        for feat in X_aligned.columns:
            mi_norm = mi_importance[feat] / mi_max
            rf_norm = rf_importance[feat] / rf_max
            priority_norm = priority_scores.get(feat, 0.5) / priority_max
            
            combined[feat] = 0.4 * mi_norm + 0.4 * rf_norm + 0.2 * priority_norm
        
        # Select top features
        selected = sorted(combined.keys(), key=lambda x: -combined[x])[:self.max_features]
        
        # Return combined scores
        importance = {feat: combined[feat] for feat in selected}
        
        return selected, importance
    
    def _log_selection_by_tier(self):
        """Log how many features were selected from each tier."""
        if self.priority is None:
            return
        
        tier1 = set(self.priority.get('tier1_features', {}).get('features', []))
        tier2 = set(self.priority.get('tier2_features', {}).get('features', []))
        tier3 = set(self.priority.get('tier3_features', {}).get('features', []))
        blacklist = set(self.priority.get('blacklist_features', {}).get('features', []))
        
        selected_set = set(self.selected_features)
        
        t1_selected = len(selected_set & tier1)
        t2_selected = len(selected_set & tier2)
        t3_selected = len(selected_set & tier3)
        bl_selected = len(selected_set & blacklist)
        other = len(selected_set) - t1_selected - t2_selected - t3_selected - bl_selected
        
        logger.info("\nSelected features by tier:")
        logger.info(f"  Tier 1 (OOS proven):  {t1_selected}/{len(tier1)}")
        logger.info(f"  Tier 2 (moderate):    {t2_selected}/{len(tier2)}")
        logger.info(f"  Tier 3 (weak):        {t3_selected}/{len(tier3)}")
        logger.info(f"  Blacklist:            {bl_selected}/{len(blacklist)}")
        logger.info(f"  Other:                {other}")
        
        # Log top 20
        logger.info("\nTop 20 selected features:")
        for i, feat in enumerate(self.selected_features[:20]):
            score = self.feature_importance.get(feat, 0)
            tier = self._get_tier(feat)
            logger.info(f"  {i+1:2d}. [{tier}] {feat[:50]:50s} {score:.4f}")
    
    def _get_tier(self, feature: str) -> str:
        """Get tier label for feature."""
        if self.priority is None:
            return "  "
        
        if feature in self.priority.get('tier1_features', {}).get('features', []):
            return "T1"
        elif feature in self.priority.get('tier2_features', {}).get('features', []):
            return "T2"
        elif feature in self.priority.get('tier3_features', {}).get('features', []):
            return "T3"
        elif feature in self.priority.get('blacklist_features', {}).get('features', []):
            return "BL"
        else:
            return "  "
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        return pd.DataFrame([
            {
                'feature': k,
                'importance': v,
                'tier': self._get_tier(k)
            }
            for k, v in sorted(self.feature_importance.items(), key=lambda x: -x[1])
        ])
