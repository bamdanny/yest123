"""
Feature Importance Discovery Module.

Discovers which features actually predict outcomes using:
- SHAP (SHapley Additive exPlanations) with XGBoost
- Permutation importance
- Mutual information
- Correlation analysis
- Recursive feature elimination

Key principle: Let the data tell us what matters, not intuition.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available. Some features will be disabled.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available. Some features will be disabled.")


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""
    feature_name: str
    shap_importance: float = 0.0
    permutation_importance: float = 0.0
    mutual_information: float = 0.0
    correlation: float = 0.0
    combined_rank: int = 0
    category: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'feature': self.feature_name,
            'shap': self.shap_importance,
            'permutation': self.permutation_importance,
            'mutual_info': self.mutual_information,
            'correlation': self.correlation,
            'combined_rank': self.combined_rank,
            'category': self.category
        }


@dataclass
class DiscoveryReport:
    """Complete feature discovery report."""
    target_variable: str
    n_features_analyzed: int
    top_features: List[FeatureImportanceResult]
    category_importance: Dict[str, float]
    recommended_features: List[str]
    dropped_features: List[str]
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureImportanceDiscovery:
    """
    Discovers which features actually predict the target.
    
    Uses multiple methods and combines them for robust feature selection.
    """
    
    # Feature categories for grouping - matches actual feature naming conventions
    CATEGORY_PREFIXES = {
        'price': ['price_', 'return_', 'atr_', 'vol_', 'ema_', 'sma_', 'bb_', 'rsi_', 'macd_', 'momentum_'],
        'derivatives': ['deriv_', 'funding_', 'oi_', 'ls_ratio_', 'taker_', 'cg_'],
        'liquidation': ['liq_', 'liquidation_'],
        'sentiment': ['sentiment_', 'fear_greed_', 'put_call_', 'iv_', 'fear_', 'greed_'],
        'macro': ['macro_', 'vix_', 'dxy_', 'spy_', 'yield_', 'fed_', 'fin_conditions_', 'yield_curve_'],
        'time': ['time_', 'hour_', 'day_', 'session_', 'is_', 'week_'],
        'interaction': ['AND_', 'divergence_', 'combo_'],
    }
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 500,
        random_state: int = 42
    ):
        """
        Initialize discovery module.
        
        Args:
            n_splits: Number of time series splits for cross-validation
            test_size: Minimum size of test set in each split
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def _get_cv_params(self, n_samples: int) -> tuple:
        """Calculate appropriate CV parameters based on sample size."""
        # TimeSeriesSplit requires: (n_splits + 1) * test_size <= n_samples
        # We want to use about 20% of data for each test fold
        
        # Start with smaller values to be safe
        test_size = min(self.test_size, n_samples // 8)  # Use 1/8 of data for each test set
        test_size = max(test_size, 50)  # At least 50 samples per test
        
        # Calculate max splits possible using the formula
        # (n_splits + 1) * test_size <= n_samples
        # n_splits <= (n_samples / test_size) - 1
        max_splits = (n_samples // test_size) - 1
        n_splits = min(self.n_splits, max_splits)
        n_splits = max(n_splits, 2)  # At least 2 splits
        
        # Double-check the formula holds
        while (n_splits + 1) * test_size > n_samples and n_splits > 2:
            n_splits -= 1
        
        return n_splits, test_size
        
    def discover(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification',
        top_k: int = 50
    ) -> DiscoveryReport:
        """
        Run complete feature importance discovery.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            task: 'classification' or 'regression'
            top_k: Number of top features to return
            
        Returns:
            DiscoveryReport with all findings
        """
        logger.info(f"Starting feature discovery for {y.name} ({task})")
        logger.info(f"Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Clean data
        X_clean, y_clean = self._clean_data(X, y)
        logger.info(f"After cleaning: {X_clean.shape[0]} samples")
        
        if len(X_clean) < 100:
            raise ValueError("Insufficient data for analysis (need at least 100 samples)")
            
        # Run all importance methods
        results = {}
        
        # 1. Correlation (fast, baseline)
        logger.info("Computing correlations...")
        results['correlation'] = self._compute_correlations(X_clean, y_clean)
        
        # 2. Mutual Information
        logger.info("Computing mutual information...")
        results['mutual_info'] = self._compute_mutual_information(
            X_clean, y_clean, task
        )
        
        # 3. Permutation Importance (with RF)
        logger.info("Computing permutation importance...")
        results['permutation'] = self._compute_permutation_importance(
            X_clean, y_clean, task
        )
        
        # 4. SHAP (with XGBoost) - Most important
        if HAS_XGB and HAS_SHAP:
            logger.info("Computing SHAP values...")
            results['shap'] = self._compute_shap_importance(
                X_clean, y_clean, task
            )
        else:
            logger.warning("SHAP analysis skipped (missing dependencies)")
            results['shap'] = pd.Series(0, index=X_clean.columns)
            
        # Combine results
        combined = self._combine_importance_scores(results, X_clean.columns)
        
        # Categorize features
        for feat in combined:
            feat.category = self._categorize_feature(feat.feature_name)
            
        # Sort by combined rank
        combined.sort(key=lambda x: x.combined_rank)
        
        # Calculate category importance
        category_importance = self._calculate_category_importance(combined)
        
        # Determine recommended features
        recommended = [f.feature_name for f in combined[:top_k]]
        dropped = [f.feature_name for f in combined[top_k:]]
        
        report = DiscoveryReport(
            target_variable=y.name,
            n_features_analyzed=len(X_clean.columns),
            top_features=combined[:top_k],
            category_importance=category_importance,
            recommended_features=recommended,
            dropped_features=dropped,
            analysis_metadata={
                'n_samples': len(X_clean),
                'task': task,
                'methods_used': list(results.keys()),
            }
        )
        
        logger.info(f"Discovery complete. Top features: {recommended[:10]}")
        return report
        
    def _clean_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove NaN values and align data."""
        # First, only keep rows where target is valid
        valid_target = ~y.isna()
        X_filtered = X[valid_target].copy()
        y_filtered = y[valid_target].copy()
        
        logger.info(f"Rows with valid target: {len(y_filtered)}")
        
        # Only keep numeric columns (drop strings, dates, objects)
        numeric_cols = X_filtered.select_dtypes(include=[np.number]).columns.tolist()
        X_filtered = X_filtered[numeric_cols]
        
        logger.info(f"Numeric features: {len(X_filtered.columns)}")
        
        # Fill NaN values in features (forward fill, then backward fill, then 0)
        X_filtered = X_filtered.ffill().bfill().fillna(0)
        
        # Replace infinite values
        X_filtered = X_filtered.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        non_constant = X_filtered.columns[X_filtered.std() > 1e-8]
        X_filtered = X_filtered[non_constant]
        
        logger.info(f"Non-constant features: {len(X_filtered.columns)}")
        
        # Remove highly correlated duplicates (keep first)
        # Only compute correlation on numeric columns with variance
        try:
            corr_matrix = X_filtered.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.99)]
            X_filtered = X_filtered.drop(columns=to_drop)
            
            if len(to_drop) > 0:
                logger.info(f"Dropped {len(to_drop)} highly correlated features")
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")
            
        return X_filtered, y_filtered
        
    def _compute_correlations(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.Series:
        """Compute absolute correlations with target."""
        correlations = X.apply(lambda col: col.corr(y))
        return correlations.abs()
        
    def _compute_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str
    ) -> pd.Series:
        """Compute mutual information scores."""
        X_scaled = self.scaler.fit_transform(X)
        
        if task == 'classification':
            # Ensure y is integer for classification
            y_int = y.astype(int) if y.dtype != int else y
            mi_scores = mutual_info_classif(
                X_scaled, y_int,
                random_state=self.random_state,
                n_neighbors=5
            )
        else:
            mi_scores = mutual_info_regression(
                X_scaled, y,
                random_state=self.random_state,
                n_neighbors=5
            )
            
        return pd.Series(mi_scores, index=X.columns)
        
    def _compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str
    ) -> pd.Series:
        """Compute permutation importance using Random Forest."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Use dynamic time series split parameters
        n_splits, test_size = self._get_cv_params(len(X))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        if task == 'classification':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            y_model = y.astype(int)
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            y_model = y
            
        # Get last fold for importance calculation
        for train_idx, test_idx in tscv.split(X_scaled):
            pass  # Get last fold
            
        model.fit(X_scaled[train_idx], y_model.iloc[train_idx])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm_importance = permutation_importance(
                model, X_scaled[test_idx], y_model.iloc[test_idx],
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            
        return pd.Series(
            perm_importance.importances_mean,
            index=X.columns
        )
        
    def _compute_shap_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str
    ) -> pd.Series:
        """Compute SHAP importance using XGBoost."""
        if not HAS_XGB or not HAS_SHAP:
            return pd.Series(0, index=X.columns)
            
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Time series split - use dynamic parameters
        n_splits, test_size = self._get_cv_params(len(X))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            y_model = y.astype(int)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            y_model = y
            
        # Get last fold
        for train_idx, test_idx in tscv.split(X_scaled):
            pass
            
        model.fit(X_scaled.iloc[train_idx], y_model.iloc[train_idx])
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        
        # Use subset for efficiency
        sample_size = min(1000, len(test_idx))
        sample_idx = np.random.choice(test_idx, sample_size, replace=False)
        
        shap_values = explainer.shap_values(X_scaled.iloc[sample_idx])
        
        # Handle classification (might return list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
            
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        return pd.Series(importance, index=X.columns)
        
    def _combine_importance_scores(
        self,
        results: Dict[str, pd.Series],
        feature_names: pd.Index
    ) -> List[FeatureImportanceResult]:
        """Combine different importance methods into final ranking."""
        # Normalize each method to 0-1 range
        normalized = {}
        for method, scores in results.items():
            scores_aligned = scores.reindex(feature_names).fillna(0)
            min_val, max_val = scores_aligned.min(), scores_aligned.max()
            if max_val > min_val:
                normalized[method] = (scores_aligned - min_val) / (max_val - min_val)
            else:
                normalized[method] = scores_aligned * 0
                
        # Weighted combination (SHAP gets highest weight)
        weights = {
            'shap': 0.4,
            'permutation': 0.25,
            'mutual_info': 0.2,
            'correlation': 0.15
        }
        
        combined_scores = pd.Series(0.0, index=feature_names)
        for method, scores in normalized.items():
            weight = weights.get(method, 0.1)
            combined_scores += weight * scores
            
        # Rank by combined score
        ranks = combined_scores.rank(ascending=False)
        
        # Create result objects
        results_list = []
        for feat in feature_names:
            result = FeatureImportanceResult(
                feature_name=feat,
                shap_importance=results.get('shap', pd.Series())[feat] if feat in results.get('shap', pd.Series()) else 0,
                permutation_importance=results.get('permutation', pd.Series())[feat] if feat in results.get('permutation', pd.Series()) else 0,
                mutual_information=results.get('mutual_info', pd.Series())[feat] if feat in results.get('mutual_info', pd.Series()) else 0,
                correlation=results.get('correlation', pd.Series())[feat] if feat in results.get('correlation', pd.Series()) else 0,
                combined_rank=int(ranks[feat])
            )
            results_list.append(result)
            
        return results_list
        
    def _categorize_feature(self, feature_name: str) -> str:
        """Assign feature to a category."""
        feature_lower = feature_name.lower()
        
        for category, prefixes in self.CATEGORY_PREFIXES.items():
            for prefix in prefixes:
                if feature_lower.startswith(prefix):
                    return category
                    
        return 'other'
        
    def _calculate_category_importance(
        self,
        results: List[FeatureImportanceResult]
    ) -> Dict[str, float]:
        """Calculate aggregate importance by category."""
        category_scores = {}
        category_counts = {}
        
        for result in results:
            cat = result.category
            score = 1.0 / (result.combined_rank + 1)  # Inverse rank
            
            if cat not in category_scores:
                category_scores[cat] = 0.0
                category_counts[cat] = 0
                
            category_scores[cat] += score
            category_counts[cat] += 1
            
        # Normalize by count
        category_importance = {}
        for cat in category_scores:
            category_importance[cat] = category_scores[cat] / category_counts[cat]
            
        # Normalize to sum to 1
        total = sum(category_importance.values())
        if total > 0:
            category_importance = {
                k: v / total for k, v in category_importance.items()
            }
            
        return dict(sorted(category_importance.items(), key=lambda x: -x[1]))


class PillarValidator:
    """
    Validates the 4-pillar hypothesis against data.
    
    Tests whether the pillar structure is optimal or if
    alternatives perform better.
    """
    
    CURRENT_PILLARS = {
        'derivatives': ['funding_', 'oi_', 'ls_ratio_'],
        'liquidations': ['liq_'],
        'technical': ['rsi_', 'ema_', 'macd_', 'bb_', 'momentum_'],
        'liquidity': ['taker_', 'orderbook_'],
    }
    
    CURRENT_WEIGHTS = {
        'derivatives': 0.35,
        'liquidations': 0.30,
        'technical': 0.25,
        'liquidity': 0.10,
    }
    
    def __init__(self, discovery_report: DiscoveryReport):
        """Initialize with discovery results."""
        self.report = discovery_report
        
    def validate_pillar_structure(self) -> Dict[str, Any]:
        """
        Compare discovered importance to assumed pillar weights.
        
        Returns analysis of whether pillar structure is supported by data.
        """
        # Map discovered categories to pillars
        category_to_pillar = {
            'derivatives': 'derivatives',
            'liquidation': 'liquidations',
            'price': 'technical',
            'sentiment': 'technical',  # Not in current model
            'macro': None,  # Not in current model
            'time': None,
            'interaction': None,
            'other': None,
        }
        
        # Calculate data-driven pillar importance
        pillar_importance = {p: 0.0 for p in self.CURRENT_PILLARS}
        unmapped_importance = 0.0
        
        for cat, importance in self.report.category_importance.items():
            pillar = category_to_pillar.get(cat)
            if pillar and pillar in pillar_importance:
                pillar_importance[pillar] += importance
            else:
                unmapped_importance += importance
                
        # Compare to assumed weights
        validation_results = {
            'assumed_weights': self.CURRENT_WEIGHTS,
            'data_driven_weights': pillar_importance,
            'unmapped_importance': unmapped_importance,
            'pillar_deviations': {},
            'recommendation': '',
        }
        
        # Calculate deviations
        max_deviation = 0
        for pillar in self.CURRENT_PILLARS:
            assumed = self.CURRENT_WEIGHTS[pillar]
            actual = pillar_importance.get(pillar, 0)
            deviation = actual - assumed
            validation_results['pillar_deviations'][pillar] = {
                'assumed': assumed,
                'actual': actual,
                'deviation': deviation,
                'deviation_pct': deviation / assumed * 100 if assumed > 0 else 0
            }
            max_deviation = max(max_deviation, abs(deviation))
            
        # Generate recommendation
        if unmapped_importance > 0.2:
            validation_results['recommendation'] = (
                f"RESTRUCTURE: {unmapped_importance:.1%} of predictive power comes from "
                "features not in the current pillar structure. Consider adding new pillars "
                "for macro, sentiment, or time features."
            )
        elif max_deviation > 0.15:
            validation_results['recommendation'] = (
                "REWEIGHT: Current pillar weights significantly differ from data-driven "
                "importance. Consider optimizing weights based on discovered importance."
            )
        else:
            validation_results['recommendation'] = (
                "VALIDATE: Current pillar structure is roughly consistent with data. "
                "Minor weight adjustments may improve performance."
            )
            
        return validation_results
        
    def suggest_alternative_structures(self) -> List[Dict[str, Any]]:
        """
        Suggest alternative grouping structures based on discovered importance.
        """
        alternatives = []
        
        # Alternative 1: Merge low-importance pillars
        if self.report.category_importance.get('liquidity', 0) < 0.05:
            alternatives.append({
                'name': '3-Pillar (Merge Liquidity)',
                'structure': {
                    'derivatives': ['derivatives', 'liquidity'],
                    'liquidations': ['liquidation'],
                    'technical': ['price', 'sentiment'],
                },
                'rationale': 'Liquidity shows <5% importance, merge into derivatives'
            })
            
        # Alternative 2: Add macro pillar
        if self.report.category_importance.get('macro', 0) > 0.1:
            alternatives.append({
                'name': '5-Pillar (Add Macro)',
                'structure': {
                    'derivatives': ['derivatives'],
                    'liquidations': ['liquidation'],
                    'technical': ['price'],
                    'sentiment': ['sentiment'],
                    'macro': ['macro'],
                },
                'rationale': 'Macro shows >10% importance, deserves own pillar'
            })
            
        # Alternative 3: Flat feature list (no pillars)
        alternatives.append({
            'name': 'No Pillars (Feature List)',
            'structure': None,
            'rationale': 'Use top-N features directly without pillar abstraction'
        })
        
        return alternatives


def run_full_feature_discovery(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    target_column: str = 'profitable_24h',
    top_k: int = 50
) -> Tuple[DiscoveryReport, Dict[str, Any]]:
    """
    Convenience function to run complete feature discovery pipeline.
    
    Args:
        features_df: DataFrame with all features
        targets_df: DataFrame with target variables
        target_column: Which target to optimize for
        top_k: Number of features to keep
        
    Returns:
        Tuple of (DiscoveryReport, pillar_validation)
    """
    # Determine task type
    target = targets_df[target_column]
    unique_vals = target.nunique()
    task = 'classification' if unique_vals <= 10 else 'regression'
    
    # Run discovery
    discovery = FeatureImportanceDiscovery()
    report = discovery.discover(features_df, target, task=task, top_k=top_k)
    
    # Validate pillars
    validator = PillarValidator(report)
    pillar_validation = validator.validate_pillar_structure()
    
    return report, pillar_validation
