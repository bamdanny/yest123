"""
Structure Discovery and Weight/Threshold Optimization.

Phase 5: Discovers optimal grouping structure (pillars vs flat)
Phase 6: Optimizes all thresholds and weights from scratch

Key insight: The 4-pillar structure with 35/30/25/10 weights is an
ASSUMPTION. We test whether it's optimal or if alternatives work better.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not available. Using scipy optimization instead.")


@dataclass
class ThresholdConfig:
    """Configuration for a threshold parameter."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step: float = 0.01
    discovered_value: Optional[float] = None
    improvement_pct: Optional[float] = None


@dataclass
class StructureConfig:
    """Configuration for a pillar/grouping structure."""
    name: str
    groups: Dict[str, List[str]]  # group_name -> list of feature prefixes
    weights: Dict[str, float]  # group_name -> weight
    score: float = 0.0


@dataclass
class OptimizationResult:
    """Results from optimization."""
    best_thresholds: Dict[str, float]
    best_weights: Dict[str, float]
    best_structure: Optional[StructureConfig]
    baseline_score: float
    optimized_score: float
    improvement_pct: float
    iterations: int
    convergence_history: List[float] = field(default_factory=list)


class ThresholdOptimizer:
    """
    Discovers optimal thresholds from data.
    
    Tests thresholds that are currently hardcoded assumptions:
    - Pillar bullish/bearish thresholds (55/45)
    - Confidence threshold (70%)
    - RSI extreme levels (30/70)
    - VIX regime breaks (15/20/30)
    - Funding rate extreme z-scores
    - And more...
    """
    
    # Current arbitrary thresholds to optimize
    THRESHOLDS_TO_OPTIMIZE = {
        'pillar_bullish': ThresholdConfig('pillar_bullish', 55, 50, 80, 1),
        'pillar_bearish': ThresholdConfig('pillar_bearish', 45, 20, 50, 1),
        'confidence_threshold': ThresholdConfig('confidence', 0.70, 0.50, 0.95, 0.05),
        'rsi_oversold': ThresholdConfig('rsi_oversold', 30, 15, 40, 1),
        'rsi_overbought': ThresholdConfig('rsi_overbought', 70, 60, 85, 1),
        'vix_low': ThresholdConfig('vix_low', 15, 10, 20, 1),
        'vix_normal': ThresholdConfig('vix_normal', 20, 15, 25, 1),
        'vix_high': ThresholdConfig('vix_high', 30, 25, 40, 1),
        'funding_extreme_zscore': ThresholdConfig('funding_extreme', 2.0, 1.0, 3.0, 0.1),
        'oi_change_significant': ThresholdConfig('oi_change', 0.05, 0.02, 0.15, 0.01),
        'fear_greed_extreme_low': ThresholdConfig('fg_extreme_low', 20, 10, 30, 5),
        'fear_greed_extreme_high': ThresholdConfig('fg_extreme_high', 80, 70, 90, 5),
    }
    
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, float], pd.DataFrame, pd.Series], float],
        n_trials: int = 100,
        random_state: int = 42
    ):
        """
        Initialize optimizer.
        
        Args:
            objective_fn: Function that takes thresholds, features, targets
                         and returns a score to maximize
            n_trials: Number of optimization trials
            random_state: Random seed
        """
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.random_state = random_state
        
    def optimize(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        thresholds_to_optimize: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Find optimal thresholds.
        
        Args:
            features: Feature DataFrame
            targets: Target variable
            thresholds_to_optimize: List of threshold names to optimize
                                   (None = optimize all)
                                   
        Returns:
            OptimizationResult with best thresholds found
        """
        if thresholds_to_optimize is None:
            thresholds_to_optimize = list(self.THRESHOLDS_TO_OPTIMIZE.keys())
            
        # Get baseline score with current thresholds
        current_thresholds = {
            name: cfg.current_value 
            for name, cfg in self.THRESHOLDS_TO_OPTIMIZE.items()
        }
        baseline_score = self.objective_fn(current_thresholds, features, targets)
        
        logger.info(f"Baseline score with current thresholds: {baseline_score:.4f}")
        
        if HAS_OPTUNA:
            result = self._optimize_optuna(
                features, targets, thresholds_to_optimize, baseline_score
            )
        else:
            result = self._optimize_scipy(
                features, targets, thresholds_to_optimize, baseline_score
            )
            
        return result
        
    def _optimize_optuna(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        threshold_names: List[str],
        baseline_score: float
    ) -> OptimizationResult:
        """Optimize using Optuna (better exploration)."""
        convergence = []
        
        def objective(trial):
            thresholds = {}
            for name in threshold_names:
                cfg = self.THRESHOLDS_TO_OPTIMIZE[name]
                thresholds[name] = trial.suggest_float(
                    name, cfg.min_value, cfg.max_value
                )
            # Add non-optimized thresholds at current values
            for name, cfg in self.THRESHOLDS_TO_OPTIMIZE.items():
                if name not in thresholds:
                    thresholds[name] = cfg.current_value
                    
            score = self.objective_fn(thresholds, features, targets)
            convergence.append(score)
            return score
            
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective, 
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
        best_thresholds = study.best_params
        
        # Add non-optimized at current values
        for name, cfg in self.THRESHOLDS_TO_OPTIMIZE.items():
            if name not in best_thresholds:
                best_thresholds[name] = cfg.current_value
                
        best_score = study.best_value
        improvement = (best_score - baseline_score) / abs(baseline_score) * 100
        
        return OptimizationResult(
            best_thresholds=best_thresholds,
            best_weights={},  # Weights optimized separately
            best_structure=None,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement_pct=improvement,
            iterations=len(study.trials),
            convergence_history=convergence
        )
        
    def _optimize_scipy(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        threshold_names: List[str],
        baseline_score: float
    ) -> OptimizationResult:
        """Optimize using scipy differential evolution."""
        convergence = []
        
        # Build bounds
        bounds = []
        for name in threshold_names:
            cfg = self.THRESHOLDS_TO_OPTIMIZE[name]
            bounds.append((cfg.min_value, cfg.max_value))
            
        def objective(x):
            thresholds = dict(zip(threshold_names, x))
            # Add non-optimized
            for name, cfg in self.THRESHOLDS_TO_OPTIMIZE.items():
                if name not in thresholds:
                    thresholds[name] = cfg.current_value
            score = self.objective_fn(thresholds, features, targets)
            convergence.append(score)
            return -score  # Minimize negative
            
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.n_trials,
            seed=self.random_state,
            workers=-1,
            updating='deferred'
        )
        
        best_thresholds = dict(zip(threshold_names, result.x))
        for name, cfg in self.THRESHOLDS_TO_OPTIMIZE.items():
            if name not in best_thresholds:
                best_thresholds[name] = cfg.current_value
                
        best_score = -result.fun
        improvement = (best_score - baseline_score) / abs(baseline_score) * 100
        
        return OptimizationResult(
            best_thresholds=best_thresholds,
            best_weights={},
            best_structure=None,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement_pct=improvement,
            iterations=result.nit,
            convergence_history=convergence
        )


class WeightOptimizer:
    """
    Discovers optimal pillar weights from data.
    
    Tests whether 35/30/25/10 is optimal or if different weights
    perform better.
    """
    
    CURRENT_WEIGHTS = {
        'derivatives': 0.35,
        'liquidations': 0.30,
        'technical': 0.25,
        'liquidity': 0.10,
    }
    
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, float], pd.DataFrame, pd.Series], float],
        n_trials: int = 100,
        random_state: int = 42
    ):
        """Initialize weight optimizer."""
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.random_state = random_state
        
    def optimize(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        pillar_features: Dict[str, List[str]]
    ) -> OptimizationResult:
        """
        Find optimal pillar weights.
        
        Args:
            features: Feature DataFrame
            targets: Target variable
            pillar_features: Dict mapping pillar name to list of feature names
            
        Returns:
            OptimizationResult with best weights
        """
        pillars = list(pillar_features.keys())
        n_pillars = len(pillars)
        
        # Baseline with current weights
        baseline_weights = {
            p: self.CURRENT_WEIGHTS.get(p, 1.0/n_pillars) 
            for p in pillars
        }
        # Normalize
        total = sum(baseline_weights.values())
        baseline_weights = {k: v/total for k, v in baseline_weights.items()}
        
        baseline_score = self._evaluate_weights(
            baseline_weights, features, targets, pillar_features
        )
        
        logger.info(f"Baseline score with current weights: {baseline_score:.4f}")
        
        # Optimize using simplex constraint (weights sum to 1)
        convergence = []
        
        if HAS_OPTUNA:
            best_weights, best_score, iterations = self._optimize_optuna(
                pillars, features, targets, pillar_features, convergence
            )
        else:
            best_weights, best_score, iterations = self._optimize_scipy(
                pillars, features, targets, pillar_features, convergence
            )
            
        improvement = (best_score - baseline_score) / abs(baseline_score) * 100
        
        return OptimizationResult(
            best_thresholds={},
            best_weights=best_weights,
            best_structure=None,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement_pct=improvement,
            iterations=iterations,
            convergence_history=convergence
        )
        
    def _evaluate_weights(
        self,
        weights: Dict[str, float],
        features: pd.DataFrame,
        targets: pd.Series,
        pillar_features: Dict[str, List[str]]
    ) -> float:
        """Evaluate a weight configuration."""
        # Create weighted pillar scores
        pillar_scores = {}
        for pillar, feat_names in pillar_features.items():
            available = [f for f in feat_names if f in features.columns]
            if available:
                # Simple average of normalized features
                pillar_data = features[available]
                pillar_scores[pillar] = pillar_data.mean(axis=1)
            else:
                pillar_scores[pillar] = pd.Series(0, index=features.index)
                
        # Weighted combination
        combined = pd.Series(0.0, index=features.index)
        for pillar, score in pillar_scores.items():
            combined += weights.get(pillar, 0) * score
            
        # Evaluate using objective function
        return self.objective_fn({'combined_score': combined}, features, targets)
        
    def _optimize_optuna(
        self,
        pillars: List[str],
        features: pd.DataFrame,
        targets: pd.Series,
        pillar_features: Dict[str, List[str]],
        convergence: List[float]
    ) -> Tuple[Dict[str, float], float, int]:
        """Optimize weights using Optuna."""
        def objective(trial):
            # Sample weights (unnormalized)
            raw_weights = {}
            for pillar in pillars:
                raw_weights[pillar] = trial.suggest_float(pillar, 0.01, 1.0)
                
            # Normalize to sum to 1
            total = sum(raw_weights.values())
            weights = {k: v/total for k, v in raw_weights.items()}
            
            score = self._evaluate_weights(weights, features, targets, pillar_features)
            convergence.append(score)
            return score
            
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
        # Get best weights (normalized)
        raw_best = study.best_params
        total = sum(raw_best.values())
        best_weights = {k: v/total for k, v in raw_best.items()}
        
        return best_weights, study.best_value, len(study.trials)
        
    def _optimize_scipy(
        self,
        pillars: List[str],
        features: pd.DataFrame,
        targets: pd.Series,
        pillar_features: Dict[str, List[str]],
        convergence: List[float]
    ) -> Tuple[Dict[str, float], float, int]:
        """Optimize weights using scipy with simplex constraint."""
        n = len(pillars)
        
        def objective(x):
            # x is already on simplex (sums to 1)
            weights = dict(zip(pillars, x))
            score = self._evaluate_weights(weights, features, targets, pillar_features)
            convergence.append(score)
            return -score
            
        # Start from current weights
        x0 = np.array([
            self.CURRENT_WEIGHTS.get(p, 1.0/n) for p in pillars
        ])
        x0 = x0 / x0.sum()  # Normalize
        
        # Constraints: sum to 1, all >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0.01, 0.99) for _ in pillars]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.n_trials}
        )
        
        best_weights = dict(zip(pillars, result.x))
        return best_weights, -result.fun, result.nit


class StructureDiscovery:
    """
    Discovers optimal feature grouping structure.
    
    Tests whether 4 pillars is optimal, or if alternatives like:
    - 3 pillars (merge liquidity)
    - 5 pillars (add macro)
    - No pillars (flat feature selection)
    - Data-driven clustering
    
    perform better.
    """
    
    def __init__(
        self,
        feature_importance: Dict[str, float],
        objective_fn: Callable,
        random_state: int = 42
    ):
        """
        Initialize structure discovery.
        
        Args:
            feature_importance: Dict of feature_name -> importance score
            objective_fn: Function to evaluate structure quality
            random_state: Random seed
        """
        self.feature_importance = feature_importance
        self.objective_fn = objective_fn
        self.random_state = random_state
        
    def discover_optimal_structure(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        max_groups: int = 6
    ) -> List[StructureConfig]:
        """
        Discover and rank different structure configurations.
        
        Returns list of StructureConfig sorted by score.
        """
        candidates = []
        
        # 1. Current 4-pillar structure
        current = StructureConfig(
            name="Current 4-Pillar",
            groups={
                'derivatives': ['funding_', 'oi_', 'ls_ratio_'],
                'liquidations': ['liq_'],
                'technical': ['rsi_', 'ema_', 'macd_', 'bb_', 'momentum_', 'return_'],
                'liquidity': ['taker_', 'orderbook_'],
            },
            weights={'derivatives': 0.35, 'liquidations': 0.30, 
                    'technical': 0.25, 'liquidity': 0.10}
        )
        candidates.append(current)
        
        # 2. 3-Pillar (merge liquidity into derivatives)
        three_pillar = StructureConfig(
            name="3-Pillar (No Liquidity)",
            groups={
                'derivatives': ['funding_', 'oi_', 'ls_ratio_', 'taker_'],
                'liquidations': ['liq_'],
                'technical': ['rsi_', 'ema_', 'macd_', 'bb_', 'momentum_', 'return_'],
            },
            weights={'derivatives': 0.40, 'liquidations': 0.35, 'technical': 0.25}
        )
        candidates.append(three_pillar)
        
        # 3. 5-Pillar (add macro)
        five_pillar = StructureConfig(
            name="5-Pillar (Add Macro)",
            groups={
                'derivatives': ['funding_', 'oi_', 'ls_ratio_'],
                'liquidations': ['liq_'],
                'technical': ['rsi_', 'ema_', 'macd_', 'bb_', 'momentum_'],
                'liquidity': ['taker_'],
                'macro': ['vix_', 'dxy_', 'spy_', 'yield_', 'fear_greed_'],
            },
            weights={'derivatives': 0.30, 'liquidations': 0.25, 
                    'technical': 0.20, 'liquidity': 0.10, 'macro': 0.15}
        )
        candidates.append(five_pillar)
        
        # 4. 2-Pillar (derivatives+liquidations vs technical)
        two_pillar = StructureConfig(
            name="2-Pillar (Derivatives vs Technical)",
            groups={
                'derivatives_combined': ['funding_', 'oi_', 'ls_ratio_', 'liq_', 'taker_'],
                'technical_combined': ['rsi_', 'ema_', 'macd_', 'bb_', 'momentum_', 'return_'],
            },
            weights={'derivatives_combined': 0.60, 'technical_combined': 0.40}
        )
        candidates.append(two_pillar)
        
        # 5. No pillars - top N features directly
        flat = StructureConfig(
            name="Flat (Top 20 Features)",
            groups={'all': list(self.feature_importance.keys())[:20]},
            weights={'all': 1.0}
        )
        candidates.append(flat)
        
        # 6. Data-driven clustering
        if HAS_CLUSTERING:
            clustered = self._discover_clustered_structure(features, max_groups)
            if clustered:
                candidates.append(clustered)
                
        # Evaluate all candidates
        for config in candidates:
            config.score = self._evaluate_structure(config, features, targets)
            
        # Sort by score
        candidates.sort(key=lambda x: -x.score)
        
        return candidates
        
    def _evaluate_structure(
        self,
        config: StructureConfig,
        features: pd.DataFrame,
        targets: pd.Series
    ) -> float:
        """Evaluate a structure configuration."""
        # Map features to groups
        group_features = {}
        for group_name, prefixes in config.groups.items():
            if isinstance(prefixes, list) and all(isinstance(p, str) for p in prefixes):
                # Prefixes mode
                matched = []
                for col in features.columns:
                    for prefix in prefixes:
                        if col.startswith(prefix):
                            matched.append(col)
                            break
                group_features[group_name] = matched
            else:
                # Direct feature list
                group_features[group_name] = [f for f in prefixes if f in features.columns]
                
        # Create weighted combination
        combined = pd.Series(0.0, index=features.index)
        for group_name, feat_list in group_features.items():
            if feat_list:
                group_data = features[feat_list]
                # Normalize each feature
                group_normalized = (group_data - group_data.mean()) / (group_data.std() + 1e-8)
                group_score = group_normalized.mean(axis=1)
                combined += config.weights.get(group_name, 0) * group_score
                
        # Evaluate
        return self.objective_fn({'combined_score': combined}, features, targets)
        
    def _discover_clustered_structure(
        self,
        features: pd.DataFrame,
        max_groups: int
    ) -> Optional[StructureConfig]:
        """Discover structure using feature clustering."""
        try:
            # Use top features
            top_features = list(self.feature_importance.keys())[:100]
            available = [f for f in top_features if f in features.columns]
            
            if len(available) < 10:
                return None
                
            # Transpose to cluster features (not samples)
            X = features[available].dropna().T
            
            if len(X) < max_groups:
                return None
                
            # Find optimal number of clusters
            best_n = 2
            best_score = -1
            
            for n in range(2, min(max_groups + 1, len(X))):
                kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            # Create clusters with best n
            kmeans = KMeans(n_clusters=best_n, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Build groups
            groups = {}
            for i in range(best_n):
                group_features = [available[j] for j in range(len(available)) if labels[j] == i]
                groups[f'cluster_{i}'] = group_features
                
            # Equal weights initially
            weights = {g: 1.0/best_n for g in groups}
            
            return StructureConfig(
                name=f"Data-Driven {best_n}-Cluster",
                groups=groups,
                weights=weights
            )
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return None


def create_default_objective_function():
    """
    Create a default objective function for optimization.
    
    Returns a function that calculates Sharpe-like score based on threshold parameters.
    """
    def objective(params: Dict, features: pd.DataFrame, targets: pd.Series) -> float:
        """
        Calculate score based on signal quality using threshold parameters.
        
        The thresholds are applied to relevant features to generate trading signals.
        
        Higher score = better signal.
        """
        try:
            # Align features and targets
            common_idx = features.index.intersection(targets.index)
            if len(common_idx) < 50:
                logger.debug(f"Insufficient data: {len(common_idx)} rows")
                return -10.0  # Bad score but not -inf
                
            features_aligned = features.loc[common_idx]
            targets_aligned = targets.loc[common_idx]
            
            # Generate signals based on thresholds
            # Use available features that match threshold concepts
            signal = pd.Series(0.0, index=common_idx)
            signal_count = 0
            
            # RSI signals (if RSI features exist)
            rsi_cols = [c for c in features_aligned.columns if 'rsi' in c.lower()]
            if rsi_cols and 'rsi_oversold' in params and 'rsi_overbought' in params:
                rsi_col = rsi_cols[0]
                rsi_values = features_aligned[rsi_col]
                # Oversold = buy signal (positive)
                signal += (rsi_values < params['rsi_oversold']).astype(float) * 0.3
                # Overbought = sell signal (negative)
                signal -= (rsi_values > params['rsi_overbought']).astype(float) * 0.3
                signal_count += 1
            
            # OI change signals
            oi_cols = [c for c in features_aligned.columns if 'oi_change' in c.lower() or 'oi_close_change' in c.lower()]
            if oi_cols and 'oi_change_significant' in params:
                oi_col = oi_cols[0]
                oi_values = features_aligned[oi_col]
                # Significant positive OI change = bullish
                signal += (oi_values > params['oi_change_significant']).astype(float) * 0.3
                # Significant negative OI change = bearish
                signal -= (oi_values < -params['oi_change_significant']).astype(float) * 0.3
                signal_count += 1
            
            # Fear & Greed signals
            fg_cols = [c for c in features_aligned.columns if 'fear_greed' in c.lower() or 'fg_' in c.lower()]
            if fg_cols and 'fear_greed_extreme_low' in params and 'fear_greed_extreme_high' in params:
                fg_col = fg_cols[0]
                fg_values = features_aligned[fg_col]
                # Extreme fear = contrarian buy
                signal += (fg_values < params['fear_greed_extreme_low']).astype(float) * 0.2
                # Extreme greed = contrarian sell
                signal -= (fg_values > params['fear_greed_extreme_high']).astype(float) * 0.2
                signal_count += 1
            
            # Funding rate signals
            funding_cols = [c for c in features_aligned.columns if 'funding' in c.lower() and 'zscore' in c.lower()]
            if funding_cols and 'funding_extreme_zscore' in params:
                funding_col = funding_cols[0]
                funding_values = features_aligned[funding_col]
                # Extreme positive funding = bearish (crowded long)
                signal -= (funding_values > params['funding_extreme_zscore']).astype(float) * 0.2
                # Extreme negative funding = bullish (crowded short)
                signal += (funding_values < -params['funding_extreme_zscore']).astype(float) * 0.2
                signal_count += 1
            
            # If no features matched, use a simple feature-based signal
            if signal_count == 0:
                # Fallback: use first numeric column
                numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col = numeric_cols[0]
                    # Use percentile-based signal
                    pct = features_aligned[col].rank(pct=True)
                    signal = (pct - 0.5) * 2  # Scale to -1, 1
            
            # Clip signal to reasonable range
            signal = signal.clip(-1, 1)
            
            # Calculate returns based on signal and target
            # For binary targets (profitable), convert to return-like
            if targets_aligned.dtype == bool or set(targets_aligned.unique()).issubset({0, 1, True, False}):
                target_returns = (targets_aligned.astype(float) - 0.5) * 2  # Scale 0/1 to -1/+1
            else:
                target_returns = targets_aligned
            
            # Calculate strategy returns (signal * target)
            strategy_returns = signal * target_returns
            
            # Calculate Sharpe using TRADE FREQUENCY
            # Count "trades" as bars where signal strength is meaningful
            active_signals = (abs(signal) > 0.1)  # Meaningful signal threshold
            n_trades = active_signals.sum()
            
            # Get trade-level returns
            trade_returns = strategy_returns[active_signals]
            
            if len(trade_returns) < 5:
                return -5.0  # Need minimum trades
            
            mean_ret = trade_returns.mean()
            std_ret = trade_returns.std()
            
            if std_ret < 1e-8 or np.isnan(std_ret):
                return -5.0  # Low score for no variance
            
            # Estimate period_days (4h bars = 6 per day)
            period_days = max(1, len(features_aligned) / 6)
            trades_per_year = (n_trades / period_days) * 365
            
            # Per-trade Sharpe annualized by trade frequency
            per_trade_sharpe = mean_ret / std_ret
            sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
            
            # Additional metrics
            win_rate = (strategy_returns > 0).mean()
            
            # Correlation with target
            corr = signal.corr(target_returns)
            if np.isnan(corr):
                corr = 0.0
            
            # Combined score
            score = (
                0.5 * np.clip(sharpe / 3.0, -1, 1) +  # Sharpe contribution (capped)
                0.3 * (win_rate - 0.5) * 2 +  # Win rate contribution
                0.2 * np.clip(corr, -1, 1)  # Correlation contribution
            )
            
            # Ensure finite score
            if not np.isfinite(score):
                logger.debug(f"Non-finite score: sharpe={sharpe}, win_rate={win_rate}, corr={corr}")
                return -5.0
                
            return float(score)
            
        except Exception as e:
            logger.warning(f"Objective function error: {e}")
            return -10.0  # Return bad score, not -inf
        
    return objective


def run_full_optimization(
    features: pd.DataFrame,
    targets: pd.Series,
    feature_importance: Dict[str, float],
    n_trials: int = 100
) -> Dict[str, Any]:
    """
    Run complete threshold, weight, and structure optimization.
    
    Returns comprehensive optimization results.
    """
    objective_fn = create_default_objective_function()
    
    results = {
        'thresholds': None,
        'weights': None,
        'structure': None,
        'recommendations': []
    }
    
    # 1. Optimize thresholds
    logger.info("Optimizing thresholds...")
    threshold_opt = ThresholdOptimizer(objective_fn, n_trials=n_trials)
    results['thresholds'] = threshold_opt.optimize(features, targets)
    
    if results['thresholds'].improvement_pct > 5:
        results['recommendations'].append(
            f"UPDATE THRESHOLDS: {results['thresholds'].improvement_pct:.1f}% improvement possible"
        )
        
    # 2. Discover optimal structure
    logger.info("Discovering optimal structure...")
    structure_discovery = StructureDiscovery(feature_importance, objective_fn)
    structures = structure_discovery.discover_optimal_structure(features, targets)
    
    if structures:
        results['structure'] = {
            'best': structures[0],
            'alternatives': structures[1:],
            'ranking': [(s.name, s.score) for s in structures]
        }
        
        if structures[0].name != "Current 4-Pillar":
            results['recommendations'].append(
                f"CONSIDER RESTRUCTURE: '{structures[0].name}' scores "
                f"{structures[0].score:.4f} vs current structure"
            )
            
    # 3. Optimize weights for best structure
    if results['structure']:
        logger.info("Optimizing weights for best structure...")
        best_structure = results['structure']['best']
        
        # Map prefix groups to actual features
        pillar_features = {}
        for group_name, prefixes in best_structure.groups.items():
            matched = []
            for col in features.columns:
                for prefix in prefixes if isinstance(prefixes, list) else [prefixes]:
                    if col.startswith(prefix):
                        matched.append(col)
                        break
            pillar_features[group_name] = matched
            
        weight_opt = WeightOptimizer(objective_fn, n_trials=n_trials)
        results['weights'] = weight_opt.optimize(features, targets, pillar_features)
        
        if results['weights'].improvement_pct > 5:
            results['recommendations'].append(
                f"UPDATE WEIGHTS: {results['weights'].improvement_pct:.1f}% improvement possible"
            )
            
    return results
