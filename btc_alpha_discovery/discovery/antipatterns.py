"""
Anti-Pattern Discovery.

Phase 9: Discovers conditions that predict losses (when NOT to trade).

Key insight: Knowing when NOT to trade is as valuable as knowing when to trade.
Markets have conditions where any strategy will fail. Identifying these
can significantly improve overall performance.

Anti-patterns include:
- Choppy/sideways markets
- High volatility regimes that whipsaw
- Specific time periods (news events, low liquidity)
- Feature combinations that precede losses
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class AntiPattern:
    """A discovered anti-pattern (when NOT to trade)."""
    pattern_id: str
    conditions: List[Dict[str, Any]]
    logic: str
    loss_rate: float  # Percentage of trades that lose
    avg_loss: float  # Average loss when pattern active
    frequency: float  # How often pattern occurs
    confidence: float  # Statistical confidence
    description: str
    
    def evaluate(self, features: pd.DataFrame) -> pd.Series:
        """Return True where pattern is active (should NOT trade)."""
        masks = []
        for cond in self.conditions:
            feat = features.get(cond['feature'])
            if feat is None:
                masks.append(pd.Series(False, index=features.index))
                continue
                
            op = cond['operator']
            thresh = cond['threshold']
            
            if op == '>':
                masks.append(feat > thresh)
            elif op == '<':
                masks.append(feat < thresh)
            elif op == '>=':
                masks.append(feat >= thresh)
            elif op == '<=':
                masks.append(feat <= thresh)
            elif op == 'between':
                masks.append((feat >= thresh[0]) & (feat <= thresh[1]))
                
        if not masks:
            return pd.Series(False, index=features.index)
            
        if self.logic == 'AND':
            result = masks[0]
            for m in masks[1:]:
                result = result & m
        else:
            result = masks[0]
            for m in masks[1:]:
                result = result | m
                
        return result
    
    def to_string(self) -> str:
        cond_strs = []
        for c in self.conditions:
            if c['operator'] == 'between':
                cond_strs.append(f"{c['threshold'][0]:.2f} <= {c['feature']} <= {c['threshold'][1]:.2f}")
            else:
                cond_strs.append(f"{c['feature']} {c['operator']} {c['threshold']:.4f}")
        return f"AVOID when ({f' {self.logic} '.join(cond_strs)})"


@dataclass
class RegimeAnalysis:
    """Analysis of market regimes and their tradability."""
    regime_id: str
    regime_type: str  # 'trending', 'ranging', 'volatile', 'calm'
    detection_features: List[str]
    win_rate: float
    avg_return: float
    frequency: float
    recommendation: str  # 'trade', 'avoid', 'reduce_size'


class AntiPatternDiscovery:
    """
    Discovers conditions where trading performs poorly.
    
    Methods:
    1. Loss clustering - Find common features in losing trades
    2. Regime detection - Identify market regimes with poor performance
    3. Anomaly detection - Find unusual conditions that precede losses
    4. Feature inversion - Take inverse of winning conditions
    """
    
    def __init__(
        self,
        min_samples: int = 50,
        loss_threshold: float = 0.6,  # Min loss rate to qualify as anti-pattern
        random_state: int = 42
    ):
        """
        Initialize discovery.
        
        Args:
            min_samples: Minimum samples for pattern validity
            loss_threshold: Minimum loss rate for anti-pattern
            random_state: Random seed
        """
        self.min_samples = min_samples
        self.loss_threshold = loss_threshold
        self.random_state = random_state
        
    def discover(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        top_features: Optional[List[str]] = None
    ) -> List[AntiPattern]:
        """
        Discover anti-patterns.
        
        Args:
            features: Feature DataFrame
            returns: Trade returns
            top_features: Features to analyze (optional)
            
        Returns:
            List of AntiPattern objects
        """
        logger.info("Discovering anti-patterns...")
        
        # Only keep numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features[numeric_cols].copy()
        
        logger.info(f"Numeric features: {len(features_numeric.columns)}, samples: {len(features_numeric)}")
        
        # Clean data - fill NaN instead of dropping
        features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)
        features_numeric = features_numeric.ffill().bfill().fillna(0)
        
        # Align with returns
        valid_idx = ~returns.isna()
        X = features_numeric[valid_idx].reset_index(drop=True)
        y = returns[valid_idx].reset_index(drop=True)
        
        logger.info(f"After cleaning: {len(X)} samples with valid returns")
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for anti-pattern discovery (need 100, have {len(X)})")
            return []
        
        if top_features:
            available = [f for f in top_features if f in X.columns]
            X = X[available[:50]]
            
        all_patterns = []
        
        # Method 1: Loss clustering (decision tree on losing trades)
        logger.info("Analyzing losing trade clusters...")
        loss_patterns = self._discover_loss_clusters(X, y)
        all_patterns.extend(loss_patterns)
        
        # Method 2: Volatility regimes
        logger.info("Analyzing volatility regimes...")
        vol_patterns = self._discover_volatility_antipatterns(X, y)
        all_patterns.extend(vol_patterns)
        
        # Method 3: Low activity / chop detection
        logger.info("Detecting choppy market conditions...")
        chop_patterns = self._discover_chop_patterns(X, y)
        all_patterns.extend(chop_patterns)
        
        # Method 4: Time-based anti-patterns
        logger.info("Analyzing time-based patterns...")
        time_patterns = self._discover_time_antipatterns(X, y)
        all_patterns.extend(time_patterns)
        
        # Method 5: Feature extreme anti-patterns
        logger.info("Analyzing feature extremes...")
        extreme_patterns = self._discover_extreme_antipatterns(X, y)
        all_patterns.extend(extreme_patterns)
        
        # Validate and filter
        valid_patterns = []
        for pattern in all_patterns:
            if self._validate_pattern(pattern, X, y):
                valid_patterns.append(pattern)
                
        # Sort by loss rate * frequency (impact)
        valid_patterns.sort(key=lambda p: -(p.loss_rate * p.frequency))
        
        logger.info(f"Discovered {len(valid_patterns)} anti-patterns")
        return valid_patterns
        
    def _discover_loss_clusters(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[AntiPattern]:
        """Find clusters of losing trades using decision trees."""
        if not HAS_SKLEARN:
            return []
            
        patterns = []
        
        # Create binary target: 1 = loss, 0 = win
        y_binary = (y < 0).astype(int)
        
        for max_depth in [2, 3, 4]:
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=max(self.min_samples // 2, 20),
                random_state=self.random_state
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree.fit(X, y_binary)
                
            # Extract rules that predict losses
            tree_patterns = self._extract_loss_rules(tree, X.columns.tolist())
            patterns.extend(tree_patterns)
            
        return patterns
        
    def _extract_loss_rules(
        self,
        tree: 'DecisionTreeClassifier',
        feature_names: List[str]
    ) -> List[AntiPattern]:
        """Extract anti-patterns from decision tree."""
        patterns = []
        
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, conditions):
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left branch
                left_conditions = conditions + [{
                    'feature': name,
                    'operator': '<=',
                    'threshold': threshold
                }]
                recurse(tree_.children_left[node], left_conditions)
                
                # Right branch
                right_conditions = conditions + [{
                    'feature': name,
                    'operator': '>',
                    'threshold': threshold
                }]
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf
                values = tree_.value[node][0]
                total = sum(values)
                if total > 0:
                    loss_rate = values[1] / total if len(values) > 1 else 0
                    n_samples = int(total)
                    
                    if loss_rate >= self.loss_threshold and n_samples >= self.min_samples // 2:
                        pattern = AntiPattern(
                            pattern_id=f"loss_cluster_{len(patterns)}",
                            conditions=conditions,
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=0.0,
                            frequency=n_samples,
                            confidence=0.0,
                            description=f"Loss cluster with {loss_rate:.1%} loss rate"
                        )
                        patterns.append(pattern)
                        
        recurse(0, [])
        return patterns
        
    def _discover_volatility_antipatterns(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[AntiPattern]:
        """Find volatility conditions that lead to losses."""
        patterns = []
        
        # Look for volatility-related features
        vol_features = [c for c in X.columns if any(v in c.lower() for v in ['vol', 'atr', 'std', 'vix'])]
        
        for feat in vol_features:
            if feat not in X.columns:
                continue
                
            feat_vals = X[feat].dropna()
            
            # Test high volatility
            for pct in [80, 90, 95]:
                threshold = np.percentile(feat_vals, pct)
                mask = X[feat] > threshold
                
                if mask.sum() >= self.min_samples // 2:
                    loss_rate = (y[mask] < 0).mean()
                    
                    if loss_rate >= self.loss_threshold:
                        patterns.append(AntiPattern(
                            pattern_id=f"high_vol_{feat}_{pct}",
                            conditions=[{
                                'feature': feat,
                                'operator': '>',
                                'threshold': threshold
                            }],
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                            frequency=mask.sum() / len(mask),
                            confidence=0.0,
                            description=f"High {feat} (>{threshold:.4f}) leads to {loss_rate:.1%} losses"
                        ))
                        
            # Test very low volatility (choppy markets)
            for pct in [5, 10, 20]:
                threshold = np.percentile(feat_vals, pct)
                mask = X[feat] < threshold
                
                if mask.sum() >= self.min_samples // 2:
                    loss_rate = (y[mask] < 0).mean()
                    
                    if loss_rate >= self.loss_threshold:
                        patterns.append(AntiPattern(
                            pattern_id=f"low_vol_{feat}_{pct}",
                            conditions=[{
                                'feature': feat,
                                'operator': '<',
                                'threshold': threshold
                            }],
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                            frequency=mask.sum() / len(mask),
                            confidence=0.0,
                            description=f"Low {feat} (<{threshold:.4f}) leads to {loss_rate:.1%} losses"
                        ))
                        
        return patterns
        
    def _discover_chop_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[AntiPattern]:
        """Detect choppy/ranging market conditions."""
        patterns = []
        
        # Look for trend strength indicators
        trend_features = [c for c in X.columns if any(t in c.lower() for t in ['adx', 'trend', 'momentum', 'roc'])]
        
        for feat in trend_features:
            if feat not in X.columns:
                continue
                
            feat_vals = X[feat].dropna()
            
            # Low trend strength = choppy
            for pct in [10, 20, 25]:
                threshold = np.percentile(feat_vals, pct)
                mask = X[feat] < threshold
                
                if mask.sum() >= self.min_samples // 2:
                    loss_rate = (y[mask] < 0).mean()
                    
                    if loss_rate >= self.loss_threshold:
                        patterns.append(AntiPattern(
                            pattern_id=f"chop_{feat}_{pct}",
                            conditions=[{
                                'feature': feat,
                                'operator': '<',
                                'threshold': threshold
                            }],
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                            frequency=mask.sum() / len(mask),
                            confidence=0.0,
                            description=f"Choppy market ({feat} < {threshold:.4f})"
                        ))
                        
        # RSI in neutral zone (no strong direction)
        rsi_features = [c for c in X.columns if 'rsi' in c.lower()]
        for feat in rsi_features:
            if feat not in X.columns:
                continue
                
            # Middle zone
            mask = (X[feat] > 40) & (X[feat] < 60)
            
            if mask.sum() >= self.min_samples // 2:
                loss_rate = (y[mask] < 0).mean()
                
                if loss_rate >= self.loss_threshold:
                    patterns.append(AntiPattern(
                        pattern_id=f"neutral_{feat}",
                        conditions=[{
                            'feature': feat,
                            'operator': 'between',
                            'threshold': [40, 60]
                        }],
                        logic='AND',
                        loss_rate=loss_rate,
                        avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                        frequency=mask.sum() / len(mask),
                        confidence=0.0,
                        description=f"Neutral {feat} (40-60) indicates choppy market"
                    ))
                    
        return patterns
        
    def _discover_time_antipatterns(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[AntiPattern]:
        """Find time periods with poor performance."""
        patterns = []
        
        # Hour of day
        hour_features = [c for c in X.columns if 'hour' in c.lower()]
        for feat in hour_features:
            if feat not in X.columns:
                continue
                
            for hour in range(24):
                mask = X[feat] == hour
                
                if mask.sum() >= self.min_samples // 3:
                    loss_rate = (y[mask] < 0).mean()
                    
                    if loss_rate >= self.loss_threshold + 0.05:  # Higher bar for time
                        patterns.append(AntiPattern(
                            pattern_id=f"hour_{hour}",
                            conditions=[{
                                'feature': feat,
                                'operator': '==',
                                'threshold': hour
                            }],
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                            frequency=mask.sum() / len(mask),
                            confidence=0.0,
                            description=f"Hour {hour} has {loss_rate:.1%} loss rate"
                        ))
                        
        # Day of week
        dow_features = [c for c in X.columns if 'day' in c.lower() and 'week' in c.lower()]
        for feat in dow_features:
            if feat not in X.columns:
                continue
                
            for day in range(7):
                mask = X[feat] == day
                
                if mask.sum() >= self.min_samples // 3:
                    loss_rate = (y[mask] < 0).mean()
                    
                    if loss_rate >= self.loss_threshold + 0.05:
                        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        patterns.append(AntiPattern(
                            pattern_id=f"dow_{day}",
                            conditions=[{
                                'feature': feat,
                                'operator': '==',
                                'threshold': day
                            }],
                            logic='AND',
                            loss_rate=loss_rate,
                            avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                            frequency=mask.sum() / len(mask),
                            confidence=0.0,
                            description=f"{day_names[day]} has {loss_rate:.1%} loss rate"
                        ))
                        
        return patterns
        
    def _discover_extreme_antipatterns(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[AntiPattern]:
        """Find extreme feature values that predict losses."""
        patterns = []
        
        for col in X.columns:
            feat_vals = X[col].dropna()
            if len(feat_vals) < 100:
                continue
                
            # Test extreme highs
            for pct in [95, 99]:
                try:
                    threshold = np.percentile(feat_vals, pct)
                    mask = X[col] > threshold
                    
                    if mask.sum() >= self.min_samples // 3:
                        loss_rate = (y[mask] < 0).mean()
                        
                        if loss_rate >= self.loss_threshold:
                            patterns.append(AntiPattern(
                                pattern_id=f"extreme_high_{col}_{pct}",
                                conditions=[{
                                    'feature': col,
                                    'operator': '>',
                                    'threshold': threshold
                                }],
                                logic='AND',
                                loss_rate=loss_rate,
                                avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                                frequency=mask.sum() / len(mask),
                                confidence=0.0,
                                description=f"Extreme high {col} (>{pct}th pct)"
                            ))
                except Exception:
                    continue
                    
            # Test extreme lows
            for pct in [1, 5]:
                try:
                    threshold = np.percentile(feat_vals, pct)
                    mask = X[col] < threshold
                    
                    if mask.sum() >= self.min_samples // 3:
                        loss_rate = (y[mask] < 0).mean()
                        
                        if loss_rate >= self.loss_threshold:
                            patterns.append(AntiPattern(
                                pattern_id=f"extreme_low_{col}_{pct}",
                                conditions=[{
                                    'feature': col,
                                    'operator': '<',
                                    'threshold': threshold
                                }],
                                logic='AND',
                                loss_rate=loss_rate,
                                avg_loss=y[mask][y[mask] < 0].mean() if (y[mask] < 0).sum() > 0 else 0,
                                frequency=mask.sum() / len(mask),
                                confidence=0.0,
                                description=f"Extreme low {col} (<{pct}th pct)"
                            ))
                except Exception:
                    continue
                    
        return patterns
        
    def _validate_pattern(
        self,
        pattern: AntiPattern,
        X: pd.DataFrame,
        y: pd.Series
    ) -> bool:
        """Validate anti-pattern with statistical testing."""
        mask = pattern.evaluate(X)
        n_samples = mask.sum()
        
        if n_samples < self.min_samples:
            return False
            
        # Calculate actual loss rate
        returns = y[mask]
        loss_rate = (returns < 0).mean()
        
        # Update pattern stats
        pattern.loss_rate = loss_rate
        pattern.avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0
        pattern.frequency = n_samples / len(mask)
        
        # Statistical test: is loss rate significantly above 50%?
        from scipy import stats
        try:
            n_losses = (returns < 0).sum()
            pvalue = stats.binom_test(n_losses, n_samples, 0.5, alternative='greater')
            pattern.confidence = 1 - pvalue
        except Exception:
            pattern.confidence = 0.5
            
        # Must have significant loss rate
        return loss_rate >= self.loss_threshold and pattern.confidence > 0.8


class RegimeDetector:
    """
    Detects market regimes and their tradability.
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        
    def detect_regimes(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> List[RegimeAnalysis]:
        """
        Detect market regimes and analyze performance in each.
        
        Args:
            features: Feature DataFrame
            returns: Forward returns
            
        Returns:
            List of RegimeAnalysis objects
        """
        logger.info("Detecting market regimes...")
        
        regimes = []
        
        # 1. Volatility regimes
        vol_regimes = self._detect_volatility_regimes(features, returns)
        regimes.extend(vol_regimes)
        
        # 2. Trend regimes
        trend_regimes = self._detect_trend_regimes(features, returns)
        regimes.extend(trend_regimes)
        
        # 3. Liquidity regimes
        liquidity_regimes = self._detect_liquidity_regimes(features, returns)
        regimes.extend(liquidity_regimes)
        
        return regimes
        
    def _detect_volatility_regimes(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> List[RegimeAnalysis]:
        """Detect volatility-based regimes."""
        regimes = []
        
        # Find volatility feature
        vol_features = [c for c in features.columns if 'vol' in c.lower() or 'atr' in c.lower()]
        
        if not vol_features:
            return regimes
            
        vol_feat = vol_features[0]
        vol = features[vol_feat].dropna()
        
        # Define regimes
        q25 = np.percentile(vol, 25)
        q75 = np.percentile(vol, 75)
        
        regime_defs = [
            ('low_vol', vol <= q25, 'calm'),
            ('normal_vol', (vol > q25) & (vol <= q75), 'ranging'),
            ('high_vol', vol > q75, 'volatile'),
        ]
        
        combined = pd.concat([features, returns.rename('ret')], axis=1).dropna()
        
        for name, mask, regime_type in regime_defs:
            regime_mask = mask.reindex(combined.index).fillna(False)
            regime_returns = combined.loc[regime_mask, 'ret']
            
            if len(regime_returns) >= 50:
                win_rate = (regime_returns > 0).mean()
                avg_return = regime_returns.mean()
                frequency = regime_mask.sum() / len(regime_mask)
                
                if win_rate < 0.45:
                    recommendation = 'avoid'
                elif win_rate < 0.50:
                    recommendation = 'reduce_size'
                else:
                    recommendation = 'trade'
                    
                regimes.append(RegimeAnalysis(
                    regime_id=name,
                    regime_type=regime_type,
                    detection_features=[vol_feat],
                    win_rate=win_rate,
                    avg_return=avg_return,
                    frequency=frequency,
                    recommendation=recommendation
                ))
                
        return regimes
        
    def _detect_trend_regimes(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> List[RegimeAnalysis]:
        """Detect trend-based regimes."""
        regimes = []
        
        # Find trend features
        trend_features = [c for c in features.columns if 'adx' in c.lower() or 'trend' in c.lower()]
        
        if not trend_features:
            return regimes
            
        trend_feat = trend_features[0]
        trend = features[trend_feat].dropna()
        
        # Define regimes
        median_trend = np.median(trend)
        
        regime_defs = [
            ('weak_trend', trend <= median_trend, 'ranging'),
            ('strong_trend', trend > median_trend, 'trending'),
        ]
        
        combined = pd.concat([features, returns.rename('ret')], axis=1).dropna()
        
        for name, mask, regime_type in regime_defs:
            regime_mask = mask.reindex(combined.index).fillna(False)
            regime_returns = combined.loc[regime_mask, 'ret']
            
            if len(regime_returns) >= 50:
                win_rate = (regime_returns > 0).mean()
                avg_return = regime_returns.mean()
                frequency = regime_mask.sum() / len(regime_mask)
                
                if win_rate < 0.45:
                    recommendation = 'avoid'
                elif win_rate < 0.50:
                    recommendation = 'reduce_size'
                else:
                    recommendation = 'trade'
                    
                regimes.append(RegimeAnalysis(
                    regime_id=name,
                    regime_type=regime_type,
                    detection_features=[trend_feat],
                    win_rate=win_rate,
                    avg_return=avg_return,
                    frequency=frequency,
                    recommendation=recommendation
                ))
                
        return regimes
        
    def _detect_liquidity_regimes(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> List[RegimeAnalysis]:
        """Detect liquidity-based regimes."""
        regimes = []
        
        # Find volume/liquidity features
        liq_features = [c for c in features.columns if 'volume' in c.lower() or 'spread' in c.lower()]
        
        if not liq_features:
            return regimes
            
        liq_feat = liq_features[0]
        liq = features[liq_feat].dropna()
        
        # Define regimes
        q25 = np.percentile(liq, 25)
        
        regime_defs = [
            ('low_liquidity', liq <= q25, 'illiquid'),
            ('normal_liquidity', liq > q25, 'liquid'),
        ]
        
        combined = pd.concat([features, returns.rename('ret')], axis=1).dropna()
        
        for name, mask, regime_type in regime_defs:
            regime_mask = mask.reindex(combined.index).fillna(False)
            regime_returns = combined.loc[regime_mask, 'ret']
            
            if len(regime_returns) >= 50:
                win_rate = (regime_returns > 0).mean()
                avg_return = regime_returns.mean()
                frequency = regime_mask.sum() / len(regime_mask)
                
                if win_rate < 0.45:
                    recommendation = 'avoid'
                elif win_rate < 0.50:
                    recommendation = 'reduce_size'
                else:
                    recommendation = 'trade'
                    
                regimes.append(RegimeAnalysis(
                    regime_id=name,
                    regime_type=regime_type,
                    detection_features=[liq_feat],
                    win_rate=win_rate,
                    avg_return=avg_return,
                    frequency=frequency,
                    recommendation=recommendation
                ))
                
        return regimes


def create_avoid_filter(
    anti_patterns: List[AntiPattern],
    max_patterns: int = 5
) -> callable:
    """
    Create a filter function from anti-patterns.
    
    Args:
        anti_patterns: List of discovered anti-patterns
        max_patterns: Maximum number of patterns to use
        
    Returns:
        Function that takes features and returns boolean mask (True = trade OK)
    """
    # Select top patterns by impact (loss_rate * frequency)
    selected = sorted(
        anti_patterns,
        key=lambda p: -(p.loss_rate * p.frequency)
    )[:max_patterns]
    
    def avoid_filter(features: pd.DataFrame) -> pd.Series:
        """Return True where trading is OK (no anti-patterns active)."""
        ok_mask = pd.Series(True, index=features.index)
        
        for pattern in selected:
            avoid_mask = pattern.evaluate(features)
            ok_mask = ok_mask & ~avoid_mask
            
        return ok_mask
        
    return avoid_filter
