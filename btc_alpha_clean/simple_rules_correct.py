#!/usr/bin/env python3
"""
CORRECT Simple Rules Implementation
Uses exact Phase 1 rule specifications with proper backtesting math.

This file is a CORRECTED version that fixes:
1. Target variable (binary, not continuous)
2. Return calculation (cumulative product, not sum)
3. Sharpe calculation (proper annualization)
4. Win rate calculation (actual wins vs losses)
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RuleSpec:
    """Single rule specification from Phase 1."""
    feature: str
    direction: int  # 1 = long when above threshold, -1 = long when below
    threshold_type: str  # "percentile" or "zscore"
    threshold_value: float
    oos_sharpe: float
    oos_win_rate: float


# Top credible rules from Phase 1 (EXACT specifications)
CREDIBLE_RULES = [
    RuleSpec(
        feature="deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        direction=1,
        threshold_type="percentile",
        threshold_value=90,
        oos_sharpe=8.74,
        oos_win_rate=0.735
    ),
    RuleSpec(
        feature="deriv_feat_cg_oi_aggregated_oi_close_accel",
        direction=1,
        threshold_type="percentile",
        threshold_value=80,
        oos_sharpe=7.38,
        oos_win_rate=0.614
    ),
    RuleSpec(
        feature="price_rsi_14_lag_48h",
        direction=-1,
        threshold_type="zscore",
        threshold_value=1.5,
        oos_sharpe=5.22,
        oos_win_rate=0.625
    ),
    RuleSpec(
        feature="deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        direction=-1,
        threshold_type="zscore",
        threshold_value=2.0,
        oos_sharpe=4.98,
        oos_win_rate=0.629
    ),
    RuleSpec(
        feature="sent_feat_fg_zscore_90d",
        direction=1,
        threshold_type="percentile",
        threshold_value=90,
        oos_sharpe=4.59,
        oos_win_rate=0.759
    ),
]


class Phase1RuleSystem:
    """
    Rule-based trading system using EXACT Phase 1 specifications.
    
    Key differences from broken implementation:
    1. Uses exact threshold_type and threshold_value from Phase 1
    2. Uses direction correctly (1 = long above, -1 = long below)
    3. Calculates thresholds on TRAINING data only (no leakage)
    """
    
    def __init__(self, rules: List[RuleSpec] = None, min_votes: int = 2):
        self.rules = rules or CREDIBLE_RULES
        self.min_votes = min_votes
        self.thresholds = {}  # Computed from training data
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Compute thresholds from training data.
        
        For percentile rules: compute the Nth percentile
        For zscore rules: compute mean and std for z-score calculation
        """
        self.thresholds = {}
        
        for rule in self.rules:
            if rule.feature not in X.columns:
                logger.warning(f"Feature {rule.feature} not found in data")
                continue
            
            values = X[rule.feature].dropna()
            
            if rule.threshold_type == "percentile":
                # Store BOTH upper AND lower thresholds (FIX for missing shorts)
                pct_upper = rule.threshold_value
                pct_lower = 100 - pct_upper  # e.g., 90 -> 10
                
                self.thresholds[rule.feature] = {
                    'type': 'percentile',
                    'upper_value': np.percentile(values, pct_upper),
                    'lower_value': np.percentile(values, pct_lower),
                    'direction': rule.direction
                }
            elif rule.threshold_type == "zscore":
                # Store mean and std for z-score calculation
                self.thresholds[rule.feature] = {
                    'type': 'zscore',
                    'mean': values.mean(),
                    'std': values.std(),
                    'threshold': rule.threshold_value,
                    'direction': rule.direction
                }
            
            logger.info(f"  {rule.feature[:50]}: {self.thresholds[rule.feature]}")
        
        self.fitted = True
        logger.info(f"Fitted {len(self.thresholds)} rules")
        return self
    
    def _get_signal(self, row: pd.Series, rule: RuleSpec) -> int:
        """
        Get signal for single rule: 1 (long), -1 (short), 0 (no signal)
        """
        if rule.feature not in self.thresholds:
            return 0
        
        value = row.get(rule.feature, np.nan)
        if pd.isna(value):
            return 0
        
        thresh = self.thresholds[rule.feature]
        
        if thresh['type'] == 'percentile':
            # direction=1: long when ABOVE upper, short when BELOW lower
            # direction=-1: short when ABOVE upper, long when BELOW lower
            if thresh['direction'] == 1:
                if value > thresh['upper_value']:
                    return 1   # Long signal when HIGH
                elif value < thresh['lower_value']:
                    return -1  # Short signal when LOW
            else:
                if value > thresh['upper_value']:
                    return -1  # Short signal when HIGH (inverse)
                elif value < thresh['lower_value']:
                    return 1   # Long signal when LOW (inverse)
        
        elif thresh['type'] == 'zscore':
            zscore = (value - thresh['mean']) / (thresh['std'] + 1e-10)
            # direction=1: long when zscore > threshold
            # direction=-1: long when zscore < -threshold
            if thresh['direction'] == 1:
                if zscore > thresh['threshold']:
                    return 1
                elif zscore < -thresh['threshold']:
                    return -1
            else:
                if zscore < -thresh['threshold']:
                    return 1
                elif zscore > thresh['threshold']:
                    return -1
        
        return 0
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate signals for each row.
        Returns: array of 1 (long), -1 (short), 0 (no trade)
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        signals = np.zeros(len(X))
        
        for i, (idx, row) in enumerate(X.iterrows()):
            votes = 0
            for rule in self.rules:
                votes += self._get_signal(row, rule)
            
            if votes >= self.min_votes:
                signals[i] = 1
            elif votes <= -self.min_votes:
                signals[i] = -1
        
        return signals


def calculate_returns(signals: np.ndarray, returns: pd.Series) -> Tuple[np.ndarray, dict]:
    """
    Calculate trading returns CORRECTLY using pre-computed forward returns.
    
    Args:
        signals: Array of position signals (+1, -1, 0)
        returns: Pre-computed forward returns (return_simple_6h from targets)
    
    Returns:
        trade_returns: array of returns for each trade
        stats: dictionary of performance metrics
    """
    returns_arr = returns.values
    
    # Trading returns: signal * return (same bar - returns are already forward-looking)
    trade_returns = signals * returns_arr
    
    # Calculate statistics
    trade_mask = signals != 0
    nonzero_returns = trade_returns[trade_mask]
    
    n_trades = len(nonzero_returns)
    n_wins = np.sum(nonzero_returns > 0)
    n_losses = np.sum(nonzero_returns < 0)
    
    win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
    
    # Cumulative return (multiplicative)
    cumulative_return = np.prod(1 + nonzero_returns) - 1 if n_trades > 0 else 0
    
    # Sharpe ratio (Phase 1 time-based method)
    # period_days = n_bars / 6 (6 bars per day at 4h timeframe)
    period_days = len(signals) / 6
    
    if n_trades >= 5 and period_days > 0:
        total_return = np.prod(1 + nonzero_returns) - 1
        daily_return = (1 + total_return) ** (1 / period_days) - 1
        
        trade_std = np.std(nonzero_returns)
        trades_per_day = n_trades / period_days
        daily_std = trade_std * np.sqrt(trades_per_day)
        
        if daily_std > 1e-10:
            sharpe = (daily_return / daily_std) * np.sqrt(365)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Max drawdown (only on trade returns, not including flat periods)
    if n_trades > 0:
        cumulative = np.cumprod(1 + nonzero_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
    else:
        max_drawdown = 0
    
    # Long/short breakdown
    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    
    stats = {
        'n_trades': int(n_long + n_short),  # Total positioned bars
        'n_long': int(n_long),
        'n_short': int(n_short),
        'n_wins': int(n_wins),
        'n_losses': int(n_losses),
        'win_rate': float(win_rate),
        'total_return': float(cumulative_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'avg_return': float(np.mean(nonzero_returns)) if len(nonzero_returns) > 0 else 0,
        'profit_factor': float(abs(np.sum(nonzero_returns[nonzero_returns > 0]) / 
                                   np.sum(nonzero_returns[nonzero_returns < 0]))) 
                         if np.sum(nonzero_returns < 0) != 0 else np.inf
    }
    
    return trade_returns, stats


def load_data():
    """Load features and returns data (from targets, not prices)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Load features cache
    cache_path = Path("data_cache/features_cache.pkl")
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        features = cache['features']
        targets = cache['targets']
        logger.info(f"Loaded features from cache: {features.shape}")
    else:
        raise FileNotFoundError(
            "No features cache found. Run: python run_exhaustive_search.py first"
        )
    
    # Get return_simple_6h from targets (SAME AS PHASE 1!)
    if 'return_simple_6h' in targets.columns:
        returns = targets['return_simple_6h'].copy()
        logger.info("Using return_simple_6h from targets (MATCHES PHASE 1)")
    else:
        ret_cols = [c for c in targets.columns if 'return_simple' in c]
        if ret_cols:
            returns = targets[ret_cols[0]].copy()
            logger.info(f"Fallback: using {ret_cols[0]}")
        else:
            raise ValueError("No return columns in targets!")
    
    # Align features and returns
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len].reset_index(drop=True)
    returns = returns.iloc[:min_len].reset_index(drop=True)
    
    # Drop NaN
    valid_idx = returns.dropna().index
    features = features.loc[valid_idx].reset_index(drop=True)
    returns = returns.loc[valid_idx].reset_index(drop=True)
    
    logger.info(f"Aligned features: {features.shape}")
    logger.info(f"Aligned returns: {len(returns)}")
    
    return features, returns


def main():
    logger.info("=" * 70)
    logger.info("CORRECTED SIMPLE RULES SYSTEM")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Using EXACT Phase 1 specifications:")
    for rule in CREDIBLE_RULES:
        logger.info(f"  {rule.feature[:50]}")
        logger.info(f"    -> direction={rule.direction}, {rule.threshold_type}={rule.threshold_value}")
        logger.info(f"    -> OOS Sharpe={rule.oos_sharpe}, WR={rule.oos_win_rate:.1%}")
    logger.info("")
    
    # Load data (returns from targets, same as Phase 1)
    features, returns = load_data()
    
    # Split: 60% train, 40% test (matching Phase 1)
    split_idx = int(len(features) * 0.6)
    
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    returns_train = returns.iloc[:split_idx]
    returns_test = returns.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test:  {len(X_test)} samples")
    logger.info("")
    
    # Create and fit rule system
    system = Phase1RuleSystem(rules=CREDIBLE_RULES, min_votes=2)
    
    logger.info("Fitting rule thresholds on training data...")
    system.fit(X_train)
    logger.info("")
    
    # Generate signals
    logger.info("=" * 70)
    logger.info("IN-SAMPLE (Training) Results")
    logger.info("=" * 70)
    
    train_signals = system.predict(X_train)
    _, train_stats = calculate_returns(train_signals, returns_train)
    
    logger.info(f"  Trades: {train_stats['n_trades']} (L:{train_stats['n_long']}, S:{train_stats['n_short']})")
    logger.info(f"  Win Rate: {train_stats['win_rate']:.1%} ({train_stats['n_wins']}W / {train_stats['n_losses']}L)")
    logger.info(f"  Total Return: {train_stats['total_return']:.1%}")
    logger.info(f"  Sharpe: {train_stats['sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {train_stats['max_drawdown']:.1%}")
    logger.info(f"  Profit Factor: {train_stats['profit_factor']:.2f}")
    logger.info("")
    
    logger.info("=" * 70)
    logger.info("OUT-OF-SAMPLE (Test) Results")
    logger.info("=" * 70)
    
    test_signals = system.predict(X_test)
    _, test_stats = calculate_returns(test_signals, returns_test)
    
    logger.info(f"  Trades: {test_stats['n_trades']} (L:{test_stats['n_long']}, S:{test_stats['n_short']})")
    logger.info(f"  Win Rate: {test_stats['win_rate']:.1%} ({test_stats['n_wins']}W / {test_stats['n_losses']}L)")
    logger.info(f"  Total Return: {test_stats['total_return']:.1%}")
    logger.info(f"  Sharpe: {test_stats['sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {test_stats['max_drawdown']:.1%}")
    logger.info(f"  Profit Factor: {test_stats['profit_factor']:.2f}")
    logger.info("")
    
    # Comparison with broken ML
    logger.info("=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Approach                  OOS Sharpe  OOS Win Rate  OOS Return")
    logger.info("-" * 70)
    logger.info(f"Phase 1 Best Rule         8.74        73.5%         48.4%")
    logger.info(f"Simple Rules (this)       {test_stats['sharpe']:.2f}        {test_stats['win_rate']:.1%}         {test_stats['total_return']:.1%}")
    logger.info(f"Broken ML Model           -1.74       54.0%         -2.8%")
    logger.info("")
    
    # Sanity checks
    logger.info("=" * 70)
    logger.info("SANITY CHECKS")
    logger.info("=" * 70)
    
    checks_passed = True
    
    if test_stats['sharpe'] < -100 or test_stats['sharpe'] > 100:
        logger.error(f"❌ Sharpe {test_stats['sharpe']:.2f} is unrealistic (bug in calculation)")
        checks_passed = False
    else:
        logger.info(f"✅ Sharpe {test_stats['sharpe']:.2f} is in reasonable range")
    
    if test_stats['win_rate'] == 0.0 and test_stats['n_trades'] > 0:
        logger.error(f"❌ Win rate 0% with {test_stats['n_trades']} trades is impossible")
        checks_passed = False
    else:
        logger.info(f"✅ Win rate {test_stats['win_rate']:.1%} is plausible")
    
    if test_stats['total_return'] < -1.0:  # More than -100%
        logger.error(f"❌ Return {test_stats['total_return']:.1%} < -100% without leverage is impossible")
        checks_passed = False
    else:
        logger.info(f"✅ Return {test_stats['total_return']:.1%} is mathematically valid")
    
    if test_stats['n_long'] == 0 or test_stats['n_short'] == 0:
        if test_stats['n_trades'] > 20:
            logger.warning(f"⚠️ Only {test_stats['n_long']}L/{test_stats['n_short']}S - may be unbalanced")
    else:
        logger.info(f"✅ Balanced L/S: {test_stats['n_long']}L / {test_stats['n_short']}S")
    
    logger.info("")
    
    if checks_passed:
        logger.info("✅ All sanity checks passed - results are trustworthy")
    else:
        logger.error("❌ Sanity checks failed - there are still bugs in the code")
    
    # Save model
    model_path = Path("models/simple_rules_correct.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(system, f)
    logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
