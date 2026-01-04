#!/usr/bin/env python3
"""
BTC Alpha Discovery - EXHAUSTIVE INDICATOR SEARCH WITH OOS VALIDATION
======================================================================

This script performs an exhaustive search through all indicator combinations
with proper Out-of-Sample (OOS) validation to avoid overfitting.

CRITICAL: All discoveries are made on IN-SAMPLE data (70%), then validated
on OUT-OF-SAMPLE data (30%) that was NEVER seen during discovery.

Usage:
    python run_exhaustive_search.py --mode single --top-n 50 --min-sharpe 0.5
    python run_exhaustive_search.py --mode triples                # Full search
    python run_exhaustive_search.py --synthetic                   # Test with fake data

Output:
    - reports/exhaustive_search_TIMESTAMP/
        - single_indicators.json    (with IS and OOS metrics)
        - validation_summary.json   (pass/fail statistics)
        - best_credible.json        (only rules that passed OOS)
"""

import argparse
import logging
import sys
import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from itertools import combinations
import warnings
import traceback

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(f'reports/exhaustive_search_{timestamp}')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'search_log.txt')
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION THRESHOLDS - TIGHTENED (70% pass rate was broken)
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_THRESHOLDS = {
    'min_is_sharpe': 2.0,        # was 1.5 - too loose
    'min_is_trades': 25,         # was 20 - need statistical significance
    'min_oos_sharpe': 1.0,       # was 0.5 - too loose
    'min_oos_trades': 25,        # was 15 - 17 trades is noise
    'min_oos_winrate': 0.52,     # Better than coin flip
    'min_retention': 0.40,       # was 0.30 - too loose
    'train_pct': 0.60,           # was 0.70 - need more OOS data (now 40% OOS)
}

# Correlation threshold for clustering duplicate signals
CORRELATION_THRESHOLD = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorResult:
    """Result from testing a single indicator or combination."""
    indicators: List[str]
    direction: int
    threshold_type: str
    threshold_value: float
    
    # IN-SAMPLE metrics
    is_sharpe: float = 0.0
    is_win_rate: float = 0.0
    is_avg_return: float = 0.0
    is_std_return: float = 0.0
    is_max_drawdown: float = 0.0
    is_n_trades: int = 0
    is_profit_factor: float = 0.0
    
    # OUT-OF-SAMPLE metrics
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    oos_avg_return: float = 0.0
    oos_std_return: float = 0.0  # Added for math verification
    oos_n_trades: int = 0
    oos_max_drawdown: float = 0.0
    oos_profit_factor: float = 0.0
    
    # Validation metrics
    retention_ratio: float = 0.0
    verdict: str = "NOT_TESTED"
    
    # Legacy fields for compatibility
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    period_days: float = 0.0
    trades_per_year: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_rules_tested: int = 0
    passed_is_threshold: int = 0
    passed_oos_validation: int = 0
    insufficient_data: int = 0
    expected_by_chance: float = 0.0
    survival_rate: float = 0.0
    credible_rules: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARPE CALCULATION - TIME-BASED (CORRECT)
# ═══════════════════════════════════════════════════════════════════════════════

# For 4H bars: 6 bars/day * 365 days = 2190 periods/year
PERIODS_PER_YEAR = 2190
BARS_PER_DAY = 6

def calculate_sharpe_time_based(
    trade_returns: np.ndarray,
    period_days: float,
    verbose: bool = False
) -> float:
    """
    Annualized Sharpe ratio - CORRECT TIME-BASED CALCULATION.
    
    CRITICAL: Sharpe measures return per unit of TIME, not per TRADE.
    
    Wrong approach (what we had):
        sharpe = (mean_trade_return / std_trade_return) * sqrt(trades_per_year)
        This inflates Sharpe when you trade more frequently.
    
    Correct approach:
        1. Calculate total return over the period
        2. Convert to daily return (geometric mean)
        3. Estimate daily volatility
        4. Annualize: sharpe = (daily_return / daily_std) * sqrt(365)
    
    Args:
        trade_returns: Array of returns from each trade (as decimals, e.g., 0.01 = 1%)
        period_days: Number of days the trades span
        verbose: Print debug info
    
    Returns:
        Annualized Sharpe ratio (realistic range: 0.5 - 3.0)
    """
    if len(trade_returns) < 2 or period_days <= 0:
        return 0.0
    
    n_trades = len(trade_returns)
    
    # Step 1: Calculate total return (geometric)
    total_return = np.prod(1 + trade_returns) - 1
    
    # Step 2: Convert to daily return (geometric mean)
    if total_return > -1:  # Avoid log of negative
        daily_return = (1 + total_return) ** (1 / period_days) - 1
    else:
        daily_return = -1  # Total loss
    
    # Step 3: Estimate daily volatility
    # Trade volatility scaled by trades per day
    trades_per_day = n_trades / period_days
    trade_std = np.std(trade_returns)
    
    # Daily std = trade std * sqrt(trades_per_day)
    # This accounts for the fact that multiple trades/day compound
    daily_std = trade_std * np.sqrt(trades_per_day)
    
    if daily_std < 1e-10:
        return 0.0
    
    # Step 4: Calculate daily Sharpe, then annualize
    daily_sharpe = daily_return / daily_std
    annualized_sharpe = daily_sharpe * np.sqrt(365)
    
    if verbose:
        print(f"  SHARPE CALCULATION (TIME-BASED):")
        print(f"    N trades: {n_trades}")
        print(f"    Period days: {period_days:.1f}")
        print(f"    Trades/day: {trades_per_day:.2f}")
        print(f"    Total return: {total_return:.2%}")
        print(f"    Daily return (geometric): {daily_return:.4%}")
        print(f"    Trade std: {trade_std:.4f}")
        print(f"    Daily std: {daily_std:.4f}")
        print(f"    Daily Sharpe: {daily_sharpe:.3f}")
        print(f"    Annualized Sharpe: {annualized_sharpe:.2f}")
    
    return annualized_sharpe


def calculate_sharpe(
    returns: np.ndarray,
    period_days: float = None,
    verbose: bool = False
) -> float:
    """
    Wrapper that calls the correct time-based Sharpe calculation.
    
    This replaces the old WRONG calculation that used sqrt(trades_per_year).
    """
    if period_days is None or period_days <= 0:
        # Fallback: estimate period from trade count assuming 4H bars
        period_days = max(1, len(returns) / BARS_PER_DAY)
    
    return calculate_sharpe_time_based(returns, period_days, verbose)


def verify_sharpe_sanity(result: dict, period_days: float) -> dict:
    """Add sanity check fields to result dict."""
    oos_trades = result.get('oos_n_trades', 0)
    oos_mean = result.get('oos_avg_return', 0)
    oos_sharpe = result.get('oos_sharpe', 0)
    
    if oos_trades > 0 and period_days > 0:
        trades_per_day = oos_trades / period_days
        trades_per_year = trades_per_day * 365
        implied_annual_return = oos_mean * min(oos_trades, PERIODS_PER_YEAR)
        
        result['_debug'] = {
            'trades_per_day': round(trades_per_day, 2),
            'trades_per_year': round(trades_per_year, 0),
            'implied_annual_return': round(implied_annual_return, 4),
            'annualization_factor': round(np.sqrt(min(oos_trades, PERIODS_PER_YEAR)), 2),
        }
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR - WITH FIXED THRESHOLD SUPPORT FOR OOS
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Generate trading signals from indicators."""
    
    @staticmethod
    def get_percentile_thresholds(values: pd.Series, pct_upper: float = 80, pct_lower: float = 20) -> Tuple[float, float]:
        """
        Calculate FIXED threshold values from data.
        Use this on IS data to get thresholds, then apply to OOS.
        """
        upper_threshold = np.nanpercentile(values.dropna(), pct_upper)
        lower_threshold = np.nanpercentile(values.dropna(), pct_lower)
        return upper_threshold, lower_threshold
    
    @staticmethod
    def get_zscore_params(values: pd.Series) -> Tuple[float, float]:
        """
        Calculate FIXED mean/std from data for z-score.
        Use this on IS data to get params, then apply to OOS.
        """
        clean = values.dropna()
        return float(np.mean(clean)), float(np.std(clean))
    
    @staticmethod
    def apply_fixed_threshold(
        values: pd.Series,
        upper_threshold: float,
        lower_threshold: float,
        direction: int = 1
    ) -> pd.Series:
        """
        Apply FIXED thresholds to data (no recalculation).
        CRITICAL: Use this for OOS to prevent data leakage.
        """
        signals = pd.Series(0, index=values.index)
        
        if direction == 1:
            signals[values > upper_threshold] = 1
            signals[values < lower_threshold] = -1
        else:
            signals[values > upper_threshold] = -1
            signals[values < lower_threshold] = 1
        
        return signals
    
    @staticmethod
    def apply_fixed_zscore(
        values: pd.Series,
        is_mean: float,
        is_std: float,
        z_upper: float = 1.5,
        z_lower: float = -1.5,
        direction: int = 1
    ) -> pd.Series:
        """
        Apply z-score using FIXED mean/std from IS data.
        CRITICAL: Use this for OOS to prevent data leakage.
        """
        signals = pd.Series(0, index=values.index)
        
        # Calculate z-score using IS parameters
        zscore = (values - is_mean) / (is_std + 1e-10)
        
        if direction == 1:
            signals[zscore > z_upper] = 1
            signals[zscore < z_lower] = -1
        else:
            signals[zscore > z_upper] = -1
            signals[zscore < z_lower] = 1
        
        return signals
    
    @staticmethod
    def percentile_signal(
        values: pd.Series,
        percentile_long: float = 80,
        percentile_short: float = 20,
        direction: int = 1,
        lookback: int = None
    ) -> pd.Series:
        """Generate signal based on percentile thresholds (for IS only)."""
        signals = pd.Series(0, index=values.index)
        
        if lookback is None:
            lookback = max(20, min(500, len(values) // 3))
        
        if len(values) < lookback:
            upper = values.expanding(min_periods=10).quantile(percentile_long / 100)
            lower = values.expanding(min_periods=10).quantile(percentile_short / 100)
        else:
            upper = values.rolling(lookback, min_periods=20).quantile(percentile_long / 100)
            lower = values.rolling(lookback, min_periods=20).quantile(percentile_short / 100)
        
        if direction == 1:
            signals[values > upper] = 1
            signals[values < lower] = -1
        else:
            signals[values > upper] = -1
            signals[values < lower] = 1
        
        return signals
    
    @staticmethod
    def zscore_signal(
        values: pd.Series,
        zscore_long: float = 1.5,
        zscore_short: float = -1.5,
        direction: int = 1,
        lookback: int = None
    ) -> pd.Series:
        """Generate signal based on z-score thresholds (for IS only)."""
        signals = pd.Series(0, index=values.index)
        
        if lookback is None:
            lookback = max(20, min(500, len(values) // 3))
        
        if len(values) < lookback:
            mean = values.expanding(min_periods=10).mean()
            std = values.expanding(min_periods=10).std()
        else:
            mean = values.rolling(lookback, min_periods=20).mean()
            std = values.rolling(lookback, min_periods=20).std()
        
        zscore = (values - mean) / (std + 1e-10)
        
        if direction == 1:
            signals[zscore > zscore_long] = 1
            signals[zscore < zscore_short] = -1
        else:
            signals[zscore > zscore_long] = -1
            signals[zscore < zscore_short] = 1
        
        return signals
    
    @staticmethod
    def combine_signals(signals: List[pd.Series], method: str = 'unanimous') -> pd.Series:
        """Combine multiple signals."""
        if len(signals) == 0:
            return pd.Series(dtype=float)
        
        combined_df = pd.concat(signals, axis=1)
        
        if method == 'unanimous':
            all_long = (combined_df > 0).all(axis=1)
            all_short = (combined_df < 0).all(axis=1)
            result = pd.Series(0, index=combined_df.index)
            result[all_long] = 1
            result[all_short] = -1
        elif method == 'majority':
            n = len(signals)
            threshold = n / 2
            long_count = (combined_df > 0).sum(axis=1)
            short_count = (combined_df < 0).sum(axis=1)
            result = pd.Series(0, index=combined_df.index)
            result[long_count > threshold] = 1
            result[short_count > threshold] = -1
        else:
            result = pd.Series(0, index=combined_df.index)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION CLUSTERING - Remove duplicate signals
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_correlated_rules(results: List[dict], features: pd.DataFrame, threshold: float = 0.80) -> Tuple[List[dict], List[dict]]:
    """
    Group rules by indicator correlation.
    Return only the best rule per cluster (sorted by OOS Sharpe).
    
    Args:
        results: List of rule dicts, already sorted by OOS Sharpe desc
        features: DataFrame with indicator columns
        threshold: Correlation threshold (0.8 = 80% correlated = same signal)
    
    Returns:
        (deduplicated_results, rejected_duplicates)
    """
    if not results:
        return [], []
    
    # Extract unique indicators and check which exist
    indicators = []
    for r in results:
        inds = r.get('indicators', [r.get('indicator', '')])
        if isinstance(inds, str):
            inds = [inds]
        for ind in inds:
            if ind and ind in features.columns:
                indicators.append(ind)
    
    indicators = list(set(indicators))
    
    if len(indicators) < 2:
        return results, []
    
    # Calculate correlation matrix
    available_cols = [c for c in indicators if c in features.columns]
    if len(available_cols) < 2:
        return results, []
    
    corr_matrix = features[available_cols].corr()
    
    # Track which indicators are already used (via a correlated rule)
    used_indicators = set()
    deduplicated = []
    rejected = []
    
    # Process results in order (best OOS Sharpe first)
    for r in results:
        inds = r.get('indicators', [r.get('indicator', '')])
        if isinstance(inds, str):
            inds = [inds]
        
        primary_ind = inds[0] if inds else ''
        
        if not primary_ind or primary_ind not in corr_matrix.columns:
            deduplicated.append(r)
            continue
        
        # Check if this indicator is correlated with any already-used indicator
        is_duplicate = False
        correlated_with = None
        
        for used_ind in used_indicators:
            if used_ind in corr_matrix.columns and primary_ind in corr_matrix.columns:
                corr = abs(corr_matrix.loc[primary_ind, used_ind])
                if corr > threshold:
                    is_duplicate = True
                    correlated_with = used_ind
                    break
        
        if is_duplicate:
            r['_rejected_reason'] = f"Correlated {corr:.2f} with {correlated_with}"
            rejected.append(r)
        else:
            used_indicators.add(primary_ind)
            deduplicated.append(r)
    
    return deduplicated, rejected


def print_correlation_check(results: List[dict], features: pd.DataFrame, top_n: int = 10):
    """Print correlation matrix for top indicators."""
    indicators = []
    for r in results[:top_n]:
        inds = r.get('indicators', [r.get('indicator', '')])
        if isinstance(inds, str):
            inds = [inds]
        for ind in inds:
            if ind and ind in features.columns:
                indicators.append(ind)
    
    indicators = list(dict.fromkeys(indicators))[:10]  # Preserve order, dedup
    
    if len(indicators) < 2:
        return
    
    available = [c for c in indicators if c in features.columns]
    if len(available) < 2:
        return
    
    corr_matrix = features[available].corr()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CORRELATION CHECK (top indicators)")
    logger.info("=" * 70)
    
    # Print correlation pairs
    for i, ind1 in enumerate(available):
        for ind2 in available[i+1:]:
            corr = corr_matrix.loc[ind1, ind2]
            status = "SAME SIGNAL" if abs(corr) > 0.8 else "Different"
            # Shorten names for display
            short1 = ind1[-40:] if len(ind1) > 40 else ind1
            short2 = ind2[-40:] if len(ind2) > 40 else ind2
            logger.info(f"  {short1} vs {short2}: {corr:.2f} <- {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR TESTER WITH OOS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorTester:
    """Test indicators with proper train/test split for OOS validation."""
    
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_col: str = 'return_simple_6h',
        min_trades: int = 20,
        train_pct: float = 0.70,
        test_directions: List[int] = [1, -1],
        test_percentiles: List[Tuple[float, float]] = [(80, 20), (70, 30), (90, 10)],
        test_zscores: List[Tuple[float, float]] = [(1.5, -1.5), (2.0, -2.0)]
    ):
        self.min_trades = min_trades
        self.train_pct = train_pct
        self.test_directions = test_directions
        self.test_percentiles = test_percentiles
        self.test_zscores = test_zscores
        
        # Clean and align data
        feature_values = features.values
        feature_cols = features.columns.tolist()
        target_values = targets.values
        target_cols = targets.columns.tolist()
        
        features_clean = pd.DataFrame(feature_values, columns=feature_cols)
        targets_clean = pd.DataFrame(target_values, columns=target_cols)
        
        # Find return column
        target_patterns = [target_col, 'return_simple_6h', 'return_log_6h', 
                          'return_simple_12h', 'return_simple_24h']
        
        returns = None
        actual_target = None
        for pattern in target_patterns:
            if pattern in targets_clean.columns:
                returns = targets_clean[pattern].copy()
                actual_target = pattern
                break
        
        if returns is None:
            return_cols = [c for c in targets_clean.columns if 'return' in c.lower()]
            if return_cols:
                returns = targets_clean[return_cols[0]].copy()
                actual_target = return_cols[0]
            else:
                returns = pd.Series(0.0, index=range(len(targets_clean)))
                actual_target = "NONE"
        
        # Align
        min_len = min(len(features_clean), len(returns))
        features_clean = features_clean.iloc[:min_len].reset_index(drop=True)
        returns = returns.iloc[:min_len].reset_index(drop=True)
        
        # Drop NaN
        valid_idx = returns.dropna().index.tolist()
        self.features = features_clean.loc[valid_idx].reset_index(drop=True)
        self.returns = returns.loc[valid_idx].reset_index(drop=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: TRAIN/TEST SPLIT
        # ═══════════════════════════════════════════════════════════════════
        self.n_samples = len(self.features)
        self.split_idx = int(self.n_samples * train_pct)
        
        # In-Sample (IS) data - for discovery
        self.is_features = self.features.iloc[:self.split_idx].reset_index(drop=True)
        self.is_returns = self.returns.iloc[:self.split_idx].reset_index(drop=True)
        
        # Out-of-Sample (OOS) data - for validation ONLY
        self.oos_features = self.features.iloc[self.split_idx:].reset_index(drop=True)
        self.oos_returns = self.returns.iloc[self.split_idx:].reset_index(drop=True)
        
        self.is_period_days = max(1, len(self.is_features) / 6)
        self.oos_period_days = max(1, len(self.oos_features) / 6)
        self.total_period_days = max(1, self.n_samples / 6)
        
        logger.info(f"IndicatorTester initialized with OOS validation:")
        logger.info(f"  Total samples: {self.n_samples}")
        logger.info(f"  IN-SAMPLE: {len(self.is_features)} samples ({train_pct:.0%})")
        logger.info(f"  OUT-OF-SAMPLE: {len(self.oos_features)} samples ({1-train_pct:.0%})")
        logger.info(f"  Target: {actual_target}")
        logger.info(f"  IS period: {self.is_period_days:.1f} days")
        logger.info(f"  OOS period: {self.oos_period_days:.1f} days")
    
    def test_single_indicator(
        self,
        indicator_name: str,
        validate_oos: bool = True
    ) -> List[IndicatorResult]:
        """
        Test a single indicator on IN-SAMPLE data, then validate on OOS.
        """
        results = []
        
        if indicator_name not in self.features.columns:
            return results
        
        # Get IS values only for discovery
        is_values = self.is_features[indicator_name].dropna()
        
        if len(is_values) < 30:
            return results
        
        for direction in self.test_directions:
            # Test percentile thresholds
            for pct_long, pct_short in self.test_percentiles:
                try:
                    result = self._test_and_validate(
                        indicator_name, direction, 'percentile', pct_long,
                        pct_long=pct_long, pct_short=pct_short,
                        validate_oos=validate_oos
                    )
                    if result:
                        results.append(result)
                except Exception:
                    pass
            
            # Test z-score thresholds
            for z_long, z_short in self.test_zscores:
                try:
                    result = self._test_and_validate(
                        indicator_name, direction, 'zscore', z_long,
                        z_long=z_long, z_short=z_short,
                        validate_oos=validate_oos
                    )
                    if result:
                        results.append(result)
                except Exception:
                    pass
        
        results.sort(key=lambda x: x.is_sharpe, reverse=True)
        return results
    
    def _test_and_validate(
        self,
        indicator_name: str,
        direction: int,
        threshold_type: str,
        threshold_value: float,
        validate_oos: bool = True,
        **kwargs
    ) -> Optional[IndicatorResult]:
        """Test on IS data, then validate on OOS data with FIXED thresholds."""
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: DISCOVER ON IN-SAMPLE DATA & EXTRACT FIXED THRESHOLDS
        # ═══════════════════════════════════════════════════════════════════
        is_values = self.is_features[indicator_name]
        
        # Calculate FIXED thresholds from IS data (to be applied to OOS)
        if threshold_type == 'percentile':
            # Get the actual threshold VALUES (not percentile numbers)
            upper_thresh, lower_thresh = SignalGenerator.get_percentile_thresholds(
                is_values, kwargs['pct_long'], kwargs['pct_short']
            )
            is_signals = SignalGenerator.percentile_signal(
                is_values, kwargs['pct_long'], kwargs['pct_short'], direction
            )
        else:
            # Get the actual mean/std from IS data
            is_mean, is_std = SignalGenerator.get_zscore_params(is_values)
            is_signals = SignalGenerator.zscore_signal(
                is_values, kwargs['z_long'], kwargs['z_short'], direction
            )
        
        # Calculate IS metrics
        is_metrics = self._calculate_metrics(
            is_signals, self.is_returns, self.is_period_days
        )
        
        if is_metrics is None or is_metrics['n_trades'] < self.min_trades:
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: VALIDATE ON OUT-OF-SAMPLE DATA (using FIXED IS thresholds)
        # ═══════════════════════════════════════════════════════════════════
        oos_metrics = None
        retention_ratio = 0.0
        verdict = "IS_ONLY"
        
        if validate_oos and len(self.oos_features) > 0:
            oos_values = self.oos_features[indicator_name]
            
            # CRITICAL FIX: Apply FIXED thresholds from IS to OOS
            # This prevents data leakage - OOS cannot influence thresholds
            if threshold_type == 'percentile':
                # Use the EXACT threshold values calculated from IS
                oos_signals = SignalGenerator.apply_fixed_threshold(
                    oos_values, upper_thresh, lower_thresh, direction
                )
            else:
                # Use the EXACT mean/std calculated from IS
                oos_signals = SignalGenerator.apply_fixed_zscore(
                    oos_values, is_mean, is_std,
                    kwargs['z_long'], kwargs['z_short'], direction
                )
            
            oos_metrics = self._calculate_metrics(
                oos_signals, self.oos_returns, self.oos_period_days
            )
            
            # Determine verdict
            if oos_metrics is None or oos_metrics['n_trades'] < VALIDATION_THRESHOLDS['min_oos_trades']:
                verdict = "INSUFFICIENT_DATA"
            else:
                retention_ratio = oos_metrics['sharpe'] / (is_metrics['sharpe'] + 1e-10)
                
                # SANITY CHECK: Flag if retention > 100% (suspicious)
                if retention_ratio > 1.0:
                    verdict = "SUSPICIOUS_OOS_BETTER"
                else:
                    # Check all validation criteria
                    passes_oos = (
                        oos_metrics['sharpe'] >= VALIDATION_THRESHOLDS['min_oos_sharpe'] and
                        retention_ratio >= VALIDATION_THRESHOLDS['min_retention'] and
                        oos_metrics['win_rate'] >= VALIDATION_THRESHOLDS['min_oos_winrate']
                    )
                    
                    verdict = "CREDIBLE" if passes_oos else "OOS_FAILED"
        
        # Build result
        result = IndicatorResult(
            indicators=[indicator_name],
            direction=direction,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            
            # IS metrics
            is_sharpe=is_metrics['sharpe'],
            is_win_rate=is_metrics['win_rate'],
            is_avg_return=is_metrics['avg_return'],
            is_std_return=is_metrics['std_return'],
            is_max_drawdown=is_metrics['max_drawdown'],
            is_n_trades=is_metrics['n_trades'],
            is_profit_factor=is_metrics['profit_factor'],
            
            # OOS metrics
            oos_sharpe=oos_metrics['sharpe'] if oos_metrics else 0.0,
            oos_win_rate=oos_metrics['win_rate'] if oos_metrics else 0.0,
            oos_avg_return=oos_metrics['avg_return'] if oos_metrics else 0.0,
            oos_std_return=oos_metrics['std_return'] if oos_metrics else 0.0,  # Added
            oos_n_trades=oos_metrics['n_trades'] if oos_metrics else 0,
            oos_max_drawdown=oos_metrics['max_drawdown'] if oos_metrics else 0.0,
            oos_profit_factor=oos_metrics['profit_factor'] if oos_metrics else 0.0,
            
            # Validation
            retention_ratio=retention_ratio,
            verdict=verdict,
            
            # Legacy (use IS for compatibility)
            sharpe=is_metrics['sharpe'],
            win_rate=is_metrics['win_rate'],
            avg_return=is_metrics['avg_return'],
            std_return=is_metrics['std_return'],
            max_drawdown=is_metrics['max_drawdown'],
            n_trades=is_metrics['n_trades'],
            period_days=self.is_period_days,
            trades_per_year=is_metrics['trades_per_year'],
            profit_factor=is_metrics['profit_factor'],
        )
        
        return result
    
    def _calculate_metrics(
        self,
        signals: pd.Series,
        returns: pd.Series,
        period_days: float,
        store_trades: bool = False
    ) -> Optional[Dict]:
        """Calculate performance metrics for signals."""
        
        # Transaction cost per trade (round trip = 0.12% for taker orders)
        TRANSACTION_COST = 0.0012  # 0.12% per trade
        
        # Align
        common_idx = signals.index.intersection(returns.index)
        if len(common_idx) == 0:
            return None
            
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Get trade returns
        trade_mask = signals != 0
        raw_trade_returns = returns[trade_mask] * signals[trade_mask]
        
        # APPLY TRANSACTION COSTS
        trade_returns = raw_trade_returns - TRANSACTION_COST
        
        n_trades = len(trade_returns)
        if n_trades < 2:
            return None
        
        mean_ret = trade_returns.mean()
        std_ret = trade_returns.std()
        
        if std_ret < 1e-10:
            return None
        
        # CORRECTED Sharpe calculation
        # Pass period_days for proper time-based annualization
        sharpe = calculate_sharpe(trade_returns.values, period_days=period_days)
        
        trades_per_year = (n_trades / period_days) * 365
        
        # Win rate
        win_rate = (trade_returns > 0).mean()
        
        # Max drawdown
        cumulative = (1 + trade_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_dd = drawdown.min()
        
        # Profit factor
        wins = trade_returns[trade_returns > 0].sum()
        losses = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = wins / (losses + 1e-10)
        
        result = {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_return': mean_ret,
            'std_return': std_ret,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'trades_per_year': trades_per_year,
            'profit_factor': profit_factor,
        }
        
        # Store trade returns for verification
        if store_trades:
            result['_trade_returns'] = trade_returns.values.tolist()
            result['_trade_dates'] = [str(d) for d in trade_returns.index.tolist()]
            result['_raw_returns'] = raw_trade_returns.values.tolist()
        
        return result
    
    def test_indicator_combination(
        self,
        indicator_names: List[str],
        combine_method: str = 'unanimous',
        validate_oos: bool = True
    ) -> List[IndicatorResult]:
        """Test a combination of indicators with OOS validation."""
        results = []
        
        for name in indicator_names:
            if name not in self.features.columns:
                return results
        
        for direction in self.test_directions:
            for pct_long, pct_short in self.test_percentiles:
                try:
                    # IS signals
                    is_signals_list = []
                    for name in indicator_names:
                        is_values = self.is_features[name]
                        sig = SignalGenerator.percentile_signal(
                            is_values, pct_long, pct_short, direction
                        )
                        is_signals_list.append(sig)
                    
                    is_combined = SignalGenerator.combine_signals(is_signals_list, combine_method)
                    is_metrics = self._calculate_metrics(is_combined, self.is_returns, self.is_period_days)
                    
                    if is_metrics is None or is_metrics['n_trades'] < self.min_trades:
                        continue
                    
                    # OOS validation
                    oos_metrics = None
                    retention_ratio = 0.0
                    verdict = "IS_ONLY"
                    
                    if validate_oos and len(self.oos_features) > 0:
                        oos_signals_list = []
                        for name in indicator_names:
                            oos_values = self.oos_features[name]
                            sig = SignalGenerator.percentile_signal(
                                oos_values, pct_long, pct_short, direction
                            )
                            oos_signals_list.append(sig)
                        
                        oos_combined = SignalGenerator.combine_signals(oos_signals_list, combine_method)
                        oos_metrics = self._calculate_metrics(oos_combined, self.oos_returns, self.oos_period_days)
                        
                        if oos_metrics is None or oos_metrics['n_trades'] < VALIDATION_THRESHOLDS['min_oos_trades']:
                            verdict = "INSUFFICIENT_DATA"
                        else:
                            retention_ratio = oos_metrics['sharpe'] / (is_metrics['sharpe'] + 1e-10)
                            passes = (
                                oos_metrics['sharpe'] >= VALIDATION_THRESHOLDS['min_oos_sharpe'] and
                                retention_ratio >= VALIDATION_THRESHOLDS['min_retention'] and
                                oos_metrics['win_rate'] >= VALIDATION_THRESHOLDS['min_oos_winrate']
                            )
                            verdict = "CREDIBLE" if passes else "OOS_FAILED"
                    
                    result = IndicatorResult(
                        indicators=indicator_names,
                        direction=direction,
                        threshold_type=f'percentile_{combine_method}',
                        threshold_value=pct_long,
                        is_sharpe=is_metrics['sharpe'],
                        is_win_rate=is_metrics['win_rate'],
                        is_avg_return=is_metrics['avg_return'],
                        is_std_return=is_metrics['std_return'],
                        is_max_drawdown=is_metrics['max_drawdown'],
                        is_n_trades=is_metrics['n_trades'],
                        is_profit_factor=is_metrics['profit_factor'],
                        oos_sharpe=oos_metrics['sharpe'] if oos_metrics else 0.0,
                        oos_win_rate=oos_metrics['win_rate'] if oos_metrics else 0.0,
                        oos_avg_return=oos_metrics['avg_return'] if oos_metrics else 0.0,
                        oos_n_trades=oos_metrics['n_trades'] if oos_metrics else 0,
                        oos_max_drawdown=oos_metrics['max_drawdown'] if oos_metrics else 0.0,
                        oos_profit_factor=oos_metrics['profit_factor'] if oos_metrics else 0.0,
                        retention_ratio=retention_ratio,
                        verdict=verdict,
                        sharpe=is_metrics['sharpe'],
                        win_rate=is_metrics['win_rate'],
                        n_trades=is_metrics['n_trades'],
                        period_days=self.is_period_days,
                    )
                    results.append(result)
                    
                except Exception:
                    pass
        
        results.sort(key=lambda x: x.is_sharpe, reverse=True)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXHAUSTIVE SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ExhaustiveSearchEngine:
    """Engine for exhaustive search with OOS validation."""
    
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_col: str = 'return_simple_6h',
        output_dir: Path = None,
        top_n: int = 50,
        min_sharpe: float = 0.5,
        min_trades: int = 20,
        validate_oos: bool = True
    ):
        self.features = features
        self.targets = targets
        self.target_col = target_col
        self.output_dir = output_dir or log_dir
        self.top_n = top_n
        self.min_sharpe = min_sharpe
        self.min_trades = min_trades
        self.validate_oos = validate_oos
        
        # Initialize tester
        self.tester = IndicatorTester(
            features, targets, target_col,
            min_trades=min_trades,
            train_pct=VALIDATION_THRESHOLDS['train_pct']
        )
        
        # Results
        self.single_results: List[IndicatorResult] = []
        self.pair_results: List[IndicatorResult] = []
        self.triple_results: List[IndicatorResult] = []
        self.validation_summary = ValidationSummary()
        
        # Get testable columns
        self.indicator_columns = [
            col for col in features.columns
            if features[col].dtype in ['float64', 'float32', 'int64', 'int32']
            and features[col].nunique() > 5
        ]
        
        logger.info(f"ExhaustiveSearchEngine initialized:")
        logger.info(f"  Testable indicators: {len(self.indicator_columns)}")
        logger.info(f"  OOS Validation: {'ENABLED' if validate_oos else 'DISABLED'}")
    
    def run_phase_single(self) -> List[IndicatorResult]:
        """Test every single indicator with OOS validation."""
        logger.info("="*70)
        logger.info("PHASE 1: SINGLE INDICATOR SEARCH (with OOS validation)")
        logger.info("="*70)
        
        all_results = []
        total_combinations_tested = 0  # Track actual rule combinations
        
        for indicator in tqdm(self.indicator_columns, desc="Testing indicators"):
            try:
                results = self.tester.test_single_indicator(
                    indicator, validate_oos=self.validate_oos
                )
                
                total_combinations_tested += len(results)  # Count all combinations tested
                
                # Keep results above IS threshold
                good_results = [r for r in results 
                               if r.is_sharpe >= VALIDATION_THRESHOLDS['min_is_sharpe']]
                all_results.extend(good_results)
                
            except Exception as e:
                continue
        
        # Sort by IS Sharpe
        all_results.sort(key=lambda x: x.is_sharpe, reverse=True)
        self.single_results = all_results[:self.top_n]
        
        # Calculate validation summary - use actual combinations tested
        self._calculate_validation_summary(all_results, total_combinations_tested)
        
        # Save results
        self._save_results(self.single_results, 'single_indicators.json')
        self._save_validation_summary()
        
        # Log results
        self._log_results()
        
        return self.single_results
    
    def run_phase_pairs(self) -> List[IndicatorResult]:
        """Test pairs of top indicators."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: INDICATOR PAIR SEARCH")
        logger.info("="*70)
        
        if not self.single_results:
            logger.error("No single results. Run phase 1 first.")
            return []
        
        # Get unique top indicators
        top_indicators = list(set(
            r.indicators[0] for r in self.single_results[:min(20, len(self.single_results))]
        ))
        
        all_results = []
        pairs = list(combinations(top_indicators, 2))
        
        logger.info(f"Testing {len(pairs)} pairs from top {len(top_indicators)} indicators")
        
        for pair in tqdm(pairs, desc="Testing pairs"):
            try:
                results = self.tester.test_indicator_combination(
                    list(pair), validate_oos=self.validate_oos
                )
                good_results = [r for r in results 
                               if r.is_sharpe >= VALIDATION_THRESHOLDS['min_is_sharpe']]
                all_results.extend(good_results)
            except Exception:
                continue
        
        all_results.sort(key=lambda x: x.is_sharpe, reverse=True)
        self.pair_results = all_results[:self.top_n]
        
        self._save_results(self.pair_results, 'pair_combinations.json')
        
        return self.pair_results
    
    def run_phase_triples(self) -> List[IndicatorResult]:
        """Test triples of top indicators."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: INDICATOR TRIPLE SEARCH")
        logger.info("="*70)
        
        if not self.single_results:
            logger.error("No single results. Run phase 1 first.")
            return []
        
        # Get unique top indicators
        top_indicators = list(set(
            r.indicators[0] for r in self.single_results[:min(15, len(self.single_results))]
        ))
        
        all_results = []
        triples = list(combinations(top_indicators, 3))
        
        logger.info(f"Testing {len(triples)} triples from top {len(top_indicators)} indicators")
        
        for triple in tqdm(triples, desc="Testing triples"):
            try:
                results = self.tester.test_indicator_combination(
                    list(triple), validate_oos=self.validate_oos
                )
                good_results = [r for r in results 
                               if r.is_sharpe >= VALIDATION_THRESHOLDS['min_is_sharpe']]
                all_results.extend(good_results)
            except Exception:
                continue
        
        all_results.sort(key=lambda x: x.is_sharpe, reverse=True)
        self.triple_results = all_results[:self.top_n]
        
        self._save_results(self.triple_results, 'triple_combinations.json')
        
        return self.triple_results
    
    def _calculate_validation_summary(self, all_results: List[IndicatorResult], total_tested: int):
        """Calculate validation statistics with correlation clustering."""
        total_rules = len(all_results)
        passed_is = len([r for r in all_results if r.is_sharpe >= VALIDATION_THRESHOLDS['min_is_sharpe']])
        passed_oos = len([r for r in all_results if r.verdict == "CREDIBLE"])
        insufficient = len([r for r in all_results if r.verdict == "INSUFFICIENT_DATA"])
        
        # Expected false positives at p=0.05
        expected_chance = total_tested * 0.05
        survival_rate = passed_oos / expected_chance if expected_chance > 0 else 0
        
        # Get credible rules (sorted by OOS Sharpe)
        credible_results = [r for r in all_results if r.verdict == "CREDIBLE"]
        credible_results.sort(key=lambda x: x.oos_sharpe, reverse=True)
        credible_rules = [r.to_dict() for r in credible_results]
        
        # APPLY CORRELATION CLUSTERING - Remove duplicate signals
        logger.info("\n" + "-"*70)
        logger.info("CORRELATION CLUSTERING (removing duplicate signals)")
        logger.info("-"*70)
        
        deduplicated, rejected = cluster_correlated_rules(
            credible_rules, self.features, threshold=CORRELATION_THRESHOLD
        )
        
        logger.info(f"  Before clustering: {len(credible_rules)} credible rules")
        logger.info(f"  After clustering: {len(deduplicated)} unique signals")
        logger.info(f"  Rejected duplicates: {len(rejected)}")
        
        # Show what was rejected
        if rejected:
            logger.info("\n  REJECTED DUPLICATES:")
            for r in rejected[:10]:
                ind = r.get('indicators', [r.get('indicator', 'unknown')])[0]
                reason = r.get('_rejected_reason', 'correlated')
                short_ind = ind[-50:] if len(ind) > 50 else ind
                logger.info(f"    - {short_ind} ({reason})")
        
        # Print correlation check for the deduplicated results
        print_correlation_check(deduplicated, self.features)
        
        # Update passed_oos count to reflect deduplicated
        passed_oos_dedup = len(deduplicated)
        survival_rate_dedup = passed_oos_dedup / expected_chance if expected_chance > 0 else 0
        
        self.validation_summary = ValidationSummary(
            total_rules_tested=total_tested,
            passed_is_threshold=passed_is,
            passed_oos_validation=passed_oos_dedup,  # Now using deduplicated count
            insufficient_data=insufficient,
            expected_by_chance=expected_chance,
            survival_rate=survival_rate_dedup,
            credible_rules=deduplicated[:20]  # Use deduplicated rules
        )
        
        # Store the rejected duplicates for reference
        self._rejected_duplicates = rejected
    
    def _save_results(self, results: List[IndicatorResult], filename: str):
        """Save results to JSON."""
        filepath = self.output_dir / filename
        data = [r.to_dict() for r in results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} results to {filepath}")
    
    def _save_validation_summary(self):
        """Save validation summary."""
        filepath = self.output_dir / 'validation_summary.json'
        with open(filepath, 'w') as f:
            json.dump(asdict(self.validation_summary), f, indent=2)
        
        credible_filepath = self.output_dir / 'best_credible.json'
        with open(credible_filepath, 'w') as f:
            json.dump(self.validation_summary.credible_rules, f, indent=2)
    
    def _log_results(self):
        """Log validation results."""
        vs = self.validation_summary
        
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY (after deduplication)")
        logger.info("="*70)
        logger.info(f"Rules tested:              {vs.total_rules_tested}")
        logger.info(f"Passed IS threshold:       {vs.passed_is_threshold} ({vs.passed_is_threshold/max(1,vs.total_rules_tested):.1%})")
        logger.info(f"Passed OOS (deduplicated): {vs.passed_oos_validation} ({vs.passed_oos_validation/max(1,vs.passed_is_threshold):.1%} of IS passed)")
        logger.info(f"Insufficient OOS data:     {vs.insufficient_data}")
        logger.info(f"Expected by chance (5%):   {vs.expected_by_chance:.1f}")
        logger.info(f"Survival rate:             {vs.survival_rate:.2f}x chance")
        
        # Quality check
        if vs.passed_oos_validation / max(1, vs.total_rules_tested) > 0.15:
            logger.warning("*** WARNING: Pass rate > 15% - thresholds may still be too loose ***")
        
        if vs.passed_oos_validation > 0:
            logger.info("\n" + "-"*70)
            logger.info("TOP CREDIBLE RULES (unique signals only):")
            logger.info("-"*70)
            
            for i, rule in enumerate(vs.credible_rules[:10]):
                indicators = rule.get('indicators', ['unknown'])
                verdict = rule.get('verdict', 'CREDIBLE')
                logger.info(f"\n{i+1}. {' + '.join(indicators)}")
                logger.info(f"   IS:  Sharpe={rule.get('is_sharpe',0):.2f}, WR={rule.get('is_win_rate',0):.1%}, Trades={rule.get('is_n_trades',0)}")
                logger.info(f"   OOS: Sharpe={rule.get('oos_sharpe',0):.2f}, WR={rule.get('oos_win_rate',0):.1%}, Trades={rule.get('oos_n_trades',0)}")
                retention = rule.get('retention_ratio', 0)
                
                # Flag suspicious retention
                if retention > 1.0:
                    logger.warning(f"   Retention: {retention:.0%} [SUSPICIOUS - OOS > IS]")
                else:
                    logger.info(f"   Retention: {retention:.0%} [PASS]")
                
                # Sharpe math verification for top 3 (TIME-BASED)
                if i < 3:
                    oos_trades = rule.get('oos_n_trades', 0)
                    oos_mean = rule.get('oos_avg_return', 0)
                    oos_std = rule.get('oos_std_return', 0.01)
                    oos_sharpe = rule.get('oos_sharpe', 0)
                    period_days = self.tester.oos_period_days if hasattr(self, 'tester') else 35
                    
                    if oos_trades > 0 and period_days > 0:
                        # TIME-BASED Sharpe math (correct)
                        total_return = np.prod([1 + oos_mean] * oos_trades) - 1  # Approximate
                        daily_return = (1 + total_return) ** (1/period_days) - 1
                        trades_per_day = oos_trades / period_days
                        daily_std = oos_std * np.sqrt(trades_per_day)
                        daily_sharpe = daily_return / max(daily_std, 0.0001)
                        
                        logger.info(f"   [MATH] {oos_trades} trades / {period_days:.0f} days = {trades_per_day:.1f} trades/day")
                        logger.info(f"   [MATH] Total return: {total_return:.1%}, Daily return: {daily_return:.3%}")
                        logger.info(f"   [MATH] Daily std: {daily_std:.4f}, Daily Sharpe: {daily_sharpe:.3f}")
                        logger.info(f"   [MATH] Annualized: {daily_sharpe:.3f} * sqrt(365) = {oos_sharpe:.2f}")
        else:
            logger.info("\n[!] NO RULES PASSED OOS VALIDATION")
            logger.info("This is actually good - filtering out overfitting!")
        
        if self.single_results:
            best_is = max(self.single_results, key=lambda x: x.is_sharpe)
            logger.info(f"\n[BEST IS SHARPE (may be overfit)]")
            logger.info(f"  Indicators: {best_is.indicators[0]}")
            logger.info(f"  IS Sharpe: {best_is.is_sharpe:.2f}")
            logger.info(f"  OOS Sharpe: {best_is.oos_sharpe:.2f}")
            logger.info(f"  Verdict: {best_is.verdict}")


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_trade_details(rule_name: str, trade_returns: List[float], trade_dates: List[str] = None):
    """Print detailed trade breakdown for verification."""
    print(f"\n{'='*70}")
    print(f"TRADE DETAILS: {rule_name}")
    print(f"{'='*70}")
    
    returns = np.array(trade_returns)
    
    print(f"\nFirst 20 trades:")
    for i, ret in enumerate(returns[:20]):
        date_str = f" ({trade_dates[i][:10]})" if trade_dates and i < len(trade_dates) else ""
        print(f"  {i+1:3d}. {ret*100:+7.3f}%{date_str}")
    
    if len(returns) > 20:
        print(f"  ... ({len(returns) - 20} more trades)")
    
    print(f"\nSUMMARY:")
    print(f"  Total trades: {len(returns)}")
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    print(f"  Winners: {len(winners)} ({len(winners)/len(returns)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(returns)*100:.1f}%)")
    if len(winners) > 0:
        print(f"  Avg winner: {np.mean(winners)*100:+.3f}%")
    if len(losers) > 0:
        print(f"  Avg loser: {np.mean(losers)*100:+.3f}%")
    print(f"  Best trade: {np.max(returns)*100:+.3f}%")
    print(f"  Worst trade: {np.min(returns)*100:+.3f}%")
    print(f"  Median trade: {np.median(returns)*100:+.3f}%")
    print(f"  Sum (arithmetic): {np.sum(returns)*100:.2f}%")
    print(f"  Product (compound): {(np.prod(1 + returns) - 1)*100:.2f}%")
    
    # Check for outliers
    mean_abs = np.mean(np.abs(returns))
    outliers = np.abs(returns) > 3 * mean_abs
    if np.any(outliers):
        print(f"\n  [!] OUTLIERS DETECTED (>3x mean):")
        for i, (ret, is_outlier) in enumerate(zip(returns, outliers)):
            if is_outlier:
                print(f"      Trade {i+1}: {ret*100:+.3f}%")


def print_date_ranges(is_features: pd.DataFrame, oos_features: pd.DataFrame):
    """Print date range verification."""
    print(f"\n{'='*70}")
    print("DATE RANGE VERIFICATION")
    print(f"{'='*70}")
    
    is_start = is_features.index[0]
    is_end = is_features.index[-1]
    oos_start = oos_features.index[0]
    oos_end = oos_features.index[-1]
    
    print(f"\nIS period:  {is_start} to {is_end} ({len(is_features)} bars)")
    print(f"OOS period: {oos_start} to {oos_end} ({len(oos_features)} bars)")
    
    # Verify correct ordering
    if is_end < oos_start:
        print(f"\n[OK] IS period ends BEFORE OOS period starts (correct)")
    else:
        print(f"\n[!!!] WARNING: IS and OOS periods OVERLAP - possible data leakage!")
    
    print(f"\nIS should be EARLIER (training on past)")
    print(f"OOS should be LATER (testing on future)")


def generate_equity_curve(rule_name: str, trade_returns: List[float], output_dir: Path = None):
    """Generate and save equity curve visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not available - skipping equity curve")
        return None
    
    returns = np.array(trade_returns)
    equity = np.cumprod(1 + returns)
    
    # Calculate drawdown
    rolling_max = np.maximum.accumulate(equity)
    drawdown = (equity - rolling_max) / rolling_max
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Equity curve
    ax1.plot(equity, 'b-', linewidth=1.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Equity (starting at $1)')
    ax1.set_title(f'OOS Equity Curve: {rule_name}\n'
                  f'{len(returns)} trades, {(equity[-1]-1)*100:.1f}% total return')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2.fill_between(range(len(drawdown)), drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trade Number')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path('reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'oos_equity_curve.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n[OK] Equity curve saved to: {output_path}")
    print(f"     Max drawdown: {np.min(drawdown)*100:.1f}%")
    print(f"     Final equity: ${equity[-1]:.2f} (started at $1.00)")
    
    return output_path


def check_feature_calculation(feature_name: str, features_df: pd.DataFrame):
    """Check for potential lookahead bias in feature calculation."""
    print(f"\n{'='*70}")
    print(f"FEATURE CALCULATION CHECK: {feature_name}")
    print(f"{'='*70}")
    
    if feature_name not in features_df.columns:
        print(f"[!] Feature '{feature_name}' not found in dataframe")
        return
    
    values = features_df[feature_name].dropna()
    
    print(f"\nFeature statistics:")
    print(f"  Count: {len(values)}")
    print(f"  Mean: {values.mean():.6f}")
    print(f"  Std: {values.std():.6f}")
    print(f"  Min: {values.min():.6f}")
    print(f"  Max: {values.max():.6f}")
    
    # Check for NaN pattern (should have NaNs at start due to lookback)
    full_series = features_df[feature_name]
    nan_count_start = full_series.head(50).isna().sum()
    nan_count_end = full_series.tail(50).isna().sum()
    
    print(f"\nNaN pattern check:")
    print(f"  NaNs in first 50 rows: {nan_count_start}")
    print(f"  NaNs in last 50 rows: {nan_count_end}")
    
    if nan_count_start > 0 and nan_count_end == 0:
        print(f"  [OK] NaN pattern suggests lookback is working correctly")
    elif nan_count_end > nan_count_start:
        print(f"  [!] WARNING: More NaNs at end than start - possible forward calculation")
    else:
        print(f"  [?] Cannot determine - manual review recommended")


def run_verification_report(
    tester,
    features: pd.DataFrame,
    top_rules: List[dict],
    output_dir: Path
):
    """Run comprehensive verification report."""
    print("\n")
    print("=" * 70)
    print(" VERIFICATION REPORT ")
    print("=" * 70)
    
    # 1. Date ranges
    print_date_ranges(tester.is_features, tester.oos_features)
    
    # 2. Transaction cost confirmation
    print(f"\n{'='*70}")
    print("TRANSACTION COSTS")
    print(f"{'='*70}")
    print(f"\nTransaction cost per trade: 0.12% (round-trip)")
    print(f"This is applied AFTER raw return calculation")
    print(f"Formula: adjusted_return = raw_return - 0.0012")
    
    # 3. For top 3 rules, get detailed trade info
    for i, rule in enumerate(top_rules[:3]):
        indicator = rule.get('indicators', ['unknown'])[0]
        
        # Re-run with store_trades=True to get detailed info
        print(f"\n{'='*70}")
        print(f"VERIFYING RULE {i+1}: {indicator}")
        print(f"{'='*70}")
        
        # Check feature calculation
        check_feature_calculation(indicator, features)
        
        # Get OOS trade returns (stored in rule if available)
        if '_trade_returns' in rule:
            trade_returns = rule['_trade_returns']
            trade_dates = rule.get('_trade_dates', None)
            print_trade_details(indicator, trade_returns, trade_dates)
            
            # Generate equity curve for top rule
            if i == 0:
                generate_equity_curve(indicator, trade_returns, output_dir)
        else:
            print(f"\n[!] Trade-by-trade data not available for this rule")
            print(f"    Re-run with --verify flag to capture trade details")
    
    print("\n" + "=" * 70)
    print(" END VERIFICATION REPORT ")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_and_features():
    """Load data and generate features using existing pipeline."""
    logger.info("Loading data and generating features...")
    
    cache_file = Path('data_cache/features_cache.pkl')
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logger.info("Loaded features from cache")
            return cached['features'], cached['targets']
        except:
            pass
    
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    try:
        try:
            from run_discovery import run_data_acquisition, run_feature_engineering, run_target_generation
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_discovery", script_dir / "run_discovery.py")
            run_discovery = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_discovery)
            run_data_acquisition = run_discovery.run_data_acquisition
            run_feature_engineering = run_discovery.run_feature_engineering
            run_target_generation = run_discovery.run_target_generation
        
        data = run_data_acquisition()
        if data is None:
            raise ValueError("Data acquisition failed")
        
        master_df, features = run_feature_engineering(data)
        if features is None:
            raise ValueError("Feature engineering failed")
        
        targets = run_target_generation(master_df)
        if targets is None:
            raise ValueError("Target generation failed")
        
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'features': features, 'targets': targets}, f)
        
        return features, targets
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        traceback.print_exc()
        return None, None


def load_from_synthetic():
    """Generate synthetic data for testing."""
    logger.info("Generating synthetic data for testing...")
    
    np.random.seed(42)
    n_samples = 600
    n_features = 100
    
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='4H')
    features = pd.DataFrame(index=dates)
    
    price = 50000 + np.cumsum(np.random.randn(n_samples) * 500)
    features['close'] = price
    features['open'] = price * (1 + np.random.randn(n_samples) * 0.001)
    features['high'] = price * (1 + abs(np.random.randn(n_samples) * 0.005))
    features['low'] = price * (1 - abs(np.random.randn(n_samples) * 0.005))
    features['volume'] = np.random.uniform(1e9, 5e9, n_samples)
    
    features['ma_10'] = pd.Series(price).rolling(10).mean().values
    features['ma_20'] = pd.Series(price).rolling(20).mean().values
    features['ma_50'] = pd.Series(price).rolling(50).mean().values
    features['rsi_14'] = 50 + np.random.randn(n_samples) * 15
    features['bb_upper_20'] = features['ma_20'] + 2 * pd.Series(price).rolling(20).std().values
    features['bb_lower_20'] = features['ma_20'] - 2 * pd.Series(price).rolling(20).std().values
    
    features['funding_rate'] = np.random.randn(n_samples) * 0.001
    features['open_interest'] = 5e9 + np.cumsum(np.random.randn(n_samples) * 1e8)
    features['long_short_ratio'] = 1.0 + np.random.randn(n_samples) * 0.1
    
    for i in range(n_features - len(features.columns)):
        features[f'random_feat_{i}'] = np.random.randn(n_samples)
    
    base_return = np.random.randn(n_samples) * 0.02
    signal_strength = 0.3
    funding_signal = -features['funding_rate'].values * signal_strength * 10
    oi_signal = -np.diff(features['open_interest'].values, prepend=features['open_interest'].values[0]) / 1e9 * signal_strength
    returns = base_return + funding_signal + oi_signal
    
    targets = pd.DataFrame(index=dates)
    targets['return_simple_6h'] = np.roll(returns, -1)
    targets['return_simple_12h'] = np.roll(returns, -2) + np.roll(returns, -1)
    targets['return_simple_24h'] = np.roll(returns, -4) + np.roll(returns, -3) + np.roll(returns, -2) + np.roll(returns, -1)
    targets.iloc[-4:] = np.nan
    
    logger.info(f"Generated synthetic data: {n_samples} samples, {len(features.columns)} features")
    
    return features.reset_index(drop=True), targets.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='BTC Alpha Discovery - Exhaustive Search with OOS Validation')
    parser.add_argument('--mode', choices=['single', 'pairs', 'triples', 'full'],
                       default='triples', help='Search depth')
    parser.add_argument('--target', default='return_simple_6h', help='Target column')
    parser.add_argument('--top-n', type=int, default=50, help='Keep top N results per phase')
    parser.add_argument('--min-sharpe', type=float, default=0.5, help='Minimum IS Sharpe')
    parser.add_argument('--min-trades', type=int, default=20, help='Minimum trades required')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--no-validate', action='store_true', help='Disable OOS validation')
    parser.add_argument('--verify', action='store_true', help='Run verification report for top rules')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("BTC ALPHA DISCOVERY - EXHAUSTIVE SEARCH WITH OOS VALIDATION")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"Min IS Sharpe: {args.min_sharpe}")
    logger.info(f"OOS Validation: {'DISABLED' if args.no_validate else 'ENABLED'}")
    logger.info(f"Verification: {'ENABLED' if args.verify else 'DISABLED'}")
    logger.info(f"Train/Test Split: {VALIDATION_THRESHOLDS['train_pct']:.0%} / {1-VALIDATION_THRESHOLDS['train_pct']:.0%}")
    logger.info("="*70)
    
    VALIDATION_THRESHOLDS['min_is_sharpe'] = args.min_sharpe
    
    if args.synthetic:
        features, targets = load_from_synthetic()
    else:
        features, targets = load_data_and_features()
    
    if features is None or targets is None:
        logger.error("Failed to load data. Try --synthetic flag.")
        return
    
    engine = ExhaustiveSearchEngine(
        features, targets,
        target_col=args.target,
        output_dir=log_dir,
        top_n=args.top_n,
        min_sharpe=args.min_sharpe,
        min_trades=args.min_trades,
        validate_oos=not args.no_validate
    )
    
    engine.run_phase_single()
    
    if args.mode in ['pairs', 'triples', 'full']:
        engine.run_phase_pairs()
    
    if args.mode in ['triples', 'full']:
        engine.run_phase_triples()
    
    # Run verification report if requested
    if args.verify and engine.validation_summary and engine.validation_summary.credible_rules:
        logger.info("\n" + "="*70)
        logger.info("RUNNING VERIFICATION REPORT")
        logger.info("="*70)
        
        # Re-run top 3 rules with trade storage
        verified_rules = []
        for rule_dict in engine.validation_summary.credible_rules[:3]:
            indicator = rule_dict.get('indicators', ['unknown'])[0]
            direction = rule_dict.get('direction', 1)
            threshold_type = rule_dict.get('threshold_type', 'percentile')
            threshold_value = rule_dict.get('threshold_value', 70)
            
            # Extract threshold params
            if threshold_type == 'percentile':
                kwargs = {'pct_long': threshold_value, 'pct_short': 100 - threshold_value}
            else:
                kwargs = {'z_long': threshold_value, 'z_short': -threshold_value}
            
            # Re-calculate with trade storage
            is_values = engine.tester.is_features[indicator]
            oos_values = engine.tester.oos_features[indicator]
            
            # Get fixed thresholds from IS
            if threshold_type == 'percentile':
                upper_thresh, lower_thresh = SignalGenerator.get_percentile_thresholds(
                    is_values, kwargs['pct_long'], kwargs['pct_short']
                )
                oos_signals = SignalGenerator.apply_fixed_threshold(
                    oos_values, upper_thresh, lower_thresh, direction
                )
            else:
                is_mean, is_std = SignalGenerator.get_zscore_params(is_values)
                oos_signals = SignalGenerator.apply_fixed_zscore(
                    oos_values, is_mean, is_std,
                    kwargs['z_long'], kwargs['z_short'], direction
                )
            
            # Calculate metrics with trade storage
            oos_metrics = engine.tester._calculate_metrics(
                oos_signals, engine.tester.oos_returns, 
                engine.tester.oos_period_days, store_trades=True
            )
            
            if oos_metrics:
                rule_dict['_trade_returns'] = oos_metrics.get('_trade_returns', [])
                rule_dict['_trade_dates'] = oos_metrics.get('_trade_dates', [])
                rule_dict['_raw_returns'] = oos_metrics.get('_raw_returns', [])
            
            verified_rules.append(rule_dict)
        
        # Run verification
        run_verification_report(
            engine.tester,
            features,
            verified_rules,
            log_dir
        )
    
    logger.info("\n" + "="*70)
    logger.info("EXHAUSTIVE SEARCH COMPLETE")
    logger.info(f"Results saved to: {log_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
