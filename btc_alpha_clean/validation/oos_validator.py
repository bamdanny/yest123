"""
Out-of-Sample Validation Module for BTC Alpha Discovery.

This module enforces proper walk-forward validation to prevent overfitting.
All indicator testing must use this module to get credible results.

Key Principles:
1. Data is split BEFORE any rule discovery (70% IS / 30% OOS)
2. Rules are discovered using ONLY in-sample data
3. Surviving rules are validated on out-of-sample data
4. Both IS and OOS metrics are reported with retention ratio

Usage:
    from validation.oos_validator import OOSValidator
    
    validator = OOSValidator(features_df, forward_returns, split_ratio=0.7)
    
    # Discover rules on IS data only
    is_results = validator.test_indicator_in_sample(indicator, thresholds, directions)
    
    # Validate survivors on OOS
    oos_results = validator.validate_out_of_sample(rule)
    
    # Get combined report
    report = validator.get_validation_report(rule)

Author: BTC Alpha Discovery Team
Version: 2.0.0 (OOS-Enforced)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Validation verdict for a rule."""
    CREDIBLE = "CREDIBLE"           # Passed OOS validation
    MARGINAL = "MARGINAL"           # Borderline - needs more data
    OVERFIT = "OVERFIT"             # Failed OOS validation
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough OOS trades


@dataclass
class ValidationMetrics:
    """Metrics from a single test period (IS or OOS)."""
    sharpe: float
    win_rate: float
    avg_return: float
    std_return: float
    n_trades: int
    max_drawdown: float
    profit_factor: float
    period_days: float
    trades_per_year: float


@dataclass
class ValidationReport:
    """Complete validation report for a rule."""
    # Rule definition
    indicator: str
    direction: int  # 1 = momentum (high values = long), -1 = mean_rev (high values = short)
    threshold_type: str
    threshold_value: float
    
    # In-sample metrics
    is_metrics: ValidationMetrics
    
    # Out-of-sample metrics
    oos_metrics: Optional[ValidationMetrics]
    
    # Validation results
    retention_ratio: float  # OOS_Sharpe / IS_Sharpe
    verdict: Verdict
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "indicator": self.indicator,
            "direction": self.direction,
            "threshold_type": self.threshold_type,
            "threshold_value": self.threshold_value,
            "is_sharpe": self.is_metrics.sharpe,
            "is_winrate": self.is_metrics.win_rate,
            "is_trades": self.is_metrics.n_trades,
            "is_avg_return": self.is_metrics.avg_return,
            "is_max_drawdown": self.is_metrics.max_drawdown,
            "oos_sharpe": self.oos_metrics.sharpe if self.oos_metrics else None,
            "oos_winrate": self.oos_metrics.win_rate if self.oos_metrics else None,
            "oos_trades": self.oos_metrics.n_trades if self.oos_metrics else None,
            "oos_avg_return": self.oos_metrics.avg_return if self.oos_metrics else None,
            "oos_max_drawdown": self.oos_metrics.max_drawdown if self.oos_metrics else None,
            "retention_ratio": self.retention_ratio,
            "verdict": self.verdict.value,
            "warnings": self.warnings
        }


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

VALIDATION_THRESHOLDS = {
    # In-sample thresholds (for discovery)
    'min_is_sharpe': 1.5,           # Minimum IS Sharpe to consider
    'min_is_trades': 20,            # Minimum trades in IS period
    'min_is_winrate': 0.52,         # Better than coin flip
    
    # Out-of-sample thresholds (for validation)
    'min_oos_sharpe': 0.5,          # Minimum OOS Sharpe
    'min_oos_trades': 15,           # Enough OOS trades
    'min_oos_winrate': 0.50,        # At least 50% in OOS
    
    # Retention thresholds
    'min_retention': 0.30,          # OOS Sharpe >= 30% of IS Sharpe
    'excellent_retention': 0.60,    # OOS Sharpe >= 60% of IS Sharpe
    
    # Sanity checks
    'max_plausible_sharpe': 5.0,    # Sharpe > 5 is suspicious
    'min_period_days': 30,          # Minimum test period
}


# =============================================================================
# SHARPE CALCULATION (FIXED)
# =============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 2190,  # 6 bars/day * 365 days for 4H
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe ratio correctly.
    
    Args:
        returns: Array of period returns (not cumulative)
        periods_per_year: Number of trading periods per year
        risk_free_rate: Annual risk-free rate (default 0)
    
    Returns:
        Annualized Sharpe ratio
    
    Notes:
        - Uses excess returns (return - risk_free_per_period)
        - Annualizes using sqrt(periods_per_year) for std
        - Caps at periods actually observed to avoid inflation
    """
    if len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    std = np.std(returns, ddof=1)  # Sample std
    if std == 0 or np.isnan(std):
        return 0.0
    
    # Risk-free rate per period
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    
    mean_excess = np.mean(excess_returns)
    
    # Annualization factor - cap at actual observations
    # This prevents inflating Sharpe for short test periods
    annualization = np.sqrt(min(len(returns), periods_per_year))
    
    sharpe = (mean_excess / std) * annualization
    
    return sharpe


def calculate_metrics(
    returns: np.ndarray,
    period_days: float,
    periods_per_year: int = 2190
) -> ValidationMetrics:
    """Calculate all validation metrics for a return series."""
    
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return ValidationMetrics(
            sharpe=0.0, win_rate=0.0, avg_return=0.0, std_return=0.0,
            n_trades=0, max_drawdown=0.0, profit_factor=0.0,
            period_days=period_days, trades_per_year=0.0
        )
    
    # Basic stats
    n_trades = len(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if n_trades > 1 else 0.0
    win_rate = np.sum(returns > 0) / n_trades if n_trades > 0 else 0.0
    
    # Sharpe
    sharpe = calculate_sharpe_ratio(returns, periods_per_year)
    
    # Drawdown (from cumulative returns)
    cum_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Profit factor
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    total_gains = np.sum(gains) if len(gains) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
    
    # Trades per year
    trades_per_year = (n_trades / period_days) * 365 if period_days > 0 else 0.0
    
    return ValidationMetrics(
        sharpe=sharpe,
        win_rate=win_rate,
        avg_return=avg_return,
        std_return=std_return,
        n_trades=n_trades,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor if profit_factor != float('inf') else 999.9,
        period_days=period_days,
        trades_per_year=trades_per_year
    )


# =============================================================================
# OOS VALIDATOR CLASS
# =============================================================================

class OOSValidator:
    """
    Enforces out-of-sample validation for all indicator testing.
    
    This class ensures that:
    1. Data is split before any rule discovery
    2. Rules are tested ONLY on in-sample data during discovery
    3. Validation happens ONLY on out-of-sample data
    4. Both metrics are reported with proper verdicts
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        split_ratio: float = 0.7,
        holding_period_bars: int = 6,  # 24h for 4H bars
        periods_per_year: int = 2190
    ):
        """
        Initialize the OOS Validator.
        
        Args:
            features: DataFrame of features (aligned with forward_returns)
            forward_returns: Series of forward returns to predict
            split_ratio: Fraction of data for in-sample (default 70%)
            holding_period_bars: Bars per trade (for overlap removal)
            periods_per_year: Trading periods per year (for annualization)
        """
        # Align data
        min_len = min(len(features), len(forward_returns))
        self.features = features.iloc[:min_len].reset_index(drop=True)
        self.forward_returns = forward_returns.iloc[:min_len].reset_index(drop=True)
        
        # Calculate split point
        self.split_idx = int(len(self.features) * split_ratio)
        
        # Split data
        self.is_features = self.features.iloc[:self.split_idx]
        self.is_returns = self.forward_returns.iloc[:self.split_idx]
        
        self.oos_features = self.features.iloc[self.split_idx:]
        self.oos_returns = self.forward_returns.iloc[self.split_idx:]
        
        # Config
        self.holding_period = holding_period_bars
        self.periods_per_year = periods_per_year
        
        # Calculate period lengths in days (assuming 4H bars)
        bars_per_day = 6
        self.is_days = len(self.is_features) / bars_per_day
        self.oos_days = len(self.oos_features) / bars_per_day
        self.total_days = len(self.features) / bars_per_day
        
        logger.info(f"OOSValidator initialized:")
        logger.info(f"  Total samples: {len(self.features)}")
        logger.info(f"  IS samples: {len(self.is_features)} ({self.is_days:.1f} days)")
        logger.info(f"  OOS samples: {len(self.oos_features)} ({self.oos_days:.1f} days)")
        logger.info(f"  Split ratio: {split_ratio:.0%}")
    
    def _generate_signals(
        self,
        indicator_values: pd.Series,
        threshold: float,
        direction: int
    ) -> pd.Series:
        """
        Generate trading signals based on indicator threshold.
        
        Args:
            indicator_values: Series of indicator values
            threshold: Threshold value for signal generation
            direction: 1 = momentum (above threshold = long)
                      -1 = mean_rev (above threshold = short/below = long)
        
        Returns:
            Series of signals (1 = long, -1 = short, 0 = no position)
        """
        signals = pd.Series(0, index=indicator_values.index)
        
        if direction == 1:  # Momentum: high values = long
            signals[indicator_values > threshold] = 1
            signals[indicator_values < -threshold] = -1 if threshold > 0 else 0
        else:  # Mean reversion: high values = short
            signals[indicator_values > threshold] = -1
            signals[indicator_values < -threshold] = 1 if threshold > 0 else 0
        
        return signals
    
    def _remove_overlapping_trades(
        self,
        signals: pd.Series,
        returns: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove overlapping trades to get independent returns.
        
        When trades overlap (holding period > 1 bar), we only count
        the first trade in each cluster to avoid autocorrelation
        in Sharpe calculation.
        """
        trade_indices = signals[signals != 0].index.tolist()
        
        if len(trade_indices) == 0:
            return np.array([]), np.array([])
        
        # Keep only non-overlapping trades
        valid_indices = []
        last_trade_end = -self.holding_period
        
        for idx in trade_indices:
            if idx >= last_trade_end + self.holding_period:
                valid_indices.append(idx)
                last_trade_end = idx
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([])
        
        # Get returns for valid trades
        trade_signals = signals.loc[valid_indices].values
        trade_returns = returns.loc[valid_indices].values
        
        # Apply direction: if signal is -1 (short), flip the return
        adjusted_returns = trade_signals * trade_returns
        
        return adjusted_returns, trade_signals
    
    def test_indicator_in_sample(
        self,
        indicator_name: str,
        threshold_type: str = "percentile",
        threshold_value: float = 70,
        direction: int = -1
    ) -> Optional[ValidationMetrics]:
        """
        Test an indicator rule on IN-SAMPLE data only.
        
        This is the discovery phase - we only use IS data.
        
        Args:
            indicator_name: Name of the indicator column
            threshold_type: "percentile" or "zscore"
            threshold_value: Threshold value (percentile 0-100 or zscore)
            direction: 1 = momentum, -1 = mean_reversion
        
        Returns:
            ValidationMetrics for IS period, or None if indicator not found
        """
        if indicator_name not in self.is_features.columns:
            return None
        
        indicator = self.is_features[indicator_name]
        
        # Calculate threshold
        if threshold_type == "percentile":
            threshold = np.nanpercentile(indicator, threshold_value)
        elif threshold_type == "zscore":
            mean = indicator.mean()
            std = indicator.std()
            threshold = mean + threshold_value * std if std > 0 else mean
        else:
            threshold = threshold_value
        
        # Generate signals
        signals = self._generate_signals(indicator, threshold, direction)
        
        # Get non-overlapping trade returns
        trade_returns, _ = self._remove_overlapping_trades(signals, self.is_returns)
        
        if len(trade_returns) < VALIDATION_THRESHOLDS['min_is_trades']:
            return None
        
        # Calculate metrics
        return calculate_metrics(trade_returns, self.is_days, self.periods_per_year)
    
    def validate_out_of_sample(
        self,
        indicator_name: str,
        threshold_type: str,
        threshold_value: float,
        direction: int,
        is_threshold: float = None  # Use IS threshold if provided
    ) -> Optional[ValidationMetrics]:
        """
        Validate a rule on OUT-OF-SAMPLE data.
        
        This uses the SAME threshold discovered in IS period,
        applied to OOS data.
        
        Args:
            indicator_name: Name of the indicator column
            threshold_type: "percentile" or "zscore"
            threshold_value: Threshold value from IS discovery
            direction: 1 = momentum, -1 = mean_reversion
            is_threshold: Exact threshold from IS (if computed)
        
        Returns:
            ValidationMetrics for OOS period
        """
        if indicator_name not in self.oos_features.columns:
            return None
        
        indicator = self.oos_features[indicator_name]
        
        # Use IS threshold if provided, otherwise recalculate
        # NOTE: For proper validation, we should use the exact IS threshold
        if is_threshold is not None:
            threshold = is_threshold
        else:
            if threshold_type == "percentile":
                # Use IS percentile applied to OOS data
                is_indicator = self.is_features[indicator_name]
                threshold = np.nanpercentile(is_indicator, threshold_value)
            elif threshold_type == "zscore":
                is_indicator = self.is_features[indicator_name]
                mean = is_indicator.mean()
                std = is_indicator.std()
                threshold = mean + threshold_value * std if std > 0 else mean
            else:
                threshold = threshold_value
        
        # Generate signals
        signals = self._generate_signals(indicator, threshold, direction)
        
        # Get non-overlapping trade returns
        trade_returns, _ = self._remove_overlapping_trades(signals, self.oos_returns)
        
        if len(trade_returns) == 0:
            return ValidationMetrics(
                sharpe=0.0, win_rate=0.0, avg_return=0.0, std_return=0.0,
                n_trades=0, max_drawdown=0.0, profit_factor=0.0,
                period_days=self.oos_days, trades_per_year=0.0
            )
        
        return calculate_metrics(trade_returns, self.oos_days, self.periods_per_year)
    
    def get_full_validation(
        self,
        indicator_name: str,
        threshold_type: str = "percentile",
        threshold_value: float = 70,
        direction: int = -1
    ) -> Optional[ValidationReport]:
        """
        Perform full IS + OOS validation for a rule.
        
        Returns:
            ValidationReport with IS metrics, OOS metrics, and verdict
        """
        # Test in-sample
        is_metrics = self.test_indicator_in_sample(
            indicator_name, threshold_type, threshold_value, direction
        )
        
        if is_metrics is None:
            return None
        
        # Check IS thresholds
        warnings = []
        
        if is_metrics.sharpe < VALIDATION_THRESHOLDS['min_is_sharpe']:
            return None  # Don't even validate OOS
        
        if is_metrics.sharpe > VALIDATION_THRESHOLDS['max_plausible_sharpe']:
            warnings.append(f"IS Sharpe {is_metrics.sharpe:.2f} suspiciously high")
        
        # Calculate IS threshold for OOS
        is_indicator = self.is_features[indicator_name]
        if threshold_type == "percentile":
            is_threshold = np.nanpercentile(is_indicator, threshold_value)
        elif threshold_type == "zscore":
            mean = is_indicator.mean()
            std = is_indicator.std()
            is_threshold = mean + threshold_value * std if std > 0 else mean
        else:
            is_threshold = threshold_value
        
        # Validate out-of-sample
        oos_metrics = self.validate_out_of_sample(
            indicator_name, threshold_type, threshold_value, direction, is_threshold
        )
        
        # Calculate retention ratio
        if oos_metrics and is_metrics.sharpe > 0:
            retention = oos_metrics.sharpe / is_metrics.sharpe
        else:
            retention = 0.0
        
        # Determine verdict
        if oos_metrics is None or oos_metrics.n_trades < VALIDATION_THRESHOLDS['min_oos_trades']:
            verdict = Verdict.INSUFFICIENT_DATA
            warnings.append(f"Only {oos_metrics.n_trades if oos_metrics else 0} OOS trades")
        elif oos_metrics.sharpe < VALIDATION_THRESHOLDS['min_oos_sharpe']:
            verdict = Verdict.OVERFIT
            warnings.append(f"OOS Sharpe {oos_metrics.sharpe:.2f} below threshold")
        elif retention < VALIDATION_THRESHOLDS['min_retention']:
            verdict = Verdict.OVERFIT
            warnings.append(f"Retention {retention:.1%} below threshold")
        elif retention >= VALIDATION_THRESHOLDS['excellent_retention']:
            verdict = Verdict.CREDIBLE
        else:
            verdict = Verdict.MARGINAL
            warnings.append(f"Marginal retention {retention:.1%}")
        
        return ValidationReport(
            indicator=indicator_name,
            direction=direction,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            retention_ratio=retention,
            verdict=verdict,
            warnings=warnings
        )


# =============================================================================
# BATCH VALIDATION HELPER
# =============================================================================

def run_exhaustive_search_with_oos(
    features: pd.DataFrame,
    forward_returns: pd.Series,
    split_ratio: float = 0.7,
    threshold_types: List[str] = ["percentile", "zscore"],
    percentile_values: List[float] = [70, 80, 90],
    zscore_values: List[float] = [1.5, 2.0],
    directions: List[int] = [1, -1],
    min_is_sharpe: float = 1.5,
    top_n: int = 50
) -> Dict[str, Any]:
    """
    Run exhaustive indicator search with proper OOS validation.
    
    Args:
        features: DataFrame of features
        forward_returns: Series of forward returns
        split_ratio: IS/OOS split ratio
        threshold_types: Types of thresholds to test
        percentile_values: Percentile thresholds to test
        zscore_values: Z-score thresholds to test
        directions: Directions to test (1=momentum, -1=mean_rev)
        min_is_sharpe: Minimum IS Sharpe to validate OOS
        top_n: Number of top results to return
    
    Returns:
        Dictionary with results and statistics
    """
    validator = OOSValidator(features, forward_returns, split_ratio)
    
    # Get testable indicators (numeric only)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    testable = [c for c in numeric_cols if not c.startswith('target_') 
                and not c.startswith('return_') and c != 'timestamp']
    
    logger.info(f"Testing {len(testable)} indicators...")
    
    results = []
    is_passed = 0
    oos_passed = 0
    total_tested = 0
    
    for indicator in testable:
        for threshold_type in threshold_types:
            values = percentile_values if threshold_type == "percentile" else zscore_values
            
            for threshold_value in values:
                for direction in directions:
                    total_tested += 1
                    
                    # Get full validation
                    report = validator.get_full_validation(
                        indicator, threshold_type, threshold_value, direction
                    )
                    
                    if report is None:
                        continue
                    
                    is_passed += 1
                    
                    if report.verdict in [Verdict.CREDIBLE, Verdict.MARGINAL]:
                        oos_passed += 1
                    
                    results.append(report)
    
    # Sort by OOS Sharpe (credible) then IS Sharpe (others)
    results.sort(key=lambda r: (
        r.verdict == Verdict.CREDIBLE,
        r.oos_metrics.sharpe if r.oos_metrics else 0,
        r.is_metrics.sharpe
    ), reverse=True)
    
    # Get top N
    top_results = results[:top_n]
    
    # Count by verdict
    verdicts = {}
    for r in results:
        v = r.verdict.value
        verdicts[v] = verdicts.get(v, 0) + 1
    
    # Expected by chance (rough estimate)
    expected_by_chance = total_tested * 0.05
    
    return {
        "results": [r.to_dict() for r in top_results],
        "statistics": {
            "total_tested": total_tested,
            "passed_is_threshold": is_passed,
            "passed_oos_validation": oos_passed,
            "expected_by_chance": expected_by_chance,
            "survival_rate": oos_passed / expected_by_chance if expected_by_chance > 0 else 0,
            "is_pass_rate": is_passed / total_tested if total_tested > 0 else 0,
            "oos_pass_rate": oos_passed / is_passed if is_passed > 0 else 0,
            "verdicts": verdicts
        },
        "config": {
            "split_ratio": split_ratio,
            "is_days": validator.is_days,
            "oos_days": validator.oos_days,
            "min_is_sharpe": min_is_sharpe,
            "thresholds": VALIDATION_THRESHOLDS
        }
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Create fake features
    features = pd.DataFrame({
        'indicator_1': np.random.randn(n_samples),
        'indicator_2': np.random.randn(n_samples).cumsum(),
        'indicator_3': np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.randn(n_samples)*0.1,
    })
    
    # Create fake forward returns (slight correlation with indicator_3)
    forward_returns = pd.Series(
        0.001 * features['indicator_3'] + np.random.randn(n_samples) * 0.02
    )
    
    # Run validation
    results = run_exhaustive_search_with_oos(
        features, forward_returns,
        split_ratio=0.7,
        min_is_sharpe=0.5,
        top_n=10
    )
    
    print("\n" + "="*60)
    print("OOS VALIDATION TEST")
    print("="*60)
    print(f"Total tested: {results['statistics']['total_tested']}")
    print(f"Passed IS: {results['statistics']['passed_is_threshold']}")
    print(f"Passed OOS: {results['statistics']['passed_oos_validation']}")
    print(f"Expected by chance: {results['statistics']['expected_by_chance']:.1f}")
    print(f"\nVerdicts: {results['statistics']['verdicts']}")
    
    print("\nTop results:")
    for r in results['results'][:5]:
        print(f"  {r['indicator']}: IS={r['is_sharpe']:.2f}, OOS={r['oos_sharpe']}, "
              f"Retention={r['retention_ratio']:.1%}, Verdict={r['verdict']}")
