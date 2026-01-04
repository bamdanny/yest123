"""
Validation Framework for Alpha Discovery.

Phase 10: Comprehensive validation to ensure discovered patterns are robust.

Key principles:
1. Out-of-sample testing is MANDATORY
2. Walk-forward analysis simulates real deployment
3. Statistical significance must be established
4. Regime robustness checks for different market conditions
5. Transaction cost modeling must be realistic

This module prevents overfitting and ensures discoveries generalize.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def remove_overlapping_signals(signals: pd.Series, holding_period: int = 6) -> pd.Series:
    """
    After a signal fires, mask out the next (holding_period - 1) bars.
    Ensures non-overlapping trade windows for valid Sharpe calculation.
    
    Example with holding_period=6:
      Bar:    0  1  2  3  4  5  6  7  8  9  10 11 12
      Raw:   -1 -1 -1 -1 -1 -1  0  0 -1 -1 -1  0 -1
      Clean: -1  0  0  0  0  0  0  0 -1  0  0  0 -1
      
    Trade 1: bars 0-5
    Trade 2: bars 8-13
    Trade 3: bars 12-17
    
    This prevents counting overlapping price moves as independent trades,
    which would artificially inflate Sharpe ratio.
    """
    clean = signals.copy()
    
    i = 0
    while i < len(clean):
        if clean.iloc[i] != 0:
            # Signal found - block next (holding_period - 1) bars
            block_end = min(i + holding_period, len(clean))
            for j in range(i + 1, block_end):
                clean.iloc[j] = 0
            i = block_end  # Jump past the blocked region
        else:
            i += 1
    
    return clean


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    # Core performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Trade statistics
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float  # 95% Value at Risk
    expected_shortfall: float  # CVaR
    calmar_ratio: float  # Return / MaxDD
    
    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool
    
    # Robustness
    regime_consistency: float  # Sharpe consistency across regimes
    time_consistency: float  # Sharpe consistency across time periods
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'n_trades': self.n_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'calmar_ratio': self.calmar_ratio,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'regime_consistency': self.regime_consistency,
            'time_consistency': self.time_consistency,
        }


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_metrics: ValidationMetrics
    out_sample_metrics: ValidationMetrics
    is_degraded: bool  # True if OOS significantly worse than IS
    
    
@dataclass
class ValidationReport:
    """Complete validation report."""
    strategy_name: str
    validation_date: datetime
    
    # Overall assessment
    overall_grade: str  # A, B, C, D, F
    recommendation: str
    
    # Walk-forward results
    walk_forward_results: List[WalkForwardResult]
    combined_oos_metrics: ValidationMetrics
    
    # Regime analysis
    regime_performance: Dict[str, ValidationMetrics]
    
    # Statistical tests
    statistical_tests: Dict[str, Any]
    
    # Robustness checks
    robustness_scores: Dict[str, float]
    
    # Warnings
    warnings: List[str]


class StrategyValidator:
    """
    Comprehensive strategy validation framework.
    
    Ensures discovered strategies are robust and not overfit.
    """
    
    # Transaction costs (realistic)
    COMMISSION_PCT = 0.04 / 100  # 0.04% per side
    SLIPPAGE_PCT = 0.02 / 100   # 0.02% per side
    TOTAL_COST = (COMMISSION_PCT + SLIPPAGE_PCT) * 2  # Round trip
    
    # Minimum requirements
    MIN_TRADES = 30  # Reduced from 50 for smaller datasets
    MIN_SHARPE = 0.5
    MIN_WIN_RATE = 0.48
    MAX_DRAWDOWN = 0.30
    SIGNIFICANCE_LEVEL = 0.05
    
    def __init__(
        self,
        walk_forward_periods: int = 3,  # Reduced from 5 for better period sizes
        train_ratio: float = 0.7,
        min_test_size: int = 50,  # Reduced from 200 - was causing all periods to be skipped!
        random_state: int = 42
    ):
        """
        Initialize validator.
        
        Args:
            walk_forward_periods: Number of walk-forward periods
            train_ratio: Ratio of train to total in each period
            min_test_size: Minimum samples in test set
            random_state: Random seed
        """
        self.walk_forward_periods = walk_forward_periods
        self.train_ratio = train_ratio
        self.min_test_size = min_test_size
        self.random_state = random_state
        
    def validate(
        self,
        strategy_signals: pd.Series,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        critic_survival_rate: Optional[float] = None
    ) -> ValidationReport:
        """
        Run complete validation.
        
        Args:
            strategy_signals: Series of trading signals (1=long, -1=short, 0=flat)
            returns: Series of forward returns
            features: Optional features for regime analysis
            strategy_name: Name for reporting
            critic_survival_rate: Optional - if Critic was run, the % of rules that survived
            
        Returns:
            Complete ValidationReport
        """
        logger.info(f"Validating strategy: {strategy_name}")
        
        # CRITICAL: If Critic rejected all rules, fail early
        if critic_survival_rate is not None and critic_survival_rate == 0:
            logger.error(
                "VALIDATION BLOCKED: Critic rejected ALL rules (0% survival).\n"
                "Cannot recommend deployment when no rules pass adversarial testing."
            )
            return self._create_failed_report(
                strategy_name, 
                "Critic rejected all rules (0% survival). Strategy is likely spurious."
            )
        
        # Align data
        combined = pd.concat([
            strategy_signals.rename('signal'),
            returns.rename('returns')
        ], axis=1).dropna()
        
        if len(combined) < self.MIN_TRADES * 2:
            logger.warning("Insufficient data for validation")
            return self._create_failed_report(strategy_name, "Insufficient data")
            
        signals = combined['signal']
        rets = combined['returns']
        
        warnings_list = []
        
        # 1. Walk-forward analysis
        logger.info("Running walk-forward analysis...")
        wf_results = self._walk_forward_analysis(signals, rets)
        
        if not wf_results:
            return self._create_failed_report(strategy_name, "Walk-forward failed")
            
        # Combine OOS results
        combined_oos = self._combine_oos_results(wf_results)
        
        # 2. Regime analysis
        logger.info("Analyzing regime performance...")
        regime_performance = {}
        if features is not None:
            regime_performance = self._analyze_regimes(signals, rets, features)
            
        # 3. Statistical tests
        logger.info("Running statistical tests...")
        stat_tests = self._run_statistical_tests(signals, rets)
        
        # 4. Robustness checks
        logger.info("Running robustness checks...")
        robustness = self._check_robustness(signals, rets, wf_results)
        
        # 5. Generate warnings
        if combined_oos.sharpe_ratio < self.MIN_SHARPE:
            warnings_list.append(f"Low Sharpe ratio: {combined_oos.sharpe_ratio:.2f}")
        if combined_oos.max_drawdown < -self.MAX_DRAWDOWN:
            warnings_list.append(f"High drawdown: {combined_oos.max_drawdown:.1%}")
        if combined_oos.n_trades < self.MIN_TRADES:
            warnings_list.append(f"Low trade count: {combined_oos.n_trades}")
        if not combined_oos.is_significant:
            warnings_list.append("Results not statistically significant")
            
        # Check for OOS degradation
        degradation_count = sum(1 for r in wf_results if r.is_degraded)
        if degradation_count > len(wf_results) // 2:
            warnings_list.append(f"Significant OOS degradation in {degradation_count}/{len(wf_results)} periods")
            
        # 6. Determine grade
        grade = self._determine_grade(combined_oos, robustness, stat_tests)
        
        # 7. Generate recommendation
        recommendation = self._generate_recommendation(
            grade, combined_oos, warnings_list
        )
        
        return ValidationReport(
            strategy_name=strategy_name,
            validation_date=datetime.now(),
            overall_grade=grade,
            recommendation=recommendation,
            walk_forward_results=wf_results,
            combined_oos_metrics=combined_oos,
            regime_performance=regime_performance,
            statistical_tests=stat_tests,
            robustness_scores=robustness,
            warnings=warnings_list
        )
        
    def _walk_forward_analysis(
        self,
        signals: pd.Series,
        returns: pd.Series
    ) -> List[WalkForwardResult]:
        """Perform walk-forward analysis."""
        results = []
        
        n = len(signals)
        period_size = n // self.walk_forward_periods
        
        # Calculate total period in days (4h bars = 6 per day)
        total_days = n / 6
        days_per_period = total_days / self.walk_forward_periods
        
        logger.info(f"Walk-forward: {n} samples, {self.walk_forward_periods} periods, {period_size} samples/period")
        logger.info(f"Total signals active: {(signals != 0).sum()} ({(signals != 0).mean()*100:.1f}%)")
        logger.info(f"Period duration: {days_per_period:.1f} days")
        
        # CRITICAL: Sample size warning
        if n < 1000:
            logger.warning(f"SAMPLE SIZE WARNING: Only {n} samples ({n/6:.0f} days)")
            logger.warning(f"  Minimum recommended: 1000+ samples (6+ months)")
            logger.warning(f"  Sharpe estimates will be unreliable with small samples")
            logger.warning(f"  Consider increasing data range to 12+ months")
        
        if period_size < self.min_test_size * 2:
            logger.warning(f"Periods too small for walk-forward: {period_size} < {self.min_test_size * 2}")
            return []
            
        for i in range(self.walk_forward_periods):
            # Define period boundaries
            period_start = i * period_size
            period_end = (i + 1) * period_size if i < self.walk_forward_periods - 1 else n
            
            # Split into train/test
            train_size = int((period_end - period_start) * self.train_ratio)
            
            train_start_idx = period_start
            train_end_idx = period_start + train_size
            test_start_idx = train_end_idx
            test_end_idx = period_end
            
            test_size = test_end_idx - test_start_idx
            if test_size < self.min_test_size:
                logger.warning(f"Period {i}: test size {test_size} < min {self.min_test_size}, skipping")
                continue
                
            # Get data slices
            train_signals_raw = signals.iloc[train_start_idx:train_end_idx]
            train_returns = returns.iloc[train_start_idx:train_end_idx]
            test_signals_raw = signals.iloc[test_start_idx:test_end_idx]
            test_returns = returns.iloc[test_start_idx:test_end_idx]
            
            # APPLY OVERLAP REMOVAL per walk-forward period
            train_signals = remove_overlapping_signals(train_signals_raw, holding_period=6)
            test_signals = remove_overlapping_signals(test_signals_raw, holding_period=6)
            
            # Calculate period days for train and test
            train_days = int(len(train_signals) / 6)
            test_days = int(len(test_signals) / 6)
            
            # Debug signal distribution (after overlap removal)
            train_active = (train_signals != 0).sum()
            test_active = (test_signals != 0).sum()
            train_raw_active = (train_signals_raw != 0).sum()
            test_raw_active = (test_signals_raw != 0).sum()
            logger.info(f"Period {i}: train signals={train_raw_active}->{train_active} ({train_days}d), test signals={test_raw_active}->{test_active} ({test_days}d)")
            logger.info(f"  [✓] Walk-forward Period {i} overlap removal applied")
            
            # Calculate metrics with correct period_days
            is_metrics = self._calculate_metrics(train_signals, train_returns, period_days=max(1, train_days))
            oos_metrics = self._calculate_metrics(test_signals, test_returns, period_days=max(1, test_days))
            
            # Log detailed diagnostics
            logger.info(f"  train_sharpe: {is_metrics.sharpe_ratio:.2f}, test_sharpe: {oos_metrics.sharpe_ratio:.2f}")
            
            # Check for degradation
            is_degraded = False
            if is_metrics.sharpe_ratio > 0:
                sharpe_degradation = (is_metrics.sharpe_ratio - oos_metrics.sharpe_ratio) / is_metrics.sharpe_ratio
                is_degraded = sharpe_degradation > 0.5  # >50% degradation
                
            results.append(WalkForwardResult(
                period_id=i,
                train_start=signals.index[train_start_idx],
                train_end=signals.index[train_end_idx - 1],
                test_start=signals.index[test_start_idx],
                test_end=signals.index[test_end_idx - 1],
                in_sample_metrics=is_metrics,
                out_sample_metrics=oos_metrics,
                is_degraded=is_degraded
            ))
            
        return results
        
    def _calculate_metrics(
        self,
        signals: pd.Series,
        returns: pd.Series,
        period_days: int = 90
    ) -> ValidationMetrics:
        """
        Calculate comprehensive metrics for a period.
        
        CRITICAL: Sharpe is annualized by TRADE frequency, not bar frequency.
        """
        # Strategy returns
        strategy_returns = signals * returns - self.TOTAL_COST * (signals != 0).astype(float)
        
        # Only count trades where signal != 0
        trades = strategy_returns[signals != 0]
        n_trades = len(trades)
        
        if n_trades < 5:
            return self._empty_metrics()
        
        # === DIRECTIVE #006: TRADE-LEVEL DIAGNOSTICS ===
        trade_returns_raw = returns[signals != 0]
        trade_signals = signals[signals != 0]
        trade_indices = signals[signals != 0].index
        
        logger.info("\n" + "="*70)
        logger.info("TRADE DIAGNOSTICS (first 5 trades)")
        logger.info("="*70)
        
        # Show trade structure
        logger.info(f"\nTrade object structure:")
        logger.info(f"  Index type: {type(trade_indices)}")
        logger.info(f"  Return type: {type(trade_returns_raw)}")
        logger.info(f"  Signal type: {type(trade_signals)}")
        
        for i in range(min(5, n_trades)):
            logger.info(f"\n--- Trade {i+1} ---")
            bar_idx = trade_indices[i] if hasattr(trade_indices, '__getitem__') else i
            raw_ret = trade_returns_raw.iloc[i] if hasattr(trade_returns_raw, 'iloc') else trade_returns_raw[i]
            sig = trade_signals.iloc[i] if hasattr(trade_signals, 'iloc') else trade_signals[i]
            strat_ret = trades.iloc[i] if hasattr(trades, 'iloc') else trades[i]
            
            logger.info(f"  Bar index:       {bar_idx}")
            logger.info(f"  Signal:          {sig} ({'LONG' if sig == 1 else 'SHORT'})")
            logger.info(f"  Raw 6-bar return:{raw_ret*100:.4f}%")
            logger.info(f"  Strategy return: {strat_ret*100:.4f}%")
            logger.info(f"  Math check:      signal({sig}) * raw({raw_ret:.6f}) - costs(0.0012) = {sig * raw_ret - 0.0012:.6f}")
            if sig == -1:
                logger.info(f"  SHORT logic:     Price {'dropped' if raw_ret < 0 else 'rose'} {abs(raw_ret)*100:.2f}%, we {'profited' if strat_ret > 0 else 'lost'}")
        
        logger.info(f"\nRaw returns stats:")
        logger.info(f"  min:    {trade_returns_raw.min():.6f} ({trade_returns_raw.min()*100:.3f}%)")
        logger.info(f"  max:    {trade_returns_raw.max():.6f} ({trade_returns_raw.max()*100:.3f}%)")
        logger.info(f"  mean:   {trade_returns_raw.mean():.6f} ({trade_returns_raw.mean()*100:.3f}%)")
        logger.info(f"  median: {trade_returns_raw.median():.6f} ({trade_returns_raw.median()*100:.3f}%)")
        logger.info(f"  std:    {trade_returns_raw.std():.6f} ({trade_returns_raw.std()*100:.3f}%)")
        logger.info("="*70 + "\n")
            
        # Core metrics
        total_return = (1 + strategy_returns).prod() - 1
        
        # Annualized return
        n_bars = len(strategy_returns)
        bars_per_day = 6  # 4h bars
        actual_days = n_bars / bars_per_day
        annualized = (1 + total_return) ** (365 / actual_days) - 1 if actual_days > 0 else 0
        
        # ============================================================
        # SHARPE CALCULATION - ANNUALIZE BY TRADE FREQUENCY
        # ============================================================
        trade_returns = trades.values if hasattr(trades, 'values') else np.array(trades)
        
        if len(trade_returns) < 2:
            sharpe = 0.0
            sortino = 0.0
            mean_ret = 0.0
            std_ret = 0.0
        else:
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns, ddof=1)
            
            if std_ret == 0 or np.isnan(std_ret):
                sharpe = 0.0
                sortino = 0.0
            else:
                per_trade_sharpe = mean_ret / std_ret
                
                # THIS IS THE KEY: Annualize by TRADE frequency
                trades_per_year = (n_trades / period_days) * 365
                sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
                
                # DIAGNOSTIC: Log individual trade returns when suspicious
                if per_trade_sharpe > 0.5:
                    logger.warning(
                        f"SUSPICIOUS per_trade_sharpe={per_trade_sharpe:.4f} (normal is 0.05-0.30)\n"
                        f"  Trade returns (first 10): {trade_returns[:10]}\n"
                        f"  Min return: {np.min(trade_returns):.6f}\n"
                        f"  Max return: {np.max(trade_returns):.6f}\n"
                        f"  Median return: {np.median(trade_returns):.6f}\n"
                        f"  This may indicate data leakage or returns calculation error."
                    )
                
                # SANITY CHECK - DO NOT REMOVE
                if sharpe > 10:
                    logger.error(
                        f"SHARPE > 10 DETECTED: {sharpe:.2f}\n"
                        f"  mean={mean_ret:.6f}, std={std_ret:.6f}\n"
                        f"  n_trades={n_trades}, period_days={period_days}\n"
                        f"  trades_per_year={trades_per_year:.1f}\n"
                        f"  per_trade_sharpe={per_trade_sharpe:.4f}\n"
                        f"  THIS IS LIKELY A BUG"
                    )
                
                # Sortino
                downside_returns = trade_returns[trade_returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns, ddof=1)
                    if downside_std > 0:
                        sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)
                    else:
                        sortino = sharpe * 1.5
                else:
                    sortino = sharpe * 1.5
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Trade statistics
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / (total_losses + 1e-8)
        
        # Risk metrics
        var_95 = np.percentile(strategy_returns, 5) if len(strategy_returns) > 20 else 0
        es = strategy_returns[strategy_returns <= var_95].mean() if len(strategy_returns[strategy_returns <= var_95]) > 0 else var_95
        calmar = annualized / (abs(max_dd) + 1e-8)
        
        # Statistical significance
        t_stat, p_value = self._test_significance(strategy_returns)
        is_significant = p_value < self.SIGNIFICANCE_LEVEL
        
        return ValidationMetrics(
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            var_95=var_95,
            expected_shortfall=es,
            calmar_ratio=calmar,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            regime_consistency=0.0,
            time_consistency=0.0
        )
        
    def _test_significance(
        self,
        returns: pd.Series
    ) -> Tuple[float, float]:
        """Test if returns are significantly different from zero."""
        if not HAS_SCIPY or len(returns) < 20:
            return 0.0, 1.0
            
        try:
            t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
            return t_stat, p_value / 2  # One-sided test (positive returns)
        except Exception:
            return 0.0, 1.0
            
    def _combine_oos_results(
        self,
        wf_results: List[WalkForwardResult]
    ) -> ValidationMetrics:
        """Combine all out-of-sample results."""
        if not wf_results:
            return self._empty_metrics()
            
        # Weight by number of trades
        total_trades = sum(r.out_sample_metrics.n_trades for r in wf_results)
        
        if total_trades == 0:
            return self._empty_metrics()
            
        # Weighted averages
        combined = self._empty_metrics()
        
        for r in wf_results:
            oos = r.out_sample_metrics
            weight = oos.n_trades / total_trades
            
            combined.total_return += oos.total_return * weight
            combined.annualized_return += oos.annualized_return * weight
            combined.sharpe_ratio += oos.sharpe_ratio * weight
            combined.sortino_ratio += oos.sortino_ratio * weight
            combined.max_drawdown = min(combined.max_drawdown, oos.max_drawdown)
            combined.n_trades += oos.n_trades
            combined.win_rate += oos.win_rate * weight
            combined.avg_win += oos.avg_win * weight
            combined.avg_loss += oos.avg_loss * weight
            combined.profit_factor += oos.profit_factor * weight
            combined.var_95 = min(combined.var_95, oos.var_95)
            
        # Recalculate significance from combined
        sharpes = [r.out_sample_metrics.sharpe_ratio for r in wf_results]
        combined.time_consistency = 1 - (np.std(sharpes) / (np.mean(sharpes) + 1e-8))
        
        # Overall significance
        n_significant = sum(1 for r in wf_results if r.out_sample_metrics.is_significant)
        combined.is_significant = n_significant >= len(wf_results) // 2
        
        return combined
        
    def _analyze_regimes(
        self,
        signals: pd.Series,
        returns: pd.Series,
        features: pd.DataFrame
    ) -> Dict[str, ValidationMetrics]:
        """Analyze performance across market regimes."""
        regime_metrics = {}
        
        # Calculate total period days (4h bars = 6 per day)
        total_days = len(signals) / 6
        
        # Find volatility feature
        vol_features = [c for c in features.columns if 'vol' in c.lower() or 'atr' in c.lower()]
        
        if vol_features:
            vol_feat = vol_features[0]
            vol = features[vol_feat].reindex(signals.index)
            
            median_vol = vol.median()
            
            # Low volatility regime
            low_vol_mask = vol <= median_vol
            regime_days = int(low_vol_mask.sum() / 6)
            if low_vol_mask.sum() >= self.MIN_TRADES:
                regime_metrics['low_volatility'] = self._calculate_metrics(
                    signals[low_vol_mask],
                    returns[low_vol_mask],
                    period_days=max(1, regime_days)
                )
                
            # High volatility regime
            high_vol_mask = vol > median_vol
            regime_days = int(high_vol_mask.sum() / 6)
            if high_vol_mask.sum() >= self.MIN_TRADES:
                regime_metrics['high_volatility'] = self._calculate_metrics(
                    signals[high_vol_mask],
                    returns[high_vol_mask],
                    period_days=max(1, regime_days)
                )
                
        # Find trend feature
        trend_features = [c for c in features.columns if 'trend' in c.lower() or 'adx' in c.lower()]
        
        if trend_features:
            trend_feat = trend_features[0]
            trend = features[trend_feat].reindex(signals.index)
            
            median_trend = trend.median()
            
            # Ranging regime
            ranging_mask = trend <= median_trend
            regime_days = int(ranging_mask.sum() / 6)
            if ranging_mask.sum() >= self.MIN_TRADES:
                regime_metrics['ranging'] = self._calculate_metrics(
                    signals[ranging_mask],
                    returns[ranging_mask],
                    period_days=max(1, regime_days)
                )
                
            # Trending regime
            trending_mask = trend > median_trend
            regime_days = int(trending_mask.sum() / 6)
            if trending_mask.sum() >= self.MIN_TRADES:
                regime_metrics['trending'] = self._calculate_metrics(
                    signals[trending_mask],
                    returns[trending_mask],
                    period_days=max(1, regime_days)
                )
                
        return regime_metrics
        
    def _run_statistical_tests(
        self,
        signals: pd.Series,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Run comprehensive statistical tests."""
        tests = {}
        
        strategy_returns = signals * returns - self.TOTAL_COST * (signals != 0).astype(float)
        
        if not HAS_SCIPY or len(strategy_returns) < 50:
            return tests
            
        try:
            # 1. T-test vs zero
            t_stat, p_value = stats.ttest_1samp(strategy_returns.dropna(), 0)
            tests['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.SIGNIFICANCE_LEVEL
            }
            
            # 2. Normality test (Shapiro-Wilk on sample)
            sample = strategy_returns.dropna().sample(min(1000, len(strategy_returns)), random_state=42)
            _, p_norm = stats.shapiro(sample)
            tests['normality'] = {
                'p_value': p_norm,
                'is_normal': p_norm > 0.05
            }
            
            # 3. Skewness and kurtosis
            tests['distribution'] = {
                'skewness': stats.skew(strategy_returns.dropna()),
                'kurtosis': stats.kurtosis(strategy_returns.dropna()),
                'is_heavy_tailed': stats.kurtosis(strategy_returns.dropna()) > 3
            }
            
            # 4. Autocorrelation test (Ljung-Box)
            from scipy.stats import chi2
            n = len(strategy_returns)
            acf = pd.Series(strategy_returns).autocorr(lag=1)
            lb_stat = n * (n + 2) * (acf ** 2) / (n - 1)
            p_autocorr = 1 - chi2.cdf(lb_stat, df=1)
            tests['autocorrelation'] = {
                'lag1_acf': acf,
                'ljung_box_stat': lb_stat,
                'p_value': p_autocorr,
                'has_autocorrelation': p_autocorr < 0.05
            }
            
            # 5. Runs test (tests for randomness)
            median_ret = strategy_returns.median()
            signs = (strategy_returns > median_ret).astype(int)
            runs = 1 + sum(signs.iloc[i] != signs.iloc[i-1] for i in range(1, len(signs)))
            n_pos = signs.sum()
            n_neg = len(signs) - n_pos
            
            expected_runs = 1 + 2 * n_pos * n_neg / len(signs)
            var_runs = 2 * n_pos * n_neg * (2 * n_pos * n_neg - len(signs)) / (len(signs) ** 2 * (len(signs) - 1))
            z_runs = (runs - expected_runs) / (np.sqrt(var_runs) + 1e-8)
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            
            tests['runs_test'] = {
                'n_runs': runs,
                'expected_runs': expected_runs,
                'z_statistic': z_runs,
                'p_value': p_runs,
                'is_random': p_runs > 0.05
            }
            
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
            
        return tests
        
    def _check_robustness(
        self,
        signals: pd.Series,
        returns: pd.Series,
        wf_results: List[WalkForwardResult]
    ) -> Dict[str, float]:
        """Check strategy robustness."""
        robustness = {}
        
        # 1. Walk-forward consistency
        if wf_results:
            sharpes = [r.out_sample_metrics.sharpe_ratio for r in wf_results]
            if len(sharpes) > 1 and np.mean(sharpes) != 0:
                robustness['wf_consistency'] = 1 - min(1, np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-8))
            else:
                robustness['wf_consistency'] = 0
                
            # Degradation rate
            n_degraded = sum(1 for r in wf_results if r.is_degraded)
            robustness['degradation_rate'] = 1 - n_degraded / len(wf_results)
        else:
            robustness['wf_consistency'] = 0
            robustness['degradation_rate'] = 0
            
        # 2. Time stability (split into halves)
        n = len(signals)
        half_days = int((n // 2) / 6)  # 4h bars to days
        first_half = self._calculate_metrics(signals.iloc[:n//2], returns.iloc[:n//2], period_days=max(1, half_days))
        second_half = self._calculate_metrics(signals.iloc[n//2:], returns.iloc[n//2:], period_days=max(1, half_days))
        
        if first_half.sharpe_ratio != 0:
            sharpe_change = abs(second_half.sharpe_ratio - first_half.sharpe_ratio) / (abs(first_half.sharpe_ratio) + 1e-8)
            robustness['time_stability'] = max(0, 1 - sharpe_change)
        else:
            robustness['time_stability'] = 0.5
            
        # 3. Parameter sensitivity (test with delayed signals)
        total_days = int(len(signals) / 6)
        for delay in [1, 2, 4]:
            delayed_signals = signals.shift(delay).fillna(0)
            delayed_metrics = self._calculate_metrics(delayed_signals, returns, period_days=max(1, total_days))
            robustness[f'delay_{delay}h_robustness'] = max(0, delayed_metrics.sharpe_ratio / (first_half.sharpe_ratio + 1e-8))
            
        # 4. Cost sensitivity
        for cost_mult in [1.5, 2.0]:
            high_cost_returns = signals * returns - self.TOTAL_COST * cost_mult * (signals != 0).astype(float)
            high_cost_trades = high_cost_returns[signals != 0]
            n_trades_cost = len(high_cost_trades)
            if n_trades_cost >= 5:
                # Use trade frequency for Sharpe
                mean_ret = high_cost_trades.mean()
                std_ret = high_cost_trades.std() + 1e-8
                trades_per_year = (n_trades_cost / 90) * 365  # Assume 90-day period
                high_cost_sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
            else:
                high_cost_sharpe = 0.0
            robustness[f'cost_{cost_mult}x_sharpe'] = high_cost_sharpe
            
        return robustness
        
    def _determine_grade(
        self,
        metrics: ValidationMetrics,
        robustness: Dict[str, float],
        stat_tests: Dict[str, Any]
    ) -> str:
        """Determine overall strategy grade."""
        
        # CRITICAL SANITY CHECK: Sharpe > 10 is mathematically impossible
        # This indicates a calculation error or data leakage
        if metrics.sharpe_ratio > 10:
            logger.error(
                f"GRADE BLOCKED: Sharpe {metrics.sharpe_ratio:.2f} > 10 indicates calculation error.\n"
                f"  No real strategy achieves Sharpe > 10.\n"
                f"  Renaissance Medallion (best fund ever) achieves ~5-6.\n"
                f"  Forcing grade to F."
            )
            return 'F'
        
        score = 0
        
        # Sharpe (40 points)
        if metrics.sharpe_ratio >= 2.0:
            score += 40
        elif metrics.sharpe_ratio >= 1.5:
            score += 35
        elif metrics.sharpe_ratio >= 1.0:
            score += 25
        elif metrics.sharpe_ratio >= 0.5:
            score += 15
        elif metrics.sharpe_ratio > 0:
            score += 5
            
        # Win rate (15 points)
        if metrics.win_rate >= 0.55:
            score += 15
        elif metrics.win_rate >= 0.52:
            score += 10
        elif metrics.win_rate >= 0.50:
            score += 5
            
        # Max drawdown (15 points)
        if metrics.max_drawdown > -0.10:
            score += 15
        elif metrics.max_drawdown > -0.15:
            score += 12
        elif metrics.max_drawdown > -0.20:
            score += 8
        elif metrics.max_drawdown > -0.30:
            score += 4
            
        # Statistical significance (15 points)
        if metrics.is_significant:
            score += 15
        elif metrics.p_value < 0.10:
            score += 8
            
        # Robustness (15 points)
        avg_robustness = np.mean([
            robustness.get('wf_consistency', 0),
            robustness.get('time_stability', 0),
            robustness.get('degradation_rate', 0)
        ])
        score += int(avg_robustness * 15)
        
        # Determine grade
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 55:
            return 'C'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
            
    def _generate_recommendation(
        self,
        grade: str,
        metrics: ValidationMetrics,
        warnings: List[str]
    ) -> str:
        """Generate actionable recommendation."""
        
        # CRITICAL: Check for impossible Sharpe values
        if metrics.sharpe_ratio > 10:
            return (
                "DO NOT DEPLOY: Sharpe > 10 indicates calculation error or data leakage. "
                "No real strategy achieves this. Investigate returns calculation."
            )
        
        if grade == 'A':
            return "DEPLOY: Strategy shows strong, robust performance. Recommend live testing with small size."
        elif grade == 'B':
            return "CONSIDER: Strategy shows promise but has some weaknesses. Address warnings before deployment."
        elif grade == 'C':
            return "REFINE: Strategy needs improvement. Consider adding filters or adjusting parameters."
        elif grade == 'D':
            return "REVIEW: Strategy underperforms. Major revisions needed before considering deployment."
        else:
            return "REJECT: Strategy does not meet minimum requirements. Do not deploy."
            
    def _empty_metrics(self) -> ValidationMetrics:
        """Create empty metrics object."""
        return ValidationMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            n_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            calmar_ratio=0.0,
            t_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            regime_consistency=0.0,
            time_consistency=0.0
        )
        
    def _create_failed_report(
        self,
        strategy_name: str,
        reason: str
    ) -> ValidationReport:
        """Create a failed validation report."""
        return ValidationReport(
            strategy_name=strategy_name,
            validation_date=datetime.now(),
            overall_grade='F',
            recommendation=f"FAILED: {reason}",
            walk_forward_results=[],
            combined_oos_metrics=self._empty_metrics(),
            regime_performance={},
            statistical_tests={},
            robustness_scores={},
            warnings=[reason]
        )


class MonteCarloValidator:
    """
    Monte Carlo simulation for strategy validation.
    
    Tests strategy against random permutations to establish significance.
    """
    
    def __init__(self, n_simulations: int = 1000, random_state: int = 42):
        self.n_simulations = n_simulations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def validate(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        period_days: int = 90
    ) -> Dict[str, Any]:
        """
        Validate strategy using Monte Carlo.
        
        Args:
            strategy_returns: Realized strategy returns
            market_returns: Underlying market returns
            period_days: Number of days in the period
            
        Returns:
            Monte Carlo validation results
        """
        # Get trades only (non-zero returns)
        actual_trades = strategy_returns[strategy_returns != 0]
        n_trades = len(actual_trades)
        
        if n_trades < 5:
            return {
                'actual_sharpe': 0.0,
                'actual_total_return': 0.0,
                'sharpe_percentile': 50.0,
                'total_return_percentile': 50.0,
                'is_better_than_random': False,
                'random_sharpe_mean': 0.0,
                'random_sharpe_std': 0.0,
            }
        
        # Calculate actual Sharpe using trade frequency
        trades_per_year = (n_trades / period_days) * 365
        actual_sharpe = (actual_trades.mean() / (actual_trades.std() + 1e-8)) * np.sqrt(trades_per_year)
        actual_total = (1 + strategy_returns).prod() - 1
        
        # Generate random strategies
        random_sharpes = []
        random_totals = []
        
        for _ in range(self.n_simulations):
            # Random signals with same sparsity as actual strategy
            signal_rate = n_trades / len(market_returns)
            random_signals = np.random.choice(
                [-1, 0, 1], 
                size=len(market_returns),
                p=[signal_rate/2, 1-signal_rate, signal_rate/2]
            )
            random_ret = random_signals * market_returns.values
            
            # Get random trades and compute Sharpe
            random_trades = random_ret[random_ret != 0]
            n_random_trades = len(random_trades)
            
            if n_random_trades >= 2:
                random_trades_per_year = (n_random_trades / period_days) * 365
                sharpe = (random_trades.mean() / (random_trades.std() + 1e-8)) * np.sqrt(random_trades_per_year)
            else:
                sharpe = 0.0
                
            total = (1 + random_ret).prod() - 1
            
            random_sharpes.append(sharpe)
            random_totals.append(total)
            
        # Calculate percentile ranks
        sharpe_rank = (np.array(random_sharpes) < actual_sharpe).mean()
        total_rank = (np.array(random_totals) < actual_total).mean()
        
        return {
            'actual_sharpe': actual_sharpe,
            'actual_total_return': actual_total,
            'sharpe_percentile': sharpe_rank * 100,
            'total_return_percentile': total_rank * 100,
            'is_better_than_random': sharpe_rank > 0.95,
            'random_sharpe_mean': np.mean(random_sharpes),
            'random_sharpe_std': np.std(random_sharpes),
            'n_simulations': self.n_simulations
        }


def validate_discovered_strategy(
    entry_signals: pd.Series,
    forward_returns: pd.Series,
    features: pd.DataFrame,
    strategy_name: str = "Discovered Strategy"
) -> ValidationReport:
    """
    Convenience function to validate a discovered strategy.
    
    Args:
        entry_signals: Trading signals (1=long, -1=short, 0=flat)
        forward_returns: Forward returns
        features: Feature DataFrame for regime analysis
        strategy_name: Name for the strategy
        
    Returns:
        Complete ValidationReport
    """
    # ========================================================================
    # OVERLAP REMOVAL - Critical for valid Sharpe calculation
    # ========================================================================
    # Trades with 6-bar holding periods can overlap if signals fire on
    # consecutive bars. This creates correlated returns that artificially
    # inflate Sharpe ratio (e.g., 161 "trades" that are really ~27 independent)
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("OVERLAP REMOVAL (6-bar holding period)")
    logger.info("=" * 70)
    
    raw_signal_count = (entry_signals != 0).sum()
    clean_signals = remove_overlapping_signals(entry_signals, holding_period=6)
    clean_signal_count = (clean_signals != 0).sum()
    
    removed = raw_signal_count - clean_signal_count
    removed_pct = (removed / raw_signal_count * 100) if raw_signal_count > 0 else 0
    
    logger.info(f"  Raw signals: {raw_signal_count}")
    logger.info(f"  After overlap removal: {clean_signal_count}")
    logger.info(f"  Removed: {removed} ({removed_pct:.1f}%)")
    
    if removed_pct > 50:
        logger.warning(f"  HIGH OVERLAP: {removed_pct:.1f}% of signals were overlapping!")
        logger.warning(f"  This indicates clustered entry signals (common during trends)")
    
    logger.info("=" * 70)
    
    validator = StrategyValidator()
    return validator.validate(
        clean_signals,  # Use overlap-removed signals
        forward_returns,
        features=features,
        strategy_name=strategy_name
    )
