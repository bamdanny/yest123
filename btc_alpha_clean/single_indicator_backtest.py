#!/usr/bin/env python3
"""
SINGLE INDICATOR BACKTEST - v37 FINAL FIX

CRITICAL BUG FOUND:
- Phase 1 uses return_simple_6h from TARGETS (6-bar forward return)
- Previous versions computed their own returns from prices (1-bar forward)

FIX: Load BOTH features AND targets from cache, use return_simple_6h

This should now match Phase 1 EXACTLY.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# EXACT Phase 1 specifications
PHASE1_RULES = [
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "expected_oos_sharpe": 8.74,
        "expected_oos_winrate": 0.735,
        "expected_oos_trades": 49
    },
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_accel",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 80,
        "expected_oos_sharpe": 7.38,
        "expected_oos_winrate": 0.614,
        "expected_oos_trades": 70
    },
    {
        "name": "price_rsi_14_lag_48h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 1.5,
        "expected_oos_sharpe": 5.22,
        "expected_oos_winrate": 0.625,
        "expected_oos_trades": 32
    },
    {
        "name": "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 2.0,
        "expected_oos_sharpe": 4.98,
        "expected_oos_winrate": 0.629,
        "expected_oos_trades": 35
    },
    {
        "name": "sent_feat_fg_zscore_90d",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "expected_oos_sharpe": 4.59,
        "expected_oos_winrate": 0.759,
        "expected_oos_trades": 29
    },
]


def load_data():
    """
    Load BOTH features AND targets from cache.
    
    CRITICAL: Phase 1 uses return_simple_6h from targets, not computed from prices!
    """
    cache_path = Path("data_cache/features_cache.pkl")
    
    if not cache_path.exists():
        raise FileNotFoundError("No features cache. Run: python run_exhaustive_search.py first")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    targets = cache['targets']
    
    # Filter to numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features = features[numeric_cols].copy()
    
    logger.info(f"Loaded features: {features.shape}")
    logger.info(f"Loaded targets: {targets.shape}")
    
    # Get return_simple_6h - this is what Phase 1 uses!
    if 'return_simple_6h' in targets.columns:
        returns = targets['return_simple_6h'].copy()
        logger.info("Using return_simple_6h from targets (MATCHES PHASE 1)")
    elif 'return_simple_4h' in targets.columns:
        returns = targets['return_simple_4h'].copy()
        logger.info("Fallback: using return_simple_4h")
    else:
        # Find any return column
        ret_cols = [c for c in targets.columns if 'return_simple' in c]
        if ret_cols:
            returns = targets[ret_cols[0]].copy()
            logger.info(f"Fallback: using {ret_cols[0]}")
        else:
            raise ValueError("No return columns found in targets!")
    
    # Align features and returns (same length, matching indices)
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len].reset_index(drop=True)
    returns = returns.iloc[:min_len].reset_index(drop=True)
    
    # Drop NaN rows from returns
    valid_idx = returns.dropna().index
    features = features.loc[valid_idx].reset_index(drop=True)
    returns = returns.loc[valid_idx].reset_index(drop=True)
    
    logger.info(f"After alignment: {len(features)} samples")
    
    return features, returns


def generate_signals_phase1(feature_values, rule, train_mask):
    """
    Generate signals EXACTLY as Phase 1 does.
    
    From run_exhaustive_search.py SignalGenerator.apply_fixed_threshold (lines 287-306):
    
    if direction == 1:
        signals[values > upper_threshold] = 1
        signals[values < lower_threshold] = -1
    else:
        signals[values > upper_threshold] = -1
        signals[values < lower_threshold] = 1
    """
    train_values = feature_values[train_mask]
    
    if rule['threshold_type'] == 'percentile':
        pct_upper = rule['threshold_value']
        pct_lower = 100 - pct_upper
        
        upper_threshold = np.nanpercentile(train_values, pct_upper)
        lower_threshold = np.nanpercentile(train_values, pct_lower)
        
        signals = np.zeros(len(feature_values))
        
        if rule['direction'] == 1:
            signals[feature_values > upper_threshold] = 1
            signals[feature_values < lower_threshold] = -1
        else:
            signals[feature_values > upper_threshold] = -1
            signals[feature_values < lower_threshold] = 1
        
        return signals, upper_threshold, lower_threshold
    
    else:  # zscore
        mean = np.nanmean(train_values)
        std = np.nanstd(train_values)
        
        if std < 1e-10:
            return np.zeros(len(feature_values)), np.nan, np.nan
        
        zscore = (feature_values - mean) / std
        
        z_upper = rule['threshold_value']
        z_lower = -rule['threshold_value']
        
        signals = np.zeros(len(feature_values))
        
        if rule['direction'] == 1:
            signals[zscore > z_upper] = 1
            signals[zscore < z_lower] = -1
        else:
            signals[zscore > z_upper] = -1
            signals[zscore < z_lower] = 1
        
        return signals, mean + z_upper * std, mean + z_lower * std


def calculate_sharpe_phase1(trade_returns, period_days):
    """
    Calculate Sharpe EXACTLY as Phase 1 does.
    
    From run_exhaustive_search.py calculate_sharpe_time_based (lines 176-219).
    """
    n_trades = len(trade_returns)
    if n_trades < 5 or period_days <= 0:
        return 0.0
    
    total_return = np.prod(1 + trade_returns) - 1
    daily_return = (1 + total_return) ** (1 / period_days) - 1
    
    trade_std = np.std(trade_returns)
    trades_per_day = n_trades / period_days
    daily_std = trade_std * np.sqrt(trades_per_day)
    
    if daily_std < 1e-10:
        return 0.0
    
    daily_sharpe = daily_return / daily_std
    return daily_sharpe * np.sqrt(365)


def backtest_phase1_exact(features, returns, rule, train_ratio=0.6):
    """
    Backtest EXACTLY as Phase 1 does.
    
    From run_exhaustive_search.py IndicatorTester._calculate_metrics (lines 813-887):
    
    Key insight: Phase 1 uses return_simple_6h which is already the forward return.
    At bar t, signals[t] predicts returns[t], where returns[t] is the 6-bar forward return.
    """
    feature_name = rule['name']
    
    if feature_name not in features.columns:
        logger.warning(f"Feature {feature_name} not found!")
        return None
    
    feature_values = features[feature_name].values
    return_values = returns.values
    
    n = len(feature_values)
    n_train = int(n * train_ratio)
    
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    
    # Generate signals using Phase 1 logic
    signals, upper_thresh, lower_thresh = generate_signals_phase1(
        feature_values, rule, train_mask
    )
    
    if np.isnan(upper_thresh):
        return None
    
    logger.info(f"  Thresholds: upper={upper_thresh:.6f}, lower={lower_thresh:.6f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: Phase 1 formula (from _calculate_metrics line 835):
    #   raw_trade_returns = returns[trade_mask] * signals[trade_mask]
    #
    # This multiplies signals at each bar by returns at THE SAME bar.
    # The returns are already forward-looking (return_simple_6h).
    # ═══════════════════════════════════════════════════════════════════
    
    # Calculate trade returns: signal[t] * returns[t]
    trade_returns = signals * return_values
    
    # Split into IS and OOS
    is_signals = signals[:n_train]
    is_returns = trade_returns[:n_train]
    
    oos_signals = signals[n_train:]
    oos_returns = trade_returns[n_train:]
    
    def calc_metrics(strat_rets, sigs, period_days):
        """Calculate metrics as Phase 1 does."""
        # Only count bars where we had a position
        trade_mask = sigs != 0
        trading_returns = strat_rets[trade_mask]
        n_trades = len(trading_returns)
        
        if n_trades < 5:
            return {
                'sharpe': 0.0, 'win_rate': 0.0, 'total_return': 0.0,
                'n_trades': n_trades, 'n_long': 0, 'n_short': 0
            }
        
        wins = np.sum(trading_returns > 0)
        losses = np.sum(trading_returns < 0)
        win_rate = wins / n_trades if n_trades > 0 else 0
        
        total_return = np.prod(1 + trading_returns) - 1
        sharpe = calculate_sharpe_phase1(trading_returns, period_days)
        
        n_long = np.sum(sigs[trade_mask] == 1)
        n_short = np.sum(sigs[trade_mask] == -1)
        
        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_return': total_return,
            'n_trades': n_trades,
            'wins': wins,
            'losses': losses,
            'n_long': n_long,
            'n_short': n_short
        }
    
    # Period in days (6 bars per day at 4h timeframe)
    train_days = n_train / 6
    test_days = (n - n_train) / 6
    
    is_metrics = calc_metrics(is_returns, is_signals, train_days)
    oos_metrics = calc_metrics(oos_returns, oos_signals, test_days)
    
    return {
        'rule': rule,
        'upper_threshold': upper_thresh,
        'lower_threshold': lower_thresh,
        'is': is_metrics,
        'oos': oos_metrics,
        'train_days': train_days,
        'test_days': test_days
    }


def main():
    logger.info("=" * 70)
    logger.info("SINGLE INDICATOR BACKTEST - v37 FINAL FIX")
    logger.info("=" * 70)
    
    logger.info("\nCRITICAL FIX: Now using return_simple_6h from targets cache")
    logger.info("This matches Phase 1 EXACTLY - same returns, same indices")
    
    # Load data
    features, returns = load_data()
    
    all_results = []
    
    for rule in PHASE1_RULES:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING: {rule['name']}")
        logger.info(f"{'='*60}")
        logger.info(f"  Config: direction={rule['direction']}, "
                   f"{rule['threshold_type']}={rule['threshold_value']}")
        logger.info(f"  Expected: Sharpe={rule['expected_oos_sharpe']:.2f}, "
                   f"WR={rule['expected_oos_winrate']*100:.1f}%, "
                   f"Trades={rule['expected_oos_trades']}")
        
        results = backtest_phase1_exact(features, returns, rule)
        
        if results is None:
            continue
        
        all_results.append(results)
        
        is_m = results['is']
        oos_m = results['oos']
        
        logger.info(f"\n  IN-SAMPLE ({results['train_days']:.1f} days):")
        logger.info(f"    Trades: {is_m['n_trades']} ({is_m['n_long']}L + {is_m['n_short']}S)")
        logger.info(f"    Win Rate: {is_m['win_rate']*100:.1f}%")
        logger.info(f"    Sharpe: {is_m['sharpe']:.2f}")
        
        logger.info(f"\n  OUT-OF-SAMPLE ({results['test_days']:.1f} days):")
        logger.info(f"    Trades: {oos_m['n_trades']} ({oos_m['n_long']}L + {oos_m['n_short']}S)")
        logger.info(f"    Win Rate: {oos_m['win_rate']*100:.1f}%")
        logger.info(f"    Total Return: {oos_m['total_return']*100:.1f}%")
        logger.info(f"    Sharpe: {oos_m['sharpe']:.2f}")
        
        # Validation
        expected = rule['expected_oos_sharpe']
        actual = oos_m['sharpe']
        expected_wr = rule['expected_oos_winrate']
        actual_wr = oos_m['win_rate']
        expected_trades = rule['expected_oos_trades']
        actual_trades = oos_m['n_trades']
        
        sharpe_diff = abs(actual - expected) / expected if expected > 0 else 1
        wr_diff = abs(actual_wr - expected_wr)
        trades_diff = abs(actual_trades - expected_trades)
        
        if sharpe_diff < 0.25 and wr_diff < 0.1 and trades_diff <= 10:
            logger.info(f"\n    ✅ MATCHES Phase 1!")
        else:
            logger.warning(f"\n    ❌ Does not match (Sharpe diff: {sharpe_diff*100:.1f}%, "
                          f"WR diff: {wr_diff*100:.1f}%, Trade diff: {trades_diff})")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n{'Indicator':<45} {'Exp':>6} {'Act':>6} {'ExpWR':>6} {'ActWR':>6} {'Trades':>8} {'Match':>6}")
    logger.info("-" * 95)
    
    matches = 0
    for r in all_results:
        name = r['rule']['name'][:43]
        expected = r['rule']['expected_oos_sharpe']
        actual = r['oos']['sharpe']
        exp_wr = r['rule']['expected_oos_winrate'] * 100
        act_wr = r['oos']['win_rate'] * 100
        exp_trades = r['rule']['expected_oos_trades']
        act_trades = r['oos']['n_trades']
        
        sharpe_ok = abs(actual - expected) / expected < 0.25 if expected > 0 else True
        wr_ok = abs(act_wr - exp_wr) < 10
        trades_ok = abs(act_trades - exp_trades) <= 10
        
        match = "✅" if sharpe_ok and wr_ok and trades_ok else "❌"
        if sharpe_ok and wr_ok and trades_ok:
            matches += 1
        
        logger.info(f"{name:<45} {expected:>6.2f} {actual:>6.2f} {exp_wr:>5.1f}% {act_wr:>5.1f}% {act_trades:>4}/{exp_trades:<3} {match:>6}")
    
    logger.info(f"\nMatched: {matches}/{len(all_results)} indicators")
    
    # Sanity checks
    logger.info(f"\n{'='*70}")
    logger.info("SANITY CHECKS")
    logger.info("=" * 70)
    
    for r in all_results:
        oos = r['oos']
        name = r['rule']['name'][:40]
        
        if oos['sharpe'] > 15:
            logger.error(f"❌ {name}: Sharpe {oos['sharpe']:.1f} > 15 - DATA LEAKAGE!")
        elif oos['sharpe'] < 0:
            logger.warning(f"⚠️  {name}: Sharpe {oos['sharpe']:.2f} is negative")
        else:
            logger.info(f"✅ {name}: Sharpe {oos['sharpe']:.2f} is plausible")
        
        # Check for inverted win rates (the bug we're fixing)
        exp_wr = r['rule']['expected_oos_winrate']
        act_wr = oos['win_rate']
        if abs(act_wr - (1 - exp_wr)) < 0.1:
            logger.error(f"   ❌ INVERTED: Actual WR {act_wr*100:.1f}% ≈ 100% - Expected {exp_wr*100:.1f}%")


if __name__ == "__main__":
    main()
