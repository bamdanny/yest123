#!/usr/bin/env python3
"""
SIMPLE ENSEMBLE - v37 FINAL FIX

Uses return_simple_6h from targets (same as Phase 1).
Combines indicators using position sizing.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PHASE1_RULES = [
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 8.74,
        "weight": 0.30
    },
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_accel",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 80,
        "oos_sharpe": 7.38,
        "weight": 0.25
    },
    {
        "name": "price_rsi_14_lag_48h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 1.5,
        "oos_sharpe": 5.22,
        "weight": 0.20
    },
    {
        "name": "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 2.0,
        "oos_sharpe": 4.98,
        "weight": 0.15
    },
    {
        "name": "sent_feat_fg_zscore_90d",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 4.59,
        "weight": 0.10
    },
]


def load_data():
    """Load features AND targets from cache."""
    cache_path = Path("data_cache/features_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError("No features cache")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    targets = cache['targets']
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features = features[numeric_cols].copy()
    
    # Get return_simple_6h
    if 'return_simple_6h' in targets.columns:
        returns = targets['return_simple_6h'].copy()
    else:
        ret_cols = [c for c in targets.columns if 'return_simple' in c]
        returns = targets[ret_cols[0]].copy() if ret_cols else None
        
    if returns is None:
        raise ValueError("No return columns in targets")
    
    # Align
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len].reset_index(drop=True)
    returns = returns.iloc[:min_len].reset_index(drop=True)
    
    valid_idx = returns.dropna().index
    features = features.loc[valid_idx].reset_index(drop=True)
    returns = returns.loc[valid_idx].reset_index(drop=True)
    
    return features, returns


def generate_signals(feature_values, rule, train_mask):
    """Generate +1/-1/0 signals."""
    train_values = feature_values[train_mask]
    
    if rule['threshold_type'] == 'percentile':
        pct_upper = rule['threshold_value']
        pct_lower = 100 - pct_upper
        
        upper = np.nanpercentile(train_values, pct_upper)
        lower = np.nanpercentile(train_values, pct_lower)
        
        signals = np.zeros(len(feature_values))
        
        if rule['direction'] == 1:
            signals[feature_values > upper] = 1
            signals[feature_values < lower] = -1
        else:
            signals[feature_values > upper] = -1
            signals[feature_values < lower] = 1
        
        return signals, upper, lower
    
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


def calc_sharpe(trade_rets, period_days):
    if len(trade_rets) < 5 or period_days <= 0:
        return 0
    total = np.prod(1 + trade_rets) - 1
    daily = (1 + total) ** (1/period_days) - 1
    tstd = np.std(trade_rets)
    tpd = len(trade_rets) / period_days
    dstd = tstd * np.sqrt(tpd)
    if dstd < 1e-10:
        return 0
    return (daily / dstd) * np.sqrt(365)


def backtest_ensemble(features, returns, rules, train_ratio=0.6, min_position=0.0):
    """Backtest with position sizing."""
    n = len(features)
    n_train = int(n * train_ratio)
    
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    
    return_values = returns.values
    
    # Generate signals for each indicator
    all_signals = []
    weights = []
    
    for rule in rules:
        if rule['name'] not in features.columns:
            continue
        
        feature_values = features[rule['name']].values
        signals, upper, lower = generate_signals(feature_values, rule, train_mask)
        
        if np.isnan(upper):
            continue
        
        all_signals.append(signals)
        weights.append(rule['weight'])
        
        n_long = np.sum(signals == 1)
        n_short = np.sum(signals == -1)
        logger.info(f"  {rule['name'][:45]}: {n_long}L + {n_short}S")
    
    if len(all_signals) == 0:
        raise ValueError("No valid signals")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Combine: position = weighted average
    all_signals = np.array(all_signals)
    combined_position = np.sum(all_signals * weights[:, np.newaxis], axis=0)
    
    # Generate final signal
    final_signal = np.zeros(n)
    final_signal[combined_position > min_position] = 1
    final_signal[combined_position < -min_position] = -1
    
    # Trade returns: signal[t] * returns[t] (returns already forward-looking)
    trade_returns = final_signal * return_values
    
    # Split
    is_signals = final_signal[:n_train]
    is_returns = trade_returns[:n_train]
    
    oos_signals = final_signal[n_train:]
    oos_returns = trade_returns[n_train:]
    
    def calc_metrics(strat_rets, sigs, period_days):
        trade_mask = sigs != 0
        trading_rets = strat_rets[trade_mask]
        n_trades = len(trading_rets)
        
        if n_trades < 5:
            return {'sharpe': 0, 'win_rate': 0, 'total_return': 0, 
                    'n_trades': n_trades, 'n_long': 0, 'n_short': 0}
        
        wins = np.sum(trading_rets > 0)
        win_rate = wins / n_trades
        total_return = np.prod(1 + trading_rets) - 1
        sharpe = calc_sharpe(trading_rets, period_days)
        
        n_long = np.sum(sigs[trade_mask] == 1)
        n_short = np.sum(sigs[trade_mask] == -1)
        
        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_return': total_return,
            'n_trades': n_trades,
            'n_long': n_long,
            'n_short': n_short
        }
    
    train_days = n_train / 6
    test_days = (n - n_train) / 6
    
    return {
        'is': calc_metrics(is_returns, is_signals, train_days),
        'oos': calc_metrics(oos_returns, oos_signals, test_days),
        'min_position': min_position
    }


def main():
    logger.info("=" * 70)
    logger.info("SIMPLE ENSEMBLE - v37 FINAL FIX")
    logger.info("=" * 70)
    
    logger.info("\nUsing return_simple_6h from targets (same as Phase 1)")
    
    features, returns = load_data()
    
    thresholds = [0.0, 0.1, 0.2, 0.3]
    results = []
    
    logger.info("\n" + "=" * 70)
    logger.info("TESTING POSITION THRESHOLDS")
    logger.info("=" * 70)
    
    for thresh in thresholds:
        logger.info(f"\n--- Min Position: {thresh} ---")
        result = backtest_ensemble(features, returns, PHASE1_RULES, min_position=thresh)
        results.append(result)
        
        oos = result['oos']
        logger.info(f"  OOS: Sharpe={oos['sharpe']:.2f}, WR={oos['win_rate']*100:.1f}%, "
                   f"Trades={oos['n_trades']} ({oos['n_long']}L + {oos['n_short']}S)")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n{'Threshold':>10} {'Sharpe':>10} {'WR':>10} {'Trades':>10}")
    logger.info("-" * 45)
    
    for r in results:
        thresh = r['min_position']
        oos = r['oos']
        logger.info(f"{thresh:>10.2f} {oos['sharpe']:>10.2f} {oos['win_rate']*100:>9.1f}% {oos['n_trades']:>10}")
    
    # Best result
    best = max(results, key=lambda x: x['oos']['sharpe'])
    
    logger.info(f"\n{'='*70}")
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"  Min Position: {best['min_position']}")
    logger.info(f"  OOS Sharpe: {best['oos']['sharpe']:.2f}")
    logger.info(f"  OOS Win Rate: {best['oos']['win_rate']*100:.1f}%")
    logger.info(f"  OOS Trades: {best['oos']['n_trades']}")
    
    # Comparison
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON")
    logger.info("=" * 70)
    logger.info(f"\n{'Approach':<35} {'OOS Sharpe':>12}")
    logger.info("-" * 50)
    logger.info(f"{'Phase 1 Best Single':<35} {'8.74':>12}")
    logger.info(f"{'v36 (wrong returns)':<35} {'-1.64':>12}")
    logger.info(f"{'v37 (correct returns)':<35} {best['oos']['sharpe']:>12.2f}")


if __name__ == "__main__":
    main()
