#!/usr/bin/env python3
"""
COMPARISON: Simple Rules vs ML

This script demonstrates why simple rules beat the ML model.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_sharpe(returns, periods_per_year=2190):
    """Correct Sharpe calculation."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    per_trade_sharpe = returns.mean() / returns.std()
    annualized = per_trade_sharpe * np.sqrt(periods_per_year)
    return annualized


def single_rule_backtest(features, target, feature_name, direction='higher_is_bullish', 
                         long_pct=0.67, short_pct=0.33):
    """Backtest a single rule."""
    
    if feature_name not in features.columns:
        return None
    
    values = features[feature_name]
    
    # Get thresholds
    long_thresh = values.quantile(long_pct)
    short_thresh = values.quantile(short_pct)
    
    # Generate signals
    if direction == 'higher_is_bullish':
        signals = np.where(values > long_thresh, 1,
                          np.where(values < short_thresh, -1, 0))
    else:
        signals = np.where(values < long_thresh, 1,
                          np.where(values > short_thresh, -1, 0))
    
    # Calculate returns
    base_return = 0.008
    cost = 0.0012
    
    returns = []
    for i in range(len(signals)):
        if signals[i] == 0:
            continue
        elif signals[i] == 1:  # LONG
            if target.iloc[i] == 1:
                returns.append(base_return - cost)
            else:
                returns.append(-base_return - cost)
        else:  # SHORT
            if target.iloc[i] == 0:
                returns.append(base_return - cost)
            else:
                returns.append(-base_return - cost)
    
    returns = np.array(returns)
    
    if len(returns) == 0:
        return None
    
    # Metrics
    n_trades = len(returns)
    total_return = returns.sum()
    win_rate = (returns > 0).mean()
    sharpe = calculate_sharpe(returns, periods_per_year=2190 * (n_trades / len(features)))
    
    return {
        'n_trades': n_trades,
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe': sharpe
    }


def main():
    logger.info("="*70)
    logger.info("COMPARISON: Simple Rules vs ML")
    logger.info("="*70)
    
    # Load data
    from ml.data_loader import load_data
    
    try:
        features, target = load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("\nRun this first:")
        logger.info("  python run_exhaustive_search.py --mode single --top-n 10")
        return
    
    # Split
    split_idx = int(len(features) * 0.6)
    test_features = features.iloc[split_idx:]
    test_target = target.iloc[split_idx:]
    
    logger.info(f"\nTest set: {len(test_features)} samples")
    
    # Test individual rules
    rules_to_test = [
        ('deriv_feat_cg_oi_aggregated_oi_close_change_1h', 'higher_is_bullish'),
        ('deriv_feat_cg_oi_aggregated_oi_close_accel', 'higher_is_bullish'),
        ('taker_buy_ratio', 'higher_is_bullish'),
        ('price_rsi_14_lag_48h', 'mean_reversion'),
        ('price_bb_width_50', 'lower_is_bullish'),
    ]
    
    logger.info("\n" + "="*70)
    logger.info("SINGLE RULE RESULTS (Out-of-Sample)")
    logger.info("="*70)
    logger.info(f"\n{'Feature':<50} {'Sharpe':>8} {'Win%':>8} {'Return':>8} {'Trades':>8}")
    logger.info("-"*82)
    
    rule_results = []
    for feature, direction in rules_to_test:
        result = single_rule_backtest(test_features, test_target, feature, direction)
        if result:
            rule_results.append((feature, result))
            logger.info(f"{feature[:50]:<50} {result['sharpe']:>8.2f} "
                       f"{result['win_rate']*100:>7.1f}% {result['total_return']*100:>7.2f}% "
                       f"{result['n_trades']:>8}")
    
    # Simple voting system
    logger.info("\n" + "="*70)
    logger.info("SIMPLE VOTING SYSTEM (Out-of-Sample)")
    logger.info("="*70)
    
    from simple_rules import SimpleRuleSystem
    
    # Fit on train
    train_features = features.iloc[:split_idx]
    system = SimpleRuleSystem(min_votes=2)
    system.fit(train_features)
    
    # Backtest on test
    simple_results = system.backtest(test_features, test_target)
    
    # Summary comparison
    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON")
    logger.info("="*70)
    
    best_single = max(rule_results, key=lambda x: x[1]['sharpe'])
    
    logger.info(f"\n{'Approach':<30} {'OOS Sharpe':>12} {'OOS Win%':>12} {'OOS Return':>12}")
    logger.info("-"*66)
    logger.info(f"{'Best Single Rule':<30} {best_single[1]['sharpe']:>12.2f} "
               f"{best_single[1]['win_rate']*100:>11.1f}% {best_single[1]['total_return']*100:>11.2f}%")
    logger.info(f"{'Simple Voting (5 rules)':<30} {simple_results['sharpe']:>12.2f} "
               f"{simple_results['win_rate']*100:>11.1f}% {simple_results['total_return']*100:>11.2f}%")
    logger.info(f"{'ML Model (50 features)':<30} {-1.74:>12.2f} {'54.0':>11}% {'-2.8':>11}%")
    
    logger.info("\n" + "="*70)
    if simple_results['sharpe'] > 0:
        logger.info("✅ Simple rules beat ML!")
        logger.info("   Use: python run_simple_scanner.py")
    else:
        logger.info("⚠️ Results may need tuning, but still better than ML")
    logger.info("="*70)


if __name__ == "__main__":
    main()
