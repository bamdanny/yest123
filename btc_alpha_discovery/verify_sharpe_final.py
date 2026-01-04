#!/usr/bin/env python3
"""
SHARPE CALCULATION VERIFICATION - FINAL CHECK

This script verifies the Sharpe calculation is mathematically correct.
Run this before deploying any strategy.
"""
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

import numpy as np
import pandas as pd
from data.synthetic import generate_test_data
from validation.framework import StrategyValidator

print("=" * 70)
print("SHARPE CALCULATION VERIFICATION")
print("=" * 70)

# Generate test data
data_dict = generate_test_data(n_days=180, seed=42)
data = data_dict['price']

# Create simple momentum strategy
returns = data['close'].pct_change().fillna(0)
momentum = returns.rolling(6).mean()
signals = pd.Series(0, index=data.index)
signals[momentum > 0] = 1
signals[momentum < 0] = -1

print(f"Data: {len(signals)} bars, {(signals != 0).sum()} active signals")

# Run validation
validator = StrategyValidator()
validator.walk_forward_periods = 5
report = validator.validate(signals, returns, data)

# Manual verification of Period 0
wf0 = report.walk_forward_results[0]
oos = wf0.out_sample_metrics

print("\n" + "=" * 70)
print("PERIOD 0 VERIFICATION")
print("=" * 70)

# Get the actual trade returns for Period 0
n = len(signals)
period_size = n // 5
train_size = int(period_size * 0.7)
test_start = train_size
test_end = period_size

test_signals = signals.iloc[test_start:test_end]
test_returns = returns.iloc[test_start:test_end]

# Calculate trade returns (same as _calculate_metrics)
TOTAL_COST = (0.04/100 + 0.02/100) * 2  # Round trip
strategy_returns = test_signals * test_returns - TOTAL_COST * (test_signals != 0).astype(float)
trades = strategy_returns[test_signals != 0]
n_trades = len(trades)

# Calculate period_days
period_days = len(test_signals) / 6  # 4h bars = 6 per day

# Manual calculation
mean_ret = np.mean(trades)
std_ret = np.std(trades, ddof=1)
per_trade_sharpe = mean_ret / std_ret
trades_per_year = (n_trades / period_days) * 365
manual_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)

print(f"n_trades = {n_trades}")
print(f"period_days = {period_days:.1f}")
print(f"mean_ret = {mean_ret:.6f}")
print(f"std_ret = {std_ret:.6f}")
print(f"per_trade_sharpe = {mean_ret:.6f} / {std_ret:.6f} = {per_trade_sharpe:.4f}")
print(f"trades_per_year = ({n_trades}/{period_days:.1f}) × 365 = {trades_per_year:.1f}")
print(f"annualized = {per_trade_sharpe:.4f} × √{trades_per_year:.1f} = {per_trade_sharpe:.4f} × {np.sqrt(trades_per_year):.2f} = {manual_sharpe:.2f}")

print(f"\nPipeline output: {oos.sharpe_ratio:.2f}")
print(f"Manual calculation: {manual_sharpe:.2f}")

diff = abs(manual_sharpe - oos.sharpe_ratio)
if diff < 0.1:
    print(f"\n✓ PASS: Difference = {diff:.4f} (< 0.1)")
else:
    print(f"\n✗ FAIL: Difference = {diff:.4f} (>= 0.1)")

# Check sanity
print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)

all_sharpes = [r.out_sample_metrics.sharpe_ratio for r in report.walk_forward_results]
print(f"Walk-forward Sharpes: {[f'{s:.2f}' for s in all_sharpes]}")

if all(abs(s) < 10 for s in all_sharpes):
    print("✓ PASS: All Sharpes < 10 (plausible)")
else:
    print("✗ FAIL: Some Sharpes >= 10 (implausible)")

print(f"\nCombined OOS Sharpe: {report.combined_oos_metrics.sharpe_ratio:.2f}")

# Industry comparison
print("\n" + "=" * 70)
print("INDUSTRY COMPARISON")
print("=" * 70)
print("Renaissance Medallion: ~5-6 Sharpe")
print("Excellent strategy: ~2-3 Sharpe")
print("Good strategy: ~1-2 Sharpe")
print(f"Our result: {report.combined_oos_metrics.sharpe_ratio:.2f} Sharpe")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
