#!/usr/bin/env python3
"""
COMPREHENSIVE SHARPE CALCULATION VERIFICATION

This script verifies that the Sharpe calculation is mathematically correct
by computing it by hand and comparing to the pipeline's calculation.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

from validation.framework import StrategyValidator

print("=" * 70)
print("SHARPE CALCULATION VERIFICATION")
print("=" * 70)

# Create test data that mimics the log output:
# Period 0: train signals=35 (20d), test signals=14 (9d)

np.random.seed(42)

# Simulate 20 days of 4h data = 120 bars
n_bars = 120
period_days = 20

# Create signals: 35 trades out of 120 bars
signals = np.zeros(n_bars)
trade_indices = np.random.choice(n_bars, 35, replace=False)
signals[trade_indices] = np.random.choice([-1, 1], 35)

# Create returns with realistic statistics
# mean_ret ~ 0.007, std ~ 0.016 (from typical crypto data)
returns = np.random.normal(0.002, 0.02, n_bars)

# Convert to pandas
signals = pd.Series(signals, name='signal')
returns_series = pd.Series(returns, name='returns')

# Calculate manually
trade_returns = returns[signals != 0] * signals[signals != 0]
n_trades = len(trade_returns)

mean_ret = np.mean(trade_returns)
std_ret = np.std(trade_returns, ddof=1)
per_trade_sharpe = mean_ret / std_ret

trades_per_year = (n_trades / period_days) * 365
expected_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)

print("\n" + "=" * 70)
print("MANUAL CALCULATION")
print("=" * 70)
print(f"  n_bars = {n_bars}")
print(f"  n_trades = {n_trades}")
print(f"  period_days = {period_days}")
print(f"  mean_ret = {mean_ret:.6f}")
print(f"  std_ret = {std_ret:.6f}")
print(f"  per_trade_sharpe = {per_trade_sharpe:.4f}")
print(f"  trades_per_year = ({n_trades} / {period_days}) * 365 = {trades_per_year:.1f}")
print(f"  annualization_factor = sqrt({trades_per_year:.1f}) = {np.sqrt(trades_per_year):.2f}")
print(f"  EXPECTED SHARPE = {per_trade_sharpe:.4f} * {np.sqrt(trades_per_year):.2f} = {expected_sharpe:.2f}")

# Calculate using the pipeline
print("\n" + "=" * 70)
print("PIPELINE CALCULATION")
print("=" * 70)

framework = StrategyValidator()
metrics = framework._calculate_metrics(signals, returns_series, period_days=period_days)

print(f"  Pipeline Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"  Pipeline n_trades: {metrics.n_trades}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"  Expected: {expected_sharpe:.2f}")
print(f"  Pipeline: {metrics.sharpe_ratio:.2f}")
print(f"  Difference: {abs(metrics.sharpe_ratio - expected_sharpe):.4f}")

if abs(metrics.sharpe_ratio - expected_sharpe) < 0.5:
    print(f"\n  ✓ PASS: Pipeline matches manual calculation")
else:
    print(f"\n  ✗ FAIL: Pipeline does NOT match manual calculation")

# Test what the OLD calculation would have produced
old_sharpe = per_trade_sharpe * np.sqrt(2190)
print(f"\n  OLD (WRONG) Sharpe would have been: {old_sharpe:.2f}")
print(f"  Inflation factor: {old_sharpe / expected_sharpe:.1f}x")

# Verify sanity check
print("\n" + "=" * 70)
print("SANITY CHECK VERIFICATION")
print("=" * 70)
if metrics.sharpe_ratio <= 10:
    print(f"  ✓ Sharpe {metrics.sharpe_ratio:.2f} <= 10 (passes sanity check)")
else:
    print(f"  ✗ Sharpe {metrics.sharpe_ratio:.2f} > 10 (FAILS sanity check - BUG!)")

# Benchmark comparison
print("\n" + "=" * 70)
print("BENCHMARK COMPARISON")
print("=" * 70)
print(f"  Renaissance Medallion Fund: ~5-6 Sharpe")
print(f"  Excellent strategy: ~2-3 Sharpe")
print(f"  Good strategy: ~1-2 Sharpe")
print(f"  Our result: {metrics.sharpe_ratio:.2f} Sharpe")

if 0.5 <= metrics.sharpe_ratio <= 8:
    print(f"\n  ✓ Result is in PLAUSIBLE range")
elif metrics.sharpe_ratio > 8:
    print(f"\n  ⚠ Result is HIGH but may be plausible for short periods")
else:
    print(f"\n  ✗ Result is SUSPICIOUSLY low or negative")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
