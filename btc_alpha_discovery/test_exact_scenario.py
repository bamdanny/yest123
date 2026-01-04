#!/usr/bin/env python3
"""
Direct test of Sharpe calculation matching the log scenario.

From logs:
  Period 0: train signals=35 (20d), test signals=14 (9d)
  train_sharpe: 20.78

We need to verify the calculation produces a PLAUSIBLE number.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

from validation.framework import StrategyValidator

print("=" * 70)
print("DIRECT SHARPE VERIFICATION - MATCHING LOG SCENARIO")
print("=" * 70)

# Create EXACT scenario from logs
# 35 trades in 20 days = 120 bars (4h bars, 6 per day)
n_bars = 120  # 20 days * 6 bars/day
period_days = 20
n_active_trades = 35

# Create signals - 35 active out of 120 bars
np.random.seed(42)
signals = np.zeros(n_bars)
active_indices = np.random.choice(n_bars, n_active_trades, replace=False)
signals[active_indices] = np.random.choice([-1, 1], n_active_trades)

# Create returns with realistic per-trade statistics
# To get the OLD Sharpe of 20.78, the per-trade ratio must be ~0.44
# mean/std = 0.44 -> if mean = 0.007, std = 0.016
returns = np.random.normal(0.002, 0.02, n_bars)

# Make the active trade returns more favorable (to simulate what the strategy found)
for idx in active_indices:
    if signals[idx] > 0:
        returns[idx] = abs(np.random.normal(0.007, 0.015))
    else:
        returns[idx] = -abs(np.random.normal(0.007, 0.015))

signals = pd.Series(signals)
returns_series = pd.Series(returns)

# Manual calculation
trade_mask = signals != 0
trade_returns = (signals * returns_series)[trade_mask]
n_trades = len(trade_returns)

mean_ret = trade_returns.mean()
std_ret = trade_returns.std(ddof=1)

if std_ret > 0:
    per_trade_sharpe = mean_ret / std_ret
else:
    per_trade_sharpe = 0

trades_per_year = (n_trades / period_days) * 365
manual_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)

print(f"\n1. SCENARIO:")
print(f"   n_bars = {n_bars}")
print(f"   n_active_trades = {n_trades}")
print(f"   period_days = {period_days}")

print(f"\n2. TRADE STATISTICS:")
print(f"   mean_return = {mean_ret:.6f}")
print(f"   std_return = {std_ret:.6f}")
print(f"   per_trade_sharpe = {per_trade_sharpe:.4f}")

print(f"\n3. ANNUALIZATION:")
print(f"   trades_per_year = ({n_trades} / {period_days}) * 365 = {trades_per_year:.1f}")
print(f"   sqrt(trades_per_year) = {np.sqrt(trades_per_year):.2f}")

print(f"\n4. MANUAL SHARPE CALCULATION:")
print(f"   sharpe = {per_trade_sharpe:.4f} × {np.sqrt(trades_per_year):.2f} = {manual_sharpe:.2f}")

# Pipeline calculation
print(f"\n5. PIPELINE CALCULATION:")
validator = StrategyValidator()
metrics = validator._calculate_metrics(signals, returns_series, period_days=period_days)
print(f"   pipeline_sharpe = {metrics.sharpe_ratio:.2f}")

# Compare
print(f"\n6. COMPARISON:")
print(f"   Manual: {manual_sharpe:.2f}")
print(f"   Pipeline: {metrics.sharpe_ratio:.2f}")
diff = abs(manual_sharpe - metrics.sharpe_ratio)
print(f"   Difference: {diff:.4f}")

if diff < 1.0:
    print(f"\n   ✓ PASS: Pipeline matches manual calculation (diff < 1.0)")
else:
    print(f"\n   ✗ FAIL: Pipeline does NOT match manual calculation")

# What would OLD formula have given?
old_sharpe = per_trade_sharpe * np.sqrt(2190)
print(f"\n7. OLD FORMULA (WRONG):")
print(f"   old_sharpe = {per_trade_sharpe:.4f} × sqrt(2190) = {per_trade_sharpe:.4f} × 46.80 = {old_sharpe:.2f}")

print(f"\n8. SANITY CHECK:")
if metrics.sharpe_ratio <= 10:
    print(f"   ✓ Sharpe {metrics.sharpe_ratio:.2f} <= 10 (PASSES)")
else:
    print(f"   ✗ Sharpe {metrics.sharpe_ratio:.2f} > 10 (FAILS - BUG!)")

print(f"\n9. BENCHMARK:")
print(f"   Renaissance Medallion: ~5-6")
print(f"   Our result: {metrics.sharpe_ratio:.2f}")

if 1 <= metrics.sharpe_ratio <= 8:
    print(f"\n   ✓ Result is PLAUSIBLE")
elif metrics.sharpe_ratio > 8:
    print(f"\n   ⚠ Result is HIGH - investigate further")
else:
    print(f"\n   Note: Low Sharpe may be due to random data")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
