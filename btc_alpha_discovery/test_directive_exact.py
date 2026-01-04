#!/usr/bin/env python3
"""
Test with EXACT numbers from the directive.

From directive:
  mean_ret = 0.001794
  std_ret = 0.004776
  n_trades = 37
  period_days = 60
  
Expected:
  OLD: 17.58
  NEW: 5.64
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

from validation.framework import StrategyValidator

print("=" * 70)
print("TEST WITH EXACT DIRECTIVE NUMBERS")
print("=" * 70)

# Create scenario matching directive EXACTLY
n_bars = 360  # 60 days * 6 bars/day
period_days = 60
n_trades = 37

# Create signals - 37 active out of 360 bars
np.random.seed(42)
signals = np.zeros(n_bars)
active_indices = np.random.choice(n_bars, n_trades, replace=False)
signals[active_indices] = 1

# Create returns to achieve EXACT statistics from directive
target_mean = 0.001794
target_std = 0.004776

# Create trade returns with exact mean and std
trade_returns_raw = np.random.normal(target_mean, target_std, n_trades)
# Adjust to hit exact mean
trade_returns_raw = trade_returns_raw - trade_returns_raw.mean() + target_mean
# Adjust to hit exact std (approximately)
scale = target_std / trade_returns_raw.std()
trade_returns_raw = (trade_returns_raw - trade_returns_raw.mean()) * scale + target_mean

returns = np.zeros(n_bars)
for i, idx in enumerate(active_indices):
    returns[idx] = trade_returns_raw[i]

signals = pd.Series(signals)
returns_series = pd.Series(returns)

# Manual verification
trade_mask = signals != 0
actual_trade_returns = returns_series[trade_mask]

print(f"\n1. SCENARIO (from directive):")
print(f"   n_bars = {n_bars}")
print(f"   period_days = {period_days}")
print(f"   n_trades = {n_trades}")

print(f"\n2. ACHIEVED STATISTICS:")
print(f"   mean_ret = {actual_trade_returns.mean():.6f} (target: {target_mean})")
print(f"   std_ret = {actual_trade_returns.std(ddof=1):.6f} (target: {target_std})")

per_trade = actual_trade_returns.mean() / actual_trade_returns.std(ddof=1)
trades_per_year = (n_trades / period_days) * 365

print(f"   per_trade_sharpe = {per_trade:.4f}")
print(f"   trades_per_year = {trades_per_year:.1f}")

print(f"\n3. EXPECTED SHARPE:")
expected_old = per_trade * np.sqrt(2190)
expected_new = per_trade * np.sqrt(trades_per_year)
print(f"   OLD (wrong): {expected_old:.2f}")
print(f"   NEW (correct): {expected_new:.2f}")

# Run through pipeline - but note pipeline applies costs!
print(f"\n4. PIPELINE CALCULATION:")
validator = StrategyValidator()

# Create a custom _calculate_metrics call that matches our scenario
# We need to account for the fact that pipeline applies TOTAL_COST
metrics = validator._calculate_metrics(signals, returns_series, period_days=period_days)
print(f"   Pipeline Sharpe (with costs): {metrics.sharpe_ratio:.2f}")

# Calculate what pipeline SHOULD get
# Pipeline: strategy_returns = signals * returns - TOTAL_COST * (signals != 0)
# TOTAL_COST = 0.0012 (0.12%)
TOTAL_COST = 0.0012
strategy_returns = signals * returns_series - TOTAL_COST * (signals != 0).astype(float)
adjusted_trade_returns = strategy_returns[signals != 0]

adj_mean = adjusted_trade_returns.mean()
adj_std = adjusted_trade_returns.std(ddof=1)
adj_per_trade = adj_mean / adj_std
adj_sharpe = adj_per_trade * np.sqrt(trades_per_year)

print(f"\n5. PIPELINE CALCULATION (traced):")
print(f"   After costs: mean={adj_mean:.6f}, std={adj_std:.6f}")
print(f"   adj_per_trade = {adj_per_trade:.4f}")
print(f"   adj_sharpe = {adj_sharpe:.2f}")

print(f"\n6. COMPARISON:")
print(f"   Expected (no costs): {expected_new:.2f}")
print(f"   Expected (with costs): {adj_sharpe:.2f}")
print(f"   Pipeline output: {metrics.sharpe_ratio:.2f}")

if abs(metrics.sharpe_ratio - adj_sharpe) < 0.5:
    print(f"\n   ✓ PASS: Pipeline matches expected calculation")
else:
    print(f"\n   ✗ FAIL: Mismatch detected")

print(f"\n7. KEY RESULT:")
print(f"   OLD formula would give: {expected_old:.2f}")
print(f"   NEW formula gives: {metrics.sharpe_ratio:.2f}")
print(f"   Reduction: {expected_old/metrics.sharpe_ratio:.1f}x")

if metrics.sharpe_ratio < 10:
    print(f"\n   ✓ Sharpe {metrics.sharpe_ratio:.2f} passes sanity check (< 10)")
else:
    print(f"\n   ⚠ Sharpe {metrics.sharpe_ratio:.2f} triggers sanity warning")

print("\n" + "=" * 70)
