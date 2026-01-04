#!/usr/bin/env python3
"""
Test with REALISTIC per-trade statistics.

From directive analysis:
  OLD Sharpe of 20.78 came from per_trade_sharpe of 0.44 × sqrt(2190)
  CORRECT Sharpe should be 0.44 × sqrt(trades_per_year) = 0.44 × sqrt(638) = 11.1

If per_trade_sharpe = 0.44, we need mean/std = 0.44
Example: mean = 0.0072, std = 0.0164
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

from validation.framework import StrategyValidator

print("=" * 70)
print("REALISTIC SHARPE CALCULATION TEST")
print("=" * 70)

# Create scenario with REALISTIC per_trade_sharpe of ~0.44
n_bars = 120  # 20 days * 6 bars/day
period_days = 20
n_active_trades = 35

# Create signals
np.random.seed(123)
signals = np.zeros(n_bars)
active_indices = np.random.choice(n_bars, n_active_trades, replace=False)
signals[active_indices] = 1  # All long for simplicity

# Create returns with per_trade_sharpe ~0.44
# mean = 0.0072, std = 0.0164 -> ratio = 0.44
target_mean = 0.0072
target_std = 0.0164
returns = np.random.normal(0, 0.01, n_bars)  # Background noise

# Set active trade returns to achieve target statistics
trade_returns_raw = np.random.normal(target_mean, target_std, n_active_trades)
for i, idx in enumerate(active_indices):
    returns[idx] = trade_returns_raw[i]

signals = pd.Series(signals)
returns_series = pd.Series(returns)

# Manual calculation
trade_mask = signals != 0
trade_returns = (signals * returns_series)[trade_mask]
n_trades = len(trade_returns)

mean_ret = trade_returns.mean()
std_ret = trade_returns.std(ddof=1)
per_trade_sharpe = mean_ret / std_ret if std_ret > 0 else 0

trades_per_year = (n_trades / period_days) * 365
manual_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)

print(f"\n1. SCENARIO (matching directive):")
print(f"   n_trades = {n_trades}")
print(f"   period_days = {period_days}")

print(f"\n2. TRADE STATISTICS:")
print(f"   mean_return = {mean_ret:.6f} (target: {target_mean})")
print(f"   std_return = {std_ret:.6f} (target: {target_std})")
print(f"   per_trade_sharpe = {per_trade_sharpe:.4f} (target: ~0.44)")

print(f"\n3. ANNUALIZATION (CORRECT formula):")
print(f"   trades_per_year = ({n_trades} / {period_days}) × 365 = {trades_per_year:.1f}")
print(f"   sqrt(trades_per_year) = {np.sqrt(trades_per_year):.2f}")

print(f"\n4. EXPECTED SHARPE:")
print(f"   CORRECT: {per_trade_sharpe:.4f} × {np.sqrt(trades_per_year):.2f} = {manual_sharpe:.2f}")

# What the OLD formula would have given
old_sharpe = per_trade_sharpe * np.sqrt(2190)
print(f"   OLD (WRONG): {per_trade_sharpe:.4f} × 46.80 = {old_sharpe:.2f}")

# Pipeline calculation
print(f"\n5. PIPELINE CALCULATION:")
validator = StrategyValidator()
metrics = validator._calculate_metrics(signals, returns_series, period_days=period_days)
print(f"   Pipeline Sharpe: {metrics.sharpe_ratio:.2f}")

# Verification
print(f"\n6. VERIFICATION:")
print(f"   Expected (manual): {manual_sharpe:.2f}")
print(f"   Pipeline output: {metrics.sharpe_ratio:.2f}")

diff = abs(manual_sharpe - metrics.sharpe_ratio)
# Account for cost adjustment in pipeline
if diff < 2.0:
    print(f"   ✓ PASS: Pipeline calculation is correct (diff={diff:.2f})")
else:
    print(f"   ✗ FAIL: Unexpected difference (diff={diff:.2f})")

# Sanity check
print(f"\n7. SANITY CHECK:")
if metrics.sharpe_ratio <= 10:
    print(f"   ✓ PASS: Sharpe {metrics.sharpe_ratio:.2f} <= 10")
else:
    print(f"   ✗ FAIL: Sharpe {metrics.sharpe_ratio:.2f} > 10")

# Benchmark
print(f"\n8. CONTEXT:")
print(f"   Renaissance Medallion: ~5-6")
print(f"   Excellent strategy: ~2-3")
print(f"   Our result: {metrics.sharpe_ratio:.2f}")

if 2 <= metrics.sharpe_ratio <= 15:
    print(f"\n   Result is in REASONABLE range for a short 20-day period")
else:
    print(f"\n   Result may need investigation")

# Final summary
print(f"\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Per-trade Sharpe: {per_trade_sharpe:.4f}")
print(f"  Trades per year: {trades_per_year:.0f}")
print(f"  ")
print(f"  OLD formula (WRONG): {old_sharpe:.2f}")
print(f"  NEW formula (CORRECT): {metrics.sharpe_ratio:.2f}")
print(f"  Reduction factor: {old_sharpe/metrics.sharpe_ratio:.2f}x")
print("=" * 70)
