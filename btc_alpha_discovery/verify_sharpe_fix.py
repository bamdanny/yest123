#!/usr/bin/env python3
"""
Verification script for Sharpe calculation fix.
Tests that the corrected formula produces plausible results.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

from utils.metrics import calculate_sharpe_ratio, calculate_sharpe_simple

print("=" * 70)
print("SHARPE CALCULATION VERIFICATION")
print("=" * 70)

# Test case from the logs:
# mean_ret = 0.001794 (0.18% per trade)
# std_ret = 0.004776 (0.48% std per trade)
# n_trades = 37
# period = ~60 days

mean_ret = 0.001794
std_ret = 0.004776
n_trades = 37
period_days = 60

# Create sample returns matching these statistics
np.random.seed(42)
returns = np.random.normal(mean_ret, std_ret, n_trades)

print("\n1. MANUAL CALCULATION (expected):")
print("-" * 40)
per_trade_sharpe = mean_ret / std_ret
trades_per_year = (n_trades / period_days) * 365
expected_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
print(f"   mean_ret = {mean_ret:.6f}")
print(f"   std_ret = {std_ret:.6f}")
print(f"   per_trade_sharpe = {per_trade_sharpe:.4f}")
print(f"   n_trades = {n_trades}")
print(f"   period_days = {period_days}")
print(f"   trades_per_year = {trades_per_year:.1f}")
print(f"   annualization_factor = sqrt({trades_per_year:.1f}) = {np.sqrt(trades_per_year):.2f}")
print(f"   EXPECTED SHARPE = {expected_sharpe:.2f}")

print("\n2. OLD CALCULATION (WRONG - what we're fixing):")
print("-" * 40)
old_sharpe = (mean_ret / std_ret) * np.sqrt(2190)
print(f"   old_sharpe = per_trade * sqrt(2190)")
print(f"   old_sharpe = {per_trade_sharpe:.4f} * {np.sqrt(2190):.2f}")
print(f"   OLD SHARPE = {old_sharpe:.2f} <-- WRONG!")

print("\n3. NEW CALCULATION (from utils.metrics):")
print("-" * 40)
new_sharpe = calculate_sharpe_ratio(
    returns=returns,
    n_trades=n_trades,
    period_days=period_days,
    context="verification test"
)
print(f"   NEW SHARPE = {new_sharpe:.2f}")

print("\n4. VERIFICATION:")
print("-" * 40)
print(f"   Expected: ~{expected_sharpe:.2f}")
print(f"   Got: {new_sharpe:.2f}")
print(f"   Difference: {abs(new_sharpe - expected_sharpe):.2f}")

# Check if new value is in plausible range
if 0.5 < new_sharpe < 10:
    print(f"\n   ✓ PASS: Sharpe {new_sharpe:.2f} is in plausible range (0.5-10)")
else:
    print(f"\n   ✗ FAIL: Sharpe {new_sharpe:.2f} is NOT in plausible range")

# Check old was wrong
if old_sharpe > 10:
    print(f"   ✓ CONFIRMED: Old calculation ({old_sharpe:.2f}) was inflated")
else:
    print(f"   ? Old calculation was: {old_sharpe:.2f}")

print("\n5. COMPARE TO BENCHMARKS:")
print("-" * 40)
print(f"   Renaissance Medallion: ~5-6 Sharpe")
print(f"   Excellent strategy: ~2-3 Sharpe")
print(f"   Good strategy: ~1-2 Sharpe")
print(f"   Our corrected result: {new_sharpe:.2f} Sharpe")

if new_sharpe > 5:
    print(f"\n   Note: Sharpe > 5 is still very high but now PLAUSIBLE")
elif new_sharpe > 3:
    print(f"\n   Note: Sharpe in excellent range")
elif new_sharpe > 1:
    print(f"\n   Note: Sharpe in good range")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
