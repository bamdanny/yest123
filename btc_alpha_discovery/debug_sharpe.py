#!/usr/bin/env python3
"""
Debug script to trace Sharpe calculation step by step.
"""

import numpy as np
import pandas as pd

# Simulate the EXACT scenario from the logs:
# Period 0: train signals=35 (20d), test signals=14 (9d)
# train_sharpe: 20.78

# This would mean:
# - 35 trades in 20 days
# - trades_per_year = (35/20) * 365 = 638
# - To get sharpe of 20.78, we need:
#   per_trade_sharpe = 20.78 / sqrt(638) = 20.78 / 25.27 = 0.82

print("=" * 70)
print("REVERSE ENGINEERING THE LOG OUTPUT")
print("=" * 70)

# From logs: train_sharpe = 20.78, n_trades = 35, period_days = 20
target_sharpe = 20.78
n_trades = 35
period_days = 20

# If using CORRECT formula:
trades_per_year_correct = (n_trades / period_days) * 365
per_trade_sharpe_if_correct = target_sharpe / np.sqrt(trades_per_year_correct)

print(f"\nIf using CORRECT formula (trade frequency):")
print(f"  trades_per_year = ({n_trades}/{period_days}) * 365 = {trades_per_year_correct:.1f}")
print(f"  sqrt(trades_per_year) = {np.sqrt(trades_per_year_correct):.2f}")
print(f"  Implied per_trade_sharpe = {target_sharpe} / {np.sqrt(trades_per_year_correct):.2f} = {per_trade_sharpe_if_correct:.4f}")
print(f"  This would require mean/std ratio of {per_trade_sharpe_if_correct:.4f}")

# If using WRONG formula (sqrt(2190)):
per_trade_sharpe_if_wrong = target_sharpe / np.sqrt(2190)

print(f"\nIf using WRONG formula (bar frequency):")
print(f"  sqrt(2190) = {np.sqrt(2190):.2f}")
print(f"  Implied per_trade_sharpe = {target_sharpe} / {np.sqrt(2190):.2f} = {per_trade_sharpe_if_wrong:.4f}")
print(f"  This would require mean/std ratio of {per_trade_sharpe_if_wrong:.4f}")

print("\n" + "=" * 70)
print("WHICH FORMULA MAKES MORE SENSE?")
print("=" * 70)

# A per-trade Sharpe of 0.82 is VERY HIGH but possible
# A per-trade Sharpe of 0.44 is MORE REASONABLE

print(f"\n  Per-trade Sharpe of {per_trade_sharpe_if_correct:.2f} (if correct formula) is VERY HIGH")
print(f"  Per-trade Sharpe of {per_trade_sharpe_if_wrong:.2f} (if wrong formula) is REASONABLE")

# So the 20.78 Sharpe MUST be coming from the wrong formula!
# Let's verify: if per_trade = 0.44 and we use sqrt(2190):
wrong_result = per_trade_sharpe_if_wrong * np.sqrt(2190)
print(f"\n  Verification: {per_trade_sharpe_if_wrong:.4f} * sqrt(2190) = {wrong_result:.2f}")

correct_result = per_trade_sharpe_if_wrong * np.sqrt(trades_per_year_correct)
print(f"  Correct result: {per_trade_sharpe_if_wrong:.4f} * sqrt({trades_per_year_correct:.0f}) = {correct_result:.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"\n  The output Sharpe of 20.78 implies the code is STILL using sqrt(2190)")
print(f"  OR the period_days value being passed is WRONG")
print(f"\n  To get Sharpe = 20.78 with n_trades=35, period_days would need to be:")
# target_sharpe = per_trade * sqrt((n_trades/period_days)*365)
# 20.78 = 0.44 * sqrt((35/period_days)*365)
# 47.2 = sqrt((35/period_days)*365)
# 2228 = (35/period_days)*365
# period_days = 35*365/2228 = 5.7 days

implied_trades_per_year = (target_sharpe / per_trade_sharpe_if_wrong) ** 2
implied_period_days = (n_trades * 365) / implied_trades_per_year
print(f"  implied_period_days = {implied_period_days:.1f} days")
print(f"  But the logs say period_days = 20")
print(f"\n  THEREFORE: The code is likely still using sqrt(2190) somewhere!")
