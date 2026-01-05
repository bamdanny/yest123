#!/usr/bin/env python3
"""
DIAGNOSTIC: Confirm the returns alignment bug

Hypothesis: The backtest uses returns[1:] instead of returns[:-1],
causing a one-bar shift that inverts the sign.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("RETURNS ALIGNMENT DIAGNOSTIC")
print("=" * 70)

# Load data
cache_path = Path("data_cache/features_cache.pkl")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)
features = cache['features']

price_path = Path("data_cache/binance/price_4h_365d.parquet")
prices = pd.read_parquet(price_path)
if 'timestamp' in prices.columns:
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    prices = prices.set_index('timestamp')

# Align
n_features = len(features)
prices_aligned = prices.iloc[-n_features:]
close = prices_aligned['close'].values

# Get indicator
indicator = "deriv_feat_cg_oi_aggregated_oi_close_change_1h"
feature_values = features[indicator].values

# Split 60/40
n = len(feature_values)
n_train = int(n * 0.6)

train_mask = np.zeros(n, dtype=bool)
train_mask[:n_train] = True

# Compute thresholds
pct_upper = 90
pct_lower = 10
upper_thresh = np.nanpercentile(feature_values[train_mask], pct_upper)
lower_thresh = np.nanpercentile(feature_values[train_mask], pct_lower)

print(f"\nIndicator: {indicator}")
print(f"Upper threshold (90th): {upper_thresh:.6f}")
print(f"Lower threshold (10th): {lower_thresh:.6f}")

# Generate signals
signals = np.zeros(n)
signals[feature_values > upper_thresh] = 1
signals[feature_values < lower_thresh] = -1

# Compute returns two ways
returns = np.zeros(n)
returns[:-1] = (close[1:] - close[:-1]) / close[:-1]

# OOS only
test_signals = signals[n_train:-1]  # Exclude last bar (no return)
test_returns_correct = returns[n_train:-1]  # Same index as signals
test_returns_wrong = returns[n_train+1:]   # Shifted by 1

print(f"\nOOS samples: {len(test_signals)}")
print(f"OOS signals: {np.sum(test_signals == 1)} long, {np.sum(test_signals == -1)} short")

# Calculate strategy returns both ways
strategy_correct = test_signals * test_returns_correct  # Signal at t * return from t to t+1
strategy_wrong = test_signals * test_returns_wrong      # Signal at t * return from t+1 to t+2

# Filter to trades only
trade_mask = test_signals != 0
correct_trades = strategy_correct[trade_mask]
wrong_trades = strategy_wrong[trade_mask]

print(f"\n{'='*70}")
print("TRADE RETURNS COMPARISON")
print("=" * 70)

print(f"\nNumber of trades: {len(correct_trades)}")

# Calculate Sharpe for both
def calc_sharpe(trade_rets, period_days):
    if len(trade_rets) < 5:
        return 0
    total_ret = np.prod(1 + trade_rets) - 1
    daily_ret = (1 + total_ret) ** (1/period_days) - 1
    trade_std = np.std(trade_rets)
    tpd = len(trade_rets) / period_days
    daily_std = trade_std * np.sqrt(tpd)
    if daily_std < 1e-10:
        return 0
    return (daily_ret / daily_std) * np.sqrt(365)

period_days = (n - n_train) / 6

sharpe_correct = calc_sharpe(correct_trades, period_days)
sharpe_wrong = calc_sharpe(wrong_trades, period_days)

win_rate_correct = np.sum(correct_trades > 0) / len(correct_trades)
win_rate_wrong = np.sum(wrong_trades > 0) / len(wrong_trades)

print(f"\nCORRECT alignment (signal[t] * return[t]):")
print(f"  Sharpe: {sharpe_correct:.2f}")
print(f"  Win Rate: {win_rate_correct*100:.1f}%")
print(f"  Total Return: {(np.prod(1+correct_trades)-1)*100:.1f}%")

print(f"\nWRONG alignment (signal[t] * return[t+1]) - what v35 was doing:")
print(f"  Sharpe: {sharpe_wrong:.2f}")
print(f"  Win Rate: {win_rate_wrong*100:.1f}%")
print(f"  Total Return: {(np.prod(1+wrong_trades)-1)*100:.1f}%")

print(f"\nPhase 1 expected:")
print(f"  Sharpe: 8.74")
print(f"  Win Rate: 73.5%")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if abs(sharpe_correct - 8.74) < abs(sharpe_wrong - 8.74):
    print("\n✅ CORRECT alignment matches Phase 1 better!")
    print("   BUG: v35 was using returns[1:] instead of returns[:-1]")
else:
    print("\n❌ Neither alignment matches - there's another bug")

# Also test the FLIPPED signals
print("\n" + "=" * 70)
print("FLIPPED SIGNALS TEST")
print("=" * 70)

flipped_signals = -test_signals
flipped_trades_correct = flipped_signals[trade_mask] * test_returns_correct[trade_mask]
flipped_trades_wrong = flipped_signals[trade_mask] * test_returns_wrong[trade_mask]

sharpe_flipped_correct = calc_sharpe(flipped_trades_correct, period_days)
sharpe_flipped_wrong = calc_sharpe(flipped_trades_wrong, period_days)

print(f"\nFlipped + correct alignment: Sharpe = {sharpe_flipped_correct:.2f}")
print(f"Flipped + wrong alignment: Sharpe = {sharpe_flipped_wrong:.2f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
                        SHARPE      Phase 1 = 8.74
----------------------------------------------------
Original + Correct:     {sharpe_correct:>6.2f}
Original + Wrong (v35): {sharpe_wrong:>6.2f}
Flipped + Correct:      {sharpe_flipped_correct:>6.2f}
Flipped + Wrong:        {sharpe_flipped_wrong:>6.2f}

The one closest to 8.74 shows the fix needed.
""")
