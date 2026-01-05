#!/usr/bin/env python3
"""
DIAGNOSTIC: Test Signal Inversion Hypothesis

Evidence:
- Same trades (49)
- Phase 1 Sharpe: +8.77, WR: 73.5%
- Replica Sharpe: -5.28, WR: 36.7% (inverted!)

If we flip all signals and get ~+8.77 Sharpe, bug confirmed.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("SIGNAL INVERSION DIAGNOSTIC")
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

# =============================================================
# CURRENT SIGNAL GENERATION (v36)
# =============================================================
print("\n" + "=" * 70)
print("CURRENT SIGNAL GENERATION (v36)")
print("=" * 70)

# Compute thresholds on train
pct_upper = 90
pct_lower = 10
upper_thresh = np.nanpercentile(feature_values[train_mask], pct_upper)
lower_thresh = np.nanpercentile(feature_values[train_mask], pct_lower)

print(f"\nThresholds (computed on train):")
print(f"  Upper (90th pct): {upper_thresh:.6f}")
print(f"  Lower (10th pct): {lower_thresh:.6f}")

# Generate signals (direction=1)
signals = np.zeros(n)
signals[feature_values > upper_thresh] = 1   # LONG when HIGH
signals[feature_values < lower_thresh] = -1  # SHORT when LOW

# OOS signals
test_signals = signals[n_train:-1]

n_long = np.sum(test_signals == 1)
n_short = np.sum(test_signals == -1)
print(f"\nOOS Signal distribution:")
print(f"  LONG (+1):  {n_long}")
print(f"  SHORT (-1): {n_short}")
print(f"  Total trades: {n_long + n_short}")

# =============================================================
# COMPUTE RETURNS
# =============================================================
print("\n" + "=" * 70)
print("RETURNS COMPUTATION")
print("=" * 70)

# Forward returns
returns = np.zeros(n)
returns[:-1] = (close[1:] - close[:-1]) / close[:-1]

# OOS returns
test_returns = returns[n_train:-1]

print(f"\nOOS period: bars {n_train} to {n-1}")
print(f"OOS returns: {len(test_returns)} bars")

# =============================================================
# CALCULATE SHARPE (ORIGINAL)
# =============================================================
def calc_sharpe(signals, returns, period_days):
    trade_mask = signals != 0
    trade_rets = signals[trade_mask] * returns[trade_mask]
    n_trades = len(trade_rets)
    
    if n_trades < 5:
        return 0, 0, trade_rets
    
    total = np.prod(1 + trade_rets) - 1
    daily = (1 + total) ** (1/period_days) - 1
    tstd = np.std(trade_rets)
    tpd = n_trades / period_days
    dstd = tstd * np.sqrt(tpd)
    
    wins = np.sum(trade_rets > 0)
    win_rate = wins / n_trades
    
    if dstd < 1e-10:
        return 0, win_rate, trade_rets
    
    sharpe = (daily / dstd) * np.sqrt(365)
    return sharpe, win_rate, trade_rets

period_days = (n - n_train - 1) / 6

print("\n" + "=" * 70)
print("ORIGINAL SIGNALS")
print("=" * 70)

sharpe_orig, wr_orig, trades_orig = calc_sharpe(test_signals, test_returns, period_days)
print(f"\nOriginal (v36):")
print(f"  Sharpe:   {sharpe_orig:.2f}")
print(f"  Win Rate: {wr_orig*100:.1f}%")
print(f"  Trades:   {len(trades_orig)}")

# =============================================================
# FLIP SIGNALS AND RECALCULATE
# =============================================================
print("\n" + "=" * 70)
print("FLIPPED SIGNALS")
print("=" * 70)

flipped_signals = -1 * test_signals

sharpe_flip, wr_flip, trades_flip = calc_sharpe(flipped_signals, test_returns, period_days)
print(f"\nFlipped (signals * -1):")
print(f"  Sharpe:   {sharpe_flip:.2f}")
print(f"  Win Rate: {wr_flip*100:.1f}%")
print(f"  Trades:   {len(trades_flip)}")

# =============================================================
# COMPARISON
# =============================================================
print("\n" + "=" * 70)
print("COMPARISON TO PHASE 1")
print("=" * 70)

phase1_sharpe = 8.74
phase1_wr = 0.735

print(f"""
                    Sharpe    Win Rate
---------------------------------------
Phase 1 (target):   {phase1_sharpe:>6.2f}    {phase1_wr*100:>5.1f}%
Original v36:       {sharpe_orig:>6.2f}    {wr_orig*100:>5.1f}%
Flipped v36:        {sharpe_flip:>6.2f}    {wr_flip*100:>5.1f}%
""")

# =============================================================
# DIAGNOSIS
# =============================================================
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

diff_orig = abs(sharpe_orig - phase1_sharpe)
diff_flip = abs(sharpe_flip - phase1_sharpe)

if diff_flip < diff_orig and diff_flip < 2.0:
    print(f"""
✅ CONFIRMED: Signals are INVERTED!

Flipped Sharpe ({sharpe_flip:.2f}) is much closer to Phase 1 ({phase1_sharpe:.2f}) 
than Original ({sharpe_orig:.2f}).

FIX: The signal generation has opposite signs. Find where direction=1 
assigns +1 to HIGH values and change it to assign -1.

Current (wrong):
    signals[feature > upper] = +1  # Should be -1 for direction=1
    signals[feature < lower] = -1  # Should be +1 for direction=1

Phase 1 (correct):
    signals[feature > upper] = +1  
    signals[feature < lower] = -1
    
Wait - that looks the same! The bug must be elsewhere...

Let me check the threshold computation...
""")
    
    # Check threshold order
    print("\nThreshold analysis:")
    print(f"  Upper: {upper_thresh:.6f}")
    print(f"  Lower: {lower_thresh:.6f}")
    
    if upper_thresh < lower_thresh:
        print("  ❌ BUG: upper < lower! Thresholds are swapped!")
    else:
        print("  ✓ upper > lower (correct)")
    
    # Check feature distribution
    print("\nFeature distribution:")
    print(f"  Min: {np.nanmin(feature_values):.6f}")
    print(f"  Max: {np.nanmax(feature_values):.6f}")
    print(f"  Mean: {np.nanmean(feature_values):.6f}")
    
    pct_positive = np.sum(feature_values > 0) / len(feature_values)
    print(f"  % positive: {pct_positive*100:.1f}%")
    
elif diff_orig < diff_flip:
    print(f"""
❌ Original is closer to Phase 1 - no simple inversion.
   There's a different bug.
""")
else:
    print(f"""
⚠️ Neither matches Phase 1 well. 
   Diff original: {diff_orig:.2f}
   Diff flipped:  {diff_flip:.2f}
   There may be multiple bugs.
""")

# =============================================================
# CHECK PHASE 1 METHODOLOGY
# =============================================================
print("\n" + "=" * 70)
print("CHECKING PHASE 1 METHODOLOGY")
print("=" * 70)

print("""
Phase 1 uses:
1. SignalGenerator.get_percentile_thresholds() - returns (upper, lower)
2. SignalGenerator.apply_fixed_threshold() - applies thresholds

Let me check if the issue is in how Phase 1 calls these functions...

From run_exhaustive_search.py line 700-707:
    if threshold_type == 'percentile':
        upper_thresh, lower_thresh = SignalGenerator.get_percentile_thresholds(
            is_values, kwargs['pct_long'], kwargs['pct_short']
        )
        is_signals = SignalGenerator.percentile_signal(
            is_values, kwargs['pct_long'], kwargs['pct_short'], direction
        )

Phase 1 passes (pct_long, pct_short) = (90, 10) for a percentile=90 rule.

From SignalGenerator.get_percentile_thresholds():
    upper_threshold = np.nanpercentile(values, pct_upper)  # 90th percentile
    lower_threshold = np.nanpercentile(values, pct_lower)  # 10th percentile

This is the same as what we do!

But wait - let me check percentile_signal() more carefully...
""")

# =============================================================
# TEST WITH ROLLING THRESHOLDS (like Phase 1 percentile_signal)
# =============================================================
print("\n" + "=" * 70)
print("TESTING ROLLING THRESHOLDS")
print("=" * 70)

print("""
Phase 1's percentile_signal() uses ROLLING thresholds:

    if len(values) < lookback:
        upper = values.expanding(min_periods=10).quantile(percentile_long / 100)
        lower = values.expanding(min_periods=10).quantile(percentile_short / 100)
    else:
        upper = values.rolling(lookback, min_periods=20).quantile(percentile_long / 100)
        lower = values.rolling(lookback, min_periods=20).quantile(percentile_short / 100)

This means the threshold CHANGES at each bar based on recent history!
But our fixed threshold is computed once on all train data.

However, for OOS, Phase 1 uses apply_fixed_threshold() which uses 
the FIXED threshold from IS. So they should be the same...

Unless the issue is that Phase 1 uses return_simple_6h (forward looking)
while we compute returns differently.
""")

# Let me check if returns direction matters
print("\n" + "=" * 70)
print("TESTING RETURNS DIRECTION")
print("=" * 70)

# Maybe returns should be backward-looking?
backward_returns = np.zeros(n)
backward_returns[1:] = (close[1:] - close[:-1]) / close[:-1]

test_backward_returns = backward_returns[n_train:-1]

sharpe_backward, wr_backward, _ = calc_sharpe(test_signals, test_backward_returns, period_days)
print(f"\nWith backward returns:")
print(f"  Sharpe:   {sharpe_backward:.2f}")
print(f"  Win Rate: {wr_backward*100:.1f}%")

sharpe_flip_back, wr_flip_back, _ = calc_sharpe(flipped_signals, test_backward_returns, period_days)
print(f"\nWith backward returns + flipped signals:")
print(f"  Sharpe:   {sharpe_flip_back:.2f}")
print(f"  Win Rate: {wr_flip_back*100:.1f}%")

print("\n" + "=" * 70)
print("SUMMARY OF ALL COMBINATIONS")
print("=" * 70)
print(f"""
                          Sharpe    Win Rate   Match Phase 1?
-------------------------------------------------------------
Phase 1 target:           {phase1_sharpe:>6.2f}    {phase1_wr*100:>5.1f}%    (target)
Original + forward ret:   {sharpe_orig:>6.2f}    {wr_orig*100:>5.1f}%    {abs(sharpe_orig - phase1_sharpe) < 2}
Flipped + forward ret:    {sharpe_flip:>6.2f}    {wr_flip*100:>5.1f}%    {abs(sharpe_flip - phase1_sharpe) < 2}
Original + backward ret:  {sharpe_backward:>6.2f}    {wr_backward*100:>5.1f}%    {abs(sharpe_backward - phase1_sharpe) < 2}
Flipped + backward ret:   {sharpe_flip_back:>6.2f}    {wr_flip_back*100:>5.1f}%    {abs(sharpe_flip_back - phase1_sharpe) < 2}
""")

# Find the best match
results = [
    ("Original + forward", sharpe_orig, wr_orig),
    ("Flipped + forward", sharpe_flip, wr_flip),
    ("Original + backward", sharpe_backward, wr_backward),
    ("Flipped + backward", sharpe_flip_back, wr_flip_back),
]

best = min(results, key=lambda x: abs(x[1] - phase1_sharpe))
print(f"BEST MATCH: {best[0]} with Sharpe={best[1]:.2f}")

if abs(best[1] - phase1_sharpe) < 2:
    print(f"\n✅ FIX IDENTIFIED: Use {best[0]}")
else:
    print(f"\n❌ No combination matches Phase 1 well. Need to investigate further.")
