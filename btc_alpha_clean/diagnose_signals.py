#!/usr/bin/env python3
"""
DIAGNOSTIC: Compare Phase 1 vs Replica Signal Generation

This proves the bug: replica is missing SHORT signals.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("SIGNAL GENERATION COMPARISON DIAGNOSTIC")
print("=" * 70)

# Load data
cache_path = Path("data_cache/features_cache.pkl")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)
features = cache['features']

# Get the indicator
indicator = "deriv_feat_cg_oi_aggregated_oi_close_change_1h"
values = features[indicator].values

# Train/test split (60/40)
n = len(values)
n_train = int(n * 0.6)

train_values = values[:n_train]
test_values = values[n_train:]

print(f"\nIndicator: {indicator}")
print(f"Total samples: {n}")
print(f"Train: {n_train}, Test: {n - n_train}")

# =====================================================
# PHASE 1 SIGNAL GENERATION (TWO thresholds)
# =====================================================
print("\n" + "=" * 70)
print("PHASE 1 SIGNAL GENERATION (TWO thresholds)")
print("=" * 70)

# Phase 1 uses percentile 90 for direction=1
# Upper threshold: 90th percentile (for LONG)
# Lower threshold: 10th percentile (for SHORT)
pct_upper = 90
pct_lower = 100 - pct_upper  # = 10

upper_threshold = np.nanpercentile(train_values, pct_upper)
lower_threshold = np.nanpercentile(train_values, pct_lower)

print(f"\nThresholds (computed on train):")
print(f"  Upper (90th percentile): {upper_threshold:.6f}")
print(f"  Lower (10th percentile): {lower_threshold:.6f}")

# Generate Phase 1 signals (direction=1)
phase1_signals = np.zeros(len(test_values))
phase1_signals[test_values > upper_threshold] = 1   # LONG
phase1_signals[test_values < lower_threshold] = -1  # SHORT

n_long = np.sum(phase1_signals == 1)
n_short = np.sum(phase1_signals == -1)
n_flat = np.sum(phase1_signals == 0)
n_trades = n_long + n_short

print(f"\nPhase 1 signal distribution (OOS):")
print(f"  +1 (LONG):  {n_long}")
print(f"  -1 (SHORT): {n_short}")
print(f"   0 (FLAT):  {n_flat}")
print(f"  TOTAL TRADES: {n_trades}")

# =====================================================
# REPLICA SIGNAL GENERATION (ONE threshold - BUGGY)
# =====================================================
print("\n" + "=" * 70)
print("REPLICA SIGNAL GENERATION (ONE threshold - BUGGY)")
print("=" * 70)

# The replica only used the upper threshold
replica_threshold = np.nanpercentile(train_values, pct_upper)

print(f"\nThreshold (computed on train):")
print(f"  Single threshold: {replica_threshold:.6f}")

# Generate replica signals (BUGGY - only long, no short)
replica_signals = np.zeros(len(test_values))
replica_signals[test_values > replica_threshold] = 1  # LONG only

n_long_rep = np.sum(replica_signals == 1)
n_short_rep = np.sum(replica_signals == -1)
n_flat_rep = np.sum(replica_signals == 0)
n_trades_rep = n_long_rep + n_short_rep

print(f"\nReplica signal distribution (OOS):")
print(f"  +1 (LONG):  {n_long_rep}")
print(f"  -1 (SHORT): {n_short_rep}  ← THIS IS THE BUG!")
print(f"   0 (FLAT):  {n_flat_rep}")
print(f"  TOTAL TRADES: {n_trades_rep}")

# =====================================================
# COMPARISON
# =====================================================
print("\n" + "=" * 70)
print("BUG IDENTIFIED")
print("=" * 70)

print(f"""
Phase 1 trades: {n_trades} ({n_long} long + {n_short} short)
Replica trades: {n_trades_rep} ({n_long_rep} long + {n_short_rep} short)

MISSING SHORT TRADES: {n_short}

The replica is missing {n_short} short signals because it only checks:
  signal = 1 if value > threshold else 0

But Phase 1 checks BOTH:
  signal = 1 if value > upper_threshold   (LONG)
  signal = -1 if value < lower_threshold  (SHORT)
""")

# Show where they diverge
print("\n" + "=" * 70)
print("FIRST 20 SIGNAL COMPARISONS")
print("=" * 70)
print(f"\n{'Index':>6} {'Value':>12} {'Phase1':>8} {'Replica':>8} {'Match':>8}")
print("-" * 50)

for i in range(min(20, len(test_values))):
    val = test_values[i]
    p1 = int(phase1_signals[i])
    rep = int(replica_signals[i])
    match = "✓" if p1 == rep else "✗"
    print(f"{i:>6} {val:>12.6f} {p1:>8} {rep:>8} {match:>8}")

# Find first divergence
for i in range(len(test_values)):
    if phase1_signals[i] != replica_signals[i]:
        print(f"\n*** FIRST DIVERGENCE at index {i}:")
        print(f"    Value: {test_values[i]:.6f}")
        print(f"    Phase 1: {int(phase1_signals[i])} (SHORT because value < {lower_threshold:.6f})")
        print(f"    Replica: {int(replica_signals[i])} (FLAT - missing the short signal)")
        break

print("\n" + "=" * 70)
print("FIX REQUIRED")
print("=" * 70)
print("""
In single_indicator_backtest.py, change:

FROM (buggy):
    if rule['direction'] == 1:
        train_signals = (train_feature > threshold).astype(float)
        test_signals = (test_feature > threshold).astype(float)

TO (correct):
    upper_threshold = np.nanpercentile(train_feature, 90)
    lower_threshold = np.nanpercentile(train_feature, 10)
    
    train_signals = np.zeros(len(train_feature))
    train_signals[train_feature > upper_threshold] = 1   # LONG
    train_signals[train_feature < lower_threshold] = -1  # SHORT
""")
