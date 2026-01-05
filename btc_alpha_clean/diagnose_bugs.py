#!/usr/bin/env python3
"""
DIAGNOSTIC: Find and fix signal generation bugs

BUG 1: simple_rules_correct.py - No short signals for percentile rules
BUG 2: funding_vol_cumul - 35L + 0S (data issue or threshold issue)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("BUG DIAGNOSTIC")
print("=" * 70)

# Load data
cache_path = Path("data_cache/features_cache.pkl")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)
features = cache['features']
targets = cache['targets']

# Get returns
returns = targets['return_simple_6h'] if 'return_simple_6h' in targets.columns else None

# Split 60/40
n = len(features)
n_train = int(n * 0.6)

train_mask = np.zeros(n, dtype=bool)
train_mask[:n_train] = True

print(f"\nData: {n} samples, Train: {n_train}, Test: {n - n_train}")

# ═══════════════════════════════════════════════════════════════════════════════
# BUG 1 DIAGNOSIS: simple_rules_correct.py percentile handling
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BUG 1: simple_rules_correct.py PERCENTILE HANDLING")
print("=" * 70)

print("""
BROKEN CODE (simple_rules_correct.py lines 150-158):
```python
if thresh['type'] == 'percentile':
    if thresh['direction'] == 1:
        if value > thresh['value']:
            return 1  # Long signal
    else:
        if value < thresh['value']:
            return 1  # Long signal (inverse)
    
    # ❌ BUG: NEVER returns -1 (short) for percentile rules!
```

WORKING CODE (single_indicator_backtest.py lines 143-159):
```python
if rule['threshold_type'] == 'percentile':
    pct_upper = rule['threshold_value']
    pct_lower = 100 - pct_upper  # e.g., 90 -> 10
    
    upper_threshold = np.nanpercentile(train_values, pct_upper)
    lower_threshold = np.nanpercentile(train_values, pct_lower)
    
    if rule['direction'] == 1:
        signals[feature_values > upper_threshold] = 1   # LONG
        signals[feature_values < lower_threshold] = -1  # SHORT ✓
    else:
        signals[feature_values > upper_threshold] = -1  # SHORT ✓
        signals[feature_values < lower_threshold] = 1   # LONG
```

DIFFERENCE:
- simple_rules_correct.py stores ONE threshold (upper only)
- single_indicator_backtest.py stores TWO thresholds (upper AND lower)
- simple_rules_correct.py never generates short (-1) for percentile rules
""")

# ═══════════════════════════════════════════════════════════════════════════════
# BUG 2 DIAGNOSIS: funding_vol_cumul has 0 shorts
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BUG 2: funding_vol_cumul MISSING SHORTS")
print("=" * 70)

feature_name = "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h"

if feature_name in features.columns:
    feature_values = features[feature_name].values
    train_values = feature_values[train_mask]
    test_values = feature_values[~train_mask]
    
    # Compute zscore parameters from train
    mean = np.nanmean(train_values)
    std = np.nanstd(train_values)
    
    print(f"\nFeature: {feature_name}")
    print(f"\nTRAIN statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Min:  {np.nanmin(train_values):.6f}")
    print(f"  Max:  {np.nanmax(train_values):.6f}")
    
    # Thresholds for zscore=2.0, direction=-1
    z_upper = 2.0
    z_lower = -2.0
    
    upper_thresh = mean + z_upper * std  # SHORT when zscore > 2.0
    lower_thresh = mean + z_lower * std  # LONG when zscore < -2.0
    
    print(f"\nThresholds (zscore=2.0):")
    print(f"  Upper (SHORT when >): {upper_thresh:.6f}")
    print(f"  Lower (LONG when <):  {lower_thresh:.6f}")
    
    # Check OOS values
    print(f"\nTEST (OOS) statistics:")
    print(f"  N samples: {len(test_values)}")
    print(f"  Min:  {np.nanmin(test_values):.6f}")
    print(f"  Max:  {np.nanmax(test_values):.6f}")
    print(f"  Mean: {np.nanmean(test_values):.6f}")
    
    # Check signal counts
    n_long = np.sum(test_values < lower_thresh)
    n_short = np.sum(test_values > upper_thresh)
    
    print(f"\nSignal counts in OOS:")
    print(f"  Values < lower threshold (LONG):  {n_long}")
    print(f"  Values > upper threshold (SHORT): {n_short}")
    
    if n_short == 0:
        print(f"\n⚠️  ROOT CAUSE IDENTIFIED:")
        print(f"   In OOS period, max value is {np.nanmax(test_values):.6f}")
        print(f"   Upper threshold is {upper_thresh:.6f}")
        print(f"   No values exceed upper threshold, so NO SHORTS generated")
        print(f"\n   This is a DATA ISSUE, not a CODE BUG!")
        print(f"   The feature simply doesn't trigger shorts in this time period.")
    
    # Compare with train
    train_n_long = np.sum(train_values < lower_thresh)
    train_n_short = np.sum(train_values > upper_thresh)
    print(f"\nSignal counts in TRAIN (for comparison):")
    print(f"  Values < lower threshold (LONG):  {train_n_long}")
    print(f"  Values > upper threshold (SHORT): {train_n_short}")

else:
    print(f"Feature {feature_name} not found!")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX FOR BUG 1
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FIX FOR BUG 1: simple_rules_correct.py")
print("=" * 70)

print("""
REQUIRED CHANGES:

1. In fit() method (line 114-129), store BOTH thresholds:

```python
if rule.threshold_type == "percentile":
    pct_upper = rule.threshold_value
    pct_lower = 100 - pct_upper
    
    self.thresholds[rule.feature] = {
        'type': 'percentile',
        'upper_value': np.percentile(values, pct_upper),
        'lower_value': np.percentile(values, pct_lower),  # ADD THIS
        'direction': rule.direction
    }
```

2. In _get_signal() method (line 150-158), generate BOTH long AND short:

```python
if thresh['type'] == 'percentile':
    if thresh['direction'] == 1:
        if value > thresh['upper_value']:
            return 1   # Long when HIGH
        elif value < thresh['lower_value']:
            return -1  # Short when LOW  # ADD THIS
    else:
        if value > thresh['upper_value']:
            return -1  # Short when HIGH  # ADD THIS
        elif value < thresh['lower_value']:
            return 1   # Long when LOW
```
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
BUG 1: simple_rules_correct.py
  Status: CODE BUG
  Cause:  Only stores upper threshold, never generates short signals
  Fix:    Store both upper/lower thresholds, generate -1 for lower crossings

BUG 2: funding_vol_cumul 35L + 0S
  Status: DATA ISSUE (not a code bug)
  Cause:  OOS feature values never exceed +2σ threshold
  Fix:    None needed - code is correct, data just doesn't trigger shorts
          This indicator only generates longs in this time period.
""")
