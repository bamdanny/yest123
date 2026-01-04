#!/usr/bin/env python3
"""
End-to-end test of Sharpe calculation.
"""
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

import numpy as np
import pandas as pd
from validation.framework import StrategyValidator

print("=" * 70)
print("GENERATING SYNTHETIC DATA")
print("=" * 70)

# Generate 180 days of data (similar to actual pipeline)
from data.synthetic import generate_test_data

data_dict = generate_test_data(n_days=180, seed=42)
# Merge all dataframes
data = data_dict['price']
for key in ['derivatives', 'sentiment']:
    if key in data_dict and data_dict[key] is not None:
        df = data_dict[key]
        for col in df.columns:
            if col not in data.columns:
                data[col] = df[col].reindex(data.index)

print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)[:10]}...")

# Create a simple strategy signal based on momentum
print("\n" + "=" * 70)
print("CREATING STRATEGY SIGNALS")
print("=" * 70)

if 'price_close' in data.columns:
    returns = data['price_close'].pct_change().fillna(0)
    # Simple momentum strategy: go long if 24h return > 0
    momentum = returns.rolling(6).mean()  # 24h momentum
    signals = pd.Series(0, index=data.index)
    signals[momentum > 0] = 1
    signals[momentum < 0] = -1
else:
    # Fallback
    returns = pd.Series(np.random.normal(0.001, 0.02, len(data)), index=data.index)
    signals = pd.Series(np.random.choice([-1, 0, 1], len(data), p=[0.1, 0.8, 0.1]), index=data.index)

print(f"Total bars: {len(signals)}")
print(f"Active signals: {(signals != 0).sum()} ({(signals != 0).mean()*100:.1f}%)")

# Run validation
print("\n" + "=" * 70)
print("RUNNING VALIDATION (watch for SHARPE DEBUG output)")
print("=" * 70)

validator = StrategyValidator()
validator.walk_forward_periods = 5  # Same as pipeline

# This will print SHARPE CALCULATION DEBUG for each period
report = validator.validate(signals, returns, data)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Combined OOS Sharpe: {report.combined_oos_metrics.sharpe_ratio:.2f}")

# Manual verification of one period
if report.walk_forward_results:
    wf = report.walk_forward_results[0]
    print(f"\nPeriod 0 Manual Check:")
    print(f"  OOS n_trades: {wf.out_sample_metrics.n_trades}")
    print(f"  OOS Sharpe from report: {wf.out_sample_metrics.sharpe_ratio:.2f}")
