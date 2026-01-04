#!/usr/bin/env python3
"""
Minimal test to trace Sharpe calculation.
"""
import sys
sys.path.insert(0, '/home/claude/btc_alpha_discovery')

import numpy as np
import pandas as pd
from validation.framework import StrategyValidator

print("Creating test data matching log output:")
print("Period 0: train signals=35 (20d), test signals=14 (9d)")

# Create 20 days of 4h bars = 120 bars
n_bars = 120
np.random.seed(42)

# Create signals: 35 non-zero out of 120
signals = np.zeros(n_bars)
trade_idx = np.random.choice(n_bars, 35, replace=False)
signals[trade_idx] = np.random.choice([-1, 1], 35)

# Create returns
returns = np.random.normal(0.002, 0.02, n_bars)

signals = pd.Series(signals)
returns = pd.Series(returns)

# Calculate metrics
validator = StrategyValidator()
metrics = validator._calculate_metrics(signals, returns, period_days=20)

print(f"\nFinal sharpe_ratio from ValidationMetrics: {metrics.sharpe_ratio:.2f}")
print(f"n_trades: {metrics.n_trades}")
