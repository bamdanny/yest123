# BTC Alpha - Corrected Implementation

## What Was Wrong

The original code had **critical bugs** that made all results garbage:

| Bug | Symptom | Root Cause |
|-----|---------|------------|
| Sharpe = -180 quadrillion | Impossible number | Division by near-zero std |
| Win Rate = 0% | Every trade "lost" | Wrong win calculation logic |
| Return = -104% | Lost more than 100% | Summing returns instead of compounding |
| Model crash | `ValueError: Unknown label type: continuous` | Passing returns to classifier |

## What's Fixed

### 1. Binary Target for Classification
```python
# WRONG
y = returns  # Continuous: 0.02, -0.01, etc

# CORRECT  
y = (returns > 0).astype(int)  # Binary: 0 or 1
```

### 2. Compound Returns
```python
# WRONG
total_return = sum(trade_returns)

# CORRECT
total_return = np.prod(1 + trade_returns) - 1
```

### 3. Proper Sharpe Calculation
```python
# WRONG (no annualization, division issues)
sharpe = mean / std

# CORRECT
sharpe = (mean / std) * np.sqrt(bars_per_year)  # 2190 for 4h bars
```

### 4. Exact Phase 1 Rule Specifications
The original code used random thresholds. The fix uses EXACT specifications from Phase 1:
- Feature name
- Direction (1 = long above, -1 = long below)
- Threshold type (percentile or zscore)
- Threshold value (90th percentile, 1.5 z-score, etc.)

## Files

| File | Purpose |
|------|---------|
| `simple_rules_correct.py` | Rule-based system using exact Phase 1 specs |
| `train_anchored_correct.py` | ML with 70% anchor on proven features |
| `ml/anchored_ensemble.py` | Reusable anchored ensemble class |
| `ml/backtester.py` | Correct backtesting with proper math |
| `config/phase1_rules.json` | Exact rule specs from Phase 1 OOS validation |
| `utils/metrics.py` | Correct Sharpe, returns, win rate calculations |
| `run_scanner.py` | Live scanner using corrected models |

## Usage

### Step 1: Test Simple Rules First
```bash
python simple_rules_correct.py
```

Expected output:
- Sharpe: 3-6 (realistic)
- Win Rate: 60-70%
- Return: Positive

If you still see impossible numbers, there's a data issue.

### Step 2: Train Anchored ML
```bash
python train_anchored_correct.py
```

Expected output:
- Sharpe: 1-3
- Win Rate: 55-65%
- Better than the broken -1.74 Sharpe

### Step 3: Run Scanner
```bash
python run_scanner.py --once
```

## Phase 1 Results (Still Valid)

The single indicators genuinely work. These are the OOS-validated rules:

| Indicator | OOS Sharpe | OOS Win Rate | Trades |
|-----------|------------|--------------|--------|
| OI close change 1h | 8.74 | 73.5% | 49 |
| OI close accel | 7.38 | 61.4% | 70 |
| RSI lag 48h | 5.22 | 62.5% | 32 |
| Funding vol cumul 168h | 4.98 | 62.9% | 35 |
| Fear/Greed z-score 90d | 4.59 | 75.9% | 29 |

The alpha is real. The implementation was just broken.

## Sanity Checks

After running, verify:
- [ ] Sharpe between -5 and +10 (anything else = bug)
- [ ] Win rate between 40% and 80% (anything else = bug)
- [ ] Return between -50% and +100% (anything else = bug)
- [ ] Both long AND short trades generated
- [ ] No NaN or Inf values
