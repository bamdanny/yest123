# BTC Alpha Discovery System

A systematic, data-driven framework for discovering trading alpha in Bitcoin markets. Built from the ground up to challenge assumptions and let data reveal what actually works.

## Core Philosophy

**"All weights, thresholds, and structures are UNVALIDATED HYPOTHESES until proven by data."**

This system does NOT validate a pre-existing trading system. Instead, it:
1. Fetches comprehensive market data
2. Engineers 500+ features
3. Discovers which features actually predict profits
4. Optimizes all thresholds from scratch
5. Discovers entry/exit conditions from data
6. Identifies when NOT to trade
7. Validates everything with rigorous out-of-sample testing

## Architecture

```
btc_alpha_discovery/
├── config.py                 # API keys, parameters, hypotheses
├── run_discovery.py          # Main pipeline orchestrator
│
├── data/                     # Phase 1: Data Acquisition
│   ├── fetcher.py           # Base fetcher with rate limiting
│   ├── coinglass.py         # Premium liquidation data (THE EDGE)
│   ├── binance.py           # OHLCV + derivatives
│   ├── polygon.py           # US markets (VIX, DXY, SPY, etc.)
│   ├── fred.py              # Macro indicators
│   ├── free_sources.py      # Sentiment data
│   ├── storage.py           # Parquet persistence + alignment
│   └── orchestrator.py      # Master coordinator
│
├── features/                 # Phase 2-3: Feature Engineering
│   ├── engineering.py       # 500+ feature generation
│   └── targets.py           # Forward returns, exit-aware targets
│
├── discovery/                # Phase 4, 7-9: Discovery Algorithms
│   ├── feature_importance.py # SHAP, permutation, mutual info
│   ├── entry_exit.py        # Entry/exit condition discovery
│   └── antipatterns.py      # When NOT to trade
│
├── optimization/             # Phase 5-6: Optimization
│   └── structure_weights.py  # Structure discovery, threshold optimization
│
├── validation/               # Phase 10: Validation
│   └── framework.py         # Walk-forward, regime analysis, stats
│
└── reports/                  # Reporting
    └── generator.py         # HTML/text report generation
```

## Data Sources

| Source | Data Type | Rate Limit | Key Features |
|--------|-----------|------------|--------------|
| CoinGlass | Liquidations, Derivatives | 10/min | Real liquidation levels (THE EDGE) |
| Binance | OHLCV, Funding, OI | 1200/min | Extended history |
| Polygon | US Markets | 5/min | VIX, DXY, SPY, QQQ |
| FRED | Macro | 120/min | Rates, yield curve, M2 |
| Free Sources | Sentiment | 1/min | Fear & Greed, Options IV |

## Feature Categories (~500+ features)

### Price Features (~150)
- Returns: 1h to 720h (log + simple)
- Volatility: ATR (7/14/21), realized, Parkinson
- Trend: EMAs (8/13/21/34/55/89/144/233), SMAs (20/50/100/200)
- Mean Reversion: Bollinger Bands, Z-scores
- Momentum: RSI (7/9/14/21/28), Stochastic, MACD, ROC, CCI

### Derivatives Features (~100)
- Funding: Z-scores, cumulative, extremes
- Open Interest: Changes, acceleration, flush detection
- Liquidations: Rolling sums, spike detection
- Long/Short Ratios: Z-scores, extreme flags

### Macro Features (~50)
- VIX: Regime classification, Z-scores, spikes
- Yield Curve: Inversion flags, changes
- DXY: Trend, momentum
- Financial Conditions: NFCI, stress indices

### Sentiment Features (~30)
- Fear & Greed: Regime, Z-scores, duration at extremes
- Options: Put/call ratio, IV metrics

### Time Features (~20)
- Hour, day, week, month (with cyclic encoding)
- Session flags: Asia, Europe, US
- Calendar: Weekend, month-end

### Interaction Features (~50)
- Price-RSI divergences
- OI-Price divergences
- Funding-Price divergences
- Multi-condition alignments

## Discovery Methods

### Feature Importance (Phase 4)
- **SHAP values** with XGBoost
- **Permutation importance** with Random Forest
- **Mutual information** for non-linear relationships
- **Correlation analysis** for baseline

### Entry Condition Discovery (Phase 7)
- **Decision tree rule extraction**
- **Threshold optimization** on individual features
- **Condition combination search**
- **Genetic programming** (optional)

### Exit Condition Discovery (Phase 8)
- **Optimal stop-loss/take-profit search**
- **Time-based exit optimization**
- **Conditional exits** (RSI cross, MACD signal)
- **Maximum Adverse Excursion analysis**

### Anti-Pattern Discovery (Phase 9)
- **Loss clustering** via decision trees
- **Volatility regime** analysis
- **Chop/ranging market** detection
- **Time-based anti-patterns**

### Structure Optimization (Phase 5-6)
- **Pillar structure validation**
- **Alternative structure testing** (3-pillar, 5-pillar, flat)
- **Threshold optimization** via Optuna/scipy
- **Weight optimization** via Bayesian methods

## Validation Framework (Phase 10)

### Walk-Forward Analysis
- 5 rolling windows (configurable)
- 70% train / 30% test per window
- Tracks in-sample vs out-of-sample degradation

### Statistical Tests
- T-test vs zero returns
- Normality test (Shapiro-Wilk)
- Autocorrelation test (Ljung-Box)
- Runs test for randomness

### Robustness Checks
- Time stability (first half vs second half)
- Parameter sensitivity (signal delays)
- Transaction cost sensitivity
- Regime consistency

### Grading System
| Grade | Criteria |
|-------|----------|
| A | Sharpe ≥1.5, Win Rate ≥55%, Max DD <15%, Significant |
| B | Sharpe ≥1.0, Win Rate ≥52%, Max DD <20%, Significant |
| C | Sharpe ≥0.5, Win Rate ≥50%, Max DD <25% |
| D | Sharpe >0, Some positive characteristics |
| F | Does not meet minimum requirements |

## Usage

### Quick Start
```bash
cd /home/claude/btc_alpha_discovery
python run_discovery.py --mode full
```

### Test Mode (verify setup)
```bash
python run_discovery.py --mode test
```

### Custom Target
```bash
python run_discovery.py --mode full --target profitable_48h
```

## Requirements

### Core (required)
- Python 3.9+
- pandas, numpy, scipy
- scikit-learn
- requests
- tqdm

### Optional (enhanced features)
- xgboost, shap (feature importance)
- optuna (optimization)
- deap (genetic programming)

## Transaction Costs

All analysis accounts for realistic costs:
- Commission: 0.04% per side
- Slippage: 0.02% per side
- **Total round-trip: 0.12%**

## Statistical Requirements

- **Minimum 50 trades** for any rule to be valid
- **p < 0.05** for statistical significance
- **Walk-forward validation** mandatory before deployment
- **Regime analysis** to ensure consistency

## Key Outputs

1. **Discovery Report** (HTML/text)
   - Feature importance rankings
   - Category importance breakdown
   - Pillar structure validation
   - Discovered entry/exit rules
   - Anti-patterns
   - Validation results

2. **Discovered Strategy**
   - Entry rules with statistics
   - Exit rules with performance
   - Feature importance from rules
   - Overall performance metrics

3. **Validation Report**
   - Overall grade (A-F)
   - Walk-forward results
   - Regime performance
   - Statistical tests
   - Robustness scores
   - Warnings and recommendations

## Philosophy

This system embodies key principles:

1. **Data-Driven**: No hardcoded thresholds are sacred. All are discovered.
2. **Skeptical**: The 4-pillar structure is a hypothesis, not a fact.
3. **Rigorous**: Out-of-sample validation is mandatory, not optional.
4. **Honest**: Anti-patterns matter as much as entry signals.
5. **Practical**: Transaction costs and slippage are always included.

## Hypotheses Being Tested

```python
HYPOTHESES_TO_TEST = {
    "pillar_structure_optimal": "Is 4-pillar structure optimal?",
    "pillar_weights_optimal": "Are 35/30/25/10 weights optimal?",
    "threshold_55_45": "Are 55/45 bullish/bearish thresholds optimal?",
    "confidence_70": "Is 70% confidence threshold optimal?",
    "rsi_30_70": "Are RSI 30/70 levels optimal?",
    "vix_15_20_30": "Are VIX regime breaks optimal?",
}
```

## Total Lines of Code
~9,500 lines across 15 Python modules

## License
For research and educational purposes. Not financial advice.
