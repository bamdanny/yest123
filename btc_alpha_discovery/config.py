"""
BTC Alpha Discovery System - Configuration
==========================================

All API keys, endpoints, and system parameters.
Treat EVERY threshold and weight as a hypothesis to be tested.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import os
from pathlib import Path

# Get project root directory (where this config.py is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# ═══════════════════════════════════════════════════════════════════════════════
# API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_KEYS = {
    "polygon": "k5kmuuIIKhypD_nx4zwq4mY3jxJcZd0s",
    "fred": "d3f2bec8332c0b77dbeb54ddd7b6a245",
    "coinglass": "c31d21c6158345f8adca60b579c6b227",
}

# Base URLs
URLS = {
    "binance_futures": "https://fapi.binance.com",
    "coinglass": "https://open-api.coinglass.com/public/v2",
    "polygon": "https://api.polygon.io",
    "fred": "https://api.stlouisfed.org/fred",
    "alternative_me": "https://api.alternative.me",
    "deribit": "https://www.deribit.com/api/v2/public",
}

# Rate limits (requests per minute)
RATE_LIMITS = {
    "binance": 2400,
    "coinglass": 30,  # Conservative estimate
    "polygon": 5,  # Free tier
    "fred": 120,
    "alternative_me": 30,
    "deribit": 60,
}


# ═══════════════════════════════════════════════════════════════════════════════
# COINGLASS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

COINGLASS_ENDPOINTS = {
    # Funding rates
    "funding": "/funding",
    "funding_history": "/funding_usd_history",
    "funding_ohlc": "/funding_ohlc",
    
    # Open Interest
    "open_interest": "/open_interest",
    "open_interest_history": "/open_interest_history",
    "oi_ohlc": "/open_interest_ohlc",
    
    # Long/Short ratios
    "long_short_ratio": "/long_short_ratio",
    "global_long_short": "/global_long_short_account_ratio",
    
    # Liquidations
    "liquidation": "/liquidation_history",
    "liquidation_map": "/liquidation_map",
    "liquidation_aggregated": "/liquidation_aggregated_history",
    
    # Taker activity
    "taker_buy_sell": "/taker_buy_sell_volume",
    "taker_history": "/taker_buy_sell_volume_history",
    
    # Exchange flows (may require higher tier)
    "exchange_flow": "/exchange_flow",
    "exchange_balance": "/exchange_balance",
    
    # Options
    "options_info": "/option/info",
    
    # Grayscale/ETF
    "grayscale": "/grayscale",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_ENDPOINTS = {
    "klines": "/fapi/v1/klines",
    "ticker_24h": "/fapi/v1/ticker/24hr",
    "mark_price": "/fapi/v1/premiumIndex",
    "funding_rate": "/fapi/v1/fundingRate",
    "open_interest": "/fapi/v1/openInterest",
    "open_interest_hist": "/futures/data/openInterestHist",
    "top_long_short": "/futures/data/topLongShortAccountRatio",
    "global_long_short": "/futures/data/globalLongShortAccountRatio",
    "taker_buy_sell": "/futures/data/takerlongshortRatio",
    "depth": "/fapi/v1/depth",
    "agg_trades": "/fapi/v1/aggTrades",
}

BINANCE_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON TICKERS
# ═══════════════════════════════════════════════════════════════════════════════

POLYGON_TICKERS = {
    # Volatility
    "VIX": "I:VIX",
    
    # Dollar
    "DXY": "I:DXY",
    
    # Equities
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    
    # Bonds
    "TLT": "TLT",
    "HYG": "HYG",
    
    # Commodities
    "GLD": "GLD",
    "USO": "USO",
    
    # Crypto proxies
    "BITO": "BITO",
    "MSTR": "MSTR",
    "COIN": "COIN",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FRED SERIES
# ═══════════════════════════════════════════════════════════════════════════════

FRED_SERIES = {
    # Interest rates
    "FEDFUNDS": "Federal Funds Rate",
    "DFF": "Fed Funds Effective (daily)",
    "GS10": "10-Year Treasury",
    "GS2": "2-Year Treasury",
    "DTB3": "3-Month T-Bill",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    
    # Inflation
    "CPIAUCSL": "CPI All Urban",
    "CPILFESL": "Core CPI",
    "T5YIE": "5Y Breakeven Inflation",
    "T10YIE": "10Y Breakeven Inflation",
    
    # Financial conditions
    "NFCI": "Chicago Fed Financial Conditions",
    "STLFSI4": "St Louis Financial Stress",
    "BAMLH0A0HYM2": "High Yield Spread",
    
    # Money supply
    "M2SL": "M2 Money Stock",
    "WALCL": "Fed Balance Sheet",
    
    # Employment (for regime detection)
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Claims",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Configuration for data fetching"""
    # Lookback
    history_days: int = 365 * 2  # 2 years of history
    
    # Primary timeframes - 4h is the main trading timeframe
    primary_timeframe: str = "4h"
    secondary_timeframes: List[str] = field(default_factory=lambda: ["1h", "1d"])
    
    # Symbol
    symbol: str = "BTCUSDT"
    
    # Storage - use project root for paths
    data_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data_cache"))
    checkpoint_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "checkpoints"))
    
    # Processing
    chunk_size: int = 10000  # Rows per chunk for memory efficiency
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """Configuration for trade simulation"""
    # Transaction costs
    commission_rate: float = 0.0004  # 0.04% per side (Binance VIP0)
    slippage_rate: float = 0.0002  # 0.02% per trade
    
    # Position sizing
    initial_capital: float = 100000.0
    position_size_pct: float = 1.0  # Full position each trade
    
    # Risk limits
    max_drawdown_pct: float = 0.20  # Stop trading at 20% drawdown
    
    # Holding periods to test (in candles for 4h timeframe)
    # 1 candle = 4h, 6 = 1 day, 42 = 1 week
    holding_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 18, 42])
    
    # Minimum trades for statistical validity
    min_trades: int = 50
    
    # Significance threshold
    p_value_threshold: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GENERATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.
    
    NOTE: All periods are in CANDLES, not hours.
    For 4h candles:
    - 6 candles = 1 day
    - 42 candles = 1 week
    - 180 candles = 1 month
    """
    # Moving average periods to test (in candles)
    ma_periods: List[int] = field(default_factory=lambda: [
        6, 12, 21, 42, 84, 180,  # 1d, 2d, 3.5d, 1w, 2w, 1m
        8, 13, 21, 34, 55, 89, 144  # Fibonacci
    ])
    
    # RSI periods to test
    rsi_periods: List[int] = field(default_factory=lambda: [6, 9, 14, 21, 42])
    
    # Lookback windows for rolling statistics (in candles)
    # For 4h: 6=1d, 42=1w, 84=2w, 180=1m
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 42, 84, 180])
    
    # Lag periods (in candles)
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 42])
    
    # Z-score windows (in candles)
    zscore_windows: List[int] = field(default_factory=lambda: [42, 84, 180])  # 1w, 2w, 1m


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimConfig:
    """Configuration for optimization"""
    # Walk-forward
    train_window_days: int = 90
    test_window_days: int = 30
    step_days: int = 30
    
    # Holdout
    holdout_pct: float = 0.2  # 20% final holdout
    
    # Feature selection
    max_features: int = 50  # After selection
    min_importance: float = 0.001  # Minimum feature importance to keep
    
    # Genetic algorithm
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    
    # Random seeds for reproducibility
    random_seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESES TO TEST (Your current system - ALL subject to invalidation)
# ═══════════════════════════════════════════════════════════════════════════════

CURRENT_HYPOTHESES = {
    "pillar_structure": {
        "derivatives": {"weight": 0.35, "hypothesis": "Derivatives most predictive"},
        "liquidations": {"weight": 0.30, "hypothesis": "Liquidations second most predictive"},
        "technical": {"weight": 0.25, "hypothesis": "Technicals useful as filter"},
        "liquidity": {"weight": 0.10, "hypothesis": "Liquidity minor factor"},
    },
    "thresholds": {
        "bullish": 55,
        "bearish": 45,
        "confidence": 0.70,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "vix_low": 15,
        "vix_normal": 20,
        "vix_high": 30,
    },
    "notes": [
        "ALL weights are arbitrary guesses",
        "ALL thresholds are arbitrary",
        "4-pillar structure is assumption, not fact",
        "No exit logic defined - must discover",
    ]
}


# Instantiate default configs
data_config = DataConfig()
sim_config = SimConfig()
feature_config = FeatureConfig()
optim_config = OptimConfig()
