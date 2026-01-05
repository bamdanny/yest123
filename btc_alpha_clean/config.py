"""
BTC Alpha Discovery - Configuration
API Keys loaded from environment variables or .env file
"""

from dataclasses import dataclass, field
from typing import List
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.resolve()

# API Keys
API_KEYS = {
    "polygon": "k5kmuuIIKhypD_nx4zwq4mY3jxJcZd0s",
    "fred": "d3f2bec8332c0b77dbeb54ddd7b6a245",
    "coinglass": "c31d21c6158345f8adca60b579c6b227",
}

URLS = {
    "binance_futures": "https://fapi.binance.com",
    "coinglass": "https://open-api.coinglass.com/public/v2",
    "polygon": "https://api.polygon.io",
    "fred": "https://api.stlouisfed.org/fred",
    "alternative_me": "https://api.alternative.me",
    "deribit": "https://www.deribit.com/api/v2/public",
}

RATE_LIMITS = {"binance": 2400, "coinglass": 30, "polygon": 5, "fred": 120, "alternative_me": 30, "deribit": 60}

COINGLASS_ENDPOINTS = {
    "funding": "/funding", "funding_history": "/funding_usd_history", "funding_ohlc": "/funding_ohlc",
    "open_interest": "/open_interest", "open_interest_history": "/open_interest_history", "oi_ohlc": "/open_interest_ohlc",
    "long_short_ratio": "/long_short_ratio", "global_long_short": "/global_long_short_account_ratio",
    "liquidation": "/liquidation_history", "liquidation_map": "/liquidation_map", "liquidation_aggregated": "/liquidation_aggregated_history",
    "taker_buy_sell": "/taker_buy_sell_volume", "taker_history": "/taker_buy_sell_volume_history",
    "exchange_flow": "/exchange_flow", "exchange_balance": "/exchange_balance",
    "options_info": "/option/info", "grayscale": "/grayscale",
}

BINANCE_ENDPOINTS = {
    "klines": "/fapi/v1/klines", "ticker_24h": "/fapi/v1/ticker/24hr", "mark_price": "/fapi/v1/premiumIndex",
    "funding_rate": "/fapi/v1/fundingRate", "open_interest": "/fapi/v1/openInterest",
    "open_interest_hist": "/futures/data/openInterestHist", "top_long_short": "/futures/data/topLongShortAccountRatio",
    "global_long_short": "/futures/data/globalLongShortAccountRatio", "taker_buy_sell": "/futures/data/takerlongshortRatio",
    "depth": "/fapi/v1/depth", "agg_trades": "/fapi/v1/aggTrades",
}

BINANCE_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

POLYGON_TICKERS = {"VIX": "I:VIX", "DXY": "I:DXY", "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM",
                   "TLT": "TLT", "HYG": "HYG", "GLD": "GLD", "USO": "USO", "BITO": "BITO", "MSTR": "MSTR", "COIN": "COIN"}

FRED_SERIES = {"FEDFUNDS": "Federal Funds Rate", "DFF": "Fed Funds Effective", "GS10": "10-Year Treasury",
               "GS2": "2-Year Treasury", "DTB3": "3-Month T-Bill", "T10Y2Y": "10Y-2Y Spread", "T10Y3M": "10Y-3M Spread",
               "CPIAUCSL": "CPI All Urban", "CPILFESL": "Core CPI", "T5YIE": "5Y Breakeven", "T10YIE": "10Y Breakeven",
               "NFCI": "Chicago Fed Financial Conditions", "STLFSI4": "St Louis Financial Stress",
               "BAMLH0A0HYM2": "High Yield Spread", "M2SL": "M2 Money Stock", "WALCL": "Fed Balance Sheet",
               "UNRATE": "Unemployment Rate", "ICSA": "Initial Claims"}

@dataclass
class DataConfig:
    history_days: int = 365 * 2
    primary_timeframe: str = "4h"
    secondary_timeframes: List[str] = field(default_factory=lambda: ["1h", "1d"])
    symbol: str = "BTCUSDT"
    data_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data_cache"))
    checkpoint_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "checkpoints"))
    chunk_size: int = 10000
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

@dataclass
class SimConfig:
    commission_rate: float = 0.0004
    slippage_rate: float = 0.0002
    initial_capital: float = 100000.0
    position_size_pct: float = 1.0
    max_drawdown_pct: float = 0.20
    holding_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 18, 42])
    min_trades: int = 50
    p_value_threshold: float = 0.05

@dataclass
class FeatureConfig:
    ma_periods: List[int] = field(default_factory=lambda: [6, 12, 21, 42, 84, 180, 8, 13, 34, 55, 89, 144])
    rsi_periods: List[int] = field(default_factory=lambda: [6, 9, 14, 21, 42])
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 42, 84, 180])
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 42])
    zscore_windows: List[int] = field(default_factory=lambda: [42, 84, 180])

@dataclass
class OptimConfig:
    train_window_days: int = 90
    test_window_days: int = 30
    step_days: int = 30
    holdout_pct: float = 0.2
    max_features: int = 50
    min_importance: float = 0.001
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    random_seed: int = 42

CURRENT_HYPOTHESES = {
    "pillar_structure": {"derivatives": {"weight": 0.35}, "liquidations": {"weight": 0.30},
                         "technical": {"weight": 0.25}, "liquidity": {"weight": 0.10}},
    "thresholds": {"bullish": 55, "bearish": 45, "confidence": 0.70, "rsi_oversold": 30, "rsi_overbought": 70,
                   "vix_low": 15, "vix_normal": 20, "vix_high": 30},
}

data_config = DataConfig()
sim_config = SimConfig()
feature_config = FeatureConfig()
optim_config = OptimConfig()
