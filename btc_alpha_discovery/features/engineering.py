"""
Feature Engineering
==================

Generates 500+ features from all data sources.
Every feature is a testable hypothesis.

Categories:
1. Price features (returns, volatility, trend, momentum, mean reversion)
2. Derivatives features (funding, OI, L/S, liquidations)
3. Macro features (VIX, DXY, yields, conditions)
4. Sentiment features (Fear & Greed, options)
5. Time features (hour, day, session, events)
6. Interaction features (divergences, confirmations)
7. Lagged features (all of above with lags)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import feature_config

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Master feature generator.
    
    Generates features in categories, each testable hypothesis.
    """
    
    def __init__(self, config=None):
        self.config = config or feature_config
        self.feature_registry = {}  # Track all generated features
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from master dataset.
        
        This is the main entry point.
        """
        logger.info("Generating feature universe...")
        logger.info(f"Input shape: {df.shape}")
        
        # Keep original columns
        original_cols = set(df.columns)
        
        # Generate each category
        with tqdm(total=7, desc="Feature categories") as pbar:
            df = self.generate_price_features(df)
            pbar.update(1)
            
            df = self.generate_derivatives_features(df)
            pbar.update(1)
            
            df = self.generate_macro_features(df)
            pbar.update(1)
            
            df = self.generate_sentiment_features(df)
            pbar.update(1)
            
            df = self.generate_time_features(df)
            pbar.update(1)
            
            df = self.generate_interaction_features(df)
            pbar.update(1)
            
            df = self.generate_lagged_features(df)
            pbar.update(1)
        
        # Count new features
        new_cols = set(df.columns) - original_cols
        logger.info(f"Generated {len(new_cols)} new features")
        logger.info(f"Total columns: {len(df.columns)}")
        
        # Register features
        self._register_features(df, new_cols)
        
        return df
    
    def _register_features(self, df: pd.DataFrame, feature_cols: set):
        """Register features with metadata"""
        for col in feature_cols:
            category = col.split("_")[0] if "_" in col else "other"
            self.feature_registry[col] = {
                "category": category,
                "dtype": str(df[col].dtype),
                "null_pct": df[col].isnull().mean(),
                "unique_values": df[col].nunique()
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all price-based features"""
        logger.info("  Generating price features...")
        
        # Need OHLCV columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing price columns: {missing}")
            return df
        
        df = df.copy()
        
        # ── Returns ──
        # Periods in candles: 1=4h, 3=12h, 6=1d, 12=2d, 42=1w, 84=2w
        for period in [1, 3, 6, 12, 42, 84]:
            df[f"price_return_{period}h"] = df["close"].pct_change(period)
            df[f"price_log_return_{period}h"] = np.log(df["close"] / df["close"].shift(period))
        
        # ── Volatility ──
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            tr = pd.concat([
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            ], axis=1).max(axis=1)
            df[f"price_atr_{period}"] = tr.rolling(period).mean()
            df[f"price_atr_{period}_pct"] = df[f"price_atr_{period}"] / df["close"]
        
        # Realized volatility (windows in candles)
        # 6=1d, 42=1w, 180=1m
        for window in [6, 42, 180]:
            df[f"price_realized_vol_{window}h"] = df["price_log_return_1h"].rolling(window).std() * np.sqrt(6 * 365)
        
        # Parkinson volatility (high-low based)
        for window in [6, 42]:
            hl_ratio = np.log(df["high"] / df["low"])
            df[f"price_parkinson_vol_{window}h"] = np.sqrt(hl_ratio.rolling(window).apply(lambda x: (x**2).sum() / (4 * np.log(2) * len(x))))
        
        # ── Trend indicators ──
        # EMAs
        for period in self.config.ma_periods:
            df[f"price_ema_{period}"] = df["close"].ewm(span=period).mean()
            df[f"price_vs_ema_{period}"] = (df["close"] - df[f"price_ema_{period}"]) / df[f"price_ema_{period}"]
            df[f"price_ema_{period}_slope"] = df[f"price_ema_{period}"].diff(4) / df[f"price_ema_{period}"]
        
        # SMAs
        for period in [20, 50, 100, 200]:
            df[f"price_sma_{period}"] = df["close"].rolling(period).mean()
            df[f"price_vs_sma_{period}"] = (df["close"] - df[f"price_sma_{period}"]) / df[f"price_sma_{period}"]
        
        # EMA alignment score
        ema_cols = [f"price_ema_{p}" for p in sorted(self.config.ma_periods)[:5]]
        if all(c in df.columns for c in ema_cols):
            def calc_alignment(row):
                vals = [row[c] for c in ema_cols]
                ascending = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
                descending = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
                if ascending:
                    return 1
                elif descending:
                    return -1
                return 0
            df["price_ema_alignment"] = df.apply(calc_alignment, axis=1)
        
        # ── Mean reversion ──
        # Bollinger Bands
        for period in [20, 50]:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"price_bb_upper_{period}"] = sma + 2 * std
            df[f"price_bb_lower_{period}"] = sma - 2 * std
            df[f"price_bb_position_{period}"] = (df["close"] - df[f"price_bb_lower_{period}"]) / (df[f"price_bb_upper_{period}"] - df[f"price_bb_lower_{period}"])
            df[f"price_bb_width_{period}"] = (df[f"price_bb_upper_{period}"] - df[f"price_bb_lower_{period}"]) / sma
        
        # Z-scores
        for window in [24, 168, 720]:
            mean = df["close"].rolling(window).mean()
            std = df["close"].rolling(window).std()
            df[f"price_zscore_{window}h"] = (df["close"] - mean) / std
        
        # Distance from high/low
        for window in [24, 168, 720]:
            rolling_high = df["high"].rolling(window).max()
            rolling_low = df["low"].rolling(window).min()
            df[f"price_dist_from_high_{window}h"] = (df["close"] - rolling_high) / rolling_high
            df[f"price_dist_from_low_{window}h"] = (df["close"] - rolling_low) / rolling_low
            df[f"price_range_position_{window}h"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        # ── Momentum oscillators ──
        # RSI (multiple periods - not just 14!)
        for period in self.config.rsi_periods:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df[f"price_rsi_{period}"] = 100 - (100 / (1 + rs))
            
            # RSI zones
            df[f"price_rsi_{period}_oversold"] = (df[f"price_rsi_{period}"] < 30).astype(int)
            df[f"price_rsi_{period}_overbought"] = (df[f"price_rsi_{period}"] > 70).astype(int)
        
        # Stochastic
        for period in [14, 28]:
            rolling_low = df["low"].rolling(period).min()
            rolling_high = df["high"].rolling(period).max()
            df[f"price_stoch_k_{period}"] = 100 * (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            df[f"price_stoch_d_{period}"] = df[f"price_stoch_k_{period}"].rolling(3).mean()
        
        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["price_macd_line"] = ema12 - ema26
        df["price_macd_signal"] = df["price_macd_line"].ewm(span=9).mean()
        df["price_macd_histogram"] = df["price_macd_line"] - df["price_macd_signal"]
        df["price_macd_histogram_slope"] = df["price_macd_histogram"].diff()
        
        # Rate of Change
        for period in [10, 20, 50]:
            df[f"price_roc_{period}"] = df["close"].pct_change(period) * 100
        
        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f"price_cci_{period}"] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # Williams %R
        for period in [14, 28]:
            rolling_high = df["high"].rolling(period).max()
            rolling_low = df["low"].rolling(period).min()
            df[f"price_williams_r_{period}"] = -100 * (rolling_high - df["close"]) / (rolling_high - rolling_low + 1e-10)
        
        # ── Volume features ──
        df["price_volume_sma_20_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        df["price_volume_trend_5"] = df["volume"].rolling(5).mean().diff()
        
        # OBV (On Balance Volume)
        df["price_obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["price_obv_slope"] = df["price_obv"].diff(5)
        
        # VWAP distance
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["price_vwap_6c"] = (typical_price * df["volume"]).rolling(6).sum() / (df["volume"].rolling(6).sum() + 1e-10)
        df["price_vwap_dist"] = (df["close"] - df["price_vwap_6c"]) / df["price_vwap_6c"]
        
        # Taker buy ratio (if available)
        if "taker_buy_ratio" in df.columns:
            df["price_taker_buy_zscore"] = (df["taker_buy_ratio"] - df["taker_buy_ratio"].rolling(42).mean()) / (df["taker_buy_ratio"].rolling(42).std() + 1e-10)
        
        # ── Market structure ──
        # Higher highs / Lower lows count
        for window in [20, 50]:
            hh = (df["high"] > df["high"].shift(1)).rolling(window).sum()
            ll = (df["low"] < df["low"].shift(1)).rolling(window).sum()
            df[f"price_hh_count_{window}"] = hh
            df[f"price_ll_count_{window}"] = ll
            df[f"price_hh_ll_diff_{window}"] = hh - ll
        
        # Consolidation (range contraction)
        for window in [20, 50]:
            range_now = df["high"].rolling(window).max() - df["low"].rolling(window).min()
            range_prev = df["high"].rolling(window * 2).max() - df["low"].rolling(window * 2).min()
            df[f"price_consolidation_{window}"] = range_now / (range_prev + 1e-10)
        
        logger.info(f"    Generated price features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DERIVATIVES FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_derivatives_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from derivatives data (CoinGlass/Binance)"""
        logger.info("  Generating derivatives features...")
        
        df = df.copy()
        
        # Find derivatives columns
        deriv_cols = [c for c in df.columns if c.startswith("deriv_")]
        
        # Process funding rate columns
        funding_cols = [c for c in deriv_cols if "funding" in c.lower()]
        for col in funding_cols:
            base_name = col.replace("deriv_", "deriv_feat_")
            
            # Z-scores
            for window in [168, 336, 720]:
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                df[f"{base_name}_zscore_{window}h"] = (df[col] - mean) / (std + 1e-10)
            
            # Cumulative
            for window in [8, 24, 72, 168]:
                df[f"{base_name}_cumul_{window}h"] = df[col].rolling(window).sum()
            
            # Extremes
            df[f"{base_name}_extreme_pos"] = (df[col] > df[col].rolling(42).mean() + 2 * df[col].rolling(42).std()).astype(int)
            df[f"{base_name}_extreme_neg"] = (df[col] < df[col].rolling(42).mean() - 2 * df[col].rolling(42).std()).astype(int)
        
        # Process OI columns
        oi_cols = [c for c in deriv_cols if "oi" in c.lower()]
        for col in oi_cols:
            base_name = col.replace("deriv_", "deriv_feat_")
            
            # Changes
            for period in [1, 4, 24, 168]:
                df[f"{base_name}_change_{period}h"] = df[col].pct_change(period)
            
            # Z-scores
            for window in [168, 720]:
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                df[f"{base_name}_zscore_{window}h"] = (df[col] - mean) / (std + 1e-10)
            
            # Acceleration
            df[f"{base_name}_accel"] = df[col].diff().diff()
            
            # Flush detection (large drop)
            df[f"{base_name}_flush"] = (df[col].pct_change(4) < -0.05).astype(int)
        
        # Process liquidation columns
        liq_cols = [c for c in deriv_cols if "liq" in c.lower()]
        for col in liq_cols:
            base_name = col.replace("deriv_", "deriv_feat_")
            
            # Rolling stats
            for window in [24, 168]:
                df[f"{base_name}_sum_{window}h"] = df[col].rolling(window).sum()
                df[f"{base_name}_max_{window}h"] = df[col].rolling(window).max()
            
            # Spike detection
            df[f"{base_name}_spike"] = (df[col] > df[col].rolling(42).mean() + 3 * df[col].rolling(42).std()).astype(int)
        
        # Process L/S ratio columns
        ls_cols = [c for c in deriv_cols if "long_short" in c.lower() or "ls_" in c.lower()]
        for col in ls_cols:
            base_name = col.replace("deriv_", "deriv_feat_")
            
            # Z-scores
            for window in [168, 336]:
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                df[f"{base_name}_zscore_{window}h"] = (df[col] - mean) / (std + 1e-10)
            
            # Extremes
            df[f"{base_name}_extreme_long"] = (df[col] > df[col].rolling(42).quantile(0.95)).astype(int)
            df[f"{base_name}_extreme_short"] = (df[col] < df[col].rolling(42).quantile(0.05)).astype(int)
        
        logger.info(f"    Generated derivatives features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MACRO FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from macro data (Polygon/FRED)"""
        logger.info("  Generating macro features...")
        
        df = df.copy()
        
        # Find macro columns
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        
        # Process VIX-related columns
        vix_cols = [c for c in macro_cols if "vix" in c.lower()]
        for col in vix_cols:
            base_name = col.replace("macro_", "macro_feat_")
            
            # Regime classification
            if df[col].dtype in [np.float64, np.int64]:
                df[f"{base_name}_regime_low"] = (df[col] < 15).astype(int)
                df[f"{base_name}_regime_normal"] = ((df[col] >= 15) & (df[col] < 20)).astype(int)
                df[f"{base_name}_regime_elevated"] = ((df[col] >= 20) & (df[col] < 30)).astype(int)
                df[f"{base_name}_regime_extreme"] = (df[col] >= 30).astype(int)
                
                # Z-scores
                for window in [30, 90, 252]:
                    mean = df[col].rolling(window).mean()
                    std = df[col].rolling(window).std()
                    df[f"{base_name}_zscore_{window}d"] = (df[col] - mean) / (std + 1e-10)
                
                # Spike
                df[f"{base_name}_spike"] = (df[col].pct_change(1) > 0.2).astype(int)
        
        # Process yield curve columns
        curve_cols = [c for c in macro_cols if "t10y" in c.lower() or "yield" in c.lower() or "curve" in c.lower()]
        for col in curve_cols:
            if df[col].dtype in [np.float64, np.int64]:
                base_name = col.replace("macro_", "macro_feat_")
                
                # Inversion
                df[f"{base_name}_inverted"] = (df[col] < 0).astype(int)
                
                # Changes
                for period in [5, 20, 60]:
                    df[f"{base_name}_change_{period}d"] = df[col].diff(period)
        
        # Process DXY columns
        dxy_cols = [c for c in macro_cols if "dxy" in c.lower()]
        for col in dxy_cols:
            if df[col].dtype in [np.float64, np.int64]:
                base_name = col.replace("macro_", "macro_feat_")
                
                # Trend
                df[f"{base_name}_vs_sma50"] = df[col] / df[col].rolling(50).mean() - 1
                
                # Changes
                for period in [1, 5, 20]:
                    df[f"{base_name}_change_{period}d"] = df[col].pct_change(period)
        
        # Process financial conditions
        fin_cols = [c for c in macro_cols if "nfci" in c.lower() or "stlfsi" in c.lower()]
        for col in fin_cols:
            if df[col].dtype in [np.float64, np.int64]:
                base_name = col.replace("macro_", "macro_feat_")
                
                # Level
                df[f"{base_name}_tight"] = (df[col] > 0).astype(int)
                df[f"{base_name}_loose"] = (df[col] < 0).astype(int)
                
                # Changes
                df[f"{base_name}_change_5d"] = df[col].diff(5)
        
        logger.info(f"    Generated macro features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SENTIMENT FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from sentiment data"""
        logger.info("  Generating sentiment features...")
        
        df = df.copy()
        
        # Find sentiment columns
        sent_cols = [c for c in df.columns if c.startswith("sent_")]
        
        # Process Fear & Greed columns
        fg_cols = [c for c in sent_cols if "fear_greed" in c.lower()]
        for col in fg_cols:
            if "value" in col.lower() and df[col].dtype in [np.float64, np.int64]:
                base_name = "sent_feat_fg"
                
                # Regime
                df[f"{base_name}_extreme_fear"] = (df[col] < 20).astype(int)
                df[f"{base_name}_fear"] = ((df[col] >= 20) & (df[col] < 40)).astype(int)
                df[f"{base_name}_neutral"] = ((df[col] >= 40) & (df[col] < 60)).astype(int)
                df[f"{base_name}_greed"] = ((df[col] >= 60) & (df[col] < 80)).astype(int)
                df[f"{base_name}_extreme_greed"] = (df[col] >= 80).astype(int)
                
                # Z-scores
                for window in [30, 90]:
                    mean = df[col].rolling(window).mean()
                    std = df[col].rolling(window).std()
                    df[f"{base_name}_zscore_{window}d"] = (df[col] - mean) / (std + 1e-10)
                
                # Changes
                df[f"{base_name}_change_7d"] = df[col].diff(7)
                
                # Days at extreme
                extreme = (df[col] < 20) | (df[col] > 80)
                df[f"{base_name}_days_at_extreme"] = extreme.groupby((~extreme).cumsum()).cumsum()
        
        # Process options columns
        opt_cols = [c for c in sent_cols if "put_call" in c.lower() or "iv" in c.lower()]
        for col in opt_cols:
            if df[col].dtype in [np.float64, np.int64]:
                base_name = col.replace("sent_", "sent_feat_")
                
                # Z-scores
                for window in [30, 90]:
                    mean = df[col].rolling(window).mean()
                    std = df[col].rolling(window).std()
                    df[f"{base_name}_zscore_{window}d"] = (df[col] - mean) / (std + 1e-10)
        
        logger.info(f"    Generated sentiment features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIME FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features"""
        logger.info("  Generating time features...")
        
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column for time features")
            return df
        
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])
        
        # Basic time
        df["time_hour"] = ts.dt.hour
        df["time_day_of_week"] = ts.dt.dayofweek
        df["time_day_of_month"] = ts.dt.day
        df["time_week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["time_month"] = ts.dt.month
        
        # Cyclic encoding (sin/cos for periodicity)
        df["time_hour_sin"] = np.sin(2 * np.pi * df["time_hour"] / 24)
        df["time_hour_cos"] = np.cos(2 * np.pi * df["time_hour"] / 24)
        df["time_dow_sin"] = np.sin(2 * np.pi * df["time_day_of_week"] / 7)
        df["time_dow_cos"] = np.cos(2 * np.pi * df["time_day_of_week"] / 7)
        df["time_month_sin"] = np.sin(2 * np.pi * df["time_month"] / 12)
        df["time_month_cos"] = np.cos(2 * np.pi * df["time_month"] / 12)
        
        # Session flags
        df["time_is_weekend"] = (df["time_day_of_week"] >= 5).astype(int)
        df["time_is_month_end"] = (df["time_day_of_month"] >= 28).astype(int)
        
        # Trading sessions (UTC)
        df["time_session_asia"] = ((df["time_hour"] >= 0) & (df["time_hour"] < 8)).astype(int)
        df["time_session_europe"] = ((df["time_hour"] >= 7) & (df["time_hour"] < 16)).astype(int)
        df["time_session_us"] = ((df["time_hour"] >= 13) & (df["time_hour"] < 21)).astype(int)
        
        # US market hours (approximately 14:30-21:00 UTC)
        df["time_us_market_open"] = ((df["time_hour"] >= 14) & (df["time_hour"] < 21) & (df["time_day_of_week"] < 5)).astype(int)
        
        logger.info(f"    Generated time features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INTERACTION FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-feature interactions"""
        logger.info("  Generating interaction features...")
        
        df = df.copy()
        
        # Price-RSI divergence
        for period in [14, 21]:
            rsi_col = f"price_rsi_{period}"
            if rsi_col in df.columns:
                # Bullish divergence: price lower low, RSI higher low
                price_ll = (df["low"] < df["low"].shift(6)) & (df["low"].shift(6) < df["low"].shift(48))
                rsi_hl = (df[rsi_col] > df[rsi_col].shift(6)) & (df[rsi_col].shift(6) < df[rsi_col].shift(48))
                df[f"interact_bullish_div_rsi{period}"] = (price_ll & rsi_hl).astype(int)
                
                # Bearish divergence
                price_hh = (df["high"] > df["high"].shift(6)) & (df["high"].shift(6) > df["high"].shift(48))
                rsi_lh = (df[rsi_col] < df[rsi_col].shift(6)) & (df[rsi_col].shift(6) > df[rsi_col].shift(48))
                df[f"interact_bearish_div_rsi{period}"] = (price_hh & rsi_lh).astype(int)
        
        # OI-Price divergence (if OI available)
        oi_cols = [c for c in df.columns if "oi" in c.lower() and "change" in c.lower()]
        if oi_cols and "price_return_6h" in df.columns:
            oi_col = oi_cols[0]
            # OI up, price down = distribution
            df["interact_oi_price_div"] = ((df[oi_col] > 0) & (df["price_return_6h"] < 0)).astype(int)
            # OI down, price up = short squeeze
            df["interact_oi_price_squeeze"] = ((df[oi_col] < 0) & (df["price_return_6h"] > 0)).astype(int)
        
        # Funding-Price divergence
        funding_cols = [c for c in df.columns if "funding" in c.lower() and df[c].dtype in [np.float64, np.int64]]
        if funding_cols and "price_return_6h" in df.columns:
            funding_col = funding_cols[0]
            # High funding, negative returns = correction incoming?
            df["interact_funding_price_div"] = ((df[funding_col] > df[funding_col].rolling(42).mean() + df[funding_col].rolling(42).std()) & (df["price_return_6h"] < 0)).astype(int)
        
        # RSI extreme + Volume spike
        if "price_rsi_14" in df.columns and "price_volume_sma_20_ratio" in df.columns:
            df["interact_rsi_oversold_vol_spike"] = ((df["price_rsi_14"] < 30) & (df["price_volume_sma_20_ratio"] > 2)).astype(int)
            df["interact_rsi_overbought_vol_spike"] = ((df["price_rsi_14"] > 70) & (df["price_volume_sma_20_ratio"] > 2)).astype(int)
        
        # Multi-condition alignment
        # All oversold
        oversold_cols = [c for c in df.columns if "oversold" in c.lower()]
        if len(oversold_cols) >= 2:
            df["interact_multi_oversold"] = df[oversold_cols].sum(axis=1)
        
        overbought_cols = [c for c in df.columns if "overbought" in c.lower()]
        if len(overbought_cols) >= 2:
            df["interact_multi_overbought"] = df[overbought_cols].sum(axis=1)
        
        logger.info(f"    Generated interaction features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAGGED FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lagged versions of key features"""
        logger.info("  Generating lagged features...")
        
        df = df.copy()
        
        # Select key features to lag (not all - too many)
        key_features = []
        
        # Price features
        for col in df.columns:
            if col.startswith("price_"):
                if any(x in col for x in ["rsi_14", "zscore_168", "bb_position", "macd_histogram", "ema_alignment"]):
                    key_features.append(col)
        
        # Derivatives features
        for col in df.columns:
            if "funding" in col.lower() and "zscore" in col.lower():
                key_features.append(col)
            if "oi" in col.lower() and "change" in col.lower():
                key_features.append(col)
        
        # Sentiment features
        for col in df.columns:
            if "fear_greed" in col.lower() and "value" in col.lower():
                key_features.append(col)
        
        # Generate lags
        lag_periods = [4, 24, 48]  # 4h, 1d, 2d
        
        for col in key_features[:30]:  # Limit to avoid explosion
            for lag in lag_periods:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
        
        logger.info(f"    Generated lagged features for {len(key_features[:30])} base features")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TARGET VARIABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_targets(self, df: pd.DataFrame, holding_periods: List[int] = None) -> pd.DataFrame:
        """
        Generate target variables for multiple holding periods.
        
        Target = future return (for regression) or profitable (for classification)
        """
        if holding_periods is None:
            holding_periods = [1, 4, 8, 24, 48, 168]  # hours
        
        logger.info(f"Generating target variables for holding periods: {holding_periods}")
        
        df = df.copy()
        
        for period in holding_periods:
            # Forward return
            df[f"target_return_{period}h"] = df["close"].shift(-period) / df["close"] - 1
            
            # Binary: profitable after costs
            cost = 0.0006  # 0.04% commission + 0.02% slippage per side = 0.06% round trip
            df[f"target_profitable_{period}h"] = (df[f"target_return_{period}h"] > cost).astype(int)
            
            # Ternary: -1 (short), 0 (no trade), 1 (long)
            threshold = 0.01  # 1% move threshold
            df[f"target_direction_{period}h"] = 0
            df.loc[df[f"target_return_{period}h"] > threshold, f"target_direction_{period}h"] = 1
            df.loc[df[f"target_return_{period}h"] < -threshold, f"target_direction_{period}h"] = -1
        
        return df
    
    def get_feature_list(self) -> Dict[str, List[str]]:
        """Get all generated features by category"""
        categories = {}
        for feat, info in self.feature_registry.items():
            cat = info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(feat)
        return categories


if __name__ == "__main__":
    # Test feature generation
    import numpy as np
    
    # Create sample data
    n = 1000
    sample_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": 40000 + np.cumsum(np.random.randn(n) * 100),
        "high": 40000 + np.cumsum(np.random.randn(n) * 100) + abs(np.random.randn(n) * 50),
        "low": 40000 + np.cumsum(np.random.randn(n) * 100) - abs(np.random.randn(n) * 50),
        "close": 40000 + np.cumsum(np.random.randn(n) * 100),
        "volume": abs(np.random.randn(n) * 1000 + 5000),
    })
    sample_df["high"] = sample_df[["open", "close", "high"]].max(axis=1)
    sample_df["low"] = sample_df[["open", "close", "low"]].min(axis=1)
    
    # Generate features
    generator = FeatureGenerator()
    result = generator.generate_all_features(sample_df)
    result = generator.generate_targets(result)
    
    print(f"\nGenerated {len(result.columns)} total columns")
    print(f"\nFeature categories:")
    for cat, feats in generator.get_feature_list().items():
        print(f"  {cat}: {len(feats)} features")
    
    print(f"\nSample columns:")
    for col in sorted(result.columns)[:50]:
        print(f"  - {col}")
