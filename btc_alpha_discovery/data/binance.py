"""
Binance Futures Data Fetcher
============================

OHLCV price data and backup derivatives data.
No API key required for public endpoints.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BaseFetcher, timestamp_to_ms, ms_to_timestamp
from config import URLS, RATE_LIMITS, BINANCE_ENDPOINTS, BINANCE_INTERVALS

logger = logging.getLogger(__name__)


class BinanceFetcher(BaseFetcher):
    """
    Fetches data from Binance Futures API.
    
    Key data:
    - OHLCV (klines) at all timeframes
    - Mark price and funding rate
    - Open interest
    - Long/short ratios (backup to CoinGlass)
    - Orderbook depth
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["binance_futures"],
            rate_limit=RATE_LIMITS["binance"],
            api_key=None  # Public endpoints
        )
        self.symbol = "BTCUSDT"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._get("/fapi/v1/ping")
            logger.info("Binance connection test: SUCCESS")
            return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # KLINES (OHLCV) DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_klines(
        self, 
        interval: str = "1h", 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1500
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV klines data.
        
        Args:
            interval: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
            start_time: Start datetime
            end_time: End datetime
            limit: Max 1500 candles per request
        """
        try:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = timestamp_to_ms(start_time)
            if end_time:
                params["endTime"] = timestamp_to_ms(end_time)
            
            result = self._get(BINANCE_ENDPOINTS["klines"], params)
            
            if not result:
                return None
            
            df = pd.DataFrame(result, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume",
                "taker_buy_quote_volume", "ignore"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume", "quote_volume", 
                       "taker_buy_volume", "taker_buy_quote_volume"]:
                df[col] = df[col].astype(float)
            df["trades"] = df["trades"].astype(int)
            
            # Keep only essential columns
            df = df[[
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"
            ]]
            
            # Derive taker sell
            df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
            df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-10)
            
            logger.debug(f"Fetched {len(df)} {interval} klines")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return None
    
    def get_klines_extended(
        self,
        interval: str = "1h",
        days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Fetch extended history by making multiple requests.
        
        Args:
            interval: Timeframe
            days: Number of days of history
        """
        logger.info(f"Fetching {days} days of {interval} klines...")
        
        # Calculate interval in milliseconds
        interval_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000,
            "1h": 3600000, "2h": 7200000, "4h": 14400000, "6h": 21600000,
            "8h": 28800000, "12h": 43200000, "1d": 86400000
        }
        
        if interval not in interval_ms:
            raise ValueError(f"Unknown interval: {interval}")
        
        ms_per_candle = interval_ms[interval]
        candles_needed = int((days * 86400000) / ms_per_candle) + 1
        
        all_data = []
        end_time = datetime.utcnow()
        
        while candles_needed > 0:
            batch_size = min(1500, candles_needed)
            
            df = self.get_klines(
                interval=interval,
                end_time=end_time,
                limit=batch_size
            )
            
            if df is None or len(df) == 0:
                break
            
            all_data.append(df)
            candles_needed -= len(df)
            
            # Move end_time to before the earliest candle we got
            end_time = df["timestamp"].min() - timedelta(milliseconds=1)
            
            logger.debug(f"  Fetched batch, {candles_needed} candles remaining")
        
        if not all_data:
            return None
        
        # Combine and deduplicate
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Total: {len(combined)} {interval} candles fetched")
        return combined
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FUNDING RATE DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_funding_rate(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical funding rates"""
        try:
            params = {"symbol": self.symbol, "limit": limit}
            result = self._get(BINANCE_ENDPOINTS["funding_rate"], params)
            
            if not result:
                return None
            
            df = pd.DataFrame(result)
            df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["funding_rate"] = df["fundingRate"].astype(float)
            df["mark_price"] = df["markPrice"].astype(float)
            
            df = df[["timestamp", "funding_rate", "mark_price"]]
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} funding rate records from Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN INTEREST DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_open_interest(self) -> Optional[Dict]:
        """Get current open interest"""
        try:
            result = self._get(
                BINANCE_ENDPOINTS["open_interest"],
                params={"symbol": self.symbol}
            )
            
            return {
                "timestamp": datetime.utcnow(),
                "oi_contracts": float(result.get("openInterest", 0)),
                "symbol": result.get("symbol", self.symbol)
            }
            
        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DEPRECATED DERIVATIVES METHODS
    # CoinGlass provides same data aggregated across all exchanges
    # These methods are kept for backward compatibility but return None
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_open_interest_history(
        self,
        period: str = "4h",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        """DEPRECATED: Use CoinGlass oi_history instead."""
        logger.warning("Binance OI history deprecated - use CoinGlass oi_history")
        return None
    
    def get_top_trader_long_short(
        self,
        period: str = "4h",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        """DEPRECATED: Use CoinGlass top_ls_history instead."""
        logger.warning("Binance top L/S deprecated - use CoinGlass top_ls_history")
        return None
    
    def get_global_long_short(
        self,
        period: str = "4h",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        """DEPRECATED: Use CoinGlass ls_history instead."""
        logger.warning("Binance global L/S deprecated - use CoinGlass ls_history")
        return None
    
    def get_taker_long_short_ratio(
        self,
        period: str = "4h",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        """DEPRECATED: Use CoinGlass taker_history instead."""
        logger.warning("Binance taker L/S deprecated - use CoinGlass taker_history")
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ORDERBOOK DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_orderbook(self, limit: int = 100) -> Optional[Dict]:
        """
        Get current orderbook snapshot.
        
        Args:
            limit: 5, 10, 20, 50, 100, 500, 1000
        """
        try:
            result = self._get(
                BINANCE_ENDPOINTS["depth"],
                params={"symbol": self.symbol, "limit": limit}
            )
            
            bids = np.array([[float(p), float(q)] for p, q in result["bids"]])
            asks = np.array([[float(p), float(q)] for p, q in result["asks"]])
            
            # Calculate metrics
            bid_volume = bids[:, 1].sum()
            ask_volume = asks[:, 1].sum()
            
            # Imbalance: positive = more bids (bullish), negative = more asks (bearish)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Depth at various levels
            mid_price = (bids[0, 0] + asks[0, 0]) / 2
            
            return {
                "timestamp": datetime.utcnow(),
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "imbalance": imbalance,
                "spread": (asks[0, 0] - bids[0, 0]) / mid_price,
                "mid_price": mid_price,
                "best_bid": bids[0, 0],
                "best_ask": asks[0, 0],
            }
            
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINED FETCH
    # ═══════════════════════════════════════════════════════════════════════════
    
    def fetch_all_historical(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available historical data from Binance.
        
        Args:
            days: Number of days of history
        """
        logger.info(f"Fetching all Binance data for past {days} days...")
        
        data = {}
        
        # OHLCV at multiple timeframes
        for tf in ["1h", "4h", "1d"]:
            data[f"klines_{tf}"] = self.get_klines_extended(interval=tf, days=days)
        
        # Funding rate (limited history from Binance)
        data["funding"] = self.get_funding_rate(limit=1000)
        
        # These have 30 day limits on Binance
        binance_days = min(days, 30)
        
        # Open interest
        data["open_interest"] = self.get_open_interest_history(days=binance_days)
        
        # Long/Short ratios
        data["ls_top_traders"] = self.get_top_trader_long_short(days=binance_days)
        data["ls_global"] = self.get_global_long_short(days=binance_days)
        
        # Taker
        data["taker"] = self.get_taker_long_short_ratio(days=binance_days)
        
        # Current snapshots
        data["current_oi"] = pd.DataFrame([self.get_open_interest()]) if self.get_open_interest() else None
        data["current_orderbook"] = pd.DataFrame([self.get_orderbook()]) if self.get_orderbook() else None
        
        # Log summary
        for name, df in data.items():
            if df is not None and isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: {len(df)} records")
            else:
                logger.warning(f"  {name}: NO DATA")
        
        return data


if __name__ == "__main__":
    # Test the fetcher
    fetcher = BinanceFetcher()
    
    print("Testing Binance API connection...")
    if fetcher.test_connection():
        print("[OK] Connection successful")
        
        print("\nFetching sample klines...")
        klines = fetcher.get_klines(interval="1h", limit=10)
        if klines is not None:
            print(f"\nKlines sample:\n{klines.head()}")
        
        print("\nFetching orderbook...")
        ob = fetcher.get_orderbook()
        if ob:
            print(f"\nOrderbook: {ob}")
    else:
        print("[X] Connection failed")
