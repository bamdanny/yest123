"""
Polygon Data Fetcher
===================

US equities, indices, and macro data.
VIX, DXY, SPY, QQQ, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BaseFetcher
from config import API_KEYS, URLS, RATE_LIMITS, POLYGON_TICKERS

logger = logging.getLogger(__name__)


class PolygonFetcher(BaseFetcher):
    """
    Fetches data from Polygon.io API.
    
    Key data:
    - VIX (volatility index)
    - DXY (dollar index)
    - SPY, QQQ, IWM (equity indices)
    - TLT, HYG (bonds)
    - GLD, USO (commodities)
    - MSTR, COIN, BITO (crypto proxies)
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["polygon"],
            rate_limit=RATE_LIMITS["polygon"],  # 5/min free tier
            api_key=API_KEYS["polygon"]
        )
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._get("/v1/marketstatus/now", params={"apiKey": self.api_key})
            logger.info("Polygon connection test: SUCCESS")
            return True
        except Exception as e:
            logger.error(f"Polygon connection test failed: {e}")
            return False
    
    def get_aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = None,
        to_date: str = None,
        limit: int = 5000
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV aggregates for a ticker.
        
        Args:
            ticker: Ticker symbol (use POLYGON_TICKERS values)
            multiplier: Size of the time window
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max results
        """
        try:
            if from_date is None:
                from_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
            if to_date is None:
                to_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
                "apiKey": self.api_key
            }
            
            result = self._get(endpoint, params)
            
            if not result.get("results"):
                logger.warning(f"No data returned for {ticker}")
                return None
            
            df = pd.DataFrame(result["results"])
            
            # Rename columns
            column_map = {
                "t": "timestamp_ms",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions"
            }
            df = df.rename(columns=column_map)
            
            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
            
            # Select columns (some may not exist)
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in ["vwap", "transactions"]:
                if col in df.columns:
                    cols.append(col)
            
            df = df[cols]
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} {timespan} bars for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def get_previous_close(self, ticker: str) -> Optional[Dict]:
        """Get previous day's close"""
        try:
            result = self._get(
                f"/v2/aggs/ticker/{ticker}/prev",
                params={"adjusted": "true", "apiKey": self.api_key}
            )
            
            if not result.get("results"):
                return None
            
            data = result["results"][0]
            return {
                "ticker": ticker,
                "timestamp": datetime.fromtimestamp(data["t"] / 1000),
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"]
            }
            
        except Exception as e:
            logger.error(f"Error fetching prev close for {ticker}: {e}")
            return None
    
    def get_market_status(self) -> Optional[Dict]:
        """Get current market status"""
        try:
            result = self._get("/v1/marketstatus/now", params={"apiKey": self.api_key})
            return {
                "market": result.get("market"),
                "exchanges": result.get("exchanges", {}),
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return None
    
    def fetch_all_tickers(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch all configured tickers.
        
        Note: Free tier has 5 calls/min limit, so this will be slow.
        """
        logger.info(f"Fetching {len(POLYGON_TICKERS)} tickers for past {days} days...")
        
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        data = {}
        
        for name, ticker in POLYGON_TICKERS.items():
            logger.info(f"  Fetching {name} ({ticker})...")
            
            df = self.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=from_date,
                to_date=to_date
            )
            
            if df is not None:
                data[name.lower()] = df
                logger.info(f"    {name}: {len(df)} records")
            else:
                logger.warning(f"    {name}: NO DATA")
        
        return data
    
    def get_vix(self, days: int = 365) -> Optional[pd.DataFrame]:
        """Get VIX specifically (commonly needed)"""
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.get_aggregates(
            ticker=POLYGON_TICKERS["VIX"],
            multiplier=1,
            timespan="day",
            from_date=from_date
        )
    
    def get_dxy(self, days: int = 365) -> Optional[pd.DataFrame]:
        """Get DXY specifically (commonly needed)"""
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.get_aggregates(
            ticker=POLYGON_TICKERS["DXY"],
            multiplier=1,
            timespan="day",
            from_date=from_date
        )


if __name__ == "__main__":
    # Test the fetcher
    fetcher = PolygonFetcher()
    
    print("Testing Polygon API connection...")
    if fetcher.test_connection():
        print("[OK] Connection successful")
        
        print("\nFetching VIX data...")
        vix = fetcher.get_vix(days=30)
        if vix is not None:
            print(f"\nVIX sample:\n{vix.tail()}")
        
        print("\nFetching SPY data...")
        spy = fetcher.get_aggregates(POLYGON_TICKERS["SPY"], timespan="day")
        if spy is not None:
            print(f"\nSPY sample:\n{spy.tail()}")
    else:
        print("[X] Connection failed")
