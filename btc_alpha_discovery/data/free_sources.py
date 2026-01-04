"""
Free Sources Data Fetcher
========================

Alternative.me - Fear & Greed Index
Deribit - Options data (BTC)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BaseFetcher
from config import URLS

logger = logging.getLogger(__name__)


class AlternativeMeFetcher(BaseFetcher):
    """
    Fetches Fear & Greed Index from Alternative.me.
    
    Index ranges from 0-100:
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["alternative_me"],
            rate_limit=30,  # Conservative
            api_key=None
        )
    
    def get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._get("/fng/", params={"limit": 1})
            success = "data" in result
            logger.info(f"Alternative.me connection test: {'SUCCESS' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"Alternative.me connection test failed: {e}")
            return False
    
    def get_fear_greed(self, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Get Fear & Greed Index history.
        
        Args:
            days: Number of days of history (max ~365)
        """
        try:
            result = self._get("/fng/", params={"limit": min(days, 500)})
            
            if not result.get("data"):
                return None
            
            records = []
            for item in result["data"]:
                records.append({
                    "timestamp": datetime.fromtimestamp(int(item["timestamp"])),
                    "fear_greed_value": int(item["value"]),
                    "fear_greed_class": item["value_classification"]
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Add derived features
            df["fg_extreme_fear"] = df["fear_greed_value"] < 25
            df["fg_extreme_greed"] = df["fear_greed_value"] > 75
            df["fg_fear"] = df["fear_greed_value"] < 45
            df["fg_greed"] = df["fear_greed_value"] > 55
            
            logger.info(f"Fetched {len(df)} Fear & Greed observations")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
            return None


class DeribitFetcher(BaseFetcher):
    """
    Fetches options data from Deribit.
    
    Key data:
    - Implied volatility
    - Put/Call ratio
    - Options open interest
    - Max pain calculation
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["deribit"],
            rate_limit=60,
            api_key=None
        )
        self.currency = "BTC"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._get("/get_index_price", params={"index_name": "btc_usd"})
            success = "result" in result
            logger.info(f"Deribit connection test: {'SUCCESS' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"Deribit connection test failed: {e}")
            return False
    
    def get_index_price(self) -> Optional[Dict]:
        """Get current BTC index price"""
        try:
            result = self._get("/get_index_price", params={"index_name": "btc_usd"})
            
            if not result.get("result"):
                return None
            
            return {
                "timestamp": datetime.utcnow(),
                "index_price": result["result"]["index_price"],
                "estimated_delivery_price": result["result"].get("estimated_delivery_price")
            }
            
        except Exception as e:
            logger.error(f"Error fetching index price: {e}")
            return None
    
    def get_historical_volatility(self) -> Optional[pd.DataFrame]:
        """Get historical volatility"""
        try:
            result = self._get(
                "/get_historical_volatility",
                params={"currency": self.currency}
            )
            
            if not result.get("result"):
                return None
            
            records = []
            for item in result["result"]:
                records.append({
                    "timestamp": datetime.fromtimestamp(item[0] / 1000),
                    "historical_volatility": item[1]
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} historical volatility observations")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical volatility: {e}")
            return None
    
    def get_instruments(self, kind: str = "option") -> Optional[pd.DataFrame]:
        """
        Get available instruments.
        
        Args:
            kind: future, option, spot, future_combo, option_combo
        """
        try:
            result = self._get(
                "/get_instruments",
                params={"currency": self.currency, "kind": kind}
            )
            
            if not result.get("result"):
                return None
            
            df = pd.DataFrame(result["result"])
            
            # Filter to relevant columns
            cols = ["instrument_name", "strike", "option_type", "expiration_timestamp",
                    "creation_timestamp", "is_active"]
            df = df[[c for c in cols if c in df.columns]]
            
            if "expiration_timestamp" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration_timestamp"], unit="ms")
            
            logger.info(f"Fetched {len(df)} {kind} instruments")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return None
    
    def get_book_summary(self, kind: str = "option") -> Optional[pd.DataFrame]:
        """
        Get book summary for all instruments.
        
        Contains:
        - Open interest
        - Mark price
        - Mark IV (implied volatility)
        - Bid/Ask IV
        """
        try:
            result = self._get(
                "/get_book_summary_by_currency",
                params={"currency": self.currency, "kind": kind}
            )
            
            if not result.get("result"):
                return None
            
            df = pd.DataFrame(result["result"])
            
            # Keep essential columns
            cols = ["instrument_name", "open_interest", "mark_price", "mark_iv",
                    "bid_price", "ask_price", "underlying_price", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            
            df["timestamp"] = datetime.utcnow()
            
            logger.info(f"Fetched book summary for {len(df)} {kind} instruments")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching book summary: {e}")
            return None
    
    def calculate_options_metrics(self) -> Optional[Dict]:
        """
        Calculate aggregated options metrics:
        - Total call OI vs put OI
        - Put/Call ratio
        - Average implied volatility
        - Max pain estimate
        """
        try:
            # Get all options
            options = self.get_book_summary(kind="option")
            if options is None or len(options) == 0:
                return None
            
            # Parse instrument names to get option type and strike
            def parse_instrument(name):
                parts = name.split("-")
                if len(parts) >= 4:
                    return {
                        "strike": float(parts[2]),
                        "option_type": parts[3]  # C or P
                    }
                return None
            
            parsed = options["instrument_name"].apply(parse_instrument)
            options["strike"] = parsed.apply(lambda x: x["strike"] if x else None)
            options["option_type"] = parsed.apply(lambda x: x["option_type"] if x else None)
            
            # Filter valid options
            options = options.dropna(subset=["strike", "option_type", "open_interest"])
            
            # Separate calls and puts
            calls = options[options["option_type"] == "C"]
            puts = options[options["option_type"] == "P"]
            
            # Calculate metrics
            call_oi = calls["open_interest"].sum()
            put_oi = puts["open_interest"].sum()
            
            put_call_ratio = put_oi / (call_oi + 1e-10)
            
            # Average IV (volume-weighted if available)
            if "mark_iv" in options.columns and "open_interest" in options.columns:
                weights = options["open_interest"]
                avg_iv = (options["mark_iv"] * weights).sum() / (weights.sum() + 1e-10)
            else:
                avg_iv = options["mark_iv"].mean() if "mark_iv" in options.columns else None
            
            # Max pain calculation (simplified)
            # Max pain = strike where total option premium is minimized
            strikes = options["strike"].unique()
            underlying = options["underlying_price"].iloc[0] if "underlying_price" in options.columns else None
            
            max_pain = None
            if underlying and len(strikes) > 0:
                # Simplified: use the strike closest to current price with highest OI
                max_pain = options.groupby("strike")["open_interest"].sum().idxmax()
            
            return {
                "timestamp": datetime.utcnow(),
                "call_oi": call_oi,
                "put_oi": put_oi,
                "put_call_ratio": put_call_ratio,
                "avg_iv": avg_iv,
                "max_pain": max_pain,
                "underlying_price": underlying,
                "total_options": len(options)
            }
            
        except Exception as e:
            logger.error(f"Error calculating options metrics: {e}")
            return None


class FreeSources:
    """Combined fetcher for all free sources"""
    
    def __init__(self):
        self.alternative_me = AlternativeMeFetcher()
        self.deribit = DeribitFetcher()
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all free source connections"""
        return {
            "alternative_me": self.alternative_me.test_connection(),
            "deribit": self.deribit.test_connection()
        }
    
    def fetch_all(self, days: int = 365) -> Dict[str, Any]:
        """Fetch all available free data"""
        logger.info("Fetching all free source data...")
        
        data = {}
        
        # Fear & Greed
        data["fear_greed"] = self.alternative_me.get_fear_greed(days=days)
        
        # Deribit options
        data["historical_vol"] = self.deribit.get_historical_volatility()
        data["options_metrics"] = self.deribit.calculate_options_metrics()
        data["options_summary"] = self.deribit.get_book_summary(kind="option")
        
        # Log summary
        for name, df in data.items():
            if df is not None:
                if isinstance(df, pd.DataFrame):
                    logger.info(f"  {name}: {len(df)} records")
                else:
                    logger.info(f"  {name}: dict data available")
            else:
                logger.warning(f"  {name}: NO DATA")
        
        return data


if __name__ == "__main__":
    # Test the fetchers
    print("Testing free source connections...")
    
    free = FreeSources()
    connections = free.test_all_connections()
    
    for source, status in connections.items():
        print(f"  {source}: {'[OK]' if status else '[X]'}")
    
    print("\nFetching sample data...")
    
    # Fear & Greed
    fg = free.alternative_me.get_fear_greed(days=30)
    if fg is not None:
        print(f"\nFear & Greed sample:\n{fg.tail()}")
    
    # Options metrics
    metrics = free.deribit.calculate_options_metrics()
    if metrics:
        print(f"\nOptions metrics:\n{metrics}")
