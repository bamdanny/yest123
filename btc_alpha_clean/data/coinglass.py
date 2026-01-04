"""
CoinGlass Data Fetcher
=====================

Premium liquidation and derivatives data - THIS IS THE EDGE.
Real liquidation levels, not estimates.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BaseFetcher, timestamp_to_ms, ms_to_timestamp
from config import API_KEYS, URLS, RATE_LIMITS, COINGLASS_ENDPOINTS

logger = logging.getLogger(__name__)


class CoinGlassFetcher(BaseFetcher):
    """
    Fetches data from CoinGlass API.
    
    Key data:
    - Liquidation levels (REAL DATA - this is the edge)
    - Funding rates
    - Open Interest
    - Long/Short ratios
    - Taker buy/sell
    - Exchange flows
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["coinglass"],
            rate_limit=RATE_LIMITS["coinglass"],
            api_key=API_KEYS["coinglass"]
        )
        self.symbol = "BTC"
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "coinglassSecret": self.api_key
        }
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._get("/funding", params={"symbol": self.symbol})
            success = result.get("success", False) or "data" in result
            if not success:
                logger.warning(f"CoinGlass API response: {result}")
            logger.info(f"CoinGlass connection test: {'SUCCESS' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"CoinGlass connection test failed: {e}")
            logger.info("CoinGlass connection test: FAILED")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FUNDING RATE DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_funding_rates(self) -> Optional[pd.DataFrame]:
        """Get current funding rates across all exchanges"""
        try:
            result = self._get("/funding", params={"symbol": self.symbol})
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "exchange": item.get("exchangeName", ""),
                    "symbol": item.get("symbol", ""),
                    "funding_rate": float(item.get("rate", 0)),
                    "next_funding_time": item.get("nextFundingTime"),
                    "timestamp": datetime.utcnow()
                })
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched funding rates from {len(df)} exchanges")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return None
    
    def get_funding_history(self, interval: str = "h8", days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get historical funding rates.
        
        Args:
            interval: h8 (8 hour), h4 (4 hour), h1 (1 hour)
            days: Number of days of history
        """
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (days * 86400)
            
            result = self._get(
                "/funding_usd_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                    "funding_rate": float(item.get("c", 0)),
                    "funding_rate_open": float(item.get("o", 0)),
                    "funding_rate_high": float(item.get("h", 0)),
                    "funding_rate_low": float(item.get("l", 0)),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Fetched {len(df)} funding rate history points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding history: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN INTEREST DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_open_interest(self) -> Optional[pd.DataFrame]:
        """Get current open interest across exchanges"""
        try:
            result = self._get("/open_interest", params={"symbol": self.symbol})
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "exchange": item.get("exchangeName", ""),
                    "oi_usd": float(item.get("openInterest", 0)),
                    "oi_btc": float(item.get("openInterestAmount", 0)),
                    "h24_change": float(item.get("h24Change", 0)),
                    "timestamp": datetime.utcnow()
                })
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched OI from {len(df)} exchanges, total: ${df['oi_usd'].sum():,.0f}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            return None
    
    def get_open_interest_history(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get historical open interest.
        
        Args:
            interval: m5, m15, h1, h4, h12, h24
            days: Number of days of history
        """
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (days * 86400)
            
            result = self._get(
                "/open_interest_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                    "oi_usd": float(item.get("c", 0)),
                    "oi_open": float(item.get("o", 0)),
                    "oi_high": float(item.get("h", 0)),
                    "oi_low": float(item.get("l", 0)),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Fetched {len(df)} OI history points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OI history: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDATION DATA (THE EDGE!)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_liquidation_history(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get historical liquidation data.
        
        This is REAL liquidation data, not estimates.
        """
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (days * 86400)
            
            result = self._get(
                "/liquidation_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                    "long_liq_usd": float(item.get("longLiquidationUsd", 0)),
                    "short_liq_usd": float(item.get("shortLiquidationUsd", 0)),
                    "total_liq_usd": float(item.get("liquidationUsd", 0)),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Derive features
            df["liq_ratio"] = df["long_liq_usd"] / (df["short_liq_usd"] + 1)  # +1 to avoid div by zero
            df["net_liq"] = df["long_liq_usd"] - df["short_liq_usd"]
            
            logger.info(f"Fetched {len(df)} liquidation history points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching liquidation history: {e}")
            return None
    
    def get_aggregated_liquidations(self, hours: int = 24) -> Optional[Dict]:
        """Get aggregated liquidation stats for past N hours"""
        try:
            result = self._get(
                "/liquidation_aggregated_history",
                params={
                    "symbol": self.symbol,
                    "range": hours
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            aggregated = {
                "long_liq_total": sum(float(x.get("longLiquidationUsd", 0)) for x in data),
                "short_liq_total": sum(float(x.get("shortLiquidationUsd", 0)) for x in data),
                "timestamp": datetime.utcnow(),
                "hours": hours
            }
            aggregated["total_liq"] = aggregated["long_liq_total"] + aggregated["short_liq_total"]
            aggregated["liq_ratio"] = aggregated["long_liq_total"] / (aggregated["short_liq_total"] + 1)
            
            logger.info(f"Aggregated {hours}h liquidations: Long ${aggregated['long_liq_total']:,.0f}, Short ${aggregated['short_liq_total']:,.0f}")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error fetching aggregated liquidations: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LONG/SHORT RATIO DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_long_short_ratio(self) -> Optional[pd.DataFrame]:
        """Get current long/short ratio across exchanges"""
        try:
            result = self._get("/long_short_ratio", params={"symbol": self.symbol})
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "exchange": item.get("exchangeName", ""),
                    "long_ratio": float(item.get("longRate", 0)),
                    "short_ratio": float(item.get("shortRate", 0)),
                    "long_short_ratio": float(item.get("longShortRatio", 1)),
                    "timestamp": datetime.utcnow()
                })
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched L/S ratio from {len(df)} exchanges")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching long/short ratio: {e}")
            return None
    
    def get_global_long_short(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Get historical global account long/short ratio"""
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (days * 86400)
            
            result = self._get(
                "/global_long_short_account_ratio",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                    "long_account_ratio": float(item.get("longAccount", 0)),
                    "short_account_ratio": float(item.get("shortAccount", 0)),
                    "long_short_ratio": float(item.get("longShortRatio", 1)),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Fetched {len(df)} global L/S ratio history points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching global L/S ratio: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAKER BUY/SELL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_taker_buy_sell_history(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Get historical taker buy/sell volume ratio"""
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (days * 86400)
            
            result = self._get(
                "/taker_buy_sell_volume_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            )
            
            if not result.get("data"):
                return None
            
            data = result["data"]
            records = []
            
            for item in data:
                records.append({
                    "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                    "buy_volume": float(item.get("buyVol", 0)),
                    "sell_volume": float(item.get("sellVol", 0)),
                    "buy_sell_ratio": float(item.get("buySellRatio", 1)),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Derive features
            df["net_taker_flow"] = df["buy_volume"] - df["sell_volume"]
            df["taker_delta_pct"] = df["net_taker_flow"] / (df["buy_volume"] + df["sell_volume"] + 1)
            
            logger.info(f"Fetched {len(df)} taker buy/sell history points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching taker buy/sell history: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCHANGE FLOW DATA (May require higher tier)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_exchange_balance(self) -> Optional[Dict]:
        """Get current BTC balance on exchanges"""
        try:
            result = self._get("/exchange_balance", params={"symbol": self.symbol})
            if not result.get("data"):
                return None
            
            data = result["data"]
            logger.info(f"Exchange balance data retrieved")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching exchange balance: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINED FETCH FOR ALL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def fetch_all_historical(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available historical data from CoinGlass.
        
        Returns dict with all dataframes.
        """
        logger.info(f"Fetching all CoinGlass data for past {days} days...")
        
        data = {}
        
        # Funding rates
        data["funding"] = self.get_funding_history(days=days)
        
        # Open Interest
        data["open_interest"] = self.get_open_interest_history(days=days)
        
        # Liquidations
        data["liquidations"] = self.get_liquidation_history(days=days)
        
        # Long/Short ratio
        data["long_short"] = self.get_global_long_short(days=days)
        
        # Taker buy/sell
        data["taker"] = self.get_taker_buy_sell_history(days=days)
        
        # Current snapshots
        data["current_funding"] = self.get_funding_rates()
        data["current_oi"] = self.get_open_interest()
        data["current_ls"] = self.get_long_short_ratio()
        
        # Log summary
        for name, df in data.items():
            if df is not None and isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: {len(df)} records")
            elif df is not None:
                logger.info(f"  {name}: dict data available")
            else:
                logger.warning(f"  {name}: NO DATA")
        
        return data


if __name__ == "__main__":
    # Test the fetcher
    fetcher = CoinGlassFetcher()
    
    print("Testing CoinGlass API connection...")
    if fetcher.test_connection():
        print("[OK] Connection successful")
        
        print("\nFetching sample data...")
        funding = fetcher.get_funding_rates()
        if funding is not None:
            print(f"\nFunding rates sample:\n{funding.head()}")
        
        oi = fetcher.get_open_interest()
        if oi is not None:
            print(f"\nOpen Interest sample:\n{oi.head()}")
    else:
        print("[X] Connection failed")
