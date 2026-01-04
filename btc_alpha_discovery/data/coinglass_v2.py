"""
CoinGlass Data Fetcher v2 - ROBUST VERSION
==========================================

Fixes from v1:
1. 3+ second delays between ALL requests
2. Exponential backoff: 5s → 10s → 20s → 40s → 80s → 120s
3. Up to 10 retries per endpoint
4. Pagination for historical data (7-day chunks)
5. Better error handling and logging
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_KEYS

logger = logging.getLogger(__name__)

# Constants
COINGLASS_BASE_URL = "https://open-api.coinglass.com/public/v2"
COINGLASS_API_KEY = API_KEYS.get("coinglass", "fb3534925c3e45e69ec3d1da7e26efe6")

# Rate limiting - CONSERVATIVE
MIN_DELAY_BETWEEN_REQUESTS = 3.0  # 3 seconds minimum
MAX_RETRIES = 10
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 120.0


class CoinGlassFetcherV2:
    """
    Robust CoinGlass data fetcher with aggressive retry logic.
    
    Key improvements:
    - 3+ second delays between requests
    - Exponential backoff up to 120 seconds
    - 10 retries per endpoint
    - Pagination for historical data
    """
    
    def __init__(self):
        self.base_url = COINGLASS_BASE_URL
        self.api_key = COINGLASS_API_KEY
        self.symbol = "BTC"
        self.last_request_time = 0
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "CG-API-KEY": self.api_key  # Try both header formats
        })
    
    def _wait_for_rate_limit(self):
        """Ensure minimum delay between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
            wait_time = MIN_DELAY_BETWEEN_REQUESTS - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _request_with_retry(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        max_retries: int = MAX_RETRIES
    ) -> Optional[Dict]:
        """
        Make request with aggressive retry logic.
        
        Backoff: 5s → 10s → 20s → 40s → 80s → 120s → 120s → ...
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                # Try both header formats
                headers = {
                    "accept": "application/json",
                    "CG-API-KEY": self.api_key,
                    "coinglassSecret": self.api_key
                }
                
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") or "data" in data:
                        return data
                    else:
                        logger.warning(f"CoinGlass API returned success=false: {data.get('msg', 'unknown')}")
                        return None
                
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code in [500, 502, 503, 504]:
                    # Server error - retry with backoff
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                    logger.warning(f"Server error ({response.status_code}). Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 404:
                    # Endpoint doesn't exist
                    logger.warning(f"Endpoint not found: {endpoint}")
                    return None
                    
                elif response.status_code == 403:
                    # Forbidden - likely API key issue or endpoint requires higher tier
                    logger.warning(f"Forbidden (403) for {endpoint} - may require higher API tier")
                    return None
                
                else:
                    logger.warning(f"Unexpected status {response.status_code} for {endpoint}")
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                logger.warning(f"Timeout. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            except requests.exceptions.RequestException as e:
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                logger.warning(f"Request error: {e}. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
        
        logger.error(f"Failed to fetch {endpoint} after {max_retries} attempts")
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            result = self._request_with_retry("/funding", params={"symbol": self.symbol}, max_retries=3)
            success = result is not None and ("data" in result or result.get("success"))
            logger.info(f"CoinGlass connection test: {'SUCCESS' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"CoinGlass connection test failed: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDATION DATA (THE EDGE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_liquidation_history_paginated(self, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch liquidation history with pagination (7-day chunks).
        
        This is THE KEY data for finding liquidation levels.
        """
        logger.info(f"Fetching liquidation history ({days} days) with pagination...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        # Fetch in 7-day chunks to avoid rate limits
        chunk_days = 7
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/liquidation_history",
                params={
                    "symbol": self.symbol,
                    "interval": "h4",  # Hobbyist plan requires 4h+
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                data = result["data"]
                for item in data:
                    try:
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "long_liq_usd": float(item.get("longLiquidationUsd", 0)),
                            "short_liq_usd": float(item.get("shortLiquidationUsd", 0)),
                            "long_liq_vol": float(item.get("longLiquidationVol", 0)),
                            "short_liq_vol": float(item.get("shortLiquidationVol", 0)),
                        })
                    except (TypeError, ValueError) as e:
                        logger.debug(f"Error parsing liquidation record: {e}")
                        continue
                
                logger.info(f"  Chunk {datetime.fromtimestamp(current_start).date()} to {datetime.fromtimestamp(current_end).date()}: {len(data)} records")
            else:
                logger.warning(f"  No data for chunk {datetime.fromtimestamp(current_start).date()}")
            
            current_start = current_end
            time.sleep(1)  # Extra delay between chunks
        
        if not all_records:
            logger.warning("No liquidation history data retrieved")
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        
        # Derive features
        df["total_liq_usd"] = df["long_liq_usd"] + df["short_liq_usd"]
        df["liq_ratio"] = df["long_liq_usd"] / (df["short_liq_usd"] + 1)
        df["net_liq"] = df["long_liq_usd"] - df["short_liq_usd"]
        
        logger.info(f"Fetched {len(df)} total liquidation history records")
        return df
    
    def get_liquidation_info(self) -> Optional[Dict]:
        """Get current liquidation levels (THE REAL EDGE)"""
        result = self._request_with_retry("/liquidation_info", params={"symbol": self.symbol})
        if result and result.get("data"):
            logger.info("Fetched current liquidation info")
            return result["data"]
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FUNDING RATE DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_funding_history_paginated(self, interval: str = "h8", days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch funding rate history with pagination"""
        logger.info(f"Fetching funding history ({days} days)...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        chunk_days = 14  # 2 week chunks for h8 data
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/funding_usd_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                for item in result["data"]:
                    try:
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "funding_rate": float(item.get("c", 0)),
                            "funding_open": float(item.get("o", 0)),
                            "funding_high": float(item.get("h", 0)),
                            "funding_low": float(item.get("l", 0)),
                        })
                    except (TypeError, ValueError):
                        continue
            
            current_start = current_end
            time.sleep(0.5)
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} funding history records")
        return df
    
    def get_funding_rates(self) -> Optional[pd.DataFrame]:
        """Get current funding rates across exchanges"""
        result = self._request_with_retry("/funding", params={"symbol": self.symbol})
        if not result or not result.get("data"):
            return None
        
        records = []
        for item in result["data"]:
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN INTEREST DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_open_interest_history_paginated(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch OI history with pagination"""
        logger.info(f"Fetching OI history ({days} days, {interval} interval)...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        chunk_days = 7
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/open_interest_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                for item in result["data"]:
                    try:
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "oi_usd": float(item.get("c", 0)),
                            "oi_open": float(item.get("o", 0)),
                            "oi_high": float(item.get("h", 0)),
                            "oi_low": float(item.get("l", 0)),
                        })
                    except (TypeError, ValueError):
                        continue
            
            current_start = current_end
            time.sleep(0.5)
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} OI history records")
        return df
    
    def get_open_interest(self) -> Optional[pd.DataFrame]:
        """Get current OI across exchanges"""
        result = self._request_with_retry("/open_interest", params={"symbol": self.symbol})
        if not result or not result.get("data"):
            return None
        
        records = []
        for item in result["data"]:
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LONG/SHORT RATIO DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_long_short_history_paginated(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch L/S account ratio history with pagination"""
        logger.info(f"Fetching L/S ratio history ({days} days, {interval} interval)...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        chunk_days = 7
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/global_long_short_account_ratio",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                for item in result["data"]:
                    try:
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "long_account_ratio": float(item.get("longAccount", 0)),
                            "short_account_ratio": float(item.get("shortAccount", 0)),
                            "long_short_ratio": float(item.get("longShortRatio", 1)),
                        })
                    except (TypeError, ValueError):
                        continue
            
            current_start = current_end
            time.sleep(0.5)
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} L/S ratio history records")
        return df
    
    def get_top_trader_long_short_history(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch top trader L/S ratio history (SMART MONEY PROXY)"""
        logger.info(f"Fetching top trader L/S history ({days} days, {interval} interval)...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        chunk_days = 7
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/top_long_short_account_ratio",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                for item in result["data"]:
                    try:
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "top_long_ratio": float(item.get("longAccount", 0)),
                            "top_short_ratio": float(item.get("shortAccount", 0)),
                            "top_ls_ratio": float(item.get("longShortRatio", 1)),
                        })
                    except (TypeError, ValueError):
                        continue
            
            current_start = current_end
            time.sleep(0.5)
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} top trader L/S records")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAKER BUY/SELL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_taker_history_paginated(self, interval: str = "h4", days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch taker buy/sell history with pagination"""
        logger.info(f"Fetching taker buy/sell history ({days} days, {interval} interval)...")
        
        all_records = []
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - (days * 86400)
        
        chunk_days = 7
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + (chunk_days * 86400), end_time)
            
            result = self._request_with_retry(
                "/taker_buy_sell_volume_history",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end
                }
            )
            
            if result and result.get("data"):
                for item in result["data"]:
                    try:
                        buy_vol = float(item.get("buyVol", 0))
                        sell_vol = float(item.get("sellVol", 0))
                        all_records.append({
                            "timestamp": datetime.fromtimestamp(item.get("t", 0) / 1000),
                            "buy_volume": buy_vol,
                            "sell_volume": sell_vol,
                            "buy_sell_ratio": buy_vol / (sell_vol + 1),
                            "net_taker_flow": buy_vol - sell_vol,
                        })
                    except (TypeError, ValueError):
                        continue
            
            current_start = current_end
            time.sleep(0.5)
        
        if not all_records:
            return None
        
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} taker history records")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINED FETCH
    # ═══════════════════════════════════════════════════════════════════════════
    
    def fetch_all_historical(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch ALL available historical data with robust retry logic.
        
        Returns dict with all dataframes and row counts.
        """
        logger.info(f"{'='*60}")
        logger.info(f"FETCHING ALL COINGLASS DATA ({days} days)")
        logger.info(f"{'='*60}")
        
        data = {}
        
        # 1. Liquidation history (THE EDGE)
        logger.info("\n[1/6] Liquidation History...")
        data["liquidation_history"] = self.get_liquidation_history_paginated(days=days)
        
        # 2. Funding history
        logger.info("\n[2/6] Funding History...")
        data["funding_history"] = self.get_funding_history_paginated(days=days)
        
        # 3. Open interest history
        logger.info("\n[3/6] Open Interest History...")
        data["oi_history"] = self.get_open_interest_history_paginated(days=days)
        
        # 4. Long/short ratio history
        logger.info("\n[4/6] Long/Short Ratio History...")
        data["ls_history"] = self.get_long_short_history_paginated(days=days)
        
        # 5. Top trader L/S history
        logger.info("\n[5/6] Top Trader L/S History...")
        data["top_ls_history"] = self.get_top_trader_long_short_history(days=days)
        
        # 6. Taker history
        logger.info("\n[6/6] Taker Buy/Sell History...")
        data["taker_history"] = self.get_taker_history_paginated(days=days)
        
        # Current snapshots (always try)
        logger.info("\n[Snapshots]...")
        data["current_funding"] = self.get_funding_rates()
        data["current_oi"] = self.get_open_interest()
        data["current_liq_info"] = None  # Will be dict
        liq_info = self.get_liquidation_info()
        if liq_info:
            data["liquidation_levels"] = liq_info
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("COINGLASS DATA SUMMARY")
        logger.info(f"{'='*60}")
        
        total_rows = 0
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                rows = len(df)
                total_rows += rows
                status = "[OK]" if rows >= 100 else "[LOW]" if rows > 0 else "[FAIL]"
                logger.info(f"  {status} {name}: {rows} rows")
            elif df is not None:
                logger.info(f"  [OK] {name}: dict data")
            else:
                logger.info(f"  [FAIL] {name}: NO DATA")
        
        logger.info(f"\nTotal historical rows: {total_rows}")
        logger.info(f"{'='*60}\n")
        
        return data


# Data validation
def validate_coinglass_data(data: Dict) -> Dict[str, Any]:
    """
    Validate CoinGlass data meets minimum requirements.
    
    Returns validation report.
    """
    MINIMUM_REQUIREMENTS = {
        "liquidation_history": 500,
        "funding_history": 500,
        "oi_history": 500,
        "ls_history": 500,
        "taker_history": 500,
    }
    
    report = {
        "passed": True,
        "failures": [],
        "warnings": [],
        "row_counts": {}
    }
    
    for name, min_rows in MINIMUM_REQUIREMENTS.items():
        df = data.get(name)
        if df is None:
            report["passed"] = False
            report["failures"].append(f"{name}: NO DATA (need {min_rows}+)")
            report["row_counts"][name] = 0
        elif isinstance(df, pd.DataFrame):
            rows = len(df)
            report["row_counts"][name] = rows
            if rows < min_rows:
                report["passed"] = False
                report["failures"].append(f"{name}: {rows} rows (need {min_rows}+)")
            elif rows < min_rows * 2:
                report["warnings"].append(f"{name}: {rows} rows (low but acceptable)")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    fetcher = CoinGlassFetcherV2()
    
    print("Testing CoinGlass API...")
    if fetcher.test_connection():
        print("\n[OK] Connected to CoinGlass")
        
        # Fetch 30 days of data as test
        print("\nFetching 30 days of data as test...")
        data = fetcher.fetch_all_historical(days=30)
        
        # Validate
        report = validate_coinglass_data(data)
        print(f"\nValidation: {'PASSED' if report['passed'] else 'FAILED'}")
        for failure in report["failures"]:
            print(f"  [X] {failure}")
        for warning in report["warnings"]:
            print(f"  [!] {warning}")
    else:
        print("[X] Failed to connect to CoinGlass")
