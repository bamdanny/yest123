"""
CoinGlass V4 Complete Client - Fetches ALL available endpoints for Hobbyist plan.

Endpoints covered:
- Open Interest (history, aggregated)
- Funding Rate (history, OI-weighted, vol-weighted)
- Long/Short Ratio (global, top trader account, top trader position)
- Liquidation (history, aggregated, heatmaps)
- Taker Buy/Sell (history, aggregated)
- Futures Basis (futures vs spot spread)
- Coinbase Premium (US retail demand)
- ETF Flows (institutional demand)
- Options (max pain, put/call)
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_KEYS

logger = logging.getLogger(__name__)

# API Configuration - Try both V3 and V4
COINGLASS_V3_BASE_URL = "https://open-api-v3.coinglass.com"
COINGLASS_V4_BASE_URL = "https://open-api-v4.coinglass.com"
COINGLASS_BASE_URL = "https://open-api.coinglass.com"
COINGLASS_API_KEY = API_KEYS.get("coinglass", "fb3534925c3e45e69ec3d1da7e26efe6")

# Rate limiting
MIN_DELAY = 0.5  # 500ms between requests
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


class CoinGlassV4Fetcher:
    """
    Complete CoinGlass client that fetches ALL available data.
    Tries multiple API versions and endpoint patterns.
    """
    
    def __init__(self):
        self.base_urls = [
            COINGLASS_BASE_URL,
            COINGLASS_V3_BASE_URL,
            COINGLASS_V4_BASE_URL,
        ]
        self.api_key = COINGLASS_API_KEY
        self.symbol = "BTC"
        self.exchange = "Binance"
        self.last_request_time = 0
        
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "CG-API-KEY": self.api_key,
            "coinglassSecret": self.api_key
        })
    
    def _wait_rate_limit(self):
        """Ensure minimum delay between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_DELAY:
            time.sleep(MIN_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Optional[Dict] = None, base_url: Optional[str] = None) -> Optional[Dict]:
        """Make request with retry logic"""
        url = f"{base_url or self.base_urls[0]}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                self._wait_rate_limit()
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "0" or data.get("success") or "data" in data:
                        return data
                    return None
                
                elif response.status_code == 429:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    
                elif response.status_code in [500, 502, 503, 504]:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    time.sleep(wait)
                    
                elif response.status_code in [403, 404]:
                    return None
                    
                else:
                    return None
                    
            except Exception as e:
                wait = BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
        
        return None
    
    def _try_endpoints(self, endpoint_configs: List[tuple]) -> Optional[Dict]:
        """Try multiple endpoint configurations until one works"""
        for base_url, endpoint, params in endpoint_configs:
            result = self._request(endpoint, params=params, base_url=base_url)
            if result and "data" in result:
                return result
        return None
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        for base_url in self.base_urls:
            result = self._request("/api/futures/supported-coins", base_url=base_url)
            if result is not None and "data" in result:
                logger.info("CoinGlass V4 connection: SUCCESS")
                return True
        logger.error("CoinGlass V4 connection: FAILED")
        return False
    
    def fetch_all_data(self, days: int = 90, interval: str = "4h") -> Dict[str, Any]:
        """
        Fetch ALL available CoinGlass data.
        Returns dict with all data sources.
        """
        limit = min(days * 6 if interval == "4h" else days * 24, 1000)
        data = {}
        
        logger.info("=" * 60)
        logger.info(f"FETCHING ALL COINGLASS V4 DATA ({days} days, {interval}, limit={limit})")
        logger.info("=" * 60)
        
        # ===== GROUP 1: OPEN INTEREST =====
        logger.info("\n[1/8] Open Interest Data...")
        
        data['oi_history'] = self._fetch_oi_history(interval, limit)
        data['oi_aggregated'] = self._fetch_oi_aggregated(interval, limit)
        
        # ===== GROUP 2: FUNDING RATE =====
        logger.info("\n[2/8] Funding Rate Data...")
        
        data['funding_history'] = self._fetch_funding_history(interval, limit)
        data['funding_oi_weighted'] = self._fetch_funding_oi_weighted(interval, limit)
        data['funding_vol_weighted'] = self._fetch_funding_vol_weighted(interval, limit)
        
        # ===== GROUP 3: LONG/SHORT RATIO =====
        logger.info("\n[3/8] Long/Short Ratio Data...")
        
        data['ls_history'] = self._fetch_ls_history(interval, limit)
        data['top_ls_history'] = self._fetch_top_ls_history(interval, limit)
        data['top_position_history'] = self._fetch_top_position_history(interval, limit)
        
        # ===== GROUP 4: LIQUIDATION =====
        logger.info("\n[4/8] Liquidation Data...")
        
        data['liquidation_history'] = self._fetch_liquidation_history(interval, limit)
        data['liquidation_aggregated'] = self._fetch_liquidation_aggregated(interval, limit)
        
        # Try to get liquidation heatmaps (snapshots)
        for model in ['model1', 'model2', 'model3']:
            heatmap = self._fetch_liq_heatmap(model)
            if heatmap is not None:
                data[f'liq_heatmap_{model}'] = heatmap
                logger.info(f"  [OK] liq_heatmap_{model}: snapshot captured")
        
        # ===== GROUP 5: TAKER BUY/SELL =====
        logger.info("\n[5/8] Taker Buy/Sell Data...")
        
        data['taker_history'] = self._fetch_taker_history(interval, limit)
        data['taker_aggregated'] = self._fetch_taker_aggregated(interval, limit)
        
        # ===== GROUP 6: MARKET INDICATORS =====
        logger.info("\n[6/8] Market Indicators...")
        
        data['futures_basis'] = self._fetch_futures_basis()
        data['coinbase_premium'] = self._fetch_coinbase_premium()
        
        # ===== GROUP 7: ETF DATA =====
        logger.info("\n[7/8] ETF Data...")
        
        data['etf_flows'] = self._fetch_etf_flows(days)
        
        # ===== GROUP 8: OPTIONS DATA =====
        logger.info("\n[8/8] Options Data...")
        
        data['options_max_pain'] = self._fetch_options_max_pain()
        data['options_oi'] = self._fetch_options_oi()
        
        # Log summary
        self._log_summary(data)
        
        return data
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN INTEREST FETCHERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_oi_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch OI OHLC history"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/openInterest/ohlc-history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/openInterest/ohlc-history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_BASE_URL, "/api/futures/openInterest/ohlc-history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ohlc_data(result, ['oi_open', 'oi_high', 'oi_low', 'oi_close'])
            if df is not None:
                logger.info(f"  [OK] oi_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_oi_aggregated(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch aggregated OI history (coin-level)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/openInterest/ohlc-aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/openInterest/ohlc-aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ohlc_data(result, ['oi_open', 'oi_high', 'oi_low', 'oi_close'])
            if df is not None:
                logger.info(f"  [OK] oi_aggregated: {len(df)} rows")
            return df
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FUNDING RATE FETCHERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_funding_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch funding rate OHLC history"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/fundingRate/ohlc-history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/fundingRate/ohlc-history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ohlc_data(result, ['funding_open', 'funding_high', 'funding_low', 'funding_close'])
            if df is not None:
                logger.info(f"  [OK] funding_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_funding_oi_weighted(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch OI-weighted funding rate (more accurate)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/fundingRate/oi-weight-ohlc-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/fundingRate/oi-weight-ohlc-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ohlc_data(result, ['funding_oi_open', 'funding_oi_high', 'funding_oi_low', 'funding_oi_close'])
            if df is not None:
                logger.info(f"  [OK] funding_oi_weighted: {len(df)} rows")
            return df
        return None
    
    def _fetch_funding_vol_weighted(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch volume-weighted funding rate"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/fundingRate/vol-weight-ohlc-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/fundingRate/vol-weight-ohlc-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ohlc_data(result, ['funding_vol_open', 'funding_vol_high', 'funding_vol_low', 'funding_vol_close'])
            if df is not None:
                logger.info(f"  [OK] funding_vol_weighted: {len(df)} rows")
            return df
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LONG/SHORT RATIO FETCHERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_ls_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch global long/short account ratio"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/global-long-short-account-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/global-long-short-account-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ls_data(result, 'global')
            if df is not None:
                logger.info(f"  [OK] ls_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_top_ls_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch top trader long/short account ratio"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/top-long-short-account-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/top-long-short-account-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ls_data(result, 'top')
            if df is not None:
                logger.info(f"  [OK] top_ls_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_top_position_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch top trader position ratio (different from account ratio)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/top-long-short-position-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/top-long-short-position-ratio/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_ls_data(result, 'position')
            if df is not None:
                logger.info(f"  [OK] top_position_history: {len(df)} rows")
            return df
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDATION FETCHERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_liquidation_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch liquidation history"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/liquidation/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/liquidation/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_liquidation_data(result)
            if df is not None:
                logger.info(f"  [OK] liquidation_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_liquidation_aggregated(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch aggregated liquidation history (coin-level)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/liquidation/aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/liquidation/aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_liquidation_data(result)
            if df is not None:
                logger.info(f"  [OK] liquidation_aggregated: {len(df)} rows")
            return df
        return None
    
    def _fetch_liq_heatmap(self, model: str) -> Optional[Dict]:
        """Fetch liquidation heatmap snapshot"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, f"/api/futures/liquidation/heatmap/{model}", {
                "symbol": self.symbol
            }),
            (COINGLASS_V4_BASE_URL, f"/api/futures/liquidation/heatmap/{model}", {
                "symbol": self.symbol
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            return result["data"]
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAKER BUY/SELL FETCHERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_taker_history(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch taker buy/sell volume history"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/taker-buy-sell-volume/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/taker-buy-sell-volume/history", {
                "exchange": self.exchange, "symbol": f"{self.symbol}USDT",
                "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_taker_data(result)
            if df is not None:
                logger.info(f"  [OK] taker_history: {len(df)} rows")
            return df
        return None
    
    def _fetch_taker_aggregated(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch aggregated taker buy/sell volume"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/taker-buy-sell-volume/aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
            (COINGLASS_V4_BASE_URL, "/api/futures/taker-buy-sell-volume/aggregated-history", {
                "symbol": self.symbol, "interval": interval, "limit": limit
            }),
        ]
        
        result = self._try_endpoints(endpoints)
        if result:
            df = self._parse_taker_data(result)
            if df is not None:
                logger.info(f"  [OK] taker_aggregated: {len(df)} rows")
            return df
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_futures_basis(self) -> Optional[Dict]:
        """Fetch futures basis (futures vs spot spread)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/futures/basis", {"symbol": self.symbol}),
            (COINGLASS_V4_BASE_URL, "/api/futures/basis", {"symbol": self.symbol}),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            logger.info("  [OK] futures_basis: snapshot captured")
            return result["data"]
        return None
    
    def _fetch_coinbase_premium(self) -> Optional[Dict]:
        """Fetch Coinbase premium index (US retail demand)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/index/coinbase-premium-index", {"symbol": self.symbol}),
            (COINGLASS_V4_BASE_URL, "/api/index/coinbase-premium-index", {"symbol": self.symbol}),
            (COINGLASS_BASE_URL, "/api/index/coinbase-premium-index", {"symbol": self.symbol}),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            logger.info("  [OK] coinbase_premium: snapshot captured")
            return result["data"]
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ETF DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_etf_flows(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch ETF flow data (institutional demand)"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/etf/history", {"symbol": self.symbol, "limit": days}),
            (COINGLASS_V4_BASE_URL, "/api/etf/history", {"symbol": self.symbol, "limit": days}),
            (COINGLASS_BASE_URL, "/api/etf/history", {"symbol": self.symbol, "limit": days}),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            try:
                data = result["data"]
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'])
                    elif 't' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    logger.info(f"  [OK] etf_flows: {len(df)} rows")
                    return df
            except Exception as e:
                logger.warning(f"Error parsing ETF data: {e}")
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIONS DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fetch_options_max_pain(self) -> Optional[Dict]:
        """Fetch options max pain level"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/option/max-pain", {"symbol": self.symbol}),
            (COINGLASS_V4_BASE_URL, "/api/option/max-pain", {"symbol": self.symbol}),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            logger.info("  [OK] options_max_pain: snapshot captured")
            return result["data"]
        return None
    
    def _fetch_options_oi(self) -> Optional[Dict]:
        """Fetch options open interest by strike"""
        endpoints = [
            (COINGLASS_V3_BASE_URL, "/api/option/oi", {"symbol": self.symbol}),
            (COINGLASS_V4_BASE_URL, "/api/option/oi", {"symbol": self.symbol}),
        ]
        
        result = self._try_endpoints(endpoints)
        if result and "data" in result:
            logger.info("  [OK] options_oi: snapshot captured")
            return result["data"]
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA PARSERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _parse_ohlc_data(self, result: Dict, columns: List[str]) -> Optional[pd.DataFrame]:
        """Parse OHLC-style data"""
        if not result or "data" not in result:
            return None
        
        records = []
        data = result["data"]
        
        if isinstance(data, list):
            for item in data:
                try:
                    ts = item.get("t") or item.get("time") or item.get("createTime") or 0
                    if ts > 1e12:
                        ts = ts / 1000
                    
                    record = {"timestamp": datetime.fromtimestamp(ts)}
                    
                    # Map OHLC values
                    ohlc_keys = [
                        ('o', 'open'), ('h', 'high'), ('l', 'low'), ('c', 'close')
                    ]
                    
                    for i, (short, long) in enumerate(ohlc_keys):
                        if i < len(columns):
                            val = item.get(short) or item.get(long) or item.get(columns[i].split('_')[-1], 0)
                            record[columns[i]] = float(val) if val else 0.0
                    
                    records.append(record)
                except Exception:
                    continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df
    
    def _parse_ls_data(self, result: Dict, prefix: str = 'global') -> Optional[pd.DataFrame]:
        """Parse long/short ratio data"""
        if not result or "data" not in result:
            return None
        
        records = []
        data = result["data"]
        
        if isinstance(data, list):
            for item in data:
                try:
                    ts = item.get("t") or item.get("time") or item.get("createTime") or 0
                    if ts > 1e12:
                        ts = ts / 1000
                    
                    long_ratio = float(item.get("longAccount", item.get("longRatio", item.get("l", 0.5))))
                    short_ratio = float(item.get("shortAccount", item.get("shortRatio", item.get("s", 0.5))))
                    
                    record = {"timestamp": datetime.fromtimestamp(ts)}
                    
                    if prefix == 'global':
                        record["long_ratio"] = long_ratio
                        record["short_ratio"] = short_ratio
                        record["long_short_ratio"] = long_ratio / short_ratio if short_ratio > 0 else 1.0
                    else:
                        record[f"{prefix}_long_ratio"] = long_ratio
                        record[f"{prefix}_short_ratio"] = short_ratio
                        record[f"{prefix}_ls_ratio"] = long_ratio / short_ratio if short_ratio > 0 else 1.0
                    
                    records.append(record)
                except Exception:
                    continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df
    
    def _parse_liquidation_data(self, result: Dict) -> Optional[pd.DataFrame]:
        """Parse liquidation data"""
        if not result or "data" not in result:
            return None
        
        records = []
        data = result["data"]
        
        if isinstance(data, list):
            for item in data:
                try:
                    ts = item.get("t") or item.get("time") or item.get("createTime") or 0
                    if ts > 1e12:
                        ts = ts / 1000
                    
                    long_liq = float(item.get("longLiquidationUsd", item.get("buyVolUsd", item.get("l", 0))))
                    short_liq = float(item.get("shortLiquidationUsd", item.get("sellVolUsd", item.get("s", 0))))
                    total = long_liq + short_liq
                    
                    records.append({
                        "timestamp": datetime.fromtimestamp(ts),
                        "long_liq_usd": long_liq,
                        "short_liq_usd": short_liq,
                        "total_liq_usd": total,
                        "liq_ratio": long_liq / total if total > 0 else 0.5,
                    })
                except Exception:
                    continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df
    
    def _parse_taker_data(self, result: Dict) -> Optional[pd.DataFrame]:
        """Parse taker buy/sell data"""
        if not result or "data" not in result:
            return None
        
        records = []
        data = result["data"]
        
        if isinstance(data, list):
            for item in data:
                try:
                    ts = item.get("t") or item.get("time") or item.get("createTime") or 0
                    if ts > 1e12:
                        ts = ts / 1000
                    
                    buy_vol = float(item.get("buyVolUsd", item.get("b", 0)))
                    sell_vol = float(item.get("sellVolUsd", item.get("s", 0)))
                    total = buy_vol + sell_vol
                    
                    records.append({
                        "timestamp": datetime.fromtimestamp(ts),
                        "taker_buy_vol": buy_vol,
                        "taker_sell_vol": sell_vol,
                        "taker_buy_sell_ratio": buy_vol / sell_vol if sell_vol > 0 else 1.0,
                        "net_taker_flow": buy_vol - sell_vol,
                    })
                except Exception:
                    continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df
    
    def _log_summary(self, data: Dict):
        """Log data summary"""
        logger.info("\n" + "=" * 60)
        logger.info("COINGLASS V4 DATA SUMMARY")
        logger.info("=" * 60)
        
        total_rows = 0
        timeseries_count = 0
        snapshot_count = 0
        
        for key, value in data.items():
            if value is not None:
                if isinstance(value, pd.DataFrame):
                    logger.info(f"  [OK] {key}: {len(value)} rows")
                    total_rows += len(value)
                    timeseries_count += 1
                elif isinstance(value, dict):
                    logger.info(f"  [OK] {key}: snapshot")
                    snapshot_count += 1
                elif isinstance(value, list):
                    logger.info(f"  [OK] {key}: {len(value)} items")
                    timeseries_count += 1
            else:
                logger.info(f"  [--] {key}: not available")
        
        logger.info(f"\nTimeseries endpoints: {timeseries_count}")
        logger.info(f"Snapshot endpoints: {snapshot_count}")
        logger.info(f"Total rows: {total_rows}")
        logger.info("=" * 60)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY INTERFACE (for backward compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_open_interest_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for OI history"""
        return self._fetch_oi_history(interval, limit)
    
    def get_funding_rate_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for funding history"""
        return self._fetch_funding_history(interval, limit)
    
    def get_global_long_short_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for global L/S history"""
        return self._fetch_ls_history(interval, limit)
    
    def get_top_trader_long_short_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for top trader L/S history"""
        return self._fetch_top_ls_history(interval, limit)
    
    def get_liquidation_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for liquidation history"""
        return self._fetch_liquidation_history(interval, limit)
    
    def get_taker_history(self, interval: str = "4h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Legacy interface for taker history"""
        return self._fetch_taker_history(interval, limit)


# Convenience function
def fetch_all_coinglass_data(days: int = 90, interval: str = "4h") -> Dict[str, Any]:
    """Convenience function to fetch all CoinGlass data"""
    client = CoinGlassV4Fetcher()
    return client.fetch_all_data(days=days, interval=interval)
