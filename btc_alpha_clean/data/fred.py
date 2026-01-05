"""
FRED Data Fetcher
================

Macroeconomic data from Federal Reserve Economic Data.
Interest rates, inflation, financial conditions, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import BaseFetcher
from config import API_KEYS, URLS, RATE_LIMITS, FRED_SERIES

logger = logging.getLogger(__name__)


class FREDFetcher(BaseFetcher):
    """
    Fetches data from FRED API.
    
    Key data:
    - Interest rates (Fed Funds, Treasury yields)
    - Yield curve (10Y-2Y, 10Y-3M spreads)
    - Inflation (CPI, PCE, breakevens)
    - Financial conditions (NFCI, STLFSI)
    - Money supply (M2, Fed balance sheet)
    """
    
    def __init__(self):
        super().__init__(
            base_url=URLS["fred"],
            rate_limit=RATE_LIMITS["fred"],  # 120/min
            api_key=API_KEYS["fred"]
        )
    
    def get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def test_connection(self) -> bool:
        """Test API connectivity"""
        try:
            # Try to fetch a simple series
            result = self._get(
                "/series/observations",
                params={
                    "series_id": "DFF",
                    "api_key": self.api_key,
                    "file_type": "json",
                    "limit": 1
                }
            )
            success = "observations" in result
            logger.info(f"FRED connection test: {'SUCCESS' if success else 'FAILED'}")
            return success
        except Exception as e:
            logger.error(f"FRED connection test failed: {e}")
            return False
    
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., "DFF", "GS10")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: d (daily), w (weekly), m (monthly), q (quarterly), a (annual)
        """
        try:
            if start_date is None:
                start_date = (datetime.utcnow() - timedelta(days=365*5)).strftime("%Y-%m-%d")
            if end_date is None:
                end_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date
            }
            
            if frequency:
                params["frequency"] = frequency
            
            result = self._get("/series/observations", params)
            
            if not result.get("observations"):
                logger.warning(f"No observations for series {series_id}")
                return None
            
            df = pd.DataFrame(result["observations"])
            
            # Convert date
            df["timestamp"] = pd.to_datetime(df["date"])
            
            # Convert value (handling "." for missing values)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Keep essential columns
            df = df[["timestamp", "value"]].rename(columns={"value": series_id.lower()})
            df = df.dropna()
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            logger.debug(f"Fetched {len(df)} observations for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return None
    
    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get metadata about a series"""
        try:
            result = self._get(
                "/series",
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json"
                }
            )
            
            if result.get("seriess"):
                series = result["seriess"][0]
                return {
                    "id": series.get("id"),
                    "title": series.get("title"),
                    "frequency": series.get("frequency"),
                    "units": series.get("units"),
                    "seasonal_adjustment": series.get("seasonal_adjustment"),
                    "last_updated": series.get("last_updated")
                }
            return None
            
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return None
    
    def fetch_all_series(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch all configured FRED series.
        
        Args:
            years: Number of years of history
        """
        logger.info(f"Fetching {len(FRED_SERIES)} FRED series for past {years} years...")
        
        start_date = (datetime.utcnow() - timedelta(days=365*years)).strftime("%Y-%m-%d")
        
        data = {}
        
        for series_id, description in FRED_SERIES.items():
            logger.info(f"  Fetching {series_id} ({description})...")
            
            df = self.get_series(series_id, start_date=start_date)
            
            if df is not None and len(df) > 0:
                data[series_id.lower()] = df
                logger.info(f"    {series_id}: {len(df)} observations")
            else:
                logger.warning(f"    {series_id}: NO DATA")
        
        return data
    
    def get_yield_curve(self) -> Optional[pd.DataFrame]:
        """
        Get yield curve spreads.
        Returns: 10Y-2Y and 10Y-3M spreads.
        """
        logger.info("Fetching yield curve data...")
        
        # Get the spread series directly from FRED
        t10y2y = self.get_series("T10Y2Y")
        t10y3m = self.get_series("T10Y3M")
        
        if t10y2y is None or t10y3m is None:
            return None
        
        # Merge
        df = t10y2y.merge(t10y3m, on="timestamp", how="outer")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Forward fill missing values
        df = df.ffill()
        
        # Add inversion flags
        df["curve_10y2y_inverted"] = df["t10y2y"] < 0
        df["curve_10y3m_inverted"] = df["t10y3m"] < 0
        
        logger.info(f"Yield curve data: {len(df)} observations")
        return df
    
    def get_financial_conditions(self) -> Optional[pd.DataFrame]:
        """
        Get financial conditions indices.
        NFCI and STLFSI.
        """
        logger.info("Fetching financial conditions data...")
        
        nfci = self.get_series("NFCI")
        stlfsi = self.get_series("STLFSI4")
        vix = self.get_series("VIXCLS")  # VIX from FRED
        
        dfs = [df for df in [nfci, stlfsi, vix] if df is not None]
        
        if not dfs:
            return None
        
        # Merge all available
        df = dfs[0]
        for other_df in dfs[1:]:
            df = df.merge(other_df, on="timestamp", how="outer")
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.ffill()
        
        # NFCI: positive = tighter than average, negative = looser
        # STLFSI: higher = more stress
        
        logger.info(f"Financial conditions data: {len(df)} observations")
        return df
    
    def get_inflation_data(self) -> Optional[pd.DataFrame]:
        """
        Get inflation-related data.
        CPI, Core CPI, breakeven inflation.
        """
        logger.info("Fetching inflation data...")
        
        data = {}
        for series in ["CPIAUCSL", "CPILFESL", "T5YIE", "T10YIE"]:
            df = self.get_series(series)
            if df is not None:
                data[series.lower()] = df
        
        if not data:
            return None
        
        # Start with the first series
        result = list(data.values())[0]
        
        # Merge the rest
        for series_name, df in list(data.items())[1:]:
            result = result.merge(df, on="timestamp", how="outer")
        
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result.ffill()
        
        # Calculate YoY changes for CPI
        if "cpiaucsl" in result.columns:
            result["cpi_yoy"] = result["cpiaucsl"].pct_change(12) * 100  # Monthly data
        if "cpilfesl" in result.columns:
            result["core_cpi_yoy"] = result["cpilfesl"].pct_change(12) * 100
        
        logger.info(f"Inflation data: {len(result)} observations")
        return result
    
    def get_rates_data(self) -> Optional[pd.DataFrame]:
        """
        Get interest rate data.
        Fed funds, Treasury yields.
        """
        logger.info("Fetching rates data...")
        
        series_list = ["DFF", "GS2", "GS10", "DTB3"]
        data = {}
        
        for series in series_list:
            df = self.get_series(series)
            if df is not None:
                data[series.lower()] = df
        
        if not data:
            return None
        
        # Merge all
        result = list(data.values())[0]
        for df in list(data.values())[1:]:
            result = result.merge(df, on="timestamp", how="outer")
        
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result.ffill()
        
        # Calculate real yield (10Y - 10Y breakeven)
        t10yie = self.get_series("T10YIE")
        if t10yie is not None and "gs10" in result.columns:
            result = result.merge(t10yie, on="timestamp", how="left")
            result["real_yield_10y"] = result["gs10"] - result["t10yie"]
        
        logger.info(f"Rates data: {len(result)} observations")
        return result


if __name__ == "__main__":
    # Test the fetcher
    fetcher = FREDFetcher()
    
    print("Testing FRED API connection...")
    if fetcher.test_connection():
        print("[OK] Connection successful")
        
        print("\nFetching yield curve data...")
        yc = fetcher.get_yield_curve()
        if yc is not None:
            print(f"\nYield curve sample:\n{yc.tail()}")
        
        print("\nFetching financial conditions...")
        fc = fetcher.get_financial_conditions()
        if fc is not None:
            print(f"\nFinancial conditions sample:\n{fc.tail()}")
        
        print("\nFetching rates data...")
        rates = fetcher.get_rates_data()
        if rates is not None:
            print(f"\nRates sample:\n{rates.tail()}")
    else:
        print("[X] Connection failed")
