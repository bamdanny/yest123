"""
Data Orchestrator
================

Master coordinator for all data fetching, storage, and preparation.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import data_config, API_KEYS
from data.coinglass import CoinGlassFetcher
from data.binance import BinanceFetcher
from data.polygon import PolygonFetcher
from data.fred import FREDFetcher
from data.free_sources import FreeSources
from data.storage import DataStorage, DataAligner, create_master_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataOrchestrator:
    """
    Master orchestrator for all data operations.
    
    Responsibilities:
    1. Test all API connections
    2. Fetch data from all sources
    3. Store data incrementally
    4. Create unified master dataset
    5. Handle resumable operations
    """
    
    def __init__(self):
        # Initialize fetchers
        self.coinglass = CoinGlassFetcher()
        self.binance = BinanceFetcher()
        self.polygon = PolygonFetcher()
        self.fred = FREDFetcher()
        self.free_sources = FreeSources()
        
        # Initialize storage
        self.storage = DataStorage()
        self.aligner = DataAligner()
        
        # State
        self.connection_status = {}
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections to all APIs"""
        logger.info("Testing API connections...")
        
        self.connection_status = {
            "binance": self.binance.test_connection(),
            "coinglass": self.coinglass.test_connection(),
            "polygon": self.polygon.test_connection(),
            "fred": self.fred.test_connection(),
            "alternative_me": self.free_sources.alternative_me.test_connection(),
            "deribit": self.free_sources.deribit.test_connection(),
        }
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("API Connection Status:")
        logger.info("="*50)
        for api, status in self.connection_status.items():
            status_str = "[OK] CONNECTED" if status else "[X] FAILED"
            logger.info(f"  {api:15s}: {status_str}")
        logger.info("="*50 + "\n")
        
        return self.connection_status
    
    def fetch_price_data(self, days: int = 365, interval: str = "4h", force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch and store price data from Binance"""
        logger.info(f"Fetching price data ({days} days, {interval} timeframe)...")
        
        cache_name = f"price_{interval}_{days}d"
        
        if not force_refresh and self.storage.exists(cache_name, "binance"):
            logger.info("Loading cached price data...")
            return self.storage.load(cache_name, "binance")
        
        df = self.binance.get_klines_extended(interval=interval, days=days)
        
        if df is not None:
            self.storage.save(df, cache_name, "binance")
        
        return df
    
    def fetch_derivatives_data(self, days: int = 90, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch derivatives data from CoinGlass and Binance.
        
        Uses CoinGlassFetcherV2 with:
        - 3+ second delays between requests
        - Exponential backoff up to 120s
        - Pagination for historical data
        """
        logger.info(f"Fetching derivatives data ({days} days)...")
        
        data = {}
        
        # Try CoinGlass V4 API (2024+ version)
        if self.connection_status.get("coinglass", False):
            try:
                from data.coinglass_v4 import CoinGlassV4Fetcher
                
                logger.info("Using CoinGlass V4 API...")
                cg_v4 = CoinGlassV4Fetcher()
                cg_data = cg_v4.fetch_all_data(interval="4h", days=days)
                
                for name, df in cg_data.items():
                    if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                        self.storage.save(df, f"cg_{name}", "coinglass")
                        data[f"cg_{name}"] = df
                        logger.info(f"  [OK] cg_{name}: {len(df)} rows")
                    elif df is not None and isinstance(df, dict):
                        # Handle snapshot data
                        data[f"cg_{name}"] = df
                        logger.info(f"  [OK] cg_{name}: snapshot")
                    else:
                        logger.debug(f"  [--] cg_{name}: not available")
                        
            except Exception as e:
                logger.error(f"CoinGlass V4 fetch failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Binance derivatives removed - CoinGlass provides same data aggregated across all exchanges
        logger.info("Skipping Binance derivatives (using CoinGlass aggregated data instead)")
        
        # Summarize derivatives data
        total_rows = sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame))
        logger.info(f"Derivatives data: {len(data)} datasets, {total_rows} total rows")
        
        return data
    
    def fetch_macro_data(self, days: int = 365, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch macro data from Polygon and FRED"""
        logger.info(f"Fetching macro data ({days} days)...")
        
        data = {}
        
        # Polygon data - skip VIX/DXY (requires paid plan)
        # Note: Using FRED VIXCLS instead for volatility
        if self.connection_status.get("polygon", False):
            logger.info("Skipping Polygon VIX/DXY (requires paid plan)...")
            # Could fetch SPY, QQQ if needed for correlation
            # For now, rely on FRED for macro data
        
        # FRED data
        if self.connection_status.get("fred", False):
            logger.info("Fetching FRED data...")
            
            yield_curve = self.fred.get_yield_curve()
            if yield_curve is not None:
                self.storage.save(yield_curve, "yield_curve", "fred")
                data["yield_curve"] = yield_curve
            
            fin_conditions = self.fred.get_financial_conditions()
            if fin_conditions is not None:
                self.storage.save(fin_conditions, "fin_conditions", "fred")
                data["fin_conditions"] = fin_conditions
            
            rates = self.fred.get_rates_data()
            if rates is not None:
                self.storage.save(rates, "rates", "fred")
                data["rates"] = rates
            
            inflation = self.fred.get_inflation_data()
            if inflation is not None:
                self.storage.save(inflation, "inflation", "fred")
                data["inflation"] = inflation
        
        return data
    
    def fetch_sentiment_data(self, days: int = 365, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch sentiment data from free sources"""
        logger.info(f"Fetching sentiment data ({days} days)...")
        
        data = {}
        
        # Fear & Greed
        if self.connection_status.get("alternative_me", False):
            fg = self.free_sources.alternative_me.get_fear_greed(days=days)
            if fg is not None:
                self.storage.save(fg, "fear_greed", "sentiment")
                data["fear_greed"] = fg
        
        # Deribit options
        if self.connection_status.get("deribit", False):
            hist_vol = self.free_sources.deribit.get_historical_volatility()
            if hist_vol is not None:
                self.storage.save(hist_vol, "historical_vol", "sentiment")
                data["historical_vol"] = hist_vol
            
            options_metrics = self.free_sources.deribit.calculate_options_metrics()
            if options_metrics:
                data["options_metrics"] = options_metrics
        
        return data
    
    def fetch_all_data(
        self,
        price_days: int = 365,
        derivatives_days: int = 90,
        macro_days: int = 365,
        sentiment_days: int = 365,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch all data from all sources.
        
        Returns dict with all data organized by category.
        """
        logger.info("\n" + "="*60)
        logger.info("FETCHING ALL DATA")
        logger.info("="*60)
        
        all_data = {
            "price": None,
            "derivatives": {},
            "macro": {},
            "sentiment": {}
        }
        
        # Test connections first
        self.test_all_connections()
        
        # Price data (Binance)
        with tqdm(total=4, desc="Data categories") as pbar:
            all_data["price"] = self.fetch_price_data(days=price_days, force_refresh=force_refresh)
            pbar.update(1)
            
            all_data["derivatives"] = self.fetch_derivatives_data(days=derivatives_days, force_refresh=force_refresh)
            pbar.update(1)
            
            all_data["macro"] = self.fetch_macro_data(days=macro_days, force_refresh=force_refresh)
            pbar.update(1)
            
            all_data["sentiment"] = self.fetch_sentiment_data(days=sentiment_days, force_refresh=force_refresh)
            pbar.update(1)
        
        # Save metadata
        metadata = {
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "price_days": price_days,
            "derivatives_days": derivatives_days,
            "macro_days": macro_days,
            "sentiment_days": sentiment_days,
            "connection_status": self.connection_status,
            "data_summary": {
                "price_rows": len(all_data["price"]) if all_data["price"] is not None else 0,
                "derivatives_datasets": len(all_data["derivatives"]),
                "macro_datasets": len(all_data["macro"]),
                "sentiment_datasets": len(all_data["sentiment"])
            }
        }
        
        with open(self.storage.base_dir / "fetch_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("\n" + "="*60)
        logger.info("DATA FETCH COMPLETE")
        logger.info("="*60)
        self._print_data_summary(all_data)
        
        return all_data
    
    def create_master_dataset(
        self,
        all_data: Optional[Dict] = None,
        timeframe: str = "4h"
    ) -> pd.DataFrame:
        """
        Create unified master dataset from all fetched data.
        
        If all_data not provided, loads from storage.
        """
        logger.info("\nCreating master dataset...")
        
        if all_data is None:
            # Load from storage
            all_data = self.load_all_from_storage()
        
        # Create master dataset
        master = create_master_dataset(
            price_data=all_data["price"],
            derivatives_data=all_data["derivatives"],
            macro_data=all_data["macro"],
            sentiment_data=all_data["sentiment"],
            timeframe=timeframe
        )
        
        # Save master dataset
        self.storage.save(master, f"master_{timeframe}", "processed")
        
        logger.info(f"\nMaster dataset created: {len(master)} rows, {len(master.columns)} columns")
        logger.info(f"Date range: {master['timestamp'].min()} to {master['timestamp'].max()}")
        
        return master
    
    def load_all_from_storage(self) -> Dict[str, Any]:
        """Load all cached data from storage"""
        logger.info("Loading data from storage...")
        
        all_data = {
            "price": None,
            "derivatives": {},
            "macro": {},
            "sentiment": {}
        }
        
        # Price data
        price_files = self.storage.list_files("binance")
        for f in price_files:
            if f.startswith("price_"):
                all_data["price"] = self.storage.load(f, "binance")
                break
        
        # Derivatives
        for subdir in ["coinglass", "binance"]:
            files = self.storage.list_files(subdir)
            for f in files:
                if not f.startswith("price_"):
                    df = self.storage.load(f, subdir)
                    if df is not None:
                        all_data["derivatives"][f] = df
        
        # Macro
        for subdir in ["polygon", "fred"]:
            files = self.storage.list_files(subdir)
            for f in files:
                df = self.storage.load(f, subdir)
                if df is not None:
                    all_data["macro"][f] = df
        
        # Sentiment
        files = self.storage.list_files("sentiment")
        for f in files:
            df = self.storage.load(f, "sentiment")
            if df is not None:
                all_data["sentiment"][f] = df
        
        return all_data
    
    def _print_data_summary(self, all_data: Dict):
        """Print summary of fetched data"""
        logger.info("\nData Summary:")
        logger.info("-" * 40)
        
        if all_data["price"] is not None:
            logger.info(f"Price data: {len(all_data['price'])} rows")
            logger.info(f"  Columns: {list(all_data['price'].columns)}")
        
        logger.info(f"\nDerivatives datasets: {len(all_data['derivatives'])}")
        for name, df in all_data["derivatives"].items():
            if isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: {len(df)} rows")
        
        logger.info(f"\nMacro datasets: {len(all_data['macro'])}")
        for name, df in all_data["macro"].items():
            if isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: {len(df)} rows")
        
        logger.info(f"\nSentiment datasets: {len(all_data['sentiment'])}")
        for name, df in all_data["sentiment"].items():
            if isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: {len(df)} rows")


def main():
    """Main entry point for data fetching"""
    orchestrator = DataOrchestrator()
    
    # Test connections
    status = orchestrator.test_all_connections()
    
    # Check if any connections work
    if not any(status.values()):
        logger.error("No API connections available!")
        return None
    
    # Fetch all data
    all_data = orchestrator.fetch_all_data(
        price_days=365,
        derivatives_days=90,
        macro_days=365,
        sentiment_days=365,
        force_refresh=False
    )
    
    # Create master dataset
    master = orchestrator.create_master_dataset(all_data)
    
    return master


if __name__ == "__main__":
    master = main()
    
    if master is not None:
        print("\n" + "="*60)
        print("MASTER DATASET PREVIEW")
        print("="*60)
        print(f"\nShape: {master.shape}")
        print(f"\nColumns ({len(master.columns)}):")
        for col in sorted(master.columns):
            print(f"  - {col}")
        print(f"\nFirst few rows:")
        print(master.head())
