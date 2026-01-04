"""
Data Storage & Alignment
========================

Handles:
- Parquet storage/retrieval
- Timestamp alignment across sources
- Data merging for feature engineering
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles parquet storage and retrieval"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or data_config.data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, df: pd.DataFrame, name: str, subdir: Optional[str] = None) -> Path:
        """
        Save DataFrame to parquet.
        
        Args:
            df: DataFrame to save
            name: Name for the file (without extension)
            subdir: Optional subdirectory
        """
        if subdir:
            save_dir = self.base_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.base_dir
        
        filepath = save_dir / f"{name}.parquet"
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return filepath
    
    def load(self, name: str, subdir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet"""
        if subdir:
            filepath = self.base_dir / subdir / f"{name}.parquet"
        else:
            filepath = self.base_dir / f"{name}.parquet"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    
    def exists(self, name: str, subdir: Optional[str] = None) -> bool:
        """Check if data file exists"""
        if subdir:
            filepath = self.base_dir / subdir / f"{name}.parquet"
        else:
            filepath = self.base_dir / f"{name}.parquet"
        return filepath.exists()
    
    def list_files(self, subdir: Optional[str] = None) -> List[str]:
        """List all parquet files in directory"""
        if subdir:
            search_dir = self.base_dir / subdir
        else:
            search_dir = self.base_dir
        
        if not search_dir.exists():
            return []
        
        return [f.stem for f in search_dir.glob("*.parquet")]
    
    def get_latest_timestamp(self, name: str, subdir: Optional[str] = None) -> Optional[datetime]:
        """Get the latest timestamp in a saved file"""
        df = self.load(name, subdir)
        if df is None or "timestamp" not in df.columns:
            return None
        return df["timestamp"].max()


class DataAligner:
    """
    Aligns data from multiple sources to a common timeline.
    
    Different sources have different frequencies:
    - Binance klines: 1m to 1d
    - CoinGlass: 5m to 1d
    - FRED: daily, weekly, monthly
    - Fear & Greed: daily
    
    This aligner handles forward-filling and proper merging.
    """
    
    def __init__(self, base_timeframe: str = "1h"):
        """
        Args:
            base_timeframe: Target timeframe for alignment (1h, 4h, 1d)
        """
        self.base_timeframe = base_timeframe
        
        self.timeframe_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
            "8h": 28800, "12h": 43200, "1d": 86400
        }
    
    def create_timeline(
        self,
        start: datetime,
        end: datetime,
        timeframe: Optional[str] = None
    ) -> pd.DatetimeIndex:
        """Create a regular timeline index"""
        tf = timeframe or self.base_timeframe
        freq_map = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h",
            "8h": "8h", "12h": "12h", "1d": "1D"
        }
        
        return pd.date_range(start=start, end=end, freq=freq_map[tf])
    
    def align_to_timeline(
        self,
        df: pd.DataFrame,
        timeline: pd.DatetimeIndex,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Align DataFrame to a timeline.
        
        Args:
            df: DataFrame with 'timestamp' column
            timeline: Target timeline
            method: 'ffill' (forward fill) or 'bfill' (backward fill)
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Remove duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        
        # Set timestamp as index
        df = df.set_index("timestamp")
        
        # Sort index to ensure proper alignment
        df = df.sort_index()
        
        # Reindex to timeline
        df = df.reindex(timeline, method=method)
        
        # Reset index
        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp"})
        
        return df
    
    def resample_to_timeframe(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str,
        agg_method: str = "last"
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: DataFrame with 'timestamp' column
            source_timeframe: Current timeframe
            target_timeframe: Target timeframe
            agg_method: 'last', 'mean', 'sum', 'first'
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        freq_map = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h",
            "8h": "8h", "12h": "12h", "1d": "1D"
        }
        
        df = df.set_index("timestamp")
        
        if agg_method == "last":
            resampled = df.resample(freq_map[target_timeframe]).last()
        elif agg_method == "mean":
            resampled = df.resample(freq_map[target_timeframe]).mean()
        elif agg_method == "sum":
            resampled = df.resample(freq_map[target_timeframe]).sum()
        elif agg_method == "first":
            resampled = df.resample(freq_map[target_timeframe]).first()
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")
        
        resampled = resampled.reset_index()
        return resampled.dropna(how="all", subset=[c for c in resampled.columns if c != "timestamp"])
    
    def merge_datasets(
        self,
        base_df: pd.DataFrame,
        other_dfs: Dict[str, pd.DataFrame],
        how: str = "left"
    ) -> pd.DataFrame:
        """
        Merge multiple DataFrames on timestamp.
        
        Args:
            base_df: Base DataFrame (determines timeline)
            other_dfs: Dict of name -> DataFrame to merge
            how: Join type ('left', 'outer', 'inner')
        """
        result = base_df.copy()
        
        for name, df in other_dfs.items():
            if df is None or len(df) == 0:
                logger.warning(f"Skipping empty dataset: {name}")
                continue
            
            if "timestamp" not in df.columns:
                logger.warning(f"Dataset {name} has no timestamp column, skipping")
                continue
            
            # Prefix columns to avoid conflicts
            rename_map = {c: f"{name}_{c}" for c in df.columns if c != "timestamp"}
            df_renamed = df.rename(columns=rename_map)
            
            result = result.merge(df_renamed, on="timestamp", how=how)
            logger.debug(f"Merged {name}: {len(result)} rows")
        
        return result


def create_master_dataset(
    price_data: pd.DataFrame,
    derivatives_data: Dict[str, pd.DataFrame],
    macro_data: Dict[str, pd.DataFrame],
    sentiment_data: Dict[str, pd.DataFrame],
    timeframe: str = "4h"
) -> pd.DataFrame:
    """
    Create a unified master dataset from all sources.
    
    This is the main data preparation function that:
    1. Uses price data as the base timeline
    2. Aligns all other data to this timeline
    3. Forward-fills missing values appropriately
    4. Returns a clean dataset ready for feature engineering
    """
    logger.info("Creating master dataset...")
    
    aligner = DataAligner(base_timeframe=timeframe)
    
    # Use price data as base
    if price_data is None or len(price_data) == 0:
        raise ValueError("Price data is required as base")
    
    master = price_data.copy()
    
    # Ensure timestamp column exists
    if "timestamp" not in master.columns:
        raise ValueError("Price data must have timestamp column")
    
    # Remove duplicate timestamps from master
    master = master.drop_duplicates(subset=["timestamp"], keep="last")
    
    # Create timeline from price data (unique, sorted)
    timeline = pd.DatetimeIndex(master["timestamp"].sort_values().unique())
    
    # Process derivatives data
    for name, df in derivatives_data.items():
        if df is None or len(df) == 0:
            continue
        
        # Skip if no timestamp column
        if "timestamp" not in df.columns:
            logger.debug(f"Skipping derivatives/{name}: no timestamp column")
            continue
        
        try:
            # Align to timeline
            aligned = aligner.align_to_timeline(df.copy(), timeline, method="ffill")
            
            # Merge
            rename_map = {c: f"deriv_{name}_{c}" for c in aligned.columns if c != "timestamp"}
            aligned = aligned.rename(columns=rename_map)
            master = master.merge(aligned, on="timestamp", how="left")
            
            logger.debug(f"Added derivatives/{name}: {len(aligned)} rows")
        except Exception as e:
            logger.warning(f"Failed to align derivatives/{name}: {e}")
    
    # Process macro data (daily -> fill forward to hourly)
    for name, df in macro_data.items():
        if df is None or len(df) == 0:
            continue
        
        # Skip if no timestamp column
        if "timestamp" not in df.columns:
            logger.debug(f"Skipping macro/{name}: no timestamp column")
            continue
        
        try:
            # These are typically daily, so resample up
            aligned = aligner.align_to_timeline(df.copy(), timeline, method="ffill")
            
            rename_map = {c: f"macro_{name}_{c}" for c in aligned.columns if c != "timestamp"}
            aligned = aligned.rename(columns=rename_map)
            master = master.merge(aligned, on="timestamp", how="left")
            
            logger.debug(f"Added macro/{name}: {len(aligned)} rows")
        except Exception as e:
            logger.warning(f"Failed to align macro/{name}: {e}")
    
    # Process sentiment data
    for name, df in sentiment_data.items():
        if df is None or len(df) == 0:
            continue
        
        if isinstance(df, dict):
            # Convert single-value dict to DataFrame
            df = pd.DataFrame([df])
        
        # Skip if no timestamp column
        if "timestamp" not in df.columns:
            logger.debug(f"Skipping sentiment/{name}: no timestamp column")
            continue
        
        try:
            aligned = aligner.align_to_timeline(df.copy(), timeline, method="ffill")
            
            rename_map = {c: f"sent_{name}_{c}" for c in aligned.columns if c != "timestamp"}
            aligned = aligned.rename(columns=rename_map)
            master = master.merge(aligned, on="timestamp", how="left")
            
            logger.debug(f"Added sentiment/{name}: {len(aligned)} rows")
        except Exception as e:
            logger.warning(f"Failed to align sentiment/{name}: {e}")
    
    # Forward fill any remaining NaNs (within reason)
    numeric_cols = master.select_dtypes(include=[np.number]).columns
    master[numeric_cols] = master[numeric_cols].ffill(limit=24)  # Max 24 hours forward fill
    
    # Drop rows with too many NaNs
    nan_threshold = 0.5  # Drop rows with >50% NaN
    master = master.dropna(thresh=int(len(master.columns) * (1 - nan_threshold)))
    
    logger.info(f"Master dataset: {len(master)} rows, {len(master.columns)} columns")
    
    return master


if __name__ == "__main__":
    # Test storage
    storage = DataStorage()
    
    # Create test data
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "value": np.random.randn(100)
    })
    
    # Save
    storage.save(test_df, "test_data")
    
    # Load
    loaded = storage.load("test_data")
    print(f"Loaded test data: {len(loaded)} rows")
    
    # Test aligner
    aligner = DataAligner("1h")
    
    # Create daily data
    daily_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1D"),
        "daily_value": np.random.randn(10)
    })
    
    # Create hourly timeline
    hourly_timeline = aligner.create_timeline(
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        "1h"
    )
    
    # Align daily to hourly
    aligned = aligner.align_to_timeline(daily_df, hourly_timeline, method="ffill")
    print(f"\nAligned daily to hourly: {len(aligned)} rows")
    print(aligned.head(30))
