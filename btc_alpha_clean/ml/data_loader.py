"""
Data Loader - Load and prepare data for ML training

This module connects to your existing data pipeline from Phase 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import pickle
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load features and create targets for ML training.
    
    Connects to existing Phase 1 data pipeline.
    """
    
    def __init__(self, config_path: str = None):
        # Find config file
        config_paths = [
            config_path,
            "config/ml_config.json",
            Path(__file__).parent.parent / "config" / "ml_config.json"
        ]
        
        self.config = {}
        for p in config_paths:
            if p and Path(p).exists():
                with open(p) as f:
                    self.config = json.load(f)
                break
        
        # Load feature priority from Phase 1 results
        priority_paths = [
            "config/feature_priority.json",
            Path(__file__).parent.parent / "config" / "feature_priority.json"
        ]
        
        self.feature_priority = {}
        for p in priority_paths:
            if Path(p).exists():
                with open(p) as f:
                    self.feature_priority = json.load(f)
                break
    
    def load_from_cache(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and returns from Phase 1 cache.
        
        Returns:
            (features, returns) tuple
        """
        # Try multiple cache locations
        cache_paths = [
            "data_cache/features_cache.pkl",
            Path(__file__).parent.parent / "data_cache" / "features_cache.pkl"
        ]
        
        for cache_path in cache_paths:
            if Path(cache_path).exists():
                logger.info(f"Loading from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                
                features = cache['features']
                returns = cache['returns']
                
                logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
                logger.info(f"Date range: {features.index[0]} to {features.index[-1]}")
                
                return features, returns
        
        raise FileNotFoundError(
            "Feature cache not found. Run Phase 1 first:\n"
            "  python run_exhaustive_search.py --mode single"
        )
    
    def load_features(self) -> pd.DataFrame:
        """
        Load all features from the existing data pipeline.
        """
        logger.info("Loading features from data pipeline...")
        
        # Try cache first
        try:
            features, _ = self.load_from_cache()
            return features
        except FileNotFoundError:
            pass
        
        # Try loading from parquet files
        feature_dirs = [
            Path("data/features"),
            Path(__file__).parent.parent / "data" / "features"
        ]
        
        for feature_dir in feature_dirs:
            if feature_dir.exists():
                feature_files = list(feature_dir.glob("*.parquet"))
                if feature_files:
                    dfs = [pd.read_parquet(f) for f in feature_files]
                    features = pd.concat(dfs, axis=1)
                    features = features.loc[:, ~features.columns.duplicated()]
                    logger.info(f"Loaded {len(features.columns)} features from {len(feature_files)} files")
                    return features
        
        # Try generating fresh using Phase 1 code
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from features.engineering import FeatureGenerator
            from data.orchestrator import DataOrchestrator
            
            orchestrator = DataOrchestrator()
            raw_data = orchestrator.fetch_all_data()
            
            generator = FeatureGenerator()
            features = generator.generate_all_features(raw_data)
            
            logger.info(f"Generated {len(features.columns)} features")
            return features
            
        except ImportError as e:
            logger.error(f"Could not import Phase 1 modules: {e}")
            raise FileNotFoundError("No feature data available. Run Phase 1 first.")
    
    def load_price_data(self) -> pd.DataFrame:
        """Load price data for target generation."""
        logger.info("Loading price data...")
        
        # Try cache first
        try:
            _, returns = self.load_from_cache()
            # Reconstruct price from returns
            price = pd.DataFrame({'close': (1 + returns).cumprod() * 10000}, index=returns.index)
            return price
        except FileNotFoundError:
            pass
        
        # Try loading from existing pipeline
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.binance import BinanceDataFetcher
            
            fetcher = BinanceDataFetcher()
            price_data = fetcher.fetch_ohlcv()
            return price_data
            
        except ImportError:
            pass
        
        # Fallback: load from file
        price_paths = [
            Path("data/price/btc_4h.parquet"),
            Path(__file__).parent.parent / "data" / "price" / "btc_4h.parquet"
        ]
        
        for price_file in price_paths:
            if price_file.exists():
                return pd.read_parquet(price_file)
        
        raise FileNotFoundError("Could not load price data")
    
    def create_target(
        self, 
        price_data: pd.DataFrame = None,
        returns: pd.Series = None,
        horizon_bars: int = 1,
        target_type: str = "direction"
    ) -> pd.Series:
        """
        Create prediction target.
        
        Args:
            price_data: DataFrame with 'close' column
            returns: Series of returns (alternative to price_data)
            horizon_bars: How many bars ahead to predict
            target_type: 'direction' (binary) or 'return' (continuous)
        
        Returns:
            Series with target values
        """
        logger.info(f"Creating {target_type} target with horizon={horizon_bars} bars")
        
        if returns is not None:
            # Use returns directly
            future_return = returns.shift(-horizon_bars)
        elif price_data is not None:
            if 'close' not in price_data.columns:
                raise ValueError("Price data must have 'close' column")
            future_return = price_data['close'].shift(-horizon_bars) / price_data['close'] - 1
        else:
            raise ValueError("Must provide either price_data or returns")
        
        if target_type == "direction":
            # Binary: 1 = up, 0 = down
            target = (future_return > 0).astype(int)
        elif target_type == "return":
            target = future_return
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Remove last rows where target would be NaN
        target = target.iloc[:-horizon_bars] if horizon_bars > 0 else target
        
        logger.info(f"Target created: {len(target)} samples")
        if target_type == "direction":
            logger.info(f"  Class balance: {target.value_counts().to_dict()}")
        
        return target
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and create target from Phase 1 cache.
        
        The Phase 1 cache (features_cache.pkl) contains:
        - 'features': DataFrame with all 431+ features
        - 'returns': Series with forward returns (might not exist)
        - 'price': Price data (might not exist)
        
        Returns:
            (features, target) tuple
        """
        import pickle
        
        # Find the cache file
        cache_paths = [
            Path("data_cache/features_cache.pkl"),
            Path(__file__).parent.parent / "data_cache" / "features_cache.pkl"
        ]
        
        cache_path = None
        for p in cache_paths:
            if p.exists():
                cache_path = p
                break
        
        if cache_path is None:
            raise FileNotFoundError(
                "Feature cache not found. Run Phase 1 first:\n"
                "  python run_exhaustive_search.py --mode single --top-n 10"
            )
        
        logger.info(f"Loading from cache: {cache_path}")
        
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        logger.info(f"Cache keys: {list(cache.keys())}")
        
        # Get features
        if 'features' in cache:
            features = cache['features']
            logger.info(f"Loaded features: {features.shape}")
        else:
            raise ValueError("Cache missing 'features' key")
        
        # Get target - try multiple approaches
        target = None
        
        # Approach 1: Use pre-computed returns if available
        if 'returns' in cache:
            returns = cache['returns']
            logger.info(f"Using cached returns: {len(returns)} samples")
            target = (returns.shift(-1) > 0).astype(int)
        
        # Approach 2: Compute from price data in cache
        elif 'price' in cache:
            price = cache['price']
            if 'close' in price.columns:
                returns = price['close'].pct_change()
                target = (returns.shift(-1) > 0).astype(int)
                logger.info(f"Generated target from cached price data")
        
        # Approach 3: Load price from parquet files
        else:
            price_paths = [
                Path("data_cache/binance/price_4h_365d.parquet"),
                Path(__file__).parent.parent / "data_cache" / "binance" / "price_4h_365d.parquet"
            ]
            
            for price_path in price_paths:
                if price_path.exists():
                    price = pd.read_parquet(price_path)
                    if 'close' in price.columns:
                        returns = price['close'].pct_change()
                        target = (returns.shift(-1) > 0).astype(int)
                        logger.info(f"Generated target from {price_path}")
                        break
        
        if target is None:
            raise ValueError(
                "Could not generate target. Cache needs either:\n"
                "  - 'returns' key with return series\n"
                "  - 'price' key with price DataFrame\n"
                "  - Or price parquet file in data_cache/binance/"
            )
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        # Drop rows where all features are NaN
        valid_rows = features.notna().any(axis=1)
        features = features.loc[valid_rows]
        target = target.loc[valid_rows]
        
        # Drop rows where target is NaN
        valid_target = target.notna()
        features = features.loc[valid_target]
        target = target.loc[valid_target]
        
        # Remove last row (no future return available)
        features = features.iloc[:-1]
        target = target.iloc[:-1]
        
        logger.info(f"Final dataset: {len(features)} samples, {len(features.columns)} features")
        logger.info(f"Date range: {features.index[0]} to {features.index[-1]}")
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        
        return features, target
    
    def get_feature_tiers(self) -> Dict[str, list]:
        """Get feature tier assignments from Phase 1 results."""
        return {
            'tier1': self.feature_priority.get('tier1_features', {}).get('features', []),
            'tier2': self.feature_priority.get('tier2_features', {}).get('features', []),
            'tier3': self.feature_priority.get('tier3_features', {}).get('features', []),
            'blacklist': self.feature_priority.get('blacklist_features', {}).get('features', [])
        }
