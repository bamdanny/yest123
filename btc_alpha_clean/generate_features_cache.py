#!/usr/bin/env python3
"""
Generate features cache for the corrected training scripts.

Run this AFTER run_exhaustive_search.py has completed.
It will create a cache file that simple_rules_correct.py and train_anchored_correct.py can use.
"""

import pickle
import logging
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("GENERATING FEATURES CACHE")
    logger.info("=" * 60)
    
    cache_path = Path("data_cache/features_cache.pkl")
    
    if cache_path.exists():
        logger.info(f"Cache already exists at {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        features = cache['features']
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Columns: {features.columns[:10].tolist()}...")
        return
    
    # Load from existing data pipeline
    logger.info("Loading data from pipeline...")
    
    try:
        from data.orchestrator import DataOrchestrator
        from features.engineering import FeatureGenerator
        from features.targets import TargetGenerator
        
        # Fetch data
        orchestrator = DataOrchestrator()
        data = orchestrator.fetch_all_data(lookback_days=90)
        
        # Create master dataset
        master_df = orchestrator.create_master_dataset(data)
        logger.info(f"Master dataset: {master_df.shape}")
        
        # Generate features
        generator = FeatureGenerator()
        features = generator.generate_all_features(master_df)
        logger.info(f"Generated features: {features.shape}")
        
        # Save cache
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'features': features}, f)
        
        logger.info(f"Saved cache to {cache_path}")
        logger.info("")
        logger.info("You can now run:")
        logger.info("  python simple_rules_correct.py")
        logger.info("  python train_anchored_correct.py")
        
    except Exception as e:
        logger.error(f"Failed to generate features: {e}")
        logger.info("")
        logger.info("Make sure you've run:")
        logger.info("  python run_exhaustive_search.py")
        logger.info("")
        logger.info("This creates the data cache that we need.")
        raise


if __name__ == "__main__":
    main()
