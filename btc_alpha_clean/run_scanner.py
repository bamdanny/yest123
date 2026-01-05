#!/usr/bin/env python3
"""
Live Scanner using corrected models.

Loads trained model and generates signals on current data.
"""

import argparse
import pickle
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_type: str = 'simple'):
    """Load trained model."""
    if model_type == 'simple':
        path = Path("models/simple_rules_correct.pkl")
    elif model_type == 'anchored':
        path = Path("models/anchored_model_correct.pkl")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            f"Run train script first."
        )
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model from {path}")
    return model


def load_latest_features():
    """Load latest feature data."""
    cache_path = Path("data_cache/features_cache.pkl")
    
    if not cache_path.exists():
        raise FileNotFoundError("No features cache. Run exhaustive search first.")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    
    # Get latest row
    latest = features.iloc[[-1]]
    timestamp = latest.index[0]
    
    logger.info(f"Latest data timestamp: {timestamp}")
    
    return latest, timestamp


def get_signal_name(signal):
    """Convert numeric signal to name."""
    if signal == 1:
        return "🟢 LONG"
    elif signal == -1:
        return "🔴 SHORT"
    else:
        return "⚪ NO TRADE"


def run_once(model_type: str = 'simple'):
    """Run single prediction."""
    logger.info("=" * 60)
    logger.info("BTC ALPHA SCANNER")
    logger.info("=" * 60)
    
    # Load model
    model = load_model(model_type)
    
    # Load latest data
    features, timestamp = load_latest_features()
    
    # Generate signal
    if model_type == 'simple':
        signals = model.predict(features)
        signal = signals[0]
        confidence = None
    else:
        proba = model.predict_proba(features)[0, 1]
        if proba > 0.55:
            signal = 1
        elif proba < 0.45:
            signal = -1
        else:
            signal = 0
        confidence = abs(proba - 0.5) * 2  # 0-1 scale
    
    # Output
    logger.info("")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Signal: {get_signal_name(signal)}")
    if confidence is not None:
        logger.info(f"Confidence: {confidence:.1%}")
    logger.info("")
    
    return signal


def run_continuous(model_type: str = 'simple', interval_minutes: int = 15):
    """Run continuously."""
    logger.info(f"Starting continuous scanner (every {interval_minutes} min)")
    logger.info("Press Ctrl+C to stop")
    
    while True:
        try:
            run_once(model_type)
            logger.info(f"Next scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(60)  # Wait 1 min on error


def main():
    parser = argparse.ArgumentParser(description="BTC Alpha Scanner")
    parser.add_argument('--model', choices=['simple', 'anchored'], 
                        default='simple', help='Model type')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--interval', type=int, default=15,
                        help='Scan interval in minutes')
    
    args = parser.parse_args()
    
    if args.once:
        run_once(args.model)
    else:
        run_continuous(args.model, args.interval)


if __name__ == "__main__":
    main()
