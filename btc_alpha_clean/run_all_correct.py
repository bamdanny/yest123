#!/usr/bin/env python3
"""
MASTER RUN SCRIPT - Corrected Implementations

This script runs all the corrected models in the right order.

Usage:
  python run_all_correct.py           # Run everything
  python run_all_correct.py --test    # Run tests only
  python run_all_correct.py --simple  # Run simple rules only
  python run_all_correct.py --ml      # Run ML only
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, desc):
    """Run a command and report results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {desc}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"FAILED: {desc}")
        return False
    
    logger.info(f"SUCCESS: {desc}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run corrected BTC Alpha models")
    parser.add_argument('--test', action='store_true', help='Run tests only')
    parser.add_argument('--simple', action='store_true', help='Run simple rules only')
    parser.add_argument('--ml', action='store_true', help='Run ML training only')
    parser.add_argument('--skip-data', action='store_true', help='Skip data fetch')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("BTC ALPHA - CORRECTED IMPLEMENTATIONS")
    logger.info("="*60)
    logger.info("")
    logger.info("This runs the FIXED code that properly calculates:")
    logger.info("  - Sharpe ratio (annualized correctly)")
    logger.info("  - Win rate (actual wins/losses)")
    logger.info("  - Returns (compound, not sum)")
    logger.info("  - Binary target (for classifiers)")
    logger.info("")
    
    if args.test:
        # Run tests only
        return run_command([sys.executable, 'test_fixes.py'], 'Math validation tests')
    
    steps = []
    
    if not args.skip_data:
        # Step 1: Make sure data exists (run exhaustive search if needed)
        price_path = Path("data_cache/binance/price_4h_365d.parquet")
        if not price_path.exists():
            steps.append(([sys.executable, 'run_exhaustive_search.py', '--mode', 'single', '--top-n', '10'],
                         'Fetch data and generate features'))
        else:
            logger.info("Data cache exists - skipping data fetch")
    
    if not args.ml:
        # Step 2: Run simple rules
        steps.append(([sys.executable, 'simple_rules_correct.py'], 
                     'Simple Rules (corrected)'))
    
    if not args.simple:
        # Step 3: Run ML training
        steps.append(([sys.executable, 'train_anchored_correct.py'],
                     'Anchored Ensemble ML (corrected)'))
    
    # Run all steps
    success = True
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            success = False
            break
    
    # Summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("")
        logger.info("Models saved to:")
        logger.info("  models/simple_rules_correct.pkl")
        logger.info("  models/anchored_model_correct.pkl")
        logger.info("")
        logger.info("To run the scanner:")
        logger.info("  python run_scanner.py --once")
    else:
        logger.error("SOME STEPS FAILED")
        logger.info("="*60)
        logger.info("Check the error messages above and fix any issues.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
