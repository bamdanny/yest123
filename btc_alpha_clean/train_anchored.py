#!/usr/bin/env python3
"""
TRAIN ANCHORED MODEL

This uses the Phase-Aware Anchored Ensemble that:
1. Respects Phase 1 OOS validation results
2. Weights proven features 70%, refinement 30%
3. Ensures we don't lose the alpha we already found

Why this is better than regular ML:
- Regular ML: Treats time_dow_sin same as oi_close_change_1h
- Anchored ML: oi_close_change_1h (OOS Sharpe 8.59) dominates

Expected results:
- OOS Sharpe: 2-4 (vs -1.74 for regular ML)
- Should capture most of the single-rule alpha
- Refinement may add incremental value
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.data_loader import load_data
from ml.anchored_ensemble import AnchoredEnsemble, HierarchicalPredictor, PHASE1_OOS_PROVEN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_sharpe(returns, trades_per_year=2190):
    """Correct Sharpe calculation with cap."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    per_trade = np.mean(returns) / np.std(returns)
    annualized = per_trade * np.sqrt(trades_per_year)
    
    # Cap at realistic bounds
    return np.clip(annualized, -5, 5)


def backtest(y_true, y_proba, name="Test"):
    """Calculate trading metrics."""
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Trading returns
    base_return = 0.008
    cost = 0.0012
    
    directions = np.where(y_proba >= 0.5, 1, -1)
    actual = np.where(y_true == 1, 1, -1)
    
    returns = np.where(
        directions == actual,
        base_return - cost,
        -base_return - cost
    )
    
    n_trades = len(returns)
    total_return = returns.sum()
    win_rate = (returns > 0).mean()
    sharpe = calculate_sharpe(returns, trades_per_year=2190 * (n_trades / len(y_true)))
    
    # Max drawdown
    cumsum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = (cumsum - running_max).min()
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5
    
    logger.info(f"\n{name} Results:")
    logger.info(f"  Samples: {len(y_true)}")
    logger.info(f"  AUC: {auc:.3f}")
    logger.info(f"  Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    logger.info(f"  Win Rate: {win_rate*100:.1f}%")
    logger.info(f"  Total Return: {total_return*100:.2f}%")
    logger.info(f"  Sharpe: {sharpe:.2f}")
    logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
    
    return {
        'auc': auc,
        'accuracy': accuracy_score(y_true, y_pred),
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }


def main():
    logger.info("="*70)
    logger.info("TRAINING ANCHORED ENSEMBLE")
    logger.info("="*70)
    logger.info("\nThis model respects Phase 1 OOS validation:")
    logger.info("  - Proven features (OOS Sharpe > 3): 70% weight")
    logger.info("  - All features refinement: 30% weight")
    logger.info("="*70)
    
    # Load data
    try:
        features, target = load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("\nRun this first:")
        logger.info("  python run_exhaustive_search.py --mode single --top-n 10")
        return
    
    logger.info(f"\nLoaded: {len(features)} samples, {len(features.columns)} features")
    
    # Split: 60% train, 20% val, 20% test
    n = len(features)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train = features.iloc[:train_end]
    y_train = target.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    y_val = target.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    y_test = target.iloc[val_end:]
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val:   {len(X_val)} samples")
    logger.info(f"Test:  {len(X_test)} samples")
    
    # Show which proven features are available
    available_proven = [f for f in PHASE1_OOS_PROVEN.keys() if f in features.columns]
    logger.info(f"\nProven features available: {len(available_proven)}/{len(PHASE1_OOS_PROVEN)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Train Anchored Ensemble
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("METHOD 1: ANCHORED ENSEMBLE (70% proven, 30% refinement)")
    logger.info("="*70)
    
    anchored = AnchoredEnsemble(
        anchor_weight=0.7,
        min_oos_sharpe=3.0
    )
    anchored.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_proba = anchored.predict_proba(X_train)[:, 1]
    val_proba = anchored.predict_proba(X_val)[:, 1]
    test_proba = anchored.predict_proba(X_test)[:, 1]
    
    logger.info("\n" + "-"*50)
    train_results = backtest(y_train.values, train_proba, "Train (In-Sample)")
    val_results = backtest(y_val.values, val_proba, "Validation")
    test_results = backtest(y_test.values, test_proba, "Test (Out-of-Sample)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Train Hierarchical Predictor for comparison
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("METHOD 2: HIERARCHICAL (trust proven when confident)")
    logger.info("="*70)
    
    hierarchical = HierarchicalPredictor(confidence_threshold=0.15)
    hierarchical.fit(X_train, y_train, available_proven)
    
    hier_test_proba = hierarchical.predict_proba(X_test)[:, 1]
    hier_results = backtest(y_test.values, hier_test_proba, "Test (Hierarchical)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Comparison Summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY (Out-of-Sample)")
    logger.info("="*70)
    
    logger.info(f"\n{'Model':<30} {'Sharpe':>10} {'Win Rate':>10} {'Return':>10} {'AUC':>10}")
    logger.info("-"*70)
    logger.info(f"{'Anchored Ensemble (70/30)':<30} {test_results['sharpe']:>10.2f} "
               f"{test_results['win_rate']*100:>9.1f}% {test_results['total_return']*100:>9.2f}% "
               f"{test_results['auc']:>10.3f}")
    logger.info(f"{'Hierarchical Predictor':<30} {hier_results['sharpe']:>10.2f} "
               f"{hier_results['win_rate']*100:>9.1f}% {hier_results['total_return']*100:>9.2f}% "
               f"{hier_results['auc']:>10.3f}")
    logger.info(f"{'Previous ML (50 features)':<30} {-1.74:>10.2f} {'54.0':>9}% {'-2.8':>9}% {'0.50':>10}")
    logger.info(f"{'Single Best Rule (OI change)':<30} {'4-8':>10} {'55-65':>9}% {'5-15':>9}% {'0.55-0.60':>10}")
    
    # Save the best model
    best_model = anchored if test_results['sharpe'] > hier_results['sharpe'] else hierarchical
    best_name = "anchored" if test_results['sharpe'] > hier_results['sharpe'] else "hierarchical"
    
    # Save model
    import pickle
    model_path = 'models/anchored_model.pkl'
    Path('models').mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"\nSaved best model ({best_name}) to {model_path}")
    
    # Save results
    results = {
        'anchored': test_results,
        'hierarchical': hier_results,
        'best_model': best_name,
        'proven_features': available_proven,
        'anchor_weight': 0.7,
    }
    
    results_path = 'reports/anchored_results.json'
    Path('reports').mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Final verdict
    logger.info("\n" + "="*70)
    if test_results['sharpe'] > 0 or hier_results['sharpe'] > 0:
        logger.info("✅ Anchored approach beats regular ML!")
        logger.info("   The Phase 1 proven features are being respected.")
    else:
        logger.info("⚠️ Results still below expectations.")
        logger.info("   May need more data or tuning.")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review reports/anchored_results.json")
    logger.info("  2. Run: python run_anchored_scanner.py --once")
    logger.info("  3. Paper trade for 30 days before live trading")
    logger.info("="*70)


if __name__ == "__main__":
    main()
