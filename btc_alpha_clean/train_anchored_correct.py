#!/usr/bin/env python3
"""
CORRECTED Anchored Model - NO Data Leakage

CRITICAL FIX: Removed LightGBM refinement model that caused data leakage.

Previous broken version had:
- 732 features with 323 samples (p >> n disaster)
- Train accuracy: 98.1% (MEMORIZATION)
- Refinement model: 100% train accuracy (SMOKING GUN)
- OOS Sharpe: 23.50 (IMPOSSIBLE - Renaissance gets ~3-4)

This version:
- Uses ONLY the 10 proven features
- Simple logistic regression (can't memorize)
- Train accuracy should be ~55-60% (realistic)
- OOS Sharpe expected: 1-4 (realistic)
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ONLY proven features from Phase 1 OOS validation
PROVEN_FEATURES = [
    "deriv_feat_cg_oi_aggregated_oi_close_change_1h",   # Sharpe 8.76
    "deriv_feat_cg_oi_aggregated_oi_close_accel",       # Sharpe 7.38
    "price_rsi_14_lag_48h",                             # Sharpe 5.22
    "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",  # Sharpe 4.98
    "sent_feat_fg_zscore_90d",                          # Sharpe 4.59
    "deriv_feat_cg_oi_aggregated_oi_high_accel",        # Sharpe 4.49
    "deriv_feat_cg_oi_aggregated_oi_low_change_1h",     # Sharpe 3.67
    "deriv_feat_cg_funding_oi_weighted_funding_oi_open_zscore_168h",  # Sharpe 3.63
    "deriv_feat_cg_oi_aggregated_oi_low_accel",         # Sharpe 3.46
]


class SimpleAnchoredModel:
    """
    Simple model using ONLY proven features.
    
    NO refinement model - that was the source of data leakage.
    """
    
    def __init__(self, proven_features=None):
        self.proven_features = proven_features or PROVEN_FEATURES
        self.model = LogisticRegression(
            C=0.1,  # Strong regularization to prevent overfit
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.available_features = []
        self.fitted = False
    
    def fit(self, X, y):
        """Fit on ONLY proven features."""
        # Find available features
        self.available_features = [f for f in self.proven_features if f in X.columns]
        
        if len(self.available_features) < 3:
            logger.warning(f"Only {len(self.available_features)} proven features available!")
        
        logger.info(f"Using {len(self.available_features)} proven features (NOT 732)")
        for f in self.available_features:
            logger.info(f"  - {f[:55]}")
        
        # Prepare data
        X_subset = X[self.available_features].copy()
        
        # Drop constant columns
        non_constant = X_subset.std() > 1e-10
        X_subset = X_subset.loc[:, non_constant]
        self.available_features = list(X_subset.columns)
        
        # Fill NaN with median
        X_subset = X_subset.fillna(X_subset.median())
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_subset)
        
        # Fit
        y_clean = y.fillna(0).astype(int).values
        self.model.fit(X_scaled, y_clean)
        
        # Report train accuracy
        train_proba = self.model.predict_proba(X_scaled)[:, 1]
        train_acc = np.mean((train_proba > 0.5).astype(int) == y_clean)
        
        logger.info(f"\nTrain accuracy: {train_acc:.1%}")
        
        # SANITY CHECK: Train accuracy should NOT be > 70% for financial data
        if train_acc > 0.70:
            logger.error(f"❌ Train accuracy {train_acc:.1%} > 70% - OVERFITTING!")
        elif train_acc > 0.60:
            logger.warning(f"⚠️  Train accuracy {train_acc:.1%} is borderline high")
        else:
            logger.info(f"✅ Train accuracy {train_acc:.1%} is realistic for financial data")
        
        self.fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if not self.fitted:
            raise ValueError("Must fit first")
        
        X_subset = X[self.available_features].copy()
        X_subset = X_subset.fillna(X_subset.median())
        X_scaled = self.scaler.transform(X_subset)
        
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


def load_data():
    """Load features and prices."""
    cache_path = Path("data_cache/features_cache.pkl")
    
    if not cache_path.exists():
        raise FileNotFoundError("No features cache. Run: python run_exhaustive_search.py first")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    features = cache['features']
    
    # Filter to numeric only
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features = features[numeric_cols]
    
    logger.info(f"Loaded features: {features.shape}")
    
    # Load prices
    price_path = Path("data_cache/binance/price_4h_365d.parquet")
    prices = pd.read_parquet(price_path)
    
    if 'timestamp' in prices.columns:
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        prices = prices.set_index('timestamp')
    
    # Align
    n_features = len(features)
    prices_aligned = prices.iloc[-n_features:].copy()
    features = features.copy()
    features.index = prices_aligned.index
    
    logger.info(f"Price range: {prices_aligned.index.min()} to {prices_aligned.index.max()}")
    
    # Create binary target
    close = prices_aligned['close']
    returns = close.pct_change().shift(-1)
    target = (returns > 0).astype(int)
    
    # Drop last row (no target)
    features = features.iloc[:-1]
    target = target.iloc[:-1]
    close = close.iloc[:-1]
    
    logger.info(f"Final dataset: {len(features)} samples")
    logger.info(f"Target: {target.sum()} up, {len(target) - target.sum()} down")
    
    return features, target, close


def evaluate(model, X, y, prices, name="Test"):
    """Evaluate with trading metrics."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba > 0.5).astype(int)
    
    accuracy = np.mean(preds == y.values)
    
    # Trading: long when pred=1, short when pred=0
    returns = prices.pct_change().fillna(0)
    common_idx = X.index.intersection(returns.index)
    returns = returns.loc[common_idx].values[1:]
    preds_aligned = preds[:-1]
    
    positions = 2 * preds_aligned - 1  # +1 for long, -1 for short
    trade_returns = positions * returns
    
    # Sharpe
    if len(trade_returns) > 1 and np.std(trade_returns) > 1e-10:
        sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(365 * 6)
    else:
        sharpe = 0
    
    wins = np.sum(trade_returns > 0)
    losses = np.sum(trade_returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    total_return = np.prod(1 + trade_returns) - 1
    
    logger.info(f"\n=== {name} Results ===")
    logger.info(f"  Accuracy: {accuracy:.1%}")
    logger.info(f"  Win Rate: {win_rate:.1%} ({wins}W / {losses}L)")
    logger.info(f"  Total Return: {total_return:.1%}")
    logger.info(f"  Sharpe: {sharpe:.2f}")
    logger.info(f"  Predictions: {np.sum(preds==1)} longs, {np.sum(preds==0)} shorts")
    
    return {
        'accuracy': accuracy,
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe
    }


def main():
    logger.info("=" * 70)
    logger.info("CORRECTED ANCHORED MODEL - NO DATA LEAKAGE")
    logger.info("=" * 70)
    
    logger.info("\nFixes applied:")
    logger.info("  ✓ Removed LightGBM refinement (source of leakage)")
    logger.info("  ✓ Using ONLY 10 proven features (not 732)")
    logger.info("  ✓ Simple logistic regression (can't memorize)")
    logger.info("  ✓ Realistic expectations (Sharpe 1-4, not 23)")
    
    # Load data
    features, target, prices = load_data()
    
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
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    
    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    
    model = SimpleAnchoredModel(proven_features=PROVEN_FEATURES)
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    train_results = evaluate(model, X_train, y_train, prices, "Train")
    val_results = evaluate(model, X_val, y_val, prices, "Validation")
    test_results = evaluate(model, X_test, y_test, prices, "Test (OOS)")
    
    # SANITY CHECKS
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECKS")
    logger.info("=" * 60)
    
    oos_sharpe = test_results['sharpe']
    
    if oos_sharpe > 15:
        logger.error(f"❌ OOS Sharpe {oos_sharpe:.1f} > 15 - IMPOSSIBLE, DATA LEAKAGE!")
    elif oos_sharpe > 10:
        logger.warning(f"⚠️  OOS Sharpe {oos_sharpe:.1f} > 10 - Very high, verify carefully")
    elif oos_sharpe > 5:
        logger.info(f"⚠️  OOS Sharpe {oos_sharpe:.2f} > 5 - High but possible in trending market")
    elif oos_sharpe > 0:
        logger.info(f"✅ OOS Sharpe {oos_sharpe:.2f} is realistic")
    else:
        logger.warning(f"⚠️  OOS Sharpe {oos_sharpe:.2f} is negative")
    
    if train_results['sharpe'] > 5 * test_results['sharpe'] and test_results['sharpe'] > 0:
        logger.warning(f"⚠️  Train Sharpe >> OOS Sharpe - possible overfit")
    
    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    
    logger.info(f"\n{'Approach':<35} {'OOS Sharpe':>12} {'OOS WR':>10}")
    logger.info("-" * 60)
    logger.info(f"{'Phase 1 Best Single':<35} {'8.76':>12} {'73.5%':>10}")
    logger.info(f"{'Broken ML (732 features)':<35} {'23.50*':>12} {'71.0%':>10}")
    logger.info(f"{'Corrected (this)':<35} {test_results['sharpe']:>12.2f} {test_results['win_rate']*100:>9.1f}%")
    logger.info("\n* 23.50 was IMPOSSIBLE and indicated data leakage")
    
    # Save model
    model_path = Path("models/anchored_model_correct.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
