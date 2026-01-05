#!/usr/bin/env python3
"""
CREATE ENSEMBLE MODEL - Production Model for Scanner

Creates a properly structured ensemble model that can be loaded by run_ml_scanner.py.
Uses the EXACT same logic as simple_ensemble.py (Sharpe 12.18 validated).

Model structure:
{
    'model_type': 'rule_ensemble',
    'indicators': [...],
    'min_position_threshold': 0.1,
    'fitted_params': {...},
    'validation_metrics': {...},
    'created_at': timestamp
}
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CONFIGURATIONS (Validated in Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

INDICATOR_CONFIGS = [
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 10.15,
        "weight": 0.30
    },
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_accel",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 80,
        "oos_sharpe": 8.41,
        "weight": 0.25
    },
    {
        "name": "price_rsi_14_lag_48h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 1.5,
        "oos_sharpe": 6.33,
        "weight": 0.20
    },
    {
        "name": "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 2.0,
        "oos_sharpe": 4.98,
        "weight": 0.15
    },
    {
        "name": "sent_feat_fg_zscore_90d",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 5.62,
        "weight": 0.10
    },
]


def load_data():
    """Load features and targets from cache."""
    cache_path = Path("data_cache/features_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError("No features cache. Run: python run_exhaustive_search.py first")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    targets = cache['targets']
    
    # Filter to numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features = features[numeric_cols].copy()
    
    # Get return_simple_6h (same as Phase 1)
    if 'return_simple_6h' in targets.columns:
        returns = targets['return_simple_6h'].copy()
    else:
        ret_cols = [c for c in targets.columns if 'return_simple' in c]
        returns = targets[ret_cols[0]].copy() if ret_cols else None
    
    if returns is None:
        raise ValueError("No return columns in targets")
    
    # Align
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len].reset_index(drop=True)
    returns = returns.iloc[:min_len].reset_index(drop=True)
    
    valid_idx = returns.dropna().index
    features = features.loc[valid_idx].reset_index(drop=True)
    returns = returns.loc[valid_idx].reset_index(drop=True)
    
    return features, returns


def fit_thresholds(features: pd.DataFrame, train_mask: np.ndarray, 
                   configs: List[Dict]) -> List[Dict]:
    """
    Fit thresholds on training data for each indicator.
    
    Returns list of indicator configs with fitted thresholds added.
    """
    fitted_indicators = []
    
    for config in configs:
        if config['name'] not in features.columns:
            logger.warning(f"Feature {config['name']} not found, skipping")
            continue
        
        feature_values = features[config['name']].values
        train_values = feature_values[train_mask]
        
        indicator = {
            'name': config['name'],
            'direction': config['direction'],
            'threshold_type': config['threshold_type'],
            'threshold_value': config['threshold_value'],
            'weight': config['weight'],
            'oos_sharpe': config['oos_sharpe']
        }
        
        if config['threshold_type'] == 'percentile':
            pct_upper = config['threshold_value']
            pct_lower = 100 - pct_upper
            
            indicator['upper_threshold'] = float(np.nanpercentile(train_values, pct_upper))
            indicator['lower_threshold'] = float(np.nanpercentile(train_values, pct_lower))
            
        else:  # zscore
            mean = float(np.nanmean(train_values))
            std = float(np.nanstd(train_values))
            z_thresh = config['threshold_value']
            
            indicator['mean'] = mean
            indicator['std'] = std
            indicator['upper_threshold'] = mean + z_thresh * std
            indicator['lower_threshold'] = mean - z_thresh * std
        
        fitted_indicators.append(indicator)
        
        logger.info(f"  {config['name'][:50]}")
        logger.info(f"    upper: {indicator['upper_threshold']:.6f}, lower: {indicator['lower_threshold']:.6f}")
    
    return fitted_indicators


def generate_signal_from_indicator(value: float, indicator: Dict) -> int:
    """Generate signal for a single indicator and value."""
    if np.isnan(value):
        return 0
    
    direction = indicator['direction']
    upper = indicator['upper_threshold']
    lower = indicator['lower_threshold']
    
    if direction == 1:
        if value > upper:
            return 1   # Long when high
        elif value < lower:
            return -1  # Short when low
    else:  # direction == -1
        if value > upper:
            return -1  # Short when high
        elif value < lower:
            return 1   # Long when low
    
    return 0


def generate_ensemble_signal(feature_row: Dict, indicators: List[Dict], 
                             min_position: float = 0.1) -> int:
    """
    Generate ensemble signal from a single row of features.
    
    Args:
        feature_row: Dict of feature_name -> value
        indicators: List of fitted indicator configs
        min_position: Minimum weighted position to trade
    
    Returns:
        1 (long), -1 (short), or 0 (no trade)
    """
    signals = []
    weights = []
    
    for ind in indicators:
        value = feature_row.get(ind['name'], np.nan)
        signal = generate_signal_from_indicator(value, ind)
        signals.append(signal)
        weights.append(ind['weight'])
    
    if len(signals) == 0:
        return 0
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average position
    position = np.sum(np.array(signals) * weights)
    
    if position > min_position:
        return 1
    elif position < -min_position:
        return -1
    return 0


def calc_sharpe(trade_returns: np.ndarray, period_days: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(trade_returns) < 5 or period_days <= 0:
        return 0.0
    
    total_return = np.prod(1 + trade_returns) - 1
    daily_return = (1 + total_return) ** (1 / period_days) - 1
    
    trade_std = np.std(trade_returns)
    trades_per_day = len(trade_returns) / period_days
    daily_std = trade_std * np.sqrt(trades_per_day)
    
    if daily_std < 1e-10:
        return 0.0
    
    return (daily_return / daily_std) * np.sqrt(365)


def validate_model(features: pd.DataFrame, returns: pd.Series, 
                   indicators: List[Dict], min_position: float,
                   train_ratio: float = 0.6) -> Dict:
    """
    Validate model produces expected results.
    
    Returns validation metrics.
    """
    n = len(features)
    n_train = int(n * train_ratio)
    
    return_values = returns.values
    
    # Generate signals for all data
    all_signals = []
    weights = []
    
    for ind in indicators:
        feature_values = features[ind['name']].values
        signals = np.zeros(len(feature_values))
        
        for i, val in enumerate(feature_values):
            signals[i] = generate_signal_from_indicator(val, ind)
        
        all_signals.append(signals)
        weights.append(ind['weight'])
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Combine
    all_signals = np.array(all_signals)
    combined_position = np.sum(all_signals * weights[:, np.newaxis], axis=0)
    
    # Final signal
    final_signal = np.zeros(n)
    final_signal[combined_position > min_position] = 1
    final_signal[combined_position < -min_position] = -1
    
    # OOS metrics
    oos_signals = final_signal[n_train:]
    oos_returns = return_values[n_train:]
    
    trade_mask = oos_signals != 0
    trading_returns = (oos_signals * oos_returns)[trade_mask]
    
    n_trades = len(trading_returns)
    n_long = np.sum(oos_signals[trade_mask] == 1)
    n_short = np.sum(oos_signals[trade_mask] == -1)
    
    period_days = (n - n_train) / 6
    
    if n_trades >= 5:
        wins = np.sum(trading_returns > 0)
        win_rate = wins / n_trades
        total_return = np.prod(1 + trading_returns) - 1
        sharpe = calc_sharpe(trading_returns, period_days)
    else:
        win_rate = 0
        total_return = 0
        sharpe = 0
    
    return {
        'oos_sharpe': sharpe,
        'oos_win_rate': win_rate,
        'oos_total_return': total_return,
        'oos_trades': n_trades,
        'oos_long': n_long,
        'oos_short': n_short,
        'period_days': period_days
    }


def create_ensemble_model(train_ratio: float = 0.6, 
                          min_position: float = 0.1) -> Dict[str, Any]:
    """
    Create and validate ensemble model.
    
    Returns model dictionary ready for saving.
    """
    logger.info("=" * 70)
    logger.info("CREATE ENSEMBLE MODEL")
    logger.info("=" * 70)
    
    # Load data
    features, returns = load_data()
    n = len(features)
    n_train = int(n * train_ratio)
    
    logger.info(f"\nData: {n} samples")
    logger.info(f"Train: {n_train} ({train_ratio*100:.0f}%)")
    logger.info(f"Test: {n - n_train} ({(1-train_ratio)*100:.0f}%)")
    
    # Create train mask
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    
    # Fit thresholds
    logger.info("\nFitting thresholds on training data:")
    fitted_indicators = fit_thresholds(features, train_mask, INDICATOR_CONFIGS)
    
    # Validate model
    logger.info("\nValidating model...")
    metrics = validate_model(features, returns, fitted_indicators, min_position, train_ratio)
    
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"  OOS Sharpe: {metrics['oos_sharpe']:.2f}")
    logger.info(f"  OOS Win Rate: {metrics['oos_win_rate']*100:.1f}%")
    logger.info(f"  OOS Total Return: {metrics['oos_total_return']*100:.1f}%")
    logger.info(f"  OOS Trades: {metrics['oos_trades']} ({metrics['oos_long']}L + {metrics['oos_short']}S)")
    
    # Create model structure
    model = {
        'model_type': 'rule_ensemble',
        'version': 'v38',
        'indicators': fitted_indicators,
        'min_position_threshold': min_position,
        'train_ratio': train_ratio,
        'n_train_samples': n_train,
        'validation_metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'description': 'Validated ensemble model using 5 Phase 1 indicators'
    }
    
    return model


def save_model(model: Dict, path: Path = None):
    """Save model to disk."""
    if path is None:
        path = Path("models/ensemble_model.pkl")
    
    path.parent.mkdir(exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"\n✅ Model saved to {path}")


def main():
    # Create model
    model = create_ensemble_model(train_ratio=0.6, min_position=0.1)
    
    # Save model
    save_model(model)
    
    # Verify saved model can be loaded
    logger.info("\nVerifying saved model...")
    with open("models/ensemble_model.pkl", 'rb') as f:
        loaded = pickle.load(f)
    
    logger.info(f"  Model type: {loaded['model_type']}")
    logger.info(f"  Indicators: {len(loaded['indicators'])}")
    logger.info(f"  Min position: {loaded['min_position_threshold']}")
    logger.info(f"  OOS Sharpe: {loaded['validation_metrics']['oos_sharpe']:.2f}")
    
    # Sanity check
    if loaded['validation_metrics']['oos_sharpe'] > 8:
        logger.info("\n✅ Model validated successfully!")
        logger.info("   Expected ~12 Sharpe, got {:.2f}".format(
            loaded['validation_metrics']['oos_sharpe']))
    else:
        logger.warning("\n⚠️ Model Sharpe lower than expected")
        logger.warning("   Expected ~12, got {:.2f}".format(
            loaded['validation_metrics']['oos_sharpe']))
    
    # Print model summary
    logger.info("\n" + "=" * 70)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nIndicators:")
    for ind in loaded['indicators']:
        logger.info(f"  {ind['name'][:50]}")
        logger.info(f"    dir={ind['direction']}, weight={ind['weight']}, "
                   f"upper={ind['upper_threshold']:.4f}, lower={ind['lower_threshold']:.4f}")
    
    logger.info(f"\nUsage:")
    logger.info(f"  from create_ensemble_model import generate_ensemble_signal")
    logger.info(f"  signal = generate_ensemble_signal(feature_dict, model['indicators'], "
               f"model['min_position_threshold'])")


if __name__ == "__main__":
    main()
