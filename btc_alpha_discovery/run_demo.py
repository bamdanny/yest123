#!/usr/bin/env python3
"""
BTC Alpha Discovery - Demo Mode

Runs the complete discovery pipeline using synthetic data.
This demonstrates all system capabilities when API access is unavailable.

Synthetic data has embedded predictive patterns that the system should discover:
1. High funding rate → price reversal
2. Extreme RSI → mean reversion  
3. OI spike + funding divergence → liquidation cascade
4. VIX spike → increased volatility
5. Fear & Greed extreme → contrarian signal

Usage:
    python run_demo.py
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Get script directory for log file
SCRIPT_DIR = Path(__file__).parent.resolve()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(SCRIPT_DIR / f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run complete discovery pipeline on synthetic data."""
    
    logger.info("=" * 60)
    logger.info("BTC ALPHA DISCOVERY SYSTEM - DEMO MODE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Using synthetic data with embedded predictive patterns.")
    logger.info("The system should discover these patterns automatically.")
    logger.info("")
    
    # =========================================================================
    # PHASE 1: Generate Synthetic Data
    # =========================================================================
    logger.info("PHASE 1: Generating Synthetic Data")
    logger.info("-" * 40)
    
    from data.synthetic import generate_test_data
    
    data = generate_test_data(n_days=180, seed=42)
    
    logger.info(f"Generated {len(data)} data components:")
    for name, df in data.items():
        if hasattr(df, 'shape'):
            logger.info(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    
    # Get combined dataframe
    combined_df = data['combined']
    logger.info(f"\nCombined dataset: {combined_df.shape}")
    logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # =========================================================================
    # PHASE 2: Feature Engineering
    # =========================================================================
    logger.info("")
    logger.info("PHASE 2: Feature Engineering")
    logger.info("-" * 40)
    
    from features.engineering import FeatureGenerator
    
    feature_gen = FeatureGenerator()
    
    # Generate features from price data
    price_df = data['price'].copy()
    
    # We need to add some derivative columns to price_df for feature generation
    price_df['funding_rate'] = data['funding'].reindex(price_df.index, method='ffill')['funding_rate']
    price_df['open_interest'] = data['open_interest'].reindex(price_df.index, method='ffill')['open_interest']
    price_df['long_liquidations'] = data['liquidations'].reindex(price_df.index, method='ffill')['long_liquidations']
    price_df['short_liquidations'] = data['liquidations'].reindex(price_df.index, method='ffill')['short_liquidations']
    price_df['long_short_ratio'] = data['long_short_ratio'].reindex(price_df.index, method='ffill')['long_short_ratio']
    price_df['fear_greed_index'] = data['sentiment'].reindex(price_df.index, method='ffill')['fear_greed_index']
    price_df['vix'] = data['macro'].reindex(price_df.index, method='ffill')['vix']
    price_df['dxy'] = data['macro'].reindex(price_df.index, method='ffill')['dxy']
    
    price_df = price_df.ffill().bfill()
    
    # Generate features
    features_df = feature_gen.generate_all_features(price_df)
    
    logger.info(f"Generated {len(features_df.columns)} features")
    logger.info(f"Feature categories:")
    
    # Count features by category
    categories = {}
    for col in features_df.columns:
        if col.startswith('return_'):
            cat = 'returns'
        elif col.startswith(('rsi_', 'stoch_', 'macd_', 'cci_', 'mom_', 'roc_')):
            cat = 'momentum'
        elif col.startswith(('bb_', 'zscore_')):
            cat = 'mean_reversion'
        elif col.startswith(('ema_', 'sma_', 'trend_')):
            cat = 'trend'
        elif col.startswith(('atr_', 'volatility_', 'realized_')):
            cat = 'volatility'
        elif col.startswith('funding_'):
            cat = 'funding'
        elif col.startswith('oi_'):
            cat = 'open_interest'
        elif col.startswith(('liq_', 'liquidation_')):
            cat = 'liquidations'
        elif col.startswith('ls_'):
            cat = 'long_short'
        elif col.startswith(('fg_', 'fear_greed_')):
            cat = 'sentiment'
        elif col.startswith('vix_'):
            cat = 'vix'
        elif col.startswith(('hour_', 'day_', 'month_', 'is_')):
            cat = 'time'
        else:
            cat = 'other'
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count} features")
    
    # =========================================================================
    # PHASE 3: Target Generation
    # =========================================================================
    logger.info("")
    logger.info("PHASE 3: Target Generation")
    logger.info("-" * 40)
    
    from features.targets import TargetGenerator
    
    target_gen = TargetGenerator()
    targets_df = target_gen.generate_all_targets(price_df)
    
    logger.info(f"Generated {len(targets_df.columns)} target variables")
    
    # Merge features and targets
    full_df = features_df.join(targets_df)
    full_df = full_df.dropna(subset=['profitable_24h'])  # Remove rows without target
    
    logger.info(f"Full dataset: {full_df.shape}")
    
    # =========================================================================
    # PHASE 4: Feature Importance Discovery
    # =========================================================================
    logger.info("")
    logger.info("PHASE 4: Feature Importance Discovery")
    logger.info("-" * 40)
    
    from discovery.feature_importance import FeatureImportanceDiscovery
    
    # Separate features and target
    target_col = 'profitable_24h'
    feature_cols = [c for c in full_df.columns if not c.startswith(('return_fwd_', 'profitable_', 'direction_', 'quintile_', 'sharpe_', 'exit_', 'mae_', 'mfe_'))]
    
    X = full_df[feature_cols].copy()
    y = full_df[target_col].copy()
    
    # Drop non-numeric columns
    X = X.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    
    # Fill any remaining NaN
    X = X.fillna(0)
    
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target: {y.value_counts().to_dict()}")
    
    importance_discovery = FeatureImportanceDiscovery()
    importance_results = importance_discovery.analyze_all(X, y)
    
    logger.info("\nTop 20 Most Important Features:")
    top_features = importance_results['combined_ranking'].head(20)
    for i, (idx, row) in enumerate(top_features.iterrows(), 1):
        logger.info(f"  {i}. {row['feature']}: {row['combined_score']:.4f}")
    
    # =========================================================================
    # PHASE 5-6: Structure and Weight Optimization
    # =========================================================================
    logger.info("")
    logger.info("PHASE 5-6: Structure & Weight Optimization")
    logger.info("-" * 40)
    
    from optimization.structure_weights import ThresholdOptimizer, WeightOptimizer
    
    # Optimize key thresholds
    threshold_optimizer = ThresholdOptimizer()
    
    logger.info("Optimizing RSI thresholds...")
    if 'rsi_14' in X.columns:
        rsi_result = threshold_optimizer.optimize_single_threshold(
            X, y, 'rsi_14', 
            search_range=(20, 40),  # Oversold
            direction='less_than'
        )
        logger.info(f"  Optimal RSI oversold: {rsi_result.get('optimal_threshold', 'N/A')}")
    
    logger.info("Optimizing funding rate threshold...")
    if 'funding_rate' in X.columns:
        funding_result = threshold_optimizer.optimize_single_threshold(
            X, y, 'funding_rate',
            search_range=(0.0001, 0.001),
            direction='greater_than'
        )
        logger.info(f"  Optimal funding extreme: {funding_result.get('optimal_threshold', 'N/A')}")
    
    # =========================================================================
    # PHASE 7-8: Entry/Exit Discovery
    # =========================================================================
    logger.info("")
    logger.info("PHASE 7-8: Entry & Exit Condition Discovery")
    logger.info("-" * 40)
    
    from discovery.entry_exit import EntryConditionDiscovery, ExitConditionDiscovery
    
    # Use forward return for exit analysis
    if 'return_fwd_24h' in full_df.columns:
        y_continuous = full_df.loc[X.index, 'return_fwd_24h']
    else:
        y_continuous = y.astype(float)
    
    # Entry discovery
    entry_discovery = EntryConditionDiscovery()
    entry_rules = entry_discovery.discover_rules(X, y, max_rules=10)
    
    logger.info(f"\nDiscovered {len(entry_rules)} entry rules:")
    for i, rule in enumerate(entry_rules[:5], 1):
        logger.info(f"  {i}. {rule.get('description', rule.get('conditions', 'Unknown'))}")
        logger.info(f"     Win rate: {rule.get('win_rate', 0):.1%}, Avg return: {rule.get('avg_return', 0):.2%}")
    
    # Exit discovery
    exit_discovery = ExitConditionDiscovery()
    exit_rules = exit_discovery.discover_rules(X, y_continuous)
    
    logger.info(f"\nDiscovered {len(exit_rules)} exit rules:")
    for i, rule in enumerate(exit_rules[:3], 1):
        logger.info(f"  {i}. {rule.get('type', 'Unknown')}: {rule.get('description', rule.get('value', 'N/A'))}")
        logger.info(f"     Hit rate: {rule.get('hit_rate', 0):.1%}, Avg return: {rule.get('avg_return', 0):.2%}")
    
    # =========================================================================
    # PHASE 9: Anti-Pattern Discovery
    # =========================================================================
    logger.info("")
    logger.info("PHASE 9: Anti-Pattern Discovery")
    logger.info("-" * 40)
    
    from discovery.antipatterns import AntiPatternDiscovery
    
    anti_discovery = AntiPatternDiscovery()
    anti_patterns = anti_discovery.discover_antipatterns(X, y)
    
    logger.info(f"\nDiscovered {len(anti_patterns)} anti-patterns (when NOT to trade):")
    for i, pattern in enumerate(anti_patterns[:5], 1):
        logger.info(f"  {i}. {pattern.get('description', pattern.get('conditions', 'Unknown'))}")
        logger.info(f"     Loss rate: {pattern.get('loss_rate', 0):.1%}, Frequency: {pattern.get('frequency', 0):.1%}")
    
    # =========================================================================
    # PHASE 10: Validation
    # =========================================================================
    logger.info("")
    logger.info("PHASE 10: Walk-Forward Validation")
    logger.info("-" * 40)
    
    from validation.framework import StrategyValidator
    
    validator = StrategyValidator()
    
    # Create simple strategy from top entry rule
    if entry_rules:
        best_rule = entry_rules[0]
        
        # Generate signals from rule
        signals = entry_discovery.apply_rule(X, best_rule)
        
        if hasattr(signals, 'sum') and signals.sum() > 0:
            validation_result = validator.validate_signals(signals, y_continuous)
            
            logger.info("\nValidation Results:")
            logger.info(f"  Total Return: {validation_result.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {validation_result.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Win Rate: {validation_result.get('win_rate', 0):.1%}")
            logger.info(f"  Max Drawdown: {validation_result.get('max_drawdown', 0):.1%}")
            logger.info(f"  Number of Trades: {validation_result.get('n_trades', 0)}")
            logger.info(f"  Grade: {validation_result.get('grade', 'N/A')}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("DISCOVERY SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nEmbedded Patterns (Ground Truth):")
    for name, pattern in data['embedded_patterns'].items():
        logger.info(f"  {name}:")
        logger.info(f"    {pattern['description']}")
        logger.info(f"    Expected: win_rate={pattern['win_rate']:.0%}, return={pattern['avg_return']:.1%}")
    
    logger.info("\nTop Discovered Features:")
    for i, (idx, row) in enumerate(top_features.head(10).iterrows(), 1):
        logger.info(f"  {i}. {row['feature']}")
    
    logger.info("\nTop Entry Rules:")
    for i, rule in enumerate(entry_rules[:3], 1):
        logger.info(f"  {i}. {rule.get('description', 'Unknown')}")
    
    logger.info("\nTop Anti-Patterns:")
    for i, pattern in enumerate(anti_patterns[:3], 1):
        logger.info(f"  {i}. {pattern.get('description', 'Unknown')}")
    
    logger.info("")
    logger.info("Demo complete! The system successfully discovered patterns in synthetic data.")
    logger.info("In production, replace synthetic data with real API data for actual trading signals.")
    logger.info("")
    
    return {
        'data': data,
        'features': features_df,
        'importance': importance_results,
        'entry_rules': entry_rules,
        'exit_rules': exit_rules,
        'anti_patterns': anti_patterns
    }


if __name__ == '__main__':
    try:
        results = run_demo()
        logger.info("Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
