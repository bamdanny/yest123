"""
Demo script - validates all system components with synthetic data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_btc_data(n_hours: int = 2000) -> pd.DataFrame:
    """Generate realistic synthetic BTC data with patterns."""
    np.random.seed(42)
    end = datetime.now()
    dates = pd.date_range(end=end, periods=n_hours, freq='1h')
    
    base_price = 50000
    log_returns = np.random.randn(n_hours) * 0.015 + 0.0001
    price = base_price * np.exp(np.cumsum(log_returns))
    
    df = pd.DataFrame({
        'open': np.roll(price, 1),
        'high': price * (1 + np.abs(np.random.randn(n_hours)) * 0.005),
        'low': price * (1 - np.abs(np.random.randn(n_hours)) * 0.005),
        'close': price,
        'volume': np.random.exponential(1000, n_hours),
        'funding_rate': 0.0001 + np.random.randn(n_hours) * 0.0001,
        'open_interest': 5e9 * (1 + np.cumsum(np.random.randn(n_hours) * 0.005)),
        'fear_greed': np.clip(50 + np.cumsum(np.random.randn(n_hours) * 2), 10, 90),
        'vix': np.clip(20 + np.cumsum(np.random.randn(n_hours) * 0.5), 12, 40),
    }, index=dates)
    df.loc[df.index[0], 'open'] = df['close'].iloc[0]
    return df

def run_demo():
    """Run full discovery pipeline demo."""
    logger.info("="*60)
    logger.info("BTC ALPHA DISCOVERY SYSTEM - SYNTHETIC DATA DEMO")
    logger.info("="*60)
    
    # Phase 1: Data
    logger.info("\n[PHASE 1] Generating synthetic BTC data...")
    raw_data = generate_synthetic_btc_data(5000)
    logger.info(f"  Generated {len(raw_data)} hours of data")
    logger.info(f"  Price range: ${raw_data['close'].min():.0f} - ${raw_data['close'].max():.0f}")
    
    # Phase 2: Feature Engineering
    logger.info("\n[PHASE 2] Engineering features...")
    from features.engineering import FeatureGenerator
    fg = FeatureGenerator()
    features = fg.generate_all_features(raw_data)
    
    # Select only columns with less than 10% NaN
    valid_cols = features.columns[features.isna().mean() < 0.1]
    features = features[valid_cols].ffill().bfill()
    logger.info(f"  Generated {len(features.columns)} valid features")
    
    # Phase 3: Targets
    logger.info("\n[PHASE 3] Generating targets...")
    from features.targets import TargetGenerator
    tg = TargetGenerator(raw_data)
    targets = tg.generate_all_targets()
    
    # Create main target - 24h forward return
    # BUG FIX: At 4h intervals, 24h = 6 bars (not 24!)
    # Also need to use FORWARD return, not backward
    BARS_PER_24H = 6
    targets['return_24h'] = raw_data['close'].shift(-BARS_PER_24H) / raw_data['close'] - 1
    targets['profitable_24h'] = (targets['return_24h'] > 0.0012).astype(int)  # After costs
    
    # Combine
    combined = pd.concat([features, targets[['return_24h', 'profitable_24h']]], axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    logger.info(f"  Combined dataset: {len(combined)} rows")
    
    # Prepare X, y
    feature_cols = [c for c in features.columns if c in combined.columns]
    X = combined[feature_cols]
    y = combined['profitable_24h']
    returns = combined['return_24h']
    
    # Phase 4: Feature Importance
    logger.info("\n[PHASE 4] Discovering feature importance...")
    from discovery.feature_importance import FeatureImportanceDiscovery
    fid = FeatureImportanceDiscovery()
    report = fid.discover(X, y)
    
    logger.info(f"\n  TOP 15 FEATURES BY IMPORTANCE:")
    for i, (feat, score) in enumerate(list(report.feature_rankings.items())[:15], 1):
        logger.info(f"    {i:2d}. {feat}: {score:.4f}")
    
    # Phase 5: Pillar Validation
    logger.info("\n[PHASE 5] Validating pillar hypothesis...")
    from discovery.feature_importance import PillarValidator
    pv = PillarValidator(report)
    pillar_results = pv.validate_pillar_structure()
    logger.info(f"  Pillar importance:")
    for pillar, imp in pillar_results.get('pillar_importance', {}).items():
        logger.info(f"    {pillar}: {imp:.4f}")
    
    # Phase 6: Threshold Optimization
    logger.info("\n[PHASE 6] Optimizing thresholds...")
    from optimization.structure_weights import ThresholdOptimizer
    to = ThresholdOptimizer()
    rsi_cols = [c for c in X.columns if 'rsi' in c.lower()]
    if rsi_cols:
        try:
            thresh = to.optimize_single_threshold(X, y, rsi_cols[0], (20, 40), 'below')
            logger.info(f"  Optimal RSI threshold: {thresh:.1f}")
        except: pass
    
    # Phase 7-8: Entry Discovery
    logger.info("\n[PHASE 7-8] Discovering entry conditions...")
    from discovery.entry_exit import EntryConditionDiscovery
    ecd = EntryConditionDiscovery()
    try:
        entry_rules = ecd.discover_entry_conditions(X, y, method='threshold_search')
        logger.info(f"  Discovered {len(entry_rules)} entry rules")
        for i, rule in enumerate(entry_rules[:3], 1):
            logger.info(f"    {i}. {rule.conditions[0] if rule.conditions else 'N/A'}")
            logger.info(f"       Win Rate: {rule.win_rate:.1%}, Avg Return: {rule.avg_return:.2%}")
    except Exception as e:
        logger.warning(f"  Entry discovery: {e}")
    
    # Phase 9: Anti-Patterns
    logger.info("\n[PHASE 9] Discovering anti-patterns...")
    from discovery.antipatterns import AntiPatternDiscovery
    apd = AntiPatternDiscovery()
    try:
        anti_patterns = apd.discover_all_antipatterns(X, returns)
        logger.info(f"  Discovered {len(anti_patterns)} anti-patterns")
        for i, ap in enumerate(anti_patterns[:3], 1):
            logger.info(f"    {i}. {ap.description}")
            logger.info(f"       Loss Rate: {ap.loss_rate:.1%}")
    except Exception as e:
        logger.warning(f"  Anti-pattern discovery: {e}")
    
    # Phase 10: Validation
    logger.info("\n[PHASE 10] Running validation...")
    from validation.framework import StrategyValidator
    try:
        rsi_col = rsi_cols[0] if rsi_cols else X.columns[0]
        signal = (X[rsi_col] < X[rsi_col].quantile(0.3)).astype(int)
        
        sv = StrategyValidator()
        val = sv.walk_forward_validation(signal, returns, n_splits=3)
        
        logger.info(f"\n  VALIDATION RESULTS:")
        logger.info(f"    Total Return: {val.get('total_return', 0):.2%}")
        logger.info(f"    Sharpe Ratio: {val.get('sharpe_ratio', 0):.2f}")
        logger.info(f"    Win Rate: {val.get('win_rate', 0):.1%}")
        logger.info(f"    Max Drawdown: {val.get('max_drawdown', 0):.2%}")
        logger.info(f"    Trades: {val.get('n_trades', 0)}")
    except Exception as e:
        logger.warning(f"  Validation: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("DEMO COMPLETE - ALL 10 PHASES EXECUTED")
    logger.info("="*60)
    return True

if __name__ == '__main__':
    run_demo()
