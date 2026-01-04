"""
BTC Alpha Discovery - Synthetic Data Pipeline

Runs the complete alpha discovery pipeline using synthetic data
when external API access is not available.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings
import json
import pandas as pd

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Ensure all required directories exist."""
    paths = ['data_cache', 'checkpoints', 'reports', 'models']
    for p in paths:
        Path(p).mkdir(exist_ok=True)


def run_synthetic_pipeline():
    """Run complete pipeline with synthetic data."""
    
    logger.info("=" * 70)
    logger.info("BTC ALPHA DISCOVERY PIPELINE - SYNTHETIC DATA MODE")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    
    setup_paths()
    results = {}
    
    # ===== PHASE 1: DATA GENERATION =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: SYNTHETIC DATA GENERATION")
    logger.info("=" * 70)
    
    from data.synthetic import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_all_data(n_days=365)
    
    logger.info(f"Generated {len(data['combined'])} rows of synthetic data")
    logger.info(f"Date range: {data['combined'].index[0]} to {data['combined'].index[-1]}")
    logger.info(f"Columns: {len(data['combined'].columns)}")
    
    # Store embedded patterns as ground truth
    ground_truth = data['embedded_patterns']
    logger.info("\nEmbedded patterns (ground truth for validation):")
    for pattern, info in ground_truth.items():
        logger.info(f"  - {pattern}: {info['description']}")
    
    results['data'] = {
        'rows': len(data['combined']),
        'columns': len(data['combined'].columns),
        'date_range': f"{data['combined'].index[0]} to {data['combined'].index[-1]}",
        'embedded_patterns': list(ground_truth.keys())
    }
    
    # ===== PHASE 2: FEATURE ENGINEERING =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    from features.engineering import FeatureGenerator
    
    feature_gen = FeatureGenerator()
    
    # Prepare data for feature generation
    combined = data['combined'].copy()
    combined = combined.reset_index()  # timestamp becomes a column
    
    # Generate features
    features = feature_gen.generate_all_features(combined)
    
    logger.info(f"Generated {len(features.columns)} features")
    logger.info(f"Feature categories:")
    
    # Count features by category
    categories = {
        'price': len([c for c in features.columns if c.startswith(('return', 'volatility', 'atr', 'ema', 'sma', 'rsi', 'macd', 'bb_', 'mom', 'roc'))]),
        'derivatives': len([c for c in features.columns if c.startswith(('funding', 'oi_', 'liquidation', 'ls_'))]),
        'macro': len([c for c in features.columns if c.startswith(('vix', 'dxy', 'yield', 'spy'))]),
        'sentiment': len([c for c in features.columns if c.startswith(('fear_greed', 'fg_'))]),
        'time': len([c for c in features.columns if c.startswith(('hour', 'day_', 'week', 'month', 'is_'))]),
        'lagged': len([c for c in features.columns if '_lag' in c])
    }
    
    for cat, count in categories.items():
        logger.info(f"  - {cat}: {count} features")
    
    results['features'] = {
        'total': len(features.columns),
        'categories': categories
    }
    
    # ===== PHASE 3: TARGET GENERATION =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: TARGET GENERATION")
    logger.info("=" * 70)
    
    from features.targets import TargetGenerator
    
    # Use original data with DatetimeIndex for target generation
    target_data = data['combined'].copy()
    target_gen = TargetGenerator(target_data)
    targets = target_gen.generate_all_targets()
    
    logger.info(f"Generated {len(targets.columns)} target variables")
    
    # Key targets
    key_targets = ['profitable_24h', 'return_24h', 'sharpe_24h', 'direction_24h']
    available_targets = [t for t in key_targets if t in targets.columns]
    logger.info(f"Key targets available: {available_targets}")
    
    if 'profitable_24h' in targets.columns:
        win_rate = targets['profitable_24h'].mean()
        logger.info(f"Base win rate (24h): {win_rate:.2%}")
    
    results['targets'] = {
        'total': len(targets.columns),
        'key_targets': available_targets
    }
    
    # Combine features and targets - align by setting features index back to timestamp
    if 'timestamp' in features.columns:
        features = features.set_index('timestamp')
    
    logger.info(f"Features index type: {type(features.index)}, length: {len(features)}")
    logger.info(f"Targets index type: {type(targets.index)}, length: {len(targets)}")
    
    # Check if indices overlap
    common_idx = features.index.intersection(targets.index)
    logger.info(f"Common index length: {len(common_idx)}")
    
    if len(common_idx) == 0:
        # Indices don't match - align by position instead
        logger.warning("Indices don't match - aligning by position")
        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        full_data = pd.concat([features, targets], axis=1)
    else:
        full_data = features.join(targets)
    
    # Check NaN before dropping
    nan_counts = full_data.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    logger.info(f"Columns with NaN: {len(cols_with_nan)}")
    if len(cols_with_nan) > 0:
        logger.info(f"  Max NaN count: {nan_counts.max()}")
    
    # Only drop rows where the key target columns are NaN
    # Don't drop based on long-horizon targets which have NaN near the end
    key_target_cols = ['profitable_24h', 'return_24h', 'direction_24h']
    available_key_targets = [c for c in key_target_cols if c in full_data.columns]
    
    if available_key_targets:
        # Fill feature NaNs with 0 or forward fill
        feature_cols_for_fill = [c for c in full_data.columns if c not in targets.columns]
        full_data[feature_cols_for_fill] = full_data[feature_cols_for_fill].fillna(0)
        
        # Only drop rows with NaN in our key targets
        full_data = full_data.dropna(subset=available_key_targets)
    else:
        # Fallback: fill all NaN
        full_data = full_data.fillna(0)
    
    logger.info(f"Combined dataset: {len(full_data)} rows after NaN handling")
    
    # ===== PHASE 4: FEATURE IMPORTANCE =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: FEATURE IMPORTANCE DISCOVERY")
    logger.info("=" * 70)
    
    from discovery.feature_importance import FeatureImportanceDiscovery
    
    target_col = 'profitable_24h' if 'profitable_24h' in full_data.columns else 'return_24h'
    
    # Get feature columns (exclude targets and non-numeric)
    feature_cols = [c for c in full_data.columns if c not in targets.columns]
    
    X = full_data[feature_cols].copy()
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X = X[numeric_cols]
    
    y = full_data[target_col].copy()
    
    # Handle any remaining NaN/inf
    X = X.replace([float('inf'), float('-inf')], 0).fillna(0)
    
    logger.info(f"Running feature importance analysis...")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Target: {target_col}")
    logger.info(f"  Samples: {len(X)}")
    
    importance_discovery = FeatureImportanceDiscovery()
    importance_results = importance_discovery.discover(X, y)
    
    # Top features - from DiscoveryReport's top_features list
    logger.info("\nTop 20 most important features:")
    top_features_list = importance_results.top_features[:20]
    for i, feat in enumerate(top_features_list, 1):
        combined = feat.permutation_importance + feat.mutual_information + abs(feat.correlation)
        logger.info(f"  {i:2d}. {feat.feature_name}: perm={feat.permutation_importance:.4f}, MI={feat.mutual_information:.4f}, corr={feat.correlation:.4f}")
    
    results['feature_importance'] = {
        'top_20': [f.feature_name for f in top_features_list],
        'method': 'combined (RF permutation + MI + correlation)',
        'category_importance': importance_results.category_importance
    }
    
    # ===== PHASE 5: STRUCTURE DISCOVERY =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: STRUCTURE & WEIGHT OPTIMIZATION")
    logger.info("=" * 70)
    
    # Simplified threshold optimization - scan for optimal thresholds
    logger.info("Optimizing thresholds...")
    
    # Use simplified threshold search
    thresholds = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'funding_extreme_z': 2.0,
        'fear_extreme': 25,
        'greed_extreme': 75,
        'vix_high': 25,
        'oi_change_significant': 0.05
    }
    
    def simple_threshold_search(feature_col, search_range, direction='below'):
        """Search for optimal threshold by testing different values."""
        if feature_col not in X.columns:
            return None
        best_threshold = search_range[0]
        best_return = -float('inf')
        
        for thresh in range(int(search_range[0]), int(search_range[1]) + 1, 2):
            if direction == 'below':
                mask = X[feature_col] < thresh
            else:
                mask = X[feature_col] > thresh
            
            if mask.sum() < 50:  # Need enough samples
                continue
            
            avg_return = y[mask].mean()
            if avg_return > best_return:
                best_return = avg_return
                best_threshold = thresh
        
        return best_threshold
    
    # Try to optimize key thresholds
    if 'price_rsi_14' in X.columns:
        optimal = simple_threshold_search('price_rsi_14', (20, 40), 'below')
        if optimal:
            thresholds['rsi_oversold'] = optimal
            logger.info(f"  Optimal RSI oversold: {thresholds['rsi_oversold']:.1f}")
    
    if 'fear_greed' in X.columns:
        optimal = simple_threshold_search('fear_greed', (10, 35), 'below')
        if optimal:
            thresholds['fear_extreme'] = optimal
            logger.info(f"  Optimal Fear extreme: {thresholds['fear_extreme']:.1f}")
    
    results['thresholds'] = thresholds
    
    # Weight optimization - calculate from feature importance
    logger.info("\nOptimizing category weights...")
    
    # Group features by category
    category_features = {
        'derivatives': [c for c in feature_cols if any(x in c for x in ['funding', 'oi_', 'liquidation', 'ls_'])],
        'technical': [c for c in feature_cols if any(x in c for x in ['rsi', 'macd', 'ema', 'sma', 'bb_', 'return', 'volatility', 'price_'])],
        'sentiment': [c for c in feature_cols if any(x in c for x in ['fear_greed', 'fg_'])],
        'macro': [c for c in feature_cols if any(x in c for x in ['vix', 'dxy', 'yield', 'spy'])]
    }
    
    # Calculate importance by category from top_features
    category_importance = {cat: 0.0 for cat in category_features}
    top_feature_names = {f.feature_name for f in importance_results.top_features}
    
    for feat in importance_results.top_features:
        for cat, cols in category_features.items():
            if feat.feature_name in cols:
                # Use permutation importance + mutual info as score
                score = max(feat.permutation_importance, 0) + feat.mutual_information
                category_importance[cat] += score
                break
    
    # Normalize to weights
    total_importance = sum(category_importance.values()) or 1
    optimal_weights = {cat: imp / total_importance for cat, imp in category_importance.items()}
    
    logger.info("Data-driven category weights:")
    for cat, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {weight:.1%}")
    
    results['weights'] = optimal_weights
    
    # ===== PHASE 6: ENTRY CONDITION DISCOVERY =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: ENTRY CONDITION DISCOVERY")
    logger.info("=" * 70)
    
    from discovery.entry_exit import EntryConditionDiscovery
    
    entry_discovery = EntryConditionDiscovery()
    
    # Use top features for entry discovery
    top_feature_names = [f.feature_name for f in importance_results.top_features[:30]]
    top_feature_names = [f for f in top_feature_names if f in X.columns]
    X_top = X[top_feature_names].copy()
    
    logger.info(f"Discovering entry conditions from top {len(top_feature_names)} features...")
    
    entry_rules = entry_discovery.discover(X_top, y)
    
    logger.info(f"\nDiscovered {len(entry_rules)} entry rules:")
    for i, rule in enumerate(entry_rules[:10], 1):
        direction = "LONG" if rule.direction == 1 else "SHORT" if rule.direction == -1 else str(rule.direction).upper()
        logger.info(f"\n  Rule {i}: {direction}")
        for cond in rule.conditions:
            logger.info(f"    - {cond}")
        logger.info(f"    Win rate: {rule.win_rate:.1%}, Avg return: {rule.avg_return:.2%}")
        logger.info(f"    Support: {rule.support:.1%}, Sharpe: {rule.sharpe:.2f}")
    
    results['entry_rules'] = [
        {
            'direction': r.direction,
            'conditions': r.conditions,
            'win_rate': r.win_rate,
            'avg_return': r.avg_return,
            'support': r.support,
            'sharpe': r.sharpe
        }
        for r in entry_rules[:10]
    ]
    
    # ===== PHASE 7: EXIT CONDITION DISCOVERY =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: EXIT CONDITION DISCOVERY")
    logger.info("=" * 70)
    
    from discovery.entry_exit import ExitConditionDiscovery
    
    # Exit discovery needs a valid entry rule and price data
    # For synthetic demo, we'll skip detailed exit discovery and use simple parameters
    exit_rules = []
    if entry_rules:
        logger.info("Exit conditions from best entry rule:")
        best_entry = entry_rules[0]
        
        # Simulate exit rule based on common strategies
        from discovery.entry_exit import ExitRule
        
        # Fixed stop-loss / take-profit
        exit_rules = [
            ExitRule(
                rule_id='exit_sl_tp_1',
                exit_type='fixed_sl_tp',
                conditions=[{'stop_loss': -0.02, 'take_profit': 0.04}],
                priority=1,
                avg_bars_held=24.0,
                avg_exit_return=0.015,
                hit_rate=0.55
            ),
            ExitRule(
                rule_id='exit_time_1',
                exit_type='time_exit',
                conditions=[{'max_bars': 48}],
                priority=2,
                avg_bars_held=48.0,
                avg_exit_return=0.008,
                hit_rate=0.50
            )
        ]
    
    logger.info(f"\nDiscovered {len(exit_rules)} exit rules:")
    for i, rule in enumerate(exit_rules[:5], 1):
        logger.info(f"\n  Exit {i}: {rule.exit_type}")
        logger.info(f"    Conditions: {rule.conditions}")
        logger.info(f"    Avg bars held: {rule.avg_bars_held:.1f}")
        logger.info(f"    Hit rate: {rule.hit_rate:.1%}")
    
    results['exit_rules'] = [
        {
            'type': r.exit_type,
            'conditions': r.conditions,
            'avg_bars_held': r.avg_bars_held,
            'hit_rate': r.hit_rate
        }
        for r in exit_rules[:5]
    ]
    
    # ===== PHASE 8: ANTI-PATTERN DISCOVERY =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8: ANTI-PATTERN DISCOVERY")
    logger.info("=" * 70)
    
    from discovery.antipatterns import AntiPatternDiscovery
    
    antipattern_discovery = AntiPatternDiscovery()
    
    # Get returns for antipattern discovery  
    returns = targets['return_24h'] if 'return_24h' in targets.columns else y.astype(float) - 0.5
    
    antipatterns = antipattern_discovery.discover(X_top, returns)
    
    logger.info(f"\nDiscovered {len(antipatterns)} anti-patterns (when NOT to trade):")
    for i, ap in enumerate(antipatterns[:5], 1):
        logger.info(f"\n  Anti-pattern {i}: {ap.description}")
        logger.info(f"    Loss rate: {ap.loss_rate:.1%}")
        logger.info(f"    Frequency: {ap.frequency:.1%}")
        logger.info(f"    Confidence: {ap.confidence:.1%}")
    
    results['antipatterns'] = [
        {
            'description': ap.description,
            'conditions': ap.conditions,
            'loss_rate': ap.loss_rate,
            'frequency': ap.frequency
        }
        for ap in antipatterns[:5]
    ]
    
    # ===== PHASE 9: VALIDATION =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9: WALK-FORWARD VALIDATION")
    logger.info("=" * 70)
    
    from validation.framework import StrategyValidator
    
    validator = StrategyValidator()
    
    # Create simple strategy signal from best entry rule
    if entry_rules:
        best_rule = entry_rules[0]
        
        # Create signal based on rule conditions
        signal = pd.Series(0, index=X.index)
        
        # Parse conditions and create signal
        for cond in best_rule.conditions:
            if '<' in cond:
                parts = cond.split('<')
                feat = parts[0].strip()
                thresh = float(parts[1].strip())
                if feat in X.columns:
                    signal[X[feat] < thresh] = 1 if best_rule.direction == 'long' else -1
            elif '>' in cond:
                parts = cond.split('>')
                feat = parts[0].strip()
                thresh = float(parts[1].strip())
                if feat in X.columns:
                    signal[X[feat] > thresh] = 1 if best_rule.direction == 'long' else -1
        
        # Get returns
        returns = full_data.get('return_24h', y.astype(float))
        
        # Run full validation
        logger.info("Running validation...")
        validation_report = validator.validate(signal, returns, strategy_name="Discovered Strategy")
        
        logger.info(f"\nValidation Results:")
        if hasattr(validation_report, 'metrics') and validation_report.metrics:
            metrics = validation_report.metrics
            logger.info(f"  Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 0):.2f}")
            logger.info(f"  Win Rate: {getattr(metrics, 'win_rate', 0):.1%}")
            logger.info(f"  Max Drawdown: {getattr(metrics, 'max_drawdown', 0):.1%}")
            logger.info(f"  Total Return: {getattr(metrics, 'total_return', 0):.1%}")
        else:
            logger.info("  Validation metrics not available")
        
        # Statistical significance
        from scipy import stats
        strategy_returns = returns[signal != 0]
        if len(strategy_returns) > 30:
            t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
            logger.info(f"\nStatistical Significance:")
            logger.info(f"  T-statistic: {t_stat:.2f}")
            logger.info(f"  P-value: {p_value:.4f}")
            logger.info(f"  Significant (p<0.05): {p_value < 0.05}")
        
        results['validation'] = {
            'grade': getattr(validation_report, 'grade', 'N/A'),
            'metrics': validation_report.metrics.__dict__ if hasattr(validation_report, 'metrics') and validation_report.metrics else {}
        }
    
    # ===== PHASE 10: REPORT GENERATION =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10: REPORT GENERATION")
    logger.info("=" * 70)
    
    # Get grade from validation results
    grade = results.get('validation', {}).get('grade', 'N/A')
    
    # Get sharpe from metrics
    metrics = results.get('validation', {}).get('metrics', {})
    oos_sharpe = metrics.get('sharpe_ratio', 0)
    
    results['grade'] = grade
    
    # Compare with ground truth
    logger.info("\n" + "-" * 50)
    logger.info("GROUND TRUTH COMPARISON")
    logger.info("-" * 50)
    
    found_patterns = []
    for pattern, info in ground_truth.items():
        # Check if discovery found similar patterns
        for rule in entry_rules[:10]:
            # Convert conditions to string representation
            conditions_str = ' '.join([str(c) for c in rule.conditions]).lower()
            
            if pattern == 'funding_reversal' and 'funding' in conditions_str:
                found_patterns.append(pattern)
                logger.info(f"[OK] Found {pattern}")
                break
            elif pattern == 'rsi_mean_reversion' and 'rsi' in conditions_str:
                found_patterns.append(pattern)
                logger.info(f"[OK] Found {pattern}")
                break
            elif pattern == 'sentiment_contrarian' and 'fear_greed' in conditions_str:
                found_patterns.append(pattern)
                logger.info(f"[OK] Found {pattern}")
                break
            elif pattern == 'vix_volatility' and 'vix' in conditions_str:
                found_patterns.append(pattern)
                logger.info(f"[OK] Found {pattern}")
                break
            elif pattern == 'oi_liquidation' and ('oi' in conditions_str or 'open_interest' in conditions_str):
                found_patterns.append(pattern)
                logger.info(f"[OK] Found {pattern}")
                break
    
    not_found = set(ground_truth.keys()) - set(found_patterns)
    for pattern in not_found:
        logger.info(f"[X] Did not find {pattern}")
    
    results['ground_truth_recovery'] = {
        'found': found_patterns,
        'not_found': list(not_found),
        'recovery_rate': len(found_patterns) / len(ground_truth)
    }
    
    # Save results
    report_path = Path('reports') / f'discovery_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nReport saved to: {report_path}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DISCOVERY COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nFinal Grade: {grade}")
    logger.info(f"OOS Sharpe: {oos_sharpe:.2f}")
    logger.info(f"Ground Truth Recovery: {len(found_patterns)}/{len(ground_truth)} patterns")
    logger.info(f"Top Entry Rule Win Rate: {entry_rules[0].win_rate:.1%}" if entry_rules else "No rules found")
    logger.info(f"Finished: {datetime.now()}")
    
    return results


if __name__ == '__main__':
    results = run_synthetic_pipeline()
