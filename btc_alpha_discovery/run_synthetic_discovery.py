#!/usr/bin/env python3
"""
Run Complete Alpha Discovery Pipeline with Synthetic Data
==========================================================

This script demonstrates the full alpha discovery system using synthetic data
when external API access is not available.

Usage:
    python run_synthetic_discovery.py
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discovery_run.log')
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the complete discovery pipeline."""
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("BTC ALPHA DISCOVERY SYSTEM - SYNTHETIC DATA RUN")
    logger.info("=" * 70)
    logger.info(f"Started at: {start_time}")
    
    results = {}
    
    # =========================================================================
    # PHASE 1: Data Generation
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: GENERATING SYNTHETIC DATA")
    logger.info("=" * 70)
    
    try:
        from data.synthetic import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        raw_data = generator.generate_all_data(n_days=365)  # 1 year of data
        
        combined_df = raw_data['combined']
        logger.info(f"Generated {len(combined_df)} rows with {len(combined_df.columns)} columns")
        logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        
        # Store embedded patterns for later validation
        embedded_patterns = raw_data['embedded_patterns']
        logger.info(f"Embedded {len(embedded_patterns)} predictive patterns for validation")
        
        results['data'] = {
            'rows': len(combined_df),
            'columns': len(combined_df.columns),
            'date_range': f"{combined_df.index.min()} to {combined_df.index.max()}"
        }
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise
    
    # =========================================================================
    # PHASE 2: Feature Engineering
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    try:
        from features.engineering import FeatureGenerator
        
        feature_gen = FeatureGenerator()
        features_df = feature_gen.generate_all_features(combined_df)
        
        logger.info(f"Generated {len(features_df.columns)} features")
        logger.info(f"Sample feature categories:")
        
        # Count features by prefix
        prefixes = {}
        for col in features_df.columns:
            prefix = col.split('_')[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {prefix}: {count} features")
        
        results['features'] = {
            'total_features': len(features_df.columns),
            'by_category': prefixes
        }
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        # Try simpler feature set
        logger.info("Using simplified feature set...")
        features_df = combined_df.copy()
        results['features'] = {'total_features': len(features_df.columns), 'simplified': True}
    
    # =========================================================================
    # PHASE 3: Target Generation
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: TARGET GENERATION")
    logger.info("=" * 70)
    
    try:
        from features.targets import TargetGenerator
        
        target_gen = TargetGenerator()
        targets_df = target_gen.generate_all_targets(features_df)
        
        # Combine features and targets
        full_df = features_df.join(targets_df, rsuffix='_target')
        
        # Get primary target
        primary_target = 'profitable_24h' if 'profitable_24h' in targets_df.columns else targets_df.columns[0]
        
        logger.info(f"Generated {len(targets_df.columns)} target variables")
        logger.info(f"Primary target: {primary_target}")
        
        if primary_target in targets_df.columns:
            target_dist = targets_df[primary_target].value_counts()
            logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        results['targets'] = {
            'total_targets': len(targets_df.columns),
            'primary_target': primary_target
        }
        
    except Exception as e:
        logger.error(f"Target generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Create simple targets WITH ACTUAL RETURNS
        logger.info("Creating simple targets...")
        # BUG FIX: At 4h intervals, 24h = 6 bars (not 24 bars!)
        # 24 bars would be 96 hours = 4 days, which inflates returns by 4x
        BARS_PER_24H = 6  # 24 hours / 4 hours per bar
        features_df['return_24h'] = features_df['close'].shift(-BARS_PER_24H) / features_df['close'] - 1
        features_df['return_net_24h'] = features_df['return_24h'] - 0.0012  # Net of costs
        features_df['profitable_24h'] = (features_df['return_24h'] > 0.0012).astype(int)
        
        # Log diagnostic info
        logger.info(f"Return calculation: using {BARS_PER_24H} bars = 24 hours")
        logger.info(f"Return stats: min={features_df['return_24h'].min():.4f}, max={features_df['return_24h'].max():.4f}, median={features_df['return_24h'].median():.4f}")
        
        full_df = features_df
        # Use ACTUAL RETURNS for Sharpe, not binary profitability
        primary_target = 'return_net_24h'  # Changed from 'profitable_24h'
        results['targets'] = {'simple_targets': True}
    
    # =========================================================================
    # PHASE 4: Feature Importance Discovery
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: FEATURE IMPORTANCE DISCOVERY")
    logger.info("=" * 70)
    
    try:
        from discovery.feature_importance import FeatureImportanceDiscovery
        
        # Prepare data for importance analysis
        feature_cols = [c for c in full_df.columns if not c.startswith('return_') and not c.startswith('profitable_')]
        X = full_df[feature_cols].dropna()
        y = full_df.loc[X.index, primary_target].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"Analysis dataset: {len(X)} samples, {len(feature_cols)} features")
        
        importance_discovery = FeatureImportanceDiscovery()
        importance_results = importance_discovery.analyze(X, y)
        
        # Display top features
        logger.info("\nTop 20 Most Important Features:")
        top_features = importance_results.get('top_features', importance_results.get('ranking', []))[:20]
        for i, feat in enumerate(top_features, 1):
            if isinstance(feat, tuple):
                logger.info(f"  {i:2d}. {feat[0]}: {feat[1]:.4f}")
            else:
                logger.info(f"  {i:2d}. {feat}")
        
        results['importance'] = {
            'n_samples': len(X),
            'top_features': top_features[:10]
        }
        
    except Exception as e:
        logger.error(f"Feature importance failed: {e}")
        import traceback
        traceback.print_exc()
        results['importance'] = {'error': str(e)}
    
    # =========================================================================
    # PHASE 5-6: Structure & Weight Optimization
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5-6: STRUCTURE & WEIGHT OPTIMIZATION")
    logger.info("=" * 70)
    
    try:
        from optimization.structure_weights import (
            ThresholdOptimizer, StructureDiscovery, WeightOptimizer
        )
        
        # Threshold optimization
        logger.info("Optimizing thresholds...")
        threshold_opt = ThresholdOptimizer()
        optimal_thresholds = threshold_opt.optimize(X, y)
        
        logger.info("Optimal thresholds found:")
        for param, value in optimal_thresholds.items():
            logger.info(f"  {param}: {value}")
        
        # Structure discovery
        logger.info("\nDiscovering optimal structure...")
        structure_disc = StructureDiscovery()
        
        # Skip structure discovery if we have limited features
        if len(feature_cols) > 50:
            structure_results = structure_disc.discover(X, y)
            logger.info(f"Best structure: {structure_results.get('best_structure', 'flat')}")
        else:
            structure_results = {'best_structure': 'flat'}
            logger.info("Using flat structure (limited features)")
        
        results['optimization'] = {
            'thresholds': optimal_thresholds,
            'structure': structure_results.get('best_structure', 'flat')
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        results['optimization'] = {'error': str(e)}
    
    # =========================================================================
    # PHASE 7-8: Entry & Exit Discovery
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7-8: ENTRY & EXIT CONDITION DISCOVERY")
    logger.info("=" * 70)
    
    try:
        from discovery.entry_exit import EntryConditionDiscovery, ExitConditionDiscovery
        
        # Entry condition discovery
        logger.info("Discovering entry conditions...")
        entry_discovery = EntryConditionDiscovery()
        entry_rules = entry_discovery.discover(X, y)
        
        logger.info(f"\nDiscovered {len(entry_rules)} entry rules:")
        for i, rule in enumerate(entry_rules[:5], 1):
            logger.info(f"  {i}. {rule}")
        
        # Exit condition discovery
        logger.info("\nDiscovering exit conditions...")
        
        # Need price data for exit discovery
        if 'close' in full_df.columns:
            exit_discovery = ExitConditionDiscovery()
            exit_rules = exit_discovery.discover(
                X, 
                full_df.loc[X.index, ['open', 'high', 'low', 'close']]
            )
            
            logger.info(f"Discovered {len(exit_rules)} exit rules:")
            for i, rule in enumerate(exit_rules[:3], 1):
                logger.info(f"  {i}. {rule}")
        else:
            exit_rules = []
            logger.info("Skipped exit discovery (no price data)")
        
        results['entry_exit'] = {
            'n_entry_rules': len(entry_rules),
            'n_exit_rules': len(exit_rules),
            'best_entry': str(entry_rules[0]) if entry_rules else None
        }
        
    except Exception as e:
        logger.error(f"Entry/Exit discovery failed: {e}")
        import traceback
        traceback.print_exc()
        results['entry_exit'] = {'error': str(e)}
    
    # =========================================================================
    # PHASE 9: Anti-Pattern Discovery
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9: ANTI-PATTERN DISCOVERY")
    logger.info("=" * 70)
    
    try:
        from discovery.antipatterns import AntiPatternDiscovery, RegimeDetector
        
        # Anti-pattern discovery
        logger.info("Discovering anti-patterns (when NOT to trade)...")
        antipattern_discovery = AntiPatternDiscovery()
        antipatterns = antipattern_discovery.discover(X, y)
        
        logger.info(f"\nDiscovered {len(antipatterns)} anti-patterns:")
        for i, ap in enumerate(antipatterns[:5], 1):
            logger.info(f"  {i}. {ap}")
        
        # Regime detection
        logger.info("\nAnalyzing market regimes...")
        regime_detector = RegimeDetector()
        
        if 'close' in full_df.columns:
            regimes = regime_detector.detect(full_df.loc[X.index, ['close']], y)
            
            logger.info("Regime performance:")
            for regime, stats in regimes.items():
                logger.info(f"  {regime}: win_rate={stats.get('win_rate', 0):.1%}, "
                          f"frequency={stats.get('frequency', 0):.1%}")
        else:
            regimes = {}
        
        results['antipatterns'] = {
            'n_antipatterns': len(antipatterns),
            'regimes': regimes
        }
        
    except Exception as e:
        logger.error(f"Anti-pattern discovery failed: {e}")
        import traceback
        traceback.print_exc()
        results['antipatterns'] = {'error': str(e)}
    
    # =========================================================================
    # PHASE 10: Validation
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10: STRATEGY VALIDATION")
    logger.info("=" * 70)
    
    try:
        from validation.framework import StrategyValidator
        
        logger.info("Running walk-forward validation...")
        validator = StrategyValidator()
        
        # Create a simple strategy from discovered rules
        if 'entry_exit' in results and results['entry_exit'].get('best_entry'):
            validation_results = validator.validate(X, y)
        else:
            # Simple baseline validation
            validation_results = validator.validate_baseline(X, y)
        
        logger.info("\nValidation Results:")
        logger.info(f"  Total Return: {validation_results.get('total_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {validation_results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Win Rate: {validation_results.get('win_rate', 0):.1%}")
        logger.info(f"  Max Drawdown: {validation_results.get('max_drawdown', 0):.2%}")
        logger.info(f"  Number of Trades: {validation_results.get('n_trades', 0)}")
        logger.info(f"  Grade: {validation_results.get('grade', 'N/A')}")
        
        results['validation'] = validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        results['validation'] = {'error': str(e)}
    
    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("DISCOVERY PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Duration: {duration}")
    logger.info(f"Completed at: {end_time}")
    
    # Print summary
    logger.info("\n" + "-" * 70)
    logger.info("SUMMARY")
    logger.info("-" * 70)
    
    for phase, phase_results in results.items():
        if isinstance(phase_results, dict):
            if 'error' in phase_results:
                logger.info(f"  {phase}: FAILED - {phase_results['error']}")
            else:
                key_metrics = {k: v for k, v in phase_results.items() 
                             if not isinstance(v, (dict, list)) or len(str(v)) < 100}
                logger.info(f"  {phase}: {key_metrics}")
        else:
            logger.info(f"  {phase}: {phase_results}")
    
    # =========================================================================
    # GENERATE REPORT FILE
    # =========================================================================
    try:
        report_path = Path('/home/claude/btc_alpha_discovery/reports')
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f'discovery_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("BTC ALPHA DISCOVERY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Duration: {duration}\n\n")
            
            for phase, phase_results in results.items():
                f.write(f"\n{phase.upper()}\n")
                f.write("-" * 40 + "\n")
                if isinstance(phase_results, dict):
                    for k, v in phase_results.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {phase_results}\n")
            
            # Embedded patterns validation
            f.write("\n\nEMBEDDED PATTERNS (Ground Truth)\n")
            f.write("-" * 40 + "\n")
            for pattern, info in embedded_patterns.items():
                f.write(f"  {pattern}:\n")
                f.write(f"    Description: {info['description']}\n")
                f.write(f"    Expected Win Rate: {info['win_rate']:.1%}\n")
                f.write(f"    Expected Avg Return: {info['avg_return']:.2%}\n")
        
        logger.info(f"\nReport saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return results


if __name__ == '__main__':
    try:
        results = run_pipeline()
        print("\n[OK] Pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
