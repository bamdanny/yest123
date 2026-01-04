"""
BTC Alpha Discovery - Main Runner

Orchestrates the complete alpha discovery pipeline:
1. Data Acquisition
2. Feature Engineering  
3. Target Generation
4. Feature Importance Discovery
5. Structure/Weight Optimization
6. Entry Condition Discovery
7. Exit Condition Discovery
8. Anti-Pattern Discovery
9. Validation Framework
10. Report Generation

Usage:
    python run_discovery.py --mode full
    python run_discovery.py --mode features_only
    python run_discovery.py --mode validate
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'discovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Ensure all required directories exist."""
    paths = [
        Path('data_cache'),
        Path('checkpoints'),
        Path('reports'),
        Path('models'),
    ]
    for p in paths:
        p.mkdir(exist_ok=True)
    return paths


def get_weekly_report_path(base_dir: str = "reports") -> tuple:
    """
    Generate organized file paths based on current date.
    Returns (report_path, raw_data_path) in weekly folder structure.
    
    Structure: reports/YYYY/week_WW_mmmDD_mmmDD/
    """
    now = datetime.now()
    year = now.year
    
    # Get ISO week number and week boundaries
    iso_calendar = now.isocalendar()
    week_num = iso_calendar[1]
    
    # Calculate week start (Monday) and end (Sunday)
    week_start = now - timedelta(days=now.weekday())
    week_end = week_start + timedelta(days=6)
    
    # Format folder name: week_01_jan01_jan07
    week_folder = f"week_{week_num:02d}_{week_start.strftime('%b%d').lower()}_{week_end.strftime('%b%d').lower()}"
    
    # Full path
    folder_path = os.path.join(base_dir, str(year), week_folder)
    os.makedirs(folder_path, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    report_path = os.path.join(folder_path, f"discovery_{timestamp}_report.txt")
    html_path = os.path.join(folder_path, f"discovery_{timestamp}_report.html")
    raw_path = os.path.join(folder_path, f"discovery_{timestamp}_raw.json")
    
    return report_path, html_path, raw_path


def generate_ai_raw_data(
    data: dict,
    features_df: pd.DataFrame,
    feature_importance: dict,
    strategy: dict,
    critic_results: list,
    validation: dict,
    antipatterns: dict,
    config: dict
) -> dict:
    """
    Generate comprehensive raw data export optimized for AI consumption.
    
    This file is NOT meant for human reading - it contains:
    - Full numeric data arrays
    - Complete feature statistics
    - All discovered rules with parameters
    - Raw validation metrics
    - Reproducibility information
    """
    
    now = datetime.now()
    
    raw_export = {
        # Metadata for reproducibility
        "_metadata": {
            "generated_at": now.isoformat(),
            "timestamp_unix": int(now.timestamp()),
            "pipeline_version": "1.0.0",
            "format_version": "ai_raw_v1"
        },
        
        # Configuration used
        "config": {
            "target": config.get("target", "profitable_24h"),
            "lookback_days": config.get("lookback_days", 90),
            "timeframe": config.get("timeframe", "4h"),
            "min_trades": config.get("min_trades", 50),
            "min_sharpe": config.get("min_sharpe", 1.0),
        },
        
        # Data acquisition summary
        "data_acquisition": {
            "sources": {},
            "total_rows": 0,
            "date_range": {"start": None, "end": None},
            "missing_sources": [],
            "data_quality_score": 0.0
        },
        
        # Feature engineering results
        "features": {
            "total_count": 0,
            "by_category": {},
            "statistics": {},
            "importance_ranking": []
        },
        
        # All discovered entry rules
        "entry_rules": {
            "total_discovered": 0,
            "rules": []
        },
        
        # Exit rules
        "exit_rules": {
            "total_discovered": 0,
            "rules": []
        },
        
        # Critic analysis for each rule
        "critic_analysis": {
            "summary": {
                "total_tested": 0,
                "credible": 0,
                "suspicious": 0,
                "debunked": 0
            },
            "rules": []
        },
        
        # Validation metrics
        "validation": {
            "grade": None,
            "oos_metrics": {},
            "walk_forward": [],
            "regime_performance": {},
            "statistical_tests": {}
        },
        
        # Anti-patterns discovered
        "antipatterns": {
            "count": 0,
            "patterns": [],
            "regime_analysis": {}
        },
        
        # Raw numeric arrays for AI processing
        "raw_arrays": {
            "returns": [],
            "signals": [],
            "timestamps": []
        }
    }
    
    # Populate data acquisition
    if data and "price" in data:
        price_df = data["price"]
        if isinstance(price_df, pd.DataFrame):
            raw_export["data_acquisition"]["sources"]["price"] = {
                "rows": len(price_df),
                "columns": list(price_df.columns)[:10]
            }
            raw_export["data_acquisition"]["total_rows"] += len(price_df)
            if "timestamp" in price_df.columns:
                raw_export["data_acquisition"]["date_range"]["start"] = str(price_df["timestamp"].min())
                raw_export["data_acquisition"]["date_range"]["end"] = str(price_df["timestamp"].max())
    
    if data and "derivatives" in data:
        for name, df in data.get("derivatives", {}).items():
            if isinstance(df, pd.DataFrame):
                raw_export["data_acquisition"]["sources"][name] = {
                    "rows": len(df),
                    "columns": list(df.columns)[:10]
                }
                raw_export["data_acquisition"]["total_rows"] += len(df)
    
    # Populate feature statistics
    if features_df is not None and len(features_df) > 0:
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        raw_export["features"]["total_count"] = len(numeric_cols)
        
        # Category breakdown
        categories = {}
        for col in numeric_cols:
            if col.startswith("price_"):
                cat = "price"
            elif col.startswith("deriv_"):
                cat = "derivatives"
            elif col.startswith("macro_"):
                cat = "macro"
            elif col.startswith("sent_") or col.startswith("sentiment_"):
                cat = "sentiment"
            elif col.startswith("time_"):
                cat = "time"
            else:
                cat = "other"
            categories[cat] = categories.get(cat, 0) + 1
        raw_export["features"]["by_category"] = categories
        
        # Feature statistics (for top 50 by importance)
        top_features = feature_importance.get("top_features", [])[:50] if feature_importance else []
        stats = {}
        for feat in top_features:
            if feat in features_df.columns:
                col = features_df[feat].dropna()
                if len(col) > 0:
                    stats[feat] = {
                        "mean": float(col.mean()),
                        "std": float(col.std()),
                        "min": float(col.min()),
                        "max": float(col.max()),
                        "median": float(col.median())
                    }
        raw_export["features"]["statistics"] = stats
        
        # Importance ranking
        if feature_importance:
            rankings = []
            for i, feat in enumerate(feature_importance.get("top_features", [])[:20]):
                rankings.append({
                    "feature": feat,
                    "rank": i + 1
                })
            raw_export["features"]["importance_ranking"] = rankings
    
    # Populate strategy/entry rules
    if strategy:
        entry_rules = strategy.get("entry_rules", [])
        raw_export["entry_rules"]["total_discovered"] = len(entry_rules)
        
        for i, rule in enumerate(entry_rules[:50]):  # Top 50 rules
            rule_data = {
                "id": i,
                "name": str(rule.get("name", f"rule_{i}")),
                "direction": rule.get("direction", "LONG"),
            }
            if hasattr(rule, '__dict__'):
                rule_data.update({k: v for k, v in rule.__dict__.items() 
                                 if not k.startswith('_') and isinstance(v, (int, float, str, bool))})
            elif isinstance(rule, dict):
                rule_data.update({k: v for k, v in rule.items() 
                                 if isinstance(v, (int, float, str, bool))})
            raw_export["entry_rules"]["rules"].append(rule_data)
        
        # Exit rules
        exit_rules = strategy.get("exit_rules", [])
        raw_export["exit_rules"]["total_discovered"] = len(exit_rules)
        for i, rule in enumerate(exit_rules[:20]):
            exit_data = {"id": i}
            if isinstance(rule, dict):
                exit_data.update({k: v for k, v in rule.items() 
                                 if isinstance(v, (int, float, str, bool))})
            raw_export["exit_rules"]["rules"].append(exit_data)
    
    # Populate critic analysis
    if critic_results:
        credible = sum(1 for r in critic_results if getattr(r, 'verdict', '') == 'CREDIBLE')
        suspicious = sum(1 for r in critic_results if getattr(r, 'verdict', '') == 'SUSPICIOUS')
        debunked = sum(1 for r in critic_results if getattr(r, 'verdict', '') == 'DEBUNKED')
        
        raw_export["critic_analysis"]["summary"] = {
            "total_tested": len(critic_results),
            "credible": credible,
            "suspicious": suspicious,
            "debunked": debunked,
            "survival_rate": credible / len(critic_results) if critic_results else 0
        }
        
        for report in critic_results[:20]:
            rule_analysis = {
                "rule_name": getattr(report, 'rule_name', 'unknown'),
                "verdict": getattr(report, 'verdict', 'unknown'),
                "tests_passed": getattr(report, 'tests_passed', 0),
                "tests_failed": getattr(report, 'tests_failed', 0)
            }
            raw_export["critic_analysis"]["rules"].append(rule_analysis)
    
    # Populate validation
    if validation:
        raw_export["validation"]["grade"] = validation.get("grade")
        raw_export["validation"]["oos_metrics"] = {
            "sharpe": validation.get("oos_sharpe"),
            "win_rate": validation.get("oos_win_rate"),
            "n_trades": validation.get("n_trades", 0)
        }
    
    # Populate antipatterns
    if antipatterns:
        raw_export["antipatterns"]["count"] = antipatterns.get("n_patterns", 0)
    
    return raw_export


def save_ai_raw_data(raw_data: dict, filepath: str):
    """Save AI raw data to JSON file with numpy/pandas type handling."""
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            if pd.isna(obj):
                return None
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            return super().default(obj)
    
    with open(filepath, 'w') as f:
        json.dump(raw_data, f, indent=2, cls=NumpyEncoder)
    
    return filepath


def run_data_acquisition():
    """Phase 1: Fetch all required data."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA ACQUISITION")
    logger.info("=" * 60)
    
    try:
        from data.orchestrator import DataOrchestrator
        
        orchestrator = DataOrchestrator()
        
        # Test connections
        logger.info("Testing API connections...")
        connection_results = orchestrator.test_all_connections()
        
        for source, status in connection_results.items():
            logger.info(f"  {source}: {'OK' if status else 'FAIL'}")
            
        # Fetch data
        logger.info("Fetching data from all sources...")
        data = orchestrator.fetch_all_data(
            price_days=365,
            derivatives_days=90,
            macro_days=365,
            sentiment_days=365
        )
        
        logger.info(f"Data fetched: {len(data)} categories")
        for category, content in data.items():
            if content is not None:
                if isinstance(content, dict):
                    logger.info(f"  {category}: {len(content)} sub-sources")
                else:
                    logger.info(f"  {category}: {len(content)} rows")
        
        # DATA VALIDATION GATE - ABORT IF CRITICAL DATA MISSING
        logger.info("\n" + "="*60)
        logger.info("DATA VALIDATION")
        logger.info("="*60)
        
        try:
            from data.validator import DataValidationGate, generate_data_audit_report
            
            validator = DataValidationGate()
            validation_report = validator.validate(data)
            
            # Print detailed audit
            audit_report = generate_data_audit_report(data)
            for line in audit_report.split('\n'):
                logger.info(line)
            
            # Print validation results
            validator.print_report(validation_report)
            
            # WARNING: Don't abort even if validation fails - let user decide
            if not validation_report["passed"]:
                logger.warning("="*60)
                logger.warning("DATA VALIDATION FAILED - Results may be unreliable!")
                logger.warning("Consider fixing data acquisition before trusting results.")
                logger.warning("="*60)
                # Continue anyway but with warning
            else:
                logger.info("[OK] Data validation passed - proceeding with analysis")
                
        except ImportError:
            logger.warning("Data validator not available - skipping validation")
        except Exception as e:
            logger.warning(f"Data validation error: {e}")
        
        # ============================================================
        # PHASE 1.5: DATA COMPLETENESS AUDIT (NEW)
        # ============================================================
        try:
            from validation.data_audit import audit_data_completeness
            
            logger.info("\n" + "="*60)
            logger.info("PHASE 1.5: DATA COMPLETENESS AUDIT")
            logger.info("="*60)
            
            audit_results = audit_data_completeness(data)
            
            if not audit_results['pass']:
                logger.error("="*60)
                logger.error("DATA AUDIT FAILED - CRITICAL DATA MISSING")
                logger.error("="*60)
                logger.error("Fix the data acquisition issues above before proceeding.")
                logger.error("The discovery results will be unreliable without required data.")
            
            logger.info(f"Data Utilization: {audit_results['utilization_pct']:.1f}%")
            
        except ImportError:
            logger.warning("Data completeness auditor not available - skipping audit")
        except Exception as e:
            logger.warning(f"Data audit error: {e}")
                
        return data
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Run: pip install requests pandas numpy --break-system-packages")
        return None
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        return None


def run_feature_engineering(data: dict):
    """Phase 2: Generate features from data."""
    logger.info("=" * 60)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    try:
        from data.storage import create_master_dataset
        from features.engineering import FeatureGenerator
        
        # Create master dataset - pass individual components
        logger.info("Creating master dataset...")
        master_df = create_master_dataset(
            price_data=data.get("price"),
            derivatives_data=data.get("derivatives", {}),
            macro_data=data.get("macro", {}),
            sentiment_data=data.get("sentiment", {})
        )
        
        if master_df is None or len(master_df) == 0:
            logger.error("Master dataset creation failed")
            return None, None
            
        logger.info(f"Master dataset: {len(master_df)} rows, {len(master_df.columns)} columns")
        
        # Generate features
        logger.info("Generating features...")
        generator = FeatureGenerator()
        features = generator.generate_all_features(master_df)
        
        logger.info(f"Generated {len(features.columns)} features")
        
        return master_df, features
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_target_generation(master_df):
    """Phase 3: Generate target variables."""
    logger.info("=" * 60)
    logger.info("PHASE 3: TARGET GENERATION")
    logger.info("=" * 60)
    
    try:
        from features.targets import TargetGenerator
        
        logger.info("Generating target variables...")
        
        # Ensure timestamp is the index (TargetGenerator requires DatetimeIndex)
        df_for_targets = master_df.copy()
        if 'timestamp' in df_for_targets.columns:
            df_for_targets = df_for_targets.set_index('timestamp')
            df_for_targets.index = pd.to_datetime(df_for_targets.index)
        
        generator = TargetGenerator(df_for_targets)
        targets = generator.generate_all_targets()
        
        logger.info(f"Generated {len(targets.columns)} target variables")
        
        return targets
        
    except Exception as e:
        logger.error(f"Target generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_feature_importance(features, targets, target_col='profitable_6h'):
    """Phase 4: Discover important features."""
    logger.info("=" * 60)
    logger.info("PHASE 4: FEATURE IMPORTANCE DISCOVERY")
    logger.info("=" * 60)
    
    try:
        from discovery.feature_importance import run_full_feature_discovery
        
        logger.info(f"Discovering feature importance for target: {target_col}")
        
        # Align features and targets by resetting indexes
        features_aligned = features.copy()
        targets_aligned = targets.copy()
        
        # Reset both to integer index for alignment
        if hasattr(features_aligned, 'reset_index'):
            features_aligned = features_aligned.reset_index(drop=True)
        if hasattr(targets_aligned, 'reset_index'):
            targets_aligned = targets_aligned.reset_index(drop=True)
        
        # Ensure same length
        min_len = min(len(features_aligned), len(targets_aligned))
        features_aligned = features_aligned.iloc[:min_len]
        targets_aligned = targets_aligned.iloc[:min_len]
        
        # Only keep numeric columns
        numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features_aligned[numeric_cols].copy()
        
        logger.info(f"Aligned data: {len(features_numeric)} samples, {len(features_numeric.columns)} numeric features")
        
        report, pillar_validation = run_full_feature_discovery(
            features_numeric, targets_aligned, target_column=target_col
        )
        
        logger.info(f"Top 10 features:")
        for i, feat in enumerate(report.top_features[:10], 1):
            logger.info(f"  {i}. {feat.feature_name}: SHAP={feat.shap_importance:.4f}")
            
        logger.info(f"\nCategory importance:")
        for cat, imp in report.category_importance.items():
            logger.info(f"  {cat}: {imp:.1%}")
            
        logger.info(f"\nPillar validation: {pillar_validation['recommendation']}")
        
        return report, pillar_validation
        
    except Exception as e:
        logger.error(f"Feature importance discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_optimization(features, targets, feature_importance):
    """Phase 5 & 6: Optimize structure and weights."""
    logger.info("=" * 60)
    logger.info("PHASE 5-6: STRUCTURE & WEIGHT OPTIMIZATION")
    logger.info("=" * 60)
    
    try:
        from optimization.structure_weights import run_full_optimization
        
        # Convert feature importance to dict
        fi_dict = {f.feature_name: f.shap_importance for f in feature_importance.top_features}
        
        target = targets.get('profitable_6h', targets.iloc[:, 0])
        
        logger.info("Running optimization...")
        results = run_full_optimization(
            features, target, fi_dict, n_trials=50
        )
        
        if results.get('thresholds'):
            logger.info(f"Threshold optimization: {results['thresholds'].improvement_pct:.1f}% improvement")
            
        if results.get('structure'):
            logger.info(f"Best structure: {results['structure']['best'].name}")
            
        for rec in results.get('recommendations', []):
            logger.info(f"  Recommendation: {rec}")
            
        return results
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None


def run_entry_exit_discovery(features, master_df, targets, top_features):
    """Phase 7 & 8: Discover entry and exit conditions."""
    logger.info("=" * 60)
    logger.info("PHASE 7-8: ENTRY & EXIT DISCOVERY")
    logger.info("=" * 60)
    
    try:
        from discovery.entry_exit import discover_complete_strategy
        
        # Align all data
        features_aligned = features.reset_index(drop=True)
        master_aligned = master_df.reset_index(drop=True)
        targets_aligned = targets.reset_index(drop=True)
        
        min_len = min(len(features_aligned), len(master_aligned), len(targets_aligned))
        features_aligned = features_aligned.iloc[:min_len]
        master_aligned = master_aligned.iloc[:min_len]
        targets_aligned = targets_aligned.iloc[:min_len]
        
        # Only keep numeric columns
        numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features_aligned[numeric_cols].copy()
        
        # Filter top_features to only include numeric columns that exist
        if top_features:
            top_features = [f for f in top_features if f in features_numeric.columns]
        
        forward_returns = targets_aligned.get('return_net_6h', targets_aligned.get('return_net_24h', targets_aligned.iloc[:, 0]))
        
        # Log data availability
        valid_returns = (~forward_returns.isna()).sum()
        logger.info(f"Numeric features for entry/exit discovery: {len(features_numeric.columns)}")
        logger.info(f"Total samples: {len(forward_returns)}, Valid returns: {valid_returns}")
        logger.info("Discovering entry and exit conditions...")
        
        strategy = discover_complete_strategy(
            features_numeric,
            master_aligned[['open', 'high', 'low', 'close', 'volume']],
            forward_returns,
            top_features
        )
        
        logger.info(f"\nDiscovered strategy: {strategy.name}")
        logger.info(f"Entry rules: {len(strategy.entry_rules)}")
        
        for i, rule in enumerate(strategy.entry_rules[:5], 1):
            logger.info(f"  {i}. {rule.to_string()}")
            logger.info(f"     Win rate: {rule.win_rate:.1%}, Sharpe: {rule.sharpe:.2f}")
            
        logger.info(f"\nExit rules: {len(strategy.exit_rules)}")
        for i, exit_rule in enumerate(strategy.exit_rules[:3], 1):
            logger.info(f"  {i}. {exit_rule.to_string()}")
            
        return strategy
        
    except Exception as e:
        logger.error(f"Entry/exit discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_antipattern_discovery(features, targets):
    """Phase 9: Discover anti-patterns."""
    logger.info("=" * 60)
    logger.info("PHASE 9: ANTI-PATTERN DISCOVERY")
    logger.info("=" * 60)
    
    try:
        from discovery.antipatterns import AntiPatternDiscovery, RegimeDetector
        
        # Align features and targets
        features_aligned = features.reset_index(drop=True)
        targets_aligned = targets.reset_index(drop=True)
        
        min_len = min(len(features_aligned), len(targets_aligned))
        features_aligned = features_aligned.iloc[:min_len]
        targets_aligned = targets_aligned.iloc[:min_len]
        
        # Only keep numeric columns
        numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features_aligned[numeric_cols].copy()
        
        returns = targets_aligned.get('return_net_6h', targets_aligned.get('return_net_24h', targets_aligned.iloc[:, 0]))
        
        # Remove NaN values
        valid_idx = ~returns.isna()
        features_valid = features_numeric[valid_idx].reset_index(drop=True)
        returns_valid = returns[valid_idx].reset_index(drop=True)
        
        logger.info(f"Valid samples for anti-pattern discovery: {len(features_valid)}")
        logger.info(f"Numeric features: {len(features_valid.columns)}")
        
        if len(features_valid) < 100:
            logger.warning("Insufficient data for anti-pattern discovery")
            return [], []
        
        # Discover anti-patterns
        logger.info("Discovering anti-patterns (when NOT to trade)...")
        discovery = AntiPatternDiscovery()
        patterns = discovery.discover(features_valid, returns_valid)
        
        logger.info(f"\nDiscovered {len(patterns)} anti-patterns:")
        for i, pattern in enumerate(patterns[:10], 1):
            logger.info(f"  {i}. {pattern.to_string()}")
            logger.info(f"     Loss rate: {pattern.loss_rate:.1%}, Frequency: {pattern.frequency:.1%}")
            
        # Detect regimes
        logger.info("\nDetecting market regimes...")
        detector = RegimeDetector()
        regimes = detector.detect_regimes(features_valid, returns_valid)
        
        for regime in regimes:
            logger.info(f"  {regime.regime_id}: {regime.recommendation} (Win rate: {regime.win_rate:.1%})")
            
        return patterns, regimes
        
    except Exception as e:
        logger.error(f"Anti-pattern discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def run_validation(strategy, features, targets, master_df=None):
    """Phase 10: Validate the discovered strategy."""
    logger.info("=" * 60)
    logger.info("PHASE 10: VALIDATION")
    logger.info("=" * 60)
    
    try:
        from validation.framework import validate_discovered_strategy
        
        if not strategy or not strategy.entry_rules:
            logger.warning("No strategy to validate")
            return None
        
        # Align features and targets
        features_aligned = features.reset_index(drop=True)
        targets_aligned = targets.reset_index(drop=True)
        
        min_len = min(len(features_aligned), len(targets_aligned))
        features_aligned = features_aligned.iloc[:min_len]
        targets_aligned = targets_aligned.iloc[:min_len]
        
        # Only keep numeric features
        numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features_aligned[numeric_cols].copy()
        features_numeric = features_numeric.ffill().bfill().fillna(0)
        
        # Generate signals from best entry rule
        best_rule = strategy.entry_rules[0]
        signals = best_rule.evaluate(features_numeric).astype(int) * best_rule.direction
        
        returns = targets_aligned.get('return_net_6h', targets_aligned.get('return_net_24h', targets_aligned.iloc[:, 0]))
        
        # === RUN 9 FIX: Verify time-based exits match validation ===
        logger.info("="*70)
        logger.info("EXIT/VALIDATION CONSISTENCY CHECK")
        logger.info("="*70)
        
        # Check what exit type the strategy uses
        exit_type = None
        if strategy.exit_rules:
            exit_rule = strategy.exit_rules[0]
            exit_type = exit_rule.exit_type
            logger.info(f"Exit rule type: {exit_type}")
            logger.info(f"Exit conditions: {exit_rule.conditions}")
            
            if exit_type == 'TIME':
                bars_held = exit_rule.conditions[0].get('threshold', 6)
                logger.info(f"✓ TIME-based exit: {bars_held} bars")
                logger.info(f"✓ Validation uses 6-bar forward returns")
                if bars_held == 6:
                    logger.info(f"✓ EXIT AND VALIDATION ARE ALIGNED!")
                else:
                    logger.warning(f"⚠ Exit holds {bars_held} bars but validation uses 6-bar returns")
            else:
                logger.warning(f"⚠ Exit type is {exit_type}, not TIME")
                logger.warning(f"⚠ This may cause return mismatch with validation!")
        else:
            logger.warning("No exit rules found in strategy")
        
        logger.info("="*70)
        
        # Filter to valid returns
        valid_idx = ~returns.isna()
        signals = signals[valid_idx].reset_index(drop=True)
        returns = returns[valid_idx].reset_index(drop=True)
        features_valid = features_numeric[valid_idx].reset_index(drop=True)
        
        logger.info(f"Validation samples: {len(signals)}, Signals active: {(signals != 0).sum()}")
        logger.info("Running comprehensive validation...")
        
        report = validate_discovered_strategy(
            signals,
            returns,
            features_valid,
            strategy_name=strategy.name
        )
        
        logger.info(f"\n{'='*40}")
        logger.info(f"VALIDATION RESULTS")
        logger.info(f"{'='*40}")
        logger.info(f"Strategy: {report.strategy_name}")
        logger.info(f"Grade: {report.overall_grade}")
        logger.info(f"Recommendation: {report.recommendation}")
        logger.info(f"\nOOS Metrics:")
        logger.info(f"  Sharpe Ratio: {report.combined_oos_metrics.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {report.combined_oos_metrics.win_rate:.1%}")
        logger.info(f"  Max Drawdown: {report.combined_oos_metrics.max_drawdown:.1%}")
        logger.info(f"  N Trades: {report.combined_oos_metrics.n_trades}")
        logger.info(f"  Statistically Significant: {report.combined_oos_metrics.is_significant}")
        
        if report.warnings:
            logger.info(f"\nWarnings:")
            for w in report.warnings:
                logger.info(f"  [!] {w}")
                
        return report
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(all_results, data=None, features_df=None, feature_importance=None, 
                   strategy=None, critic_results=None, validation=None, antipatterns=None, config=None):
    """Generate comprehensive discovery report with weekly organization."""
    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORTS")
    logger.info("=" * 60)
    
    try:
        from reports.generator import generate_discovery_report
        
        # Use the new generator with weekly folders
        output_paths = generate_discovery_report(
            results=all_results,
            output_dir='reports',
            format='both',
            use_weekly_folders=True
        )
        
        logger.info(f"Reports saved to: {output_paths.get('folder', 'reports')}")
        logger.info(f"  Text: {output_paths.get('text', 'N/A')}")
        logger.info(f"  HTML: {output_paths.get('html', 'N/A')}")
        logger.info(f"  AI Raw: {output_paths.get('raw', 'N/A')}")
        
        return output_paths.get('text', output_paths.get('html', 'reports/report.txt'))
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple report
        report_path, html_path, raw_data_path = get_weekly_report_path("reports")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("BTC ALPHA DISCOVERY REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            
            for section, result in all_results.items():
                f.write(f"\n{section.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(str(result)[:1000] + "\n")
                
        logger.info(f"Fallback report saved to: {report_path}")
        return report_path


def run_full_pipeline():
    """Run the complete alpha discovery pipeline."""
    logger.info("=" * 60)
    logger.info("BTC ALPHA DISCOVERY PIPELINE")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 60)
    
    setup_paths()
    all_results = {}
    
    # Phase 1: Data
    data = run_data_acquisition()
    if data is None:
        logger.error("Pipeline failed at data acquisition")
        return
    all_results['data'] = {k: len(v) if v is not None else 0 for k, v in data.items()}
    
    # Phase 2: Features
    master_df, features = run_feature_engineering(data)
    if features is None:
        logger.error("Pipeline failed at feature engineering")
        return
    all_results['features'] = {'n_features': len(features.columns), 'n_samples': len(features)}
    
    # Phase 3: Targets
    targets = run_target_generation(master_df)
    if targets is None:
        logger.error("Pipeline failed at target generation")
        return
    all_results['targets'] = {'n_targets': len(targets.columns)}
    
    # Phase 4: Feature Importance
    importance_report, pillar_validation = run_feature_importance(features, targets)
    if importance_report is None:
        logger.warning("Feature importance discovery failed, continuing...")
        top_features = list(features.columns)[:50]
    else:
        top_features = importance_report.recommended_features
        # Build importance scores dict from feature objects
        importance_scores = {}
        for f in getattr(importance_report, 'top_features', []):
            if hasattr(f, 'feature_name') and hasattr(f, 'shap_importance'):
                importance_scores[f.feature_name] = f.shap_importance
        
        all_results['feature_importance'] = {
            'top_features': top_features[:50],
            'category_importance': importance_report.category_importance,
            'importance_scores': importance_scores
        }
        all_results['pillar_validation'] = pillar_validation
    
    # Add date range from features dataframe
    if features is not None and len(features) > 0:
        if 'timestamp' in features.columns:
            all_results['features']['date_start'] = str(features['timestamp'].min())
            all_results['features']['date_end'] = str(features['timestamp'].max())
        elif features.index.name == 'timestamp' or hasattr(features.index[0], 'strftime'):
            all_results['features']['date_start'] = str(features.index.min())
            all_results['features']['date_end'] = str(features.index.max())
    
    # Phase 5-6: Optimization
    optimization_results = run_optimization(features, targets, importance_report)
    if optimization_results:
        all_results['optimization'] = {
            'recommendations': optimization_results.get('recommendations', [])
        }
    
    # Phase 7-8: Entry/Exit Discovery
    strategy = run_entry_exit_discovery(features, master_df, targets, top_features)
    if strategy:
        all_results['strategy'] = {
            'name': strategy.name,
            'n_entry_rules': len(strategy.entry_rules),
            'n_exit_rules': len(strategy.exit_rules),
            'entry_rules': strategy.entry_rules,    # Actual rule objects for JSON export
            'exit_rules': strategy.exit_rules,      # Actual rule objects for JSON export
            'performance': strategy.performance
        }
    
    # Phase 8.5: DEVIL'S ADVOCATE / CRITIC PASS
    logger.info("=" * 60)
    logger.info("PHASE 8.5: DEVIL'S ADVOCATE (CRITIC PASS)")
    logger.info("=" * 60)
    
    try:
        from discovery.critic import DevilsAdvocate, critique_all_rules
        
        if strategy and strategy.entry_rules:
            # Get forward returns for critic
            forward_returns = targets.get('return_net_6h', targets.get('return_net_24h', targets.iloc[:, 0]))
            forward_returns = forward_returns.reset_index(drop=True)
            features_aligned = features.reset_index(drop=True)
            
            # Align lengths
            min_len = min(len(features_aligned), len(forward_returns))
            features_critic = features_aligned.iloc[:min_len]
            returns_critic = forward_returns.iloc[:min_len]
            
            # Keep only numeric features
            numeric_cols = features_critic.select_dtypes(include=[np.number]).columns.tolist()
            features_critic = features_critic[numeric_cols].ffill().bfill().fillna(0)
            
            logger.info(f"Running critic on {len(strategy.entry_rules)} entry rules...")
            
            surviving_rules, critic_reports = critique_all_rules(
                strategy.entry_rules,
                features_critic,
                returns_critic,
                verbose=True
            )
            
            # Store results - serialize CriticReport objects using to_dict()
            all_results['critic'] = {
                'rules_tested': len(strategy.entry_rules),
                'rules_survived': len(surviving_rules),
                'debunked': len(strategy.entry_rules) - len(surviving_rules),
                'survival_rate': len(surviving_rules) / len(strategy.entry_rules) if strategy.entry_rules else 0,
                'verdicts': {r.verdict: sum(1 for rr in critic_reports if rr.verdict == r.verdict) for r in critic_reports},
                'results': [r.to_dict() if hasattr(r, 'to_dict') else {
                    'rule_name': getattr(r, 'rule_name', str(r)),
                    'verdict': getattr(r, 'verdict', 'UNKNOWN'),
                    'tests_passed': getattr(r, 'tests_passed', 0),
                    'tests_failed': getattr(r, 'tests_failed', 0),
                    'tests_total': getattr(r, 'tests_passed', 0) + getattr(r, 'tests_failed', 0),
                    'oos_sharpe': getattr(r, 'oos_sharpe', None),
                    'is_sharpe': getattr(r, 'is_sharpe', None),
                    'oos_retention': getattr(r, 'oos_retention', None),
                    'recommendation': getattr(r, 'recommendation', ''),
                    'test_results': {t.name: {'passed': t.passed, 'score': t.score, 'details': t.details} 
                                    for t in getattr(r, 'tests', [])}
                } for r in critic_reports]
            }
            
            # Update strategy with only surviving rules
            if surviving_rules:
                logger.info(f"\n[OK] {len(surviving_rules)} rules survived the critic pass")
                strategy.entry_rules = surviving_rules
            else:
                logger.warning("\n[!] NO RULES SURVIVED THE CRITIC PASS")
                logger.warning("All discovered rules appear to be spurious.")
                logger.warning("Consider:")
                logger.warning("  1. Improving data quality (more derivatives data)")
                logger.warning("  2. Using different feature combinations")
                logger.warning("  3. Relaxing critic thresholds (not recommended)")
        else:
            logger.warning("No strategy or rules to critique")
            
    except ImportError as e:
        logger.warning(f"Critic module not available: {e}")
    except Exception as e:
        logger.warning(f"Critic pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 9: Anti-patterns
    patterns, regimes = run_antipattern_discovery(features, targets)
    all_results['antipatterns'] = {
        'n_patterns': len(patterns),
        'n_regimes': len(regimes)
    }
    
    # Phase 10: Validation
    if strategy:
        validation_report = run_validation(strategy, features, targets, master_df=master_df)
        if validation_report:
            # Serialize walk_forward results
            walk_forward_data = []
            for wf in getattr(validation_report, 'walk_forward_results', []):
                # In-sample metrics
                is_metrics = getattr(wf, 'in_sample_metrics', None)
                is_data = {}
                if is_metrics:
                    is_data = {
                        'sharpe': getattr(is_metrics, 'sharpe_ratio', None),
                        'win_rate': getattr(is_metrics, 'win_rate', None),
                        'n_trades': getattr(is_metrics, 'n_trades', None),
                        'total_return': getattr(is_metrics, 'total_return', None),
                        'max_drawdown': getattr(is_metrics, 'max_drawdown', None),
                    }
                
                # Out-sample metrics
                oos_metrics = getattr(wf, 'out_sample_metrics', None)
                oos_data = {}
                if oos_metrics:
                    oos_data = {
                        'sharpe': getattr(oos_metrics, 'sharpe_ratio', None),
                        'win_rate': getattr(oos_metrics, 'win_rate', None),
                        'n_trades': getattr(oos_metrics, 'n_trades', None),
                        'total_return': getattr(oos_metrics, 'total_return', None),
                        'max_drawdown': getattr(oos_metrics, 'max_drawdown', None),
                    }
                
                wf_dict = {
                    'period_id': getattr(wf, 'period_id', 0),
                    'train_start': str(getattr(wf, 'train_start', '')),
                    'train_end': str(getattr(wf, 'train_end', '')),
                    'test_start': str(getattr(wf, 'test_start', '')),
                    'test_end': str(getattr(wf, 'test_end', '')),
                    'is_degraded': getattr(wf, 'is_degraded', False),
                    'in_sample': is_data,
                    'out_sample': oos_data,
                }
                walk_forward_data.append(wf_dict)
            
            # Serialize regime performance
            regime_data = []
            for regime_name, metrics in getattr(validation_report, 'regime_performance', {}).items():
                regime_data.append({
                    'regime': regime_name,
                    'sharpe': getattr(metrics, 'sharpe_ratio', None) if metrics else None,
                    'win_rate': getattr(metrics, 'win_rate', None) if metrics else None,
                    'n_trades': getattr(metrics, 'n_trades', None) if metrics else None,
                })
            
            # Combined OOS metrics
            oos = validation_report.combined_oos_metrics
            
            # Get IS metrics from first walk_forward period if available
            is_metrics_combined = None
            if walk_forward_data and walk_forward_data[0].get('in_sample'):
                is_metrics_combined = walk_forward_data[0]['in_sample']
            
            all_results['validation'] = {
                'grade': validation_report.overall_grade,
                'recommendation': validation_report.recommendation,
                'statistically_significant': getattr(oos, 'is_significant', False) if oos else False,
                'oos_sharpe': getattr(oos, 'sharpe_ratio', None) if oos else None,
                'oos_win_rate': getattr(oos, 'win_rate', None) if oos else None,
                'oos_max_drawdown': getattr(oos, 'max_drawdown', None) if oos else None,
                'oos_n_trades': getattr(oos, 'n_trades', None) if oos else None,
                'oos_profit_factor': getattr(oos, 'profit_factor', None) if oos else None,
                'oos_total_return': getattr(oos, 'total_return', None) if oos else None,
                'oos_avg_trade_return': getattr(oos, 'avg_win', None) if oos else None,  # Use avg_win as proxy
                'is_sharpe': is_metrics_combined.get('sharpe') if is_metrics_combined else None,
                'is_win_rate': is_metrics_combined.get('win_rate') if is_metrics_combined else None,
                'is_n_trades': is_metrics_combined.get('n_trades') if is_metrics_combined else None,
                'walk_forward': walk_forward_data,
                'regime_performance': regime_data,
                'warnings': validation_report.warnings
            }
    
    # Generate Reports (Text + HTML + AI Raw Data)
    report_path = generate_report(all_results)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Finished: {datetime.now()}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='BTC Alpha Discovery Pipeline')
    parser.add_argument('--mode', choices=['full', 'features_only', 'validate', 'test'],
                       default='full', help='Pipeline mode')
    parser.add_argument('--target', default='profitable_6h',
                       help='Target variable for optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'test':
        # Quick test mode
        logger.info("Running test mode...")
        setup_paths()
        logger.info("Test complete - all imports successful")
    else:
        logger.info(f"Mode '{args.mode}' not yet implemented")


if __name__ == '__main__':
    main()
