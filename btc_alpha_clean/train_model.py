#!/usr/bin/env python3
"""
Main Training Script for BTC Alpha ML Model

This script trains a multi-feature ML model using walk-forward validation.
Integrates Phase 1 insights for feature prioritization.

Usage:
    python train_model.py                    # Full training with real data
    python train_model.py --mode quick       # Quick test with less data
    python train_model.py --synthetic        # Use synthetic data for testing
    
Requirements:
    pip install xgboost lightgbm scikit-learn joblib
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging():
    """Setup logging to file and console."""
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Clear existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config():
    """Load ML configuration."""
    config_path = PROJECT_ROOT / 'config' / 'ml_config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 100):
    """Generate synthetic data for testing the pipeline."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='4h')
    features = pd.DataFrame(index=dates)
    
    # Base signal
    signal = np.random.randn(n_samples).cumsum() / 10
    
    # Create Tier 1 features (predictive - matching Phase 1 names)
    tier1_names = [
        'deriv_feat_cg_oi_aggregated_oi_close_change_1h',
        'deriv_cg_liquidation_aggregated_liq_ratio',
        'deriv_feat_cg_oi_aggregated_oi_close_accel'
    ]
    
    for name in tier1_names:
        noise = np.random.randn(n_samples) * 0.3
        features[name] = signal + noise  # Predictive
    
    # Create Tier 2 features (moderately predictive)
    tier2_names = [
        'price_bb_width_50',
        'deriv_feat_cg_oi_aggregated_oi_high_accel',
        'price_parkinson_vol_42h'
    ]
    
    for name in tier2_names:
        noise = np.random.randn(n_samples) * 0.5
        features[name] = signal * 0.5 + noise
    
    # Add blacklist features (not predictive)
    blacklist_names = [
        'macro_feat_fin_conditions_vixcls_zscore_90d',
        'macro_fin_conditions_vixcls'
    ]
    
    for name in blacklist_names:
        features[name] = np.random.randn(n_samples)  # Pure noise
    
    # Add other noise features
    for i in range(n_features - len(tier1_names) - len(tier2_names) - len(blacklist_names)):
        features[f'noise_feature_{i}'] = np.random.randn(n_samples)
    
    # Generate target
    true_signal = features[tier1_names].mean(axis=1)
    target_prob = 1 / (1 + np.exp(-true_signal))
    target = (np.random.rand(n_samples) < target_prob).astype(int)
    
    # Create returns based on target
    returns = pd.Series(
        np.where(target == 1, 0.008, -0.008) + np.random.randn(n_samples) * 0.003,
        index=dates
    )
    
    logging.info(f"Generated synthetic data: {n_samples} samples, {len(features.columns)} features")
    logging.info(f"Target distribution: {pd.Series(target).value_counts().to_dict()}")
    
    return features, returns


def main(args):
    logger = setup_logging()
    config = load_config()
    
    logger.info("=" * 70)
    logger.info("BTC ALPHA ML MODEL - TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Synthetic: {args.synthetic}")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 1] LOADING DATA")
    logger.info("=" * 60)
    
    if args.synthetic:
        n_samples = 500 if args.mode == 'quick' else 1500
        n_features = 50 if args.mode == 'quick' else 100
        features, returns = generate_synthetic_data(n_samples, n_features)
        
        # Create target
        target = (returns.shift(-1) > 0).astype(int)
        target = target.iloc[:-1]
        features = features.iloc[:-1]
    else:
        try:
            from ml.data_loader import DataLoader
            loader = DataLoader()
            features, target = loader.prepare_data()
        except (FileNotFoundError, ImportError, AttributeError, Exception) as e:
            logger.error(f"Error loading data: {e}")
            logger.error("")
            logger.error("=" * 60)
            logger.error("TO FIX THIS, YOU HAVE TWO OPTIONS:")
            logger.error("=" * 60)
            logger.error("")
            logger.error("OPTION 1: Run Phase 1 to generate data cache:")
            logger.error("  python run_exhaustive_search.py --mode single --top-n 10")
            logger.error("")
            logger.error("OPTION 2: Test with synthetic data first:")
            logger.error("  python train_model.py --synthetic")
            logger.error("")
            logger.error("=" * 60)
            return None
    
    logger.info(f"Loaded: {len(features)} samples, {len(features.columns)} features")
    logger.info(f"Date range: {features.index[0]} to {features.index[-1]}")
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")
    
    # VALIDATION: Check if data looks real or synthetic
    noise_features = [c for c in features.columns if 'noise_feature' in c.lower()]
    if noise_features:
        logger.warning(f"[!] Found {len(noise_features)} noise_feature columns - data may be synthetic!")
    
    real_features = [c for c in features.columns if c.startswith(('deriv_', 'price_', 'macro_', 'sent_'))]
    logger.info(f"Real feature prefixes found: {len(real_features)}")
    
    # =========================================================================
    # STEP 2: ENGINEER FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 2] ENGINEERING FEATURES")
    logger.info("=" * 60)
    
    from ml.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.engineer_features(features)
    
    logger.info(f"After engineering: {len(features.columns)} features")
    
    # =========================================================================
    # STEP 3: SELECT FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 3] SELECTING FEATURES (with Phase 1 priority)")
    logger.info("=" * 60)
    
    # Use first 60% for feature selection (avoid leakage)
    selection_cutoff = int(len(features) * 0.6)
    
    from ml.feature_selector import FeatureSelector
    max_features = 30 if args.mode == 'quick' else 50
    selector = FeatureSelector(
        max_features=max_features,
        variance_threshold=config.get('features', {}).get('variance_threshold', 0.001),
        correlation_threshold=config.get('features', {}).get('correlation_threshold', 0.85)
    )
    selector.fit(
        features.iloc[:selection_cutoff],
        target.iloc[:selection_cutoff]
    )
    
    selected_features = selector.transform(features)
    
    # Save feature scores
    scores_df = selector.get_feature_importance()
    scores_df.to_csv("reports/feature_scores.csv", index=False)
    logger.info(f"Feature scores saved to reports/feature_scores.csv")
    
    # =========================================================================
    # STEP 4: WALK-FORWARD VALIDATION
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 4] WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    
    from ml.dataset_builder import DatasetBuilder
    from ml.walk_forward import WalkForwardValidator
    from ml.models.ensemble import EnsembleModel
    
    dataset_builder = DatasetBuilder(
        features=selected_features,
        target=target,
        purge_gap=config.get('splits', {}).get('purge_gap_bars', 6)
    )
    
    model = EnsembleModel()
    
    n_splits = 3 if args.mode == 'quick' else config.get('walk_forward', {}).get('n_splits', 5)
    train_size = 150 if args.mode == 'quick' else config.get('walk_forward', {}).get('min_train_samples', 300)
    
    validator = WalkForwardValidator(
        model=model,
        dataset_builder=dataset_builder,
        n_splits=n_splits,
        train_size=train_size,
        val_size=config.get('walk_forward', {}).get('val_samples', 50),
        test_size=config.get('walk_forward', {}).get('test_samples', 50)
    )
    
    wf_results = validator.run()
    
    if 'error' in wf_results:
        logger.error(f"Walk-forward failed: {wf_results['error']}")
        return
    
    # Save walk-forward results
    with open("reports/walk_forward_results.json", 'w') as f:
        json.dump({
            'aggregate': wf_results['aggregate'],
            'folds': wf_results['folds']
        }, f, indent=2, default=str)
    logger.info("Walk-forward results saved to reports/walk_forward_results.json")
    
    # =========================================================================
    # STEP 5: TRAIN FINAL MODEL
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 5] TRAINING FINAL MODEL")
    logger.info("=" * 60)
    
    train_data, val_data, test_data = dataset_builder.create_single_split(
        train_ratio=config.get('splits', {}).get('train_ratio', 0.6),
        val_ratio=config.get('splits', {}).get('val_ratio', 0.2),
        test_ratio=config.get('splits', {}).get('test_ratio', 0.2)
    )
    
    final_model = EnsembleModel()
    final_model.fit(
        train_data['X'], train_data['y'],
        val_data['X'], val_data['y']
    )
    
    # =========================================================================
    # STEP 6: CALIBRATE PROBABILITIES
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 6] CALIBRATING PROBABILITIES")
    logger.info("=" * 60)
    
    from ml.calibration import ProbabilityCalibrator
    
    val_proba = final_model.predict_proba(val_data['X'])[:, 1]
    
    calibrator = ProbabilityCalibrator()
    calibrator.fit(val_proba, val_data['y'].values)
    
    # =========================================================================
    # STEP 7: FINAL EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 7] FINAL EVALUATION")
    logger.info("=" * 60)
    
    from ml.evaluation import ModelEvaluator
    
    test_proba_raw = final_model.predict_proba(test_data['X'])[:, 1]
    test_proba = calibrator.transform(test_proba_raw)
    test_pred = (test_proba >= 0.5).astype(int)
    
    evaluator = ModelEvaluator()
    test_results = evaluator.evaluate(
        test_data['y'].values,
        test_pred,
        test_proba,
        split_name="Final Test"
    )
    
    # Save evaluation report
    evaluator.generate_report(test_results, "reports/final_evaluation.json")
    
    # =========================================================================
    # STEP 8: SAVE ARTIFACTS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 8] SAVING ARTIFACTS")
    logger.info("=" * 60)
    
    model_dir = Path("models")
    
    # Save model
    final_model.save(model_dir / "ensemble_model.pkl")
    logger.info(f"  Saved: {model_dir / 'ensemble_model.pkl'}")
    
    # Save calibrator
    joblib.dump(calibrator, model_dir / "calibrator.pkl")
    logger.info(f"  Saved: {model_dir / 'calibrator.pkl'}")
    
    # Save selector
    joblib.dump(selector, model_dir / "feature_selector.pkl")
    logger.info(f"  Saved: {model_dir / 'feature_selector.pkl'}")
    
    # Save model config
    # Validate that we're not accidentally saving synthetic results
    noise_count = len([f for f in selected_features.columns if 'noise_feature' in f.lower()])
    data_type = 'SYNTHETIC' if args.synthetic or noise_count > 5 else 'REAL'
    
    model_config = {
        'training_date': datetime.now().isoformat(),
        'data_type': data_type,
        'n_samples': len(target),
        'n_features': len(selected_features.columns),
        'selected_features': list(selected_features.columns),
        'walk_forward': wf_results['aggregate'],
        'test_results': {
            'accuracy': test_results['accuracy'],
            'auc': test_results['auc'],
            'trading': test_results.get('trading', {})
        },
        'mode': args.mode,
        'synthetic': args.synthetic,
        'phase1_priority_used': True
    }
    
    if data_type == 'SYNTHETIC' and not args.synthetic:
        logger.warning("[!] Model trained on what appears to be SYNTHETIC data!")
        logger.warning("    Re-run Phase 1 to generate real features")
    
    with open(model_dir / "model_config.json", 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    logger.info(f"  Saved: {model_dir / 'model_config.json'}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"\nWalk-Forward Results ({wf_results['aggregate']['n_folds']} folds):")
    logger.info(f"  Overall AUC:     {wf_results['aggregate']['overall_auc']:.3f}")
    logger.info(f"  Avg Sharpe:      {wf_results['aggregate']['avg_sharpe']:.2f}")
    logger.info(f"  Avg Win Rate:    {wf_results['aggregate']['avg_win_rate']*100:.1f}%")
    logger.info(f"  Total Return:    {wf_results['aggregate'].get('total_return', 0)*100:.1f}%")
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"  AUC:             {test_results['auc']:.3f}")
    logger.info(f"  Accuracy:        {test_results['accuracy']:.3f}")
    if 'trading' in test_results:
        logger.info(f"  Trading Sharpe:  {test_results['trading'].get('sharpe', 0):.2f}")
        logger.info(f"  Trading Return:  {test_results['trading'].get('total_return', 0)*100:.1f}%")
    
    # =========================================================================
    # SANITY CHECKS
    # =========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("SANITY CHECKS")
    logger.info("-" * 70)
    
    warnings_found = False
    
    if wf_results['aggregate']['overall_auc'] > 0.70:
        logger.warning("[!] AUC > 0.70 is suspiciously high!")
        logger.warning("    Check for data leakage before deploying!")
        warnings_found = True
    elif wf_results['aggregate']['overall_auc'] > 0.55:
        logger.info("[OK] AUC > 0.55 indicates predictive power")
    else:
        logger.warning("[!] AUC < 0.55 - model may not be profitable after costs")
        warnings_found = True
    
    if wf_results['aggregate']['avg_sharpe'] > 5:
        logger.warning("[!] Sharpe > 5 is unrealistic!")
        logger.warning("    Verify Sharpe calculation!")
        warnings_found = True
    elif wf_results['aggregate']['avg_sharpe'] > 1.5:
        logger.info("[OK] Sharpe > 1.5 is good")
    
    if wf_results['aggregate'].get('std_sharpe', 0) > abs(wf_results['aggregate']['avg_sharpe']):
        logger.warning("[!] High variance across folds - strategy may be unstable")
        warnings_found = True
    
    if not warnings_found:
        logger.info("\n[OK] All sanity checks passed!")
    
    logger.info("\n" + "=" * 70)
    if not args.synthetic:
        # Check if data was actually real
        if data_type == 'REAL':
            logger.info("Model trained on REAL DATA - ready for paper trading!")
            logger.info("Next steps:")
            logger.info("  1. Review reports/walk_forward_results.json")
            logger.info("  2. Check reports/feature_scores.csv for Phase 1 alignment")
            logger.info("  3. Paper trade for 30+ days before live trading")
            logger.info("  4. Run: python run_ml_scanner.py")
        else:
            logger.warning("=" * 70)
            logger.warning("MODEL MAY BE TRAINED ON SYNTHETIC DATA!")
            logger.warning("=" * 70)
            logger.warning("The selected features include 'noise_feature' patterns")
            logger.warning("This indicates synthetic data, not real market data.")
            logger.warning("")
            logger.warning("To fix, regenerate Phase 1 cache:")
            logger.warning("  python run_exhaustive_search.py --mode single --top-n 10")
            logger.warning("")
            logger.warning("Then retrain:")
            logger.warning("  python train_model.py")
    else:
        logger.info("Synthetic test complete!")
        logger.info("Run with real data: python train_model.py")
    logger.info("=" * 70)
    
    return wf_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BTC Alpha ML Model')
    parser.add_argument('--mode', choices=['full', 'quick'], default='full',
                       help='Training mode (quick for testing)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    
    args = parser.parse_args()
    main(args)
