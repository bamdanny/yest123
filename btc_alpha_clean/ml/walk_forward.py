"""
Walk-Forward Optimization

This is THE MOST IMPORTANT validation method for trading strategies.

Instead of a single train/test split, we:
1. Train on period 1, test on period 2
2. Train on periods 1-2, test on period 3
3. Train on periods 1-3, test on period 4
...

This simulates real trading where we retrain periodically.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from .dataset_builder import DatasetBuilder
from .models.base_model import BaseModel
from .calibration import ProbabilityCalibrator, get_confidence_level
import logging
import copy

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for robust performance estimation.
    """
    
    TRANSACTION_COST = 0.0012  # 0.12% round-trip
    
    def __init__(
        self,
        model: BaseModel,
        dataset_builder: DatasetBuilder,
        n_splits: int = 5,
        train_size: int = 200,
        val_size: int = 50,
        test_size: int = 50
    ):
        self.model = model
        self.dataset_builder = dataset_builder
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        self.results: List[Dict] = []
        
    def run(self) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Returns:
            Dictionary with aggregate results and per-fold details
        """
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)
        
        # Create splits
        splits = self.dataset_builder.create_walk_forward_splits(
            n_splits=self.n_splits,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size
        )
        
        if not splits:
            logger.error("No splits created - not enough data")
            return {'error': 'Insufficient data for walk-forward validation'}
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        fold_results = []
        
        for i, (train_data, val_data, test_data) in enumerate(splits):
            logger.info(f"\n--- Fold {i+1}/{len(splits)} ---")
            logger.info(f"Train: {len(train_data['y'])} samples, "
                       f"{train_data['dates'][0]} to {train_data['dates'][-1]}")
            logger.info(f"Test:  {len(test_data['y'])} samples, "
                       f"{test_data['dates'][0]} to {test_data['dates'][-1]}")
            
            # Create fresh model for this fold
            model_copy = type(self.model)()
            model_copy.fit(
                train_data['X'], train_data['y'],
                val_data['X'], val_data['y']
            )
            
            # Predict on test
            test_proba = model_copy.predict_proba(test_data['X'])[:, 1]
            test_pred = (test_proba >= 0.5).astype(int)
            test_actual = test_data['y'].values
            
            # Calculate classification metrics
            accuracy = accuracy_score(test_actual, test_pred)
            try:
                auc = roc_auc_score(test_actual, test_proba)
            except:
                auc = 0.5  # If only one class present
            
            # Calculate trading metrics
            returns = self._calculate_returns(test_proba, test_actual, test_data['dates'])
            
            fold_result = {
                'fold': i + 1,
                'n_train': len(train_data['y']),
                'n_test': len(test_data['y']),
                'train_start': str(train_data['dates'][0]),
                'train_end': str(train_data['dates'][-1]),
                'test_start': str(test_data['dates'][0]),
                'test_end': str(test_data['dates'][-1]),
                'accuracy': accuracy,
                'auc': auc,
                'total_return': returns['total_return'],
                'sharpe': returns['sharpe'],
                'win_rate': returns['win_rate'],
                'n_trades': returns['n_trades'],
                'max_drawdown': returns['max_drawdown']
            }
            fold_results.append(fold_result)
            
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  AUC: {auc:.3f}")
            logger.info(f"  Return: {returns['total_return']*100:.1f}%")
            logger.info(f"  Sharpe: {returns['sharpe']:.2f}")
            logger.info(f"  Win Rate: {returns['win_rate']*100:.1f}% ({returns['n_trades']} trades)")
            logger.info(f"  Max DD: {returns['max_drawdown']*100:.1f}%")
            
            all_predictions.extend(test_proba)
            all_actuals.extend(test_actual)
            all_dates.extend(test_data['dates'])
        
        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        aggregate = {
            'n_folds': len(splits),
            'total_samples': len(all_actuals),
            'overall_accuracy': accuracy_score(all_actuals, (all_predictions >= 0.5).astype(int)),
            'overall_auc': roc_auc_score(all_actuals, all_predictions) if len(np.unique(all_actuals)) > 1 else 0.5,
            'avg_return': np.mean([f['total_return'] for f in fold_results]),
            'avg_sharpe': np.mean([f['sharpe'] for f in fold_results]),
            'avg_win_rate': np.mean([f['win_rate'] for f in fold_results]),
            'std_return': np.std([f['total_return'] for f in fold_results]),
            'min_return': np.min([f['total_return'] for f in fold_results]),
            'max_return': np.max([f['total_return'] for f in fold_results]),
            'avg_max_drawdown': np.mean([f['max_drawdown'] for f in fold_results]),
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Folds: {aggregate['n_folds']}")
        logger.info(f"Total test samples: {aggregate['total_samples']}")
        logger.info(f"Overall Accuracy: {aggregate['overall_accuracy']:.3f}")
        logger.info(f"Overall AUC: {aggregate['overall_auc']:.3f}")
        logger.info(f"Avg Return per fold: {aggregate['avg_return']*100:.1f}% ± {aggregate['std_return']*100:.1f}%")
        logger.info(f"Avg Sharpe: {aggregate['avg_sharpe']:.2f}")
        logger.info(f"Avg Win Rate: {aggregate['avg_win_rate']*100:.1f}%")
        logger.info(f"Return range: [{aggregate['min_return']*100:.1f}%, {aggregate['max_return']*100:.1f}%]")
        logger.info(f"Avg Max Drawdown: {aggregate['avg_max_drawdown']*100:.1f}%")
        
        # Sanity checks
        if aggregate['avg_sharpe'] > 5:
            logger.warning("\n[!] WARNING: Sharpe > 5 is suspiciously high!")
            logger.warning("   Check for data leakage or overfitting!")
        
        if aggregate['overall_auc'] > 0.70:
            logger.warning("\n[!] WARNING: AUC > 0.70 may indicate overfitting!")
        
        return {
            'aggregate': aggregate,
            'folds': fold_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'dates': all_dates
        }
    
    def _calculate_returns(
        self, 
        probabilities: np.ndarray, 
        actuals: np.ndarray,
        dates: pd.Index
    ) -> Dict[str, float]:
        """Calculate trading returns from predictions."""
        
        # Only trade when confident (>60% or <40%)
        confident_long = probabilities > 0.60
        confident_short = probabilities < 0.40
        confident_mask = confident_long | confident_short
        
        if confident_mask.sum() == 0:
            return {
                'total_return': 0.0,
                'sharpe': 0.0,
                'win_rate': 0.0,
                'n_trades': 0,
                'max_drawdown': 0.0
            }
        
        # Direction: 1 if prob > 0.5, -1 otherwise
        directions = np.where(probabilities > 0.5, 1, -1)
        
        # Actual return: 1 if up, -1 if down (simplified)
        # In real use, this should be actual price returns
        actual_directions = np.where(actuals == 1, 1, -1)
        
        # Trade return: positive if correct direction, minus transaction cost
        # Using simplified 1% gain/loss model for demonstration
        base_return = 0.01  # 1% per trade
        trade_returns = np.where(
            directions == actual_directions,
            base_return - self.TRANSACTION_COST,  # Win
            -base_return - self.TRANSACTION_COST  # Lose
        )
        
        # Apply confidence filter
        trade_returns = trade_returns[confident_mask]
        
        # Calculate metrics
        total_return = np.sum(trade_returns)
        win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
        
        # Sharpe (annualized assuming 6 trades per day)
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(365 * 6)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'n_trades': len(trade_returns),
            'max_drawdown': max_drawdown
        }
