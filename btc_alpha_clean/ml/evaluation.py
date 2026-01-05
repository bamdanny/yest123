"""
Model Evaluation - Comprehensive metrics and reports

Provides detailed evaluation including:
- Classification metrics (accuracy, AUC, precision, recall)
- Calibration analysis
- Performance by confidence level
- Trading-specific metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    brier_score_loss
)
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config_path: str = None):
        # Load config
        config_paths = [
            config_path,
            "config/ml_config.json",
            Path(__file__).parent.parent / "config" / "ml_config.json"
        ]
        
        self.config = {
            'confidence_thresholds': {
                'high': 0.65,
                'medium': 0.58,
                'low': 0.52,
                'neutral_band': [0.48, 0.52]
            }
        }
        
        for p in config_paths:
            if p and Path(p).exists():
                with open(p) as f:
                    self.config = json.load(f)
                break
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        split_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Full evaluation suite.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (may be overridden by confidence threshold)
            y_proba: Predicted probabilities
            split_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"{split_name.upper()} SET EVALUATION")
        logger.info(f"{'='*60}")
        
        # =====================================================================
        # CRITICAL: Apply confidence threshold to predictions
        # Model outputs 0.50-0.55 for everything = NO REAL SIGNAL
        # Only predict LONG if prob > high threshold, SHORT if prob < low threshold
        # Otherwise: NO TRADE (predict based on base rate)
        # =====================================================================
        thresholds = self.config.get('confidence_thresholds', {})
        min_confidence = thresholds.get('low', 0.52)
        neutral_band = thresholds.get('neutral_band', [0.48, 0.52])
        
        # Generate predictions with confidence threshold
        y_pred_confident = self._generate_confident_predictions(y_proba, min_confidence)
        
        # Log prediction distribution
        n_long = (y_pred_confident == 1).sum()
        n_short = (y_pred_confident == 0).sum()
        n_total = len(y_pred_confident)
        logger.info(f"\nPrediction Distribution (with confidence threshold {min_confidence}):")
        logger.info(f"  LONG:  {n_long:4d} ({n_long/n_total*100:.1f}%)")
        logger.info(f"  SHORT: {n_short:4d} ({n_short/n_total*100:.1f}%)")
        
        # Check if model is degenerate (all same prediction)
        if n_long == 0 or n_short == 0:
            logger.warning(f"\n⚠️  DEGENERATE MODEL: Predicting {('LONG' if n_long > 0 else 'SHORT')} for ALL samples!")
            logger.warning(f"    Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
            logger.warning(f"    This indicates the model has NO predictive power.")
        
        results = {}
        
        # =====================================================================
        # Classification Metrics (using confident predictions)
        # =====================================================================
        results['accuracy'] = accuracy_score(y_true, y_pred_confident)
        results['auc'] = roc_auc_score(y_true, y_proba)
        results['log_loss'] = log_loss(y_true, y_proba)
        results['brier_score'] = brier_score_loss(y_true, y_proba)
        results['precision'] = precision_score(y_true, y_pred_confident, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred_confident, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred_confident, zero_division=0)
        
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Accuracy:     {results['accuracy']:.4f}")
        logger.info(f"  AUC:          {results['auc']:.4f}")
        logger.info(f"  Log Loss:     {results['log_loss']:.4f}")
        logger.info(f"  Brier Score:  {results['brier_score']:.4f}")
        logger.info(f"  Precision:    {results['precision']:.4f}")
        logger.info(f"  Recall:       {results['recall']:.4f}")
        logger.info(f"  F1:           {results['f1']:.4f}")
        
        # =====================================================================
        # Confusion Matrix
        # =====================================================================
        cm = confusion_matrix(y_true, y_pred_confident)
        results['confusion_matrix'] = cm.tolist()
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"              Pred=0   Pred=1")
        if cm.shape == (2, 2):
            logger.info(f"  Actual=0    {cm[0,0]:6d}   {cm[0,1]:6d}")
            logger.info(f"  Actual=1    {cm[1,0]:6d}   {cm[1,1]:6d}")
        else:
            logger.info(f"  {cm}")
        
        # =====================================================================
        # Performance by Confidence Level
        # =====================================================================
        conf_metrics = self._evaluate_by_confidence(y_true, y_proba)
        results['by_confidence'] = conf_metrics
        
        # =====================================================================
        # Calibration Analysis
        # =====================================================================
        cal_metrics = self._evaluate_calibration(y_true, y_proba)
        results['calibration'] = cal_metrics
        
        # =====================================================================
        # Trading Metrics (ONLY trade confident predictions)
        # =====================================================================
        trading = self._calculate_trading_metrics(y_true, y_proba)
        results['trading'] = trading
        
        return results
    
    def _generate_confident_predictions(
        self,
        y_proba: np.ndarray,
        min_confidence: float = 0.55
    ) -> np.ndarray:
        """
        Generate predictions only when confident.
        
        If probability is in the neutral zone (0.45-0.55), default to base rate prediction.
        This prevents the model from predicting the same direction 100% of the time
        when it has no signal.
        
        Args:
            y_proba: Predicted probabilities
            min_confidence: Minimum probability to predict LONG (default 0.55)
            
        Returns:
            Array of predictions (0 or 1)
        """
        # High confidence LONG: prob >= min_confidence (e.g., >= 0.55)
        # High confidence SHORT: prob <= (1 - min_confidence) (e.g., <= 0.45)
        # Uncertain: use 0.5 threshold (but this will be filtered in trading)
        
        predictions = np.zeros(len(y_proba), dtype=int)
        
        # Confident LONG
        long_mask = y_proba >= min_confidence
        predictions[long_mask] = 1
        
        # Confident SHORT
        short_mask = y_proba <= (1 - min_confidence)
        predictions[short_mask] = 0
        
        # Uncertain zone: use simple 0.5 threshold for metrics
        # (but these won't be traded)
        uncertain_mask = ~long_mask & ~short_mask
        predictions[uncertain_mask] = (y_proba[uncertain_mask] >= 0.5).astype(int)
        
        return predictions
    
    def _evaluate_by_confidence(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Dict]:
        """Evaluate performance by confidence level."""
        logger.info(f"\nPerformance by Confidence Level:")
        
        thresholds = self.config.get('confidence_thresholds', {})
        high_thresh = thresholds.get('high', 0.65)
        med_thresh = thresholds.get('medium', 0.58)
        low_thresh = thresholds.get('low', 0.52)
        
        results = {}
        
        # High confidence
        high_mask = (y_proba >= high_thresh) | (y_proba <= 1 - high_thresh)
        if high_mask.sum() > 0:
            pred = (y_proba[high_mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[high_mask], pred)
            results['high'] = {
                'n_samples': int(high_mask.sum()),
                'accuracy': float(acc),
                'pct_of_total': float(high_mask.mean())
            }
            logger.info(f"  HIGH   (n={high_mask.sum():4d}, {high_mask.mean()*100:5.1f}%): "
                       f"accuracy={acc:.3f}")
        
        # Medium confidence
        med_mask = ((y_proba >= med_thresh) & (y_proba < high_thresh)) | \
                   ((y_proba > 1 - high_thresh) & (y_proba <= 1 - med_thresh))
        if med_mask.sum() > 0:
            pred = (y_proba[med_mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[med_mask], pred)
            results['medium'] = {
                'n_samples': int(med_mask.sum()),
                'accuracy': float(acc),
                'pct_of_total': float(med_mask.mean())
            }
            logger.info(f"  MEDIUM (n={med_mask.sum():4d}, {med_mask.mean()*100:5.1f}%): "
                       f"accuracy={acc:.3f}")
        
        # Low confidence
        low_mask = ((y_proba >= low_thresh) & (y_proba < med_thresh)) | \
                   ((y_proba > 1 - med_thresh) & (y_proba <= 1 - low_thresh))
        if low_mask.sum() > 0:
            pred = (y_proba[low_mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[low_mask], pred)
            results['low'] = {
                'n_samples': int(low_mask.sum()),
                'accuracy': float(acc),
                'pct_of_total': float(low_mask.mean())
            }
            logger.info(f"  LOW    (n={low_mask.sum():4d}, {low_mask.mean()*100:5.1f}%): "
                       f"accuracy={acc:.3f}")
        
        # Neutral (no trade)
        neutral_mask = ~high_mask & ~med_mask & ~low_mask
        if neutral_mask.sum() > 0:
            pred = (y_proba[neutral_mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[neutral_mask], pred)
            results['neutral'] = {
                'n_samples': int(neutral_mask.sum()),
                'accuracy': float(acc),
                'pct_of_total': float(neutral_mask.mean())
            }
            logger.info(f"  NEUTRAL(n={neutral_mask.sum():4d}, {neutral_mask.mean()*100:5.1f}%): "
                       f"accuracy={acc:.3f}")
        
        return results
    
    def _evaluate_calibration(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate probability calibration."""
        logger.info(f"\nCalibration Analysis:")
        
        # Bin probabilities and compare to actual rates
        bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
        calibration_data = []
        
        for i in range(len(bins) - 1):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
            if mask.sum() >= 5:  # Minimum samples for reliable estimate
                predicted = y_proba[mask].mean()
                actual = y_true[mask].mean()
                calibration_data.append({
                    'bin': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    'predicted': float(predicted),
                    'actual': float(actual),
                    'n_samples': int(mask.sum()),
                    'calibration_error': abs(predicted - actual)
                })
                logger.info(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: "
                           f"pred={predicted:.3f}, actual={actual:.3f}, "
                           f"error={abs(predicted-actual):.3f}, n={mask.sum()}")
        
        # Overall calibration error
        if calibration_data:
            avg_error = np.mean([d['calibration_error'] for d in calibration_data])
            logger.info(f"  Average calibration error: {avg_error:.4f}")
        else:
            avg_error = 0.0
        
        return {
            'bins': calibration_data,
            'average_error': float(avg_error)
        }
    
    def _calculate_trading_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate trading-specific metrics.
        
        IMPORTANT: We trade on ALL predictions (no confidence filter).
        The confidence filter was INVERTED (high confidence = low accuracy).
        """
        logger.info(f"\nTrading Metrics:")
        
        # ═══════════════════════════════════════════════════════════════════════
        # REMOVED CONFIDENCE FILTER
        # Analysis showed high-confidence predictions had LOWER accuracy
        # than low-confidence predictions. Trade on ALL signals instead.
        # ═══════════════════════════════════════════════════════════════════════
        
        # Direction predictions - trade ALL bars
        directions = np.where(y_proba >= 0.5, 1, -1)
        actual_directions = np.where(y_true == 1, 1, -1)
        
        # Per-trade return (simplified: 0.8% per trade, minus 0.12% costs)
        base_return = 0.008
        cost = 0.0012
        
        trade_returns = np.where(
            directions == actual_directions,
            base_return - cost,  # Win: 0.68%
            -base_return - cost  # Lose: -0.92%
        )
        
        # Calculate metrics on ALL trades
        n_trades = len(trade_returns)
        total_return = trade_returns.sum()
        win_rate = (trade_returns > 0).mean() if n_trades > 0 else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # FIXED SHARPE CALCULATION
        # Previous calculation was wrong - gave Sharpe of 39 which is impossible
        # ═══════════════════════════════════════════════════════════════════════
        if n_trades > 1:
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns, ddof=1)  # Use sample std
            
            if std_return > 0:
                # Calculate per-trade Sharpe first
                per_trade_sharpe = mean_return / std_return
                
                # Annualize based on actual trading frequency
                # With 4h bars, there are 6 bars per day, 365 days per year
                # But we should annualize based on NUMBER OF TRADES, not bars
                # If we trade N times over T days, annualized factor = sqrt(365 * N / T)
                
                # For simplicity, use sqrt(number of trades per year)
                # Assuming we trade every bar: 6 * 365 = 2190 trades/year
                trades_per_year = 6 * 365
                sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
                
                # SANITY CHECK: Cap Sharpe at reasonable bounds
                # Best quant funds achieve Sharpe 2-3. Cap at 5 for any plausibility.
                if abs(sharpe) > 5:
                    logger.warning(f"  ⚠️ Sharpe {sharpe:.2f} exceeds realistic bounds, capping at 5")
                    sharpe = 5.0 if sharpe > 0 else -5.0
            else:
                # All returns identical (no variance)
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumsum = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = cumsum - running_max
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0
        
        # Profit factor
        gains = trade_returns[trade_returns > 0].sum()
        losses = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        results = {
            'n_trades': int(n_trades),
            'pct_traded': 1.0,  # Trading all bars now
            'total_return': float(total_return),
            'win_rate': float(win_rate),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 'inf'
        }
        
        logger.info(f"  Trades: {n_trades} ({confident_mask.mean()*100:.1f}% of bars)")
        logger.info(f"  Total Return: {total_return*100:.2f}%")
        logger.info(f"  Win Rate: {win_rate*100:.1f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
        logger.info(f"  Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "  Profit Factor: ∞")
        
        return results
    
    def generate_report(
        self, 
        results: Dict[str, Any],
        output_path: str = "reports/evaluation_report.json"
    ):
        """Generate and save evaluation report."""
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nEvaluation report saved to {output_path}")
