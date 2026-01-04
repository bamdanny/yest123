"""
Live Predictor - Real-time inference for Telegram alerts

Loads trained model and makes predictions on current market data.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class LivePredictor:
    """
    Real-time prediction using trained ML model.
    
    Usage:
        predictor = LivePredictor('models')
        prediction = predictor.predict(current_features)
        message = predictor.format_telegram_message(prediction, btc_price)
    """
    
    def __init__(self, model_dir: str = 'models'):
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load artifacts
        model_path = model_dir / 'ensemble_model.pkl'
        calibrator_path = model_dir / 'calibrator.pkl'
        selector_path = model_dir / 'feature_selector.pkl'
        config_path = model_dir / 'model_config.json'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
        else:
            self.calibrator = None
            logger.warning("No calibrator found - using raw probabilities")
        
        if selector_path.exists():
            self.selector = joblib.load(selector_path)
        else:
            self.selector = None
            logger.warning("No feature selector found - using all features")
        
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        self.feature_names = self.config.get('selected_features', [])
        
        logger.info(f"LivePredictor loaded from {model_dir}")
        logger.info(f"  Features: {len(self.feature_names) if self.feature_names else 'all'}")
        logger.info(f"  Training date: {self.config.get('training_date', 'unknown')}")
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make prediction on current market data.
        
        Args:
            features: DataFrame with all generated features (single row or multiple)
            
        Returns:
            {
                'direction': 'UP' or 'DOWN',
                'probability': 0.0-1.0,
                'raw_probability': 0.0-1.0 (before calibration),
                'confidence': 'LOW'/'MEDIUM'/'HIGH'/'NEUTRAL',
                'timestamp': datetime string,
                'top_factors': [(feature, importance, current_value), ...]
            }
        """
        # Select features if selector available
        if self.selector is not None:
            selected = self.selector.transform(features)
        else:
            selected = features
        
        # Get raw probability
        proba_raw = self.model.predict_proba(selected)[:, 1]
        
        # Calibrate if available
        if self.calibrator is not None:
            proba = self.calibrator.transform(proba_raw)
        else:
            proba = proba_raw
        
        # Get the last prediction (most recent)
        prob = float(proba[-1]) if len(proba) > 0 else 0.5
        raw_prob = float(proba_raw[-1]) if len(proba_raw) > 0 else 0.5
        
        # Determine direction and confidence
        if prob >= 0.5:
            direction = 'UP'
            edge = prob - 0.5
        else:
            direction = 'DOWN'
            edge = 0.5 - prob
        
        if edge >= 0.20:
            confidence = 'HIGH'
        elif edge >= 0.10:
            confidence = 'MEDIUM'
        elif edge >= 0.05:
            confidence = 'LOW'
        else:
            confidence = 'NEUTRAL'
        
        # Get top contributing factors
        try:
            importance = self.model.get_feature_importance()
            current_values = selected.iloc[-1] if len(selected) > 0 else pd.Series()
            
            top_factors = []
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                val = current_values.get(feat, 0)
                top_factors.append({
                    'feature': feat,
                    'importance': imp,
                    'current_value': float(val) if pd.notna(val) else 0
                })
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            top_factors = []
        
        return {
            'direction': direction,
            'probability': prob,
            'raw_probability': raw_prob,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat(),
            'top_factors': top_factors
        }
    
    def format_telegram_message(self, prediction: Dict, btc_price: float, 
                                 extra_info: Optional[Dict] = None) -> str:
        """
        Format prediction for Telegram.
        
        Args:
            prediction: Output from predict()
            btc_price: Current BTC price
            extra_info: Optional dict with additional market data
        """
        emoji = "🟢" if prediction['direction'] == 'UP' else "🔴"
        conf_emoji = {
            "HIGH": "🔥", 
            "MEDIUM": "✅", 
            "LOW": "⚡", 
            "NEUTRAL": "😐"
        }
        
        # Format top factors
        factors_text = ""
        for f in prediction['top_factors'][:3]:
            feat_name = f['feature'][:30]
            factors_text += f"• {feat_name}: {f['importance']*100:.1f}%\n"
        
        # Extra market info
        extra_text = ""
        if extra_info:
            if 'funding_rate' in extra_info:
                extra_text += f"• Funding: {extra_info['funding_rate']*100:.4f}%\n"
            if 'oi_change' in extra_info:
                extra_text += f"• OI Change: {extra_info['oi_change']*100:.2f}%\n"
            if 'price_change_4h' in extra_info:
                extra_text += f"• 4H Change: {extra_info['price_change_4h']*100:+.2f}%\n"
        
        msg = f"""
{emoji} <b>ML PREDICTION: {prediction['direction']}</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

<b>Probability:</b> {prediction['probability']*100:.1f}%
<b>Confidence:</b> {conf_emoji.get(prediction['confidence'], '')} {prediction['confidence']}
<b>Raw Prob:</b> {prediction['raw_probability']*100:.1f}%

<b>BTC Price:</b> ${btc_price:,.0f}
{extra_text}
<b>Top Factors:</b>
{factors_text}
━━━━━━━━━━━━━━━━━━━━━━
<i>ML Model v1 | Paper trade first</i>
"""
        return msg.strip()
    
    def should_trade(self, prediction: Dict, min_confidence: str = 'LOW') -> bool:
        """
        Determine if prediction warrants a trade.
        
        Args:
            prediction: Output from predict()
            min_confidence: Minimum confidence level to trade
            
        Returns:
            True if should trade, False otherwise
        """
        confidence_order = ['NEUTRAL', 'LOW', 'MEDIUM', 'HIGH']
        
        pred_level = confidence_order.index(prediction['confidence'])
        min_level = confidence_order.index(min_confidence)
        
        return pred_level >= min_level
    
    def get_position_size(self, prediction: Dict, max_position: float = 0.10) -> float:
        """
        Calculate position size based on confidence.
        
        Args:
            prediction: Output from predict()
            max_position: Maximum position size (fraction of capital)
            
        Returns:
            Position size (0 to max_position)
        """
        confidence_sizes = {
            'NEUTRAL': 0.0,
            'LOW': 0.25,
            'MEDIUM': 0.50,
            'HIGH': 1.0
        }
        
        multiplier = confidence_sizes.get(prediction['confidence'], 0)
        return max_position * multiplier
