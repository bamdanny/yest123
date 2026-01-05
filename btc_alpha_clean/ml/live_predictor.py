"""
Live Predictor - Real-time inference for Telegram alerts

Loads trained model and makes predictions on current market data.
Supports both ML models and rule-based ensemble models.
"""

import joblib
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class RuleEnsemblePredictor:
    """
    Predictor for rule-based ensemble model (from create_ensemble_model.py).
    
    Provides the same interface as ML-based LivePredictor.
    """
    
    def __init__(self, model_path: Path):
        """Load rule ensemble model from pickle file."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if self.model.get('model_type') != 'rule_ensemble':
            raise ValueError(f"Expected rule_ensemble model, got {self.model.get('model_type')}")
        
        self.indicators = self.model['indicators']
        self.min_position = self.model.get('min_position_threshold', 0.1)
        self.validation_metrics = self.model.get('validation_metrics', {})
        
        logger.info(f"Rule ensemble loaded: {len(self.indicators)} indicators")
        logger.info(f"  Min position: {self.min_position}")
        logger.info(f"  Validated OOS Sharpe: {self.validation_metrics.get('oos_sharpe', 'N/A')}")
    
    def _generate_signal_from_indicator(self, value: float, indicator: Dict) -> int:
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
    
    def _generate_ensemble_signal(self, feature_row: pd.Series) -> tuple:
        """
        Generate ensemble signal from a single row of features.
        
        Returns:
            (signal, position, active_indicators)
        """
        signals = []
        weights = []
        active_indicators = []
        
        for ind in self.indicators:
            value = feature_row.get(ind['name'], np.nan)
            signal = self._generate_signal_from_indicator(value, ind)
            signals.append(signal)
            weights.append(ind['weight'])
            
            if signal != 0:
                active_indicators.append({
                    'name': ind['name'],
                    'signal': signal,
                    'weight': ind['weight'],
                    'value': float(value) if pd.notna(value) else 0
                })
        
        if len(signals) == 0:
            return 0, 0.0, []
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average position
        position = np.sum(np.array(signals) * weights)
        
        if position > self.min_position:
            signal = 1
        elif position < -self.min_position:
            signal = -1
        else:
            signal = 0
        
        return signal, position, active_indicators
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make prediction on current market data.
        
        Args:
            features: DataFrame with all generated features
            
        Returns:
            Same format as ML-based LivePredictor.predict()
        """
        # Use the last row (most recent data)
        if len(features) == 0:
            return {
                'direction': 'NEUTRAL',
                'probability': 0.5,
                'raw_probability': 0.5,
                'confidence': 'NEUTRAL',
                'timestamp': datetime.utcnow().isoformat(),
                'top_factors': []
            }
        
        last_row = features.iloc[-1]
        signal, position, active_indicators = self._generate_ensemble_signal(last_row)
        
        # Convert position to probability-like metric
        # position ranges from -1 to +1
        # map to 0-1 where 0.5 is neutral
        prob = 0.5 + (position / 2)  # Maps [-1, 1] to [0, 1]
        prob = max(0.0, min(1.0, prob))  # Clamp
        
        # Determine direction
        if signal == 1:
            direction = 'UP'
        elif signal == -1:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Determine confidence based on position strength
        edge = abs(position)
        if edge >= 0.40:
            confidence = 'HIGH'
        elif edge >= 0.25:
            confidence = 'MEDIUM'
        elif edge >= self.min_position:
            confidence = 'LOW'
        else:
            confidence = 'NEUTRAL'
        
        # Format top factors
        top_factors = []
        for ind in sorted(active_indicators, key=lambda x: -x['weight'])[:5]:
            top_factors.append({
                'feature': ind['name'][:40],
                'importance': ind['weight'],
                'current_value': ind['value'],
                'signal': 'LONG' if ind['signal'] == 1 else 'SHORT'
            })
        
        return {
            'direction': direction,
            'probability': prob,
            'raw_probability': prob,
            'confidence': confidence,
            'position': position,
            'signal': signal,
            'timestamp': datetime.utcnow().isoformat(),
            'top_factors': top_factors,
            'model_type': 'rule_ensemble'
        }
    
    def format_telegram_message(self, prediction: Dict, btc_price: float,
                                 extra_info: Optional[Dict] = None) -> str:
        """Format prediction for Telegram."""
        emoji_map = {'UP': '🟢', 'DOWN': '🔴', 'NEUTRAL': '⚪'}
        emoji = emoji_map.get(prediction['direction'], '⚪')
        
        conf_emoji = {
            "HIGH": "🔥", 
            "MEDIUM": "✅", 
            "LOW": "⚡", 
            "NEUTRAL": "😐"
        }
        
        # Format top factors
        factors_text = ""
        for f in prediction.get('top_factors', [])[:3]:
            signal_emoji = "🟢" if f.get('signal') == 'LONG' else "🔴"
            feat_name = f['feature'][:25]
            factors_text += f"  {signal_emoji} {feat_name}: {f['importance']*100:.0f}%\n"
        
        # Extra market info
        extra_text = ""
        if extra_info:
            if 'funding_rate' in extra_info:
                extra_text += f"• Funding: {extra_info['funding_rate']*100:.4f}%\n"
            if 'oi_change' in extra_info:
                extra_text += f"• OI Change: {extra_info['oi_change']*100:.2f}%\n"
        
        position_str = f"{prediction.get('position', 0)*100:+.0f}%" if 'position' in prediction else ''
        
        msg = f"""
{emoji} <b>SIGNAL: {prediction['direction']}</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

<b>Position:</b> {position_str}
<b>Confidence:</b> {conf_emoji.get(prediction['confidence'], '')} {prediction['confidence']}
<b>BTC Price:</b> ${btc_price:,.0f}
{extra_text}
<b>Active Indicators:</b>
{factors_text}
━━━━━━━━━━━━━━━━━━━━━━
<i>Rule Ensemble v38 | Sharpe {self.validation_metrics.get('oos_sharpe', 0):.1f}</i>
"""
        return msg.strip()
    
    def should_trade(self, prediction: Dict, min_confidence: str = 'LOW') -> bool:
        """Determine if prediction warrants a trade."""
        confidence_order = ['NEUTRAL', 'LOW', 'MEDIUM', 'HIGH']
        
        try:
            pred_level = confidence_order.index(prediction['confidence'])
            min_level = confidence_order.index(min_confidence)
            return pred_level >= min_level
        except ValueError:
            return False
    
    def get_position_size(self, prediction: Dict, max_position: float = 0.10) -> float:
        """Calculate position size based on confidence."""
        confidence_sizes = {
            'NEUTRAL': 0.0,
            'LOW': 0.25,
            'MEDIUM': 0.50,
            'HIGH': 1.0
        }
        
        multiplier = confidence_sizes.get(prediction['confidence'], 0)
        return max_position * multiplier


class LivePredictor:
    """
    Real-time prediction using trained ML model or rule ensemble.
    
    Usage:
        predictor = LivePredictor('models')
        prediction = predictor.predict(current_features)
        message = predictor.format_telegram_message(prediction, btc_price)
    """
    
    def __init__(self, model_dir: str = 'models'):
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Try to load ensemble_model.pkl first
        model_path = model_dir / 'ensemble_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Check if it's a rule ensemble or ML model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and model_data.get('model_type') == 'rule_ensemble':
            # Use rule ensemble predictor
            logger.info("Loading rule ensemble model...")
            self._predictor = RuleEnsemblePredictor(model_path)
            self.model_type = 'rule_ensemble'
        else:
            # Use ML model predictor (original logic)
            logger.info("Loading ML model...")
            self._predictor = None
            self.model = joblib.load(model_path)
            self.model_type = 'ml'
            
            # Load supporting artifacts
            calibrator_path = model_dir / 'calibrator.pkl'
            selector_path = model_dir / 'feature_selector.pkl'
            config_path = model_dir / 'model_config.json'
            
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
        logger.info(f"  Model type: {self.model_type}")
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make prediction on current market data.
        
        Delegates to appropriate predictor based on model type.
        """
        if self._predictor is not None:
            return self._predictor.predict(features)
        
        # Original ML prediction logic
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
            'top_factors': top_factors,
            'model_type': 'ml'
        }
    
    def format_telegram_message(self, prediction: Dict, btc_price: float, 
                                 extra_info: Optional[Dict] = None) -> str:
        """Format prediction for Telegram."""
        if self._predictor is not None:
            return self._predictor.format_telegram_message(prediction, btc_price, extra_info)
        
        # Original ML formatting
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
        """Determine if prediction warrants a trade."""
        if self._predictor is not None:
            return self._predictor.should_trade(prediction, min_confidence)
        
        confidence_order = ['NEUTRAL', 'LOW', 'MEDIUM', 'HIGH']
        
        pred_level = confidence_order.index(prediction['confidence'])
        min_level = confidence_order.index(min_confidence)
        
        return pred_level >= min_level
    
    def get_position_size(self, prediction: Dict, max_position: float = 0.10) -> float:
        """Calculate position size based on confidence."""
        if self._predictor is not None:
            return self._predictor.get_position_size(prediction, max_position)
        
        confidence_sizes = {
            'NEUTRAL': 0.0,
            'LOW': 0.25,
            'MEDIUM': 0.50,
            'HIGH': 1.0
        }
        
        multiplier = confidence_sizes.get(prediction['confidence'], 0)
        return max_position * multiplier
