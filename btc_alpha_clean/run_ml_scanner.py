"""
ML-based BTC Alpha Scanner with Telegram Alerts

Uses trained ML model for predictions instead of rule-based signals.
Sends alerts based on model confidence levels.

Usage:
    python run_ml_scanner.py
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import schedule

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLScanner:
    """
    Scanner that uses trained ML model for BTC direction predictions.
    Sends Telegram alerts for high/medium confidence predictions.
    """
    
    def __init__(self, config_path: str = None, model_dir: str = "models"):
        # Load config
        config_paths = [
            config_path,
            "config/ml_config.json",
            Path(__file__).parent / "config" / "ml_config.json"
        ]
        
        self.config = {}
        for p in config_paths:
            if p and Path(p).exists():
                with open(p) as f:
                    self.config = json.load(f)
                break
        
        # Telegram settings
        telegram_config = self.config.get('telegram', {})
        self.bot_token = telegram_config.get('bot_token', '')
        self.chat_id = telegram_config.get('chat_id', '')
        self.min_confidence = telegram_config.get('min_confidence', 'medium').upper()
        
        # Load ML model
        self.predictor = None
        self.model_dir = model_dir
        self._load_model()
        
        # Cooldown tracking
        self.last_alert_time = None
        self.alert_cooldown_hours = 4  # Don't send duplicate alerts within 4 hours
        
        logger.info("MLScanner initialized")
        logger.info(f"  Min confidence for alerts: {self.min_confidence}")
    
    def _load_model(self):
        """Load the trained ML model."""
        try:
            from ml.live_predictor import LivePredictor
            self.predictor = LivePredictor(self.model_dir)
            logger.info(f"ML model loaded from {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            logger.error("Run 'python train_model.py' first to train the model")
            self.predictor = None
    
    def fetch_current_data(self) -> pd.DataFrame:
        """
        Fetch current market data and generate features.
        Uses the same data pipeline as training for consistency.
        
        Returns DataFrame with all features for prediction.
        """
        try:
            # Use the orchestrator for consistent data fetching
            from data.orchestrator import DataOrchestrator
            from data.storage import create_master_dataset
            from features.engineering import FeatureGenerator
            
            logger.info("Fetching data using orchestrator...")
            orchestrator = DataOrchestrator()
            
            # Fetch all data (uses cache if available and fresh)
            all_data = orchestrator.fetch_all(
                days=90,  # Same as training
                interval='4h',
                use_cache=True,
                max_cache_age_hours=4  # Use cache if less than 4 hours old
            )
            
            if all_data is None or 'price' not in all_data:
                logger.error("Failed to fetch data")
                return None
            
            # Create master dataset (same as training)
            master_df = orchestrator.create_master_dataset(all_data)
            
            if master_df is None or len(master_df) < 50:
                logger.error(f"Insufficient data: {len(master_df) if master_df is not None else 0} rows")
                return None
            
            logger.info(f"Master DataFrame: {master_df.shape}")
            
            # Generate features (same as training)
            engineer = FeatureGenerator()
            features = engineer.generate_all_features(master_df)
            
            # Add engineered features from ML module
            from ml.feature_engineer import FeatureEngineer as MLEngineer
            ml_engineer = MLEngineer()
            features = ml_engineer.engineer_features(features)
            
            logger.info(f"Generated {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_btc_price(self) -> float:
        """Get current BTC price."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data['price'])
        except:
            return 0.0
    
    def make_prediction(self, features: pd.DataFrame) -> dict:
        """
        Make prediction using ML model.
        
        Returns prediction dict with direction, probability, confidence.
        """
        if self.predictor is None:
            logger.error("No ML model loaded")
            return None
        
        try:
            prediction = self.predictor.predict(features)
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def should_send_alert(self, prediction: dict) -> bool:
        """
        Determine if we should send an alert based on confidence.
        
        Also handles cooldown to avoid spam.
        """
        if prediction is None:
            return False
        
        confidence = prediction.get('confidence', 'NEUTRAL')
        
        # Check confidence level
        confidence_order = ['NEUTRAL', 'LOW', 'MEDIUM', 'HIGH']
        try:
            pred_level = confidence_order.index(confidence)
            min_level = confidence_order.index(self.min_confidence)
            
            if pred_level < min_level:
                logger.info(f"Confidence {confidence} below minimum {self.min_confidence}")
                return False
        except ValueError:
            return False
        
        # Check cooldown
        if self.last_alert_time:
            hours_since = (datetime.now() - self.last_alert_time).total_seconds() / 3600
            if hours_since < self.alert_cooldown_hours:
                logger.info(f"In cooldown period ({hours_since:.1f}h since last alert)")
                return False
        
        return True
    
    def format_alert(self, prediction: dict, btc_price: float) -> str:
        """Format prediction as Telegram message."""
        if self.predictor:
            return self.predictor.format_telegram_message(prediction, btc_price)
        
        # Fallback formatting
        emoji = "🟢" if prediction['direction'] == 'UP' else "🔴"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "✅", "LOW": "⚡", "NEUTRAL": "😐"}
        
        msg = f"""
{emoji} <b>ML PREDICTION: {prediction['direction']}</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

<b>Probability:</b> {prediction['probability']*100:.1f}%
<b>Confidence:</b> {conf_emoji.get(prediction['confidence'], '')} {prediction['confidence']}
<b>BTC Price:</b> ${btc_price:,.0f}
━━━━━━━━━━━━━━━━━━━━━━
<i>ML Model | Paper trade first</i>
"""
        return msg.strip()
    
    def send_telegram(self, message: str) -> bool:
        """Send message to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                self.last_alert_time = datetime.now()
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def scan(self):
        """
        Run a single scan: fetch data, predict, alert if confident.
        """
        logger.info("=" * 50)
        logger.info(f"Starting ML scan at {datetime.now()}")
        
        # Check if model is loaded
        if self.predictor is None:
            logger.error("No ML model loaded - skipping scan")
            return
        
        # Get current BTC price
        btc_price = self.get_btc_price()
        logger.info(f"BTC Price: ${btc_price:,.0f}")
        
        # Fetch current data
        features = self.fetch_current_data()
        
        if features is None or len(features) == 0:
            logger.error("Failed to fetch features")
            return
        
        # Make prediction
        prediction = self.make_prediction(features)
        
        if prediction is None:
            logger.error("Prediction failed")
            return
        
        logger.info(f"Prediction: {prediction['direction']} "
                   f"({prediction['probability']*100:.1f}%) "
                   f"Confidence: {prediction['confidence']}")
        
        # Check if we should alert
        if self.should_send_alert(prediction):
            message = self.format_alert(prediction, btc_price)
            self.send_telegram(message)
            logger.info("Alert sent!")
        else:
            logger.info("No alert sent (confidence below threshold or in cooldown)")
        
        logger.info("Scan complete")
    
    def run_scheduled(self, interval_minutes: int = 15):
        """
        Run scanner on a schedule.
        
        Args:
            interval_minutes: Minutes between scans
        """
        logger.info(f"Starting ML scanner with {interval_minutes}min interval")
        
        # Run immediately
        self.scan()
        
        # Schedule future runs
        schedule.every(interval_minutes).minutes.do(self.scan)
        
        logger.info(f"Scheduled to run every {interval_minutes} minutes")
        logger.info("Press Ctrl+C to stop")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML-based BTC Scanner')
    parser.add_argument('--interval', type=int, default=15,
                       help='Scan interval in minutes (default: 15)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--model-dir', default='models',
                       help='Directory containing trained model')
    
    args = parser.parse_args()
    
    scanner = MLScanner(model_dir=args.model_dir)
    
    if args.once:
        scanner.scan()
    else:
        scanner.run_scheduled(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
