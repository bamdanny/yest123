#!/usr/bin/env python3
"""
ANCHORED MODEL SCANNER - Live Trading

Uses the Phase-Aware Anchored Ensemble for predictions.
70% weight to proven features, 30% to ML refinement.
"""

import os
import sys
import time
import json
import pickle
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

from data.orchestrator import DataOrchestrator
from features.engineering import FeatureGenerator
from ml.anchored_ensemble import AnchoredEnsemble, PHASE1_OOS_PROVEN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnchoredScanner:
    """Live scanner using anchored ensemble."""
    
    def __init__(self, config_path: str = 'config/ml_config.json'):
        # Load config
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Load model
        model_path = 'models/anchored_model.pkl'
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Loaded anchored model from disk")
        else:
            logger.error(f"No model found at {model_path}")
            logger.info("Run: python train_anchored.py")
            raise FileNotFoundError(model_path)
        
        # Data components
        self.orchestrator = DataOrchestrator()
        self.feature_generator = FeatureGenerator()
        
        # Telegram
        self.telegram_token = self.config.get('telegram', {}).get('bot_token')
        self.telegram_chat_id = self.config.get('telegram', {}).get('chat_id')
    
    def scan(self) -> Dict:
        """Run a single scan."""
        logger.info("\n" + "="*60)
        logger.info("ANCHORED MODEL SCAN")
        logger.info(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("="*60)
        
        try:
            # Fetch data
            data = self.orchestrator.fetch_all_data(lookback_days=30)
            
            if 'price' not in data or data['price'] is None:
                logger.error("No price data")
                return {'error': 'No price data'}
            
            # Create master dataset
            master_df = self.orchestrator.create_master_dataset(data)
            
            # Generate features
            features_df = self.feature_generator.generate_all_features(master_df)
            
            # Get latest
            latest = features_df.iloc[[-1]]
            current_price = data['price']['close'].iloc[-1]
            
            # Predict
            proba = self.model.predict_proba(latest)[0, 1]
            
            # Determine signal
            if proba >= 0.55:
                signal = 'LONG'
                confidence = 'HIGH' if proba >= 0.60 else 'MEDIUM'
            elif proba <= 0.45:
                signal = 'SHORT'
                confidence = 'HIGH' if proba <= 0.40 else 'MEDIUM'
            else:
                signal = 'NO_TRADE'
                confidence = 'LOW'
            
            logger.info(f"\nPrice: ${current_price:,.2f}")
            logger.info(f"Probability (UP): {proba:.3f}")
            logger.info(f"Signal: {signal}")
            logger.info(f"Confidence: {confidence}")
            
            # Show proven feature values
            logger.info("\nProven Feature Values:")
            for feat, oos_sharpe in sorted(PHASE1_OOS_PROVEN.items(), key=lambda x: -x[1])[:5]:
                if feat in latest.columns:
                    val = latest[feat].iloc[0]
                    logger.info(f"  {feat[:40]}: {val:.4f} (OOS Sharpe: {oos_sharpe:.2f})")
            
            result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'price': current_price,
                'probability': proba,
                'signal': signal,
                'confidence': confidence,
            }
            
            # Send Telegram if signal
            if signal != 'NO_TRADE':
                self._send_telegram(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _send_telegram(self, result: Dict):
        """Send notification."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        signal = result['signal']
        emoji = "🟢" if signal == "LONG" else "🔴"
        
        message = f"""
{emoji} **{signal} SIGNAL** {emoji}

📊 **Anchored Ensemble**
💰 Price: ${result['price']:,.2f}
📈 Probability: {result['probability']:.1%}
🎯 Confidence: {result['confidence']}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
"""
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            })
            logger.info("Telegram sent")
        except Exception as e:
            logger.error(f"Telegram failed: {e}")
    
    def run_continuous(self, interval_minutes: int = 240):
        """Run continuously."""
        logger.info(f"Starting continuous scanner (interval: {interval_minutes} min)")
        
        while True:
            try:
                self.scan()
                logger.info(f"\nNext scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Stopped")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run once')
    parser.add_argument('--interval', type=int, default=240, help='Interval in minutes')
    
    args = parser.parse_args()
    
    scanner = AnchoredScanner()
    
    if args.once:
        result = scanner.scan()
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    else:
        scanner.run_continuous(args.interval)


if __name__ == "__main__":
    main()
