#!/usr/bin/env python3
"""
SIMPLE RULES SCANNER - Live Trading

Uses the simple voting-based system with OOS-validated rules.
No ML. No complexity. Just the rules that work.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_rules import SimpleRuleSystem, OOS_VALIDATED_RULES
from data.orchestrator import DataOrchestrator
from features.engineering import FeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRulesScanner:
    """Live scanner using simple voting rules."""
    
    def __init__(self, config_path: str = 'config/ml_config.json'):
        # Load config for telegram settings
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Load or create the rule system
        model_path = 'models/simple_rules.pkl'
        if Path(model_path).exists():
            self.system = SimpleRuleSystem.load(model_path)
            logger.info("Loaded simple rules from disk")
        else:
            logger.warning("No saved model found - will fit on first run")
            self.system = SimpleRuleSystem(min_votes=2)
            self._fit_system()
        
        # Data components
        self.orchestrator = DataOrchestrator()
        self.feature_generator = FeatureGenerator()
        
        # Telegram settings
        self.telegram_token = self.config.get('telegram', {}).get('bot_token')
        self.telegram_chat_id = self.config.get('telegram', {}).get('chat_id')
    
    def _fit_system(self):
        """Fit the rule system on available data."""
        logger.info("Fitting simple rules system...")
        
        try:
            # Fetch recent data
            data = self.orchestrator.fetch_all_data(lookback_days=90)
            
            if 'price' not in data or data['price'] is None:
                logger.error("No price data available")
                return
            
            # Create master dataset and generate features
            master_df = self.orchestrator.create_master_dataset(data)
            features_df = self.feature_generator.generate_all_features(master_df)
            
            # Fit system
            self.system.fit(features_df)
            self.system.save()
            
            logger.info("System fitted and saved")
            
        except Exception as e:
            logger.error(f"Failed to fit system: {e}")
    
    def scan(self) -> Dict:
        """
        Run a single scan and generate signal.
        
        Returns:
            Dictionary with signal and details
        """
        logger.info("\n" + "="*60)
        logger.info("SIMPLE RULES SCAN")
        logger.info(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("="*60)
        
        try:
            # Fetch latest data
            data = self.orchestrator.fetch_all_data(lookback_days=30)
            
            if 'price' not in data or data['price'] is None:
                logger.error("No price data")
                return {'error': 'No price data'}
            
            # Create master dataset and generate features
            master_df = self.orchestrator.create_master_dataset(data)
            features_df = self.feature_generator.generate_all_features(master_df)
            
            # Get latest row
            latest = features_df.iloc[-1].to_dict()
            current_price = data['price']['close'].iloc[-1]
            
            # Generate signal
            signal, total_votes, individual_votes = self.system.predict_single(latest)
            
            # Log results
            logger.info(f"\nCurrent Price: ${current_price:,.2f}")
            logger.info(f"\nRule Votes:")
            for rule_name, vote in individual_votes.items():
                vote_str = "BULLISH" if vote == 1 else "BEARISH" if vote == -1 else "NEUTRAL"
                feature = OOS_VALIDATED_RULES[rule_name]['feature']
                value = latest.get(feature, None)
                if value is not None:
                    logger.info(f"  {rule_name}: {vote_str} (value={value:.4f})")
                else:
                    logger.info(f"  {rule_name}: {vote_str} (no data)")
            
            logger.info(f"\nTotal Votes: {total_votes}")
            logger.info(f"Signal: {signal}")
            
            result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'price': current_price,
                'signal': signal,
                'total_votes': total_votes,
                'individual_votes': individual_votes,
                'feature_values': {
                    rule['feature']: latest.get(rule['feature'])
                    for rule in OOS_VALIDATED_RULES.values()
                }
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
        """Send signal to Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.info("Telegram not configured")
            return
        
        signal = result['signal']
        emoji = "🟢" if signal == "LONG" else "🔴"
        
        message = f"""
{emoji} **{signal} SIGNAL** {emoji}

📊 **Simple Rules System**
💰 Price: ${result['price']:,.2f}
🗳️ Total Votes: {result['total_votes']}

**Vote Breakdown:**
"""
        for rule, vote in result['individual_votes'].items():
            vote_emoji = "✅" if vote == 1 else "❌" if vote == -1 else "➖"
            message += f"{vote_emoji} {rule}: {vote}\n"
        
        message += f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            })
            logger.info("Telegram notification sent")
        except Exception as e:
            logger.error(f"Telegram failed: {e}")
    
    def run_continuous(self, interval_minutes: int = 240):
        """Run scanner continuously."""
        logger.info(f"Starting continuous scanner (interval: {interval_minutes} min)")
        
        while True:
            try:
                self.scan()
                
                logger.info(f"\nNext scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Scanner stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                time.sleep(60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Rules Scanner')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=240, help='Scan interval in minutes')
    parser.add_argument('--refit', action='store_true', help='Refit the model before scanning')
    
    args = parser.parse_args()
    
    scanner = SimpleRulesScanner()
    
    if args.refit:
        scanner._fit_system()
    
    if args.once:
        result = scanner.scan()
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    else:
        scanner.run_continuous(args.interval)


if __name__ == "__main__":
    main()
