#!/usr/bin/env python3
"""
TELEGRAM SCANNER - 10 Minute Push Notifications

Simple scanner that sends BTC trading signals to Telegram every 10 minutes.

Setup:
1. Create a Telegram bot via @BotFather
2. Get your chat_id via @userinfobot or the get_chat_id.py script
3. Add to .env file:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id

Usage:
    python telegram_scanner.py              # Run continuously (every 10 min)
    python telegram_scanner.py --once       # Run once and exit
    python telegram_scanner.py --test       # Send test message
"""

import os
import sys
import time
import pickle
import requests
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Telegram settings - your bot token is pre-configured
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8580722750:AAFSFgP3CZOTZL9N4hU6mxIFxpwl0_qG9Zw')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')  # Get this by running: python get_chat_id.py

# Scanner settings
SCAN_INTERVAL_MINUTES = 10  # Send update every 10 minutes
ALWAYS_SEND = True          # Send even if no signal (status update)


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CONFIGURATIONS (Same as validated ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

INDICATORS = [
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        "short_name": "OI Change 1h",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "weight": 0.30
    },
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_accel",
        "short_name": "OI Accel",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 80,
        "weight": 0.25
    },
    {
        "name": "price_rsi_14_lag_48h",
        "short_name": "RSI 14 (48h lag)",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 1.5,
        "weight": 0.20
    },
    {
        "name": "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        "short_name": "Funding Cumul",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 2.0,
        "weight": 0.15
    },
    {
        "name": "sent_feat_fg_zscore_90d",
        "short_name": "Fear/Greed Z",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "weight": 0.10
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram not configured! Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Telegram message sent")
            return True
        else:
            logger.error(f"Telegram error: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def test_telegram():
    """Send test message to verify Telegram setup."""
    message = """
🔔 <b>TEST MESSAGE</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {time} UTC

✅ Telegram is working!
Scanner will send updates every {interval} minutes.
━━━━━━━━━━━━━━━━━━━━━━
<i>BTC Alpha Scanner v38</i>
""".format(
        time=datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
        interval=SCAN_INTERVAL_MINUTES
    )
    
    return send_telegram(message.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_btc_price() -> float:
    """Get current BTC price from Binance."""
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        logger.error(f"Failed to get BTC price: {e}")
        return 0.0


def get_24h_change() -> float:
    """Get 24h price change percentage."""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['priceChangePercent'])
    except:
        return 0.0


def load_model_and_data():
    """Load ensemble model and latest features."""
    model_path = Path("models/ensemble_model.pkl")
    cache_path = Path("data_cache/features_cache.pkl")
    
    if not model_path.exists():
        logger.warning("No model found. Run: python create_ensemble_model.py")
        return None, None
    
    if not cache_path.exists():
        logger.warning("No data cache. Run: python run_exhaustive_search.py")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    
    return model, features


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signal(model: dict, features: pd.DataFrame) -> dict:
    """
    Generate trading signal from model and latest features.
    
    Returns dict with signal, confidence, and indicator breakdown.
    """
    if model is None or features is None:
        return {
            'signal': 0,
            'direction': 'NEUTRAL',
            'confidence': 'NONE',
            'position': 0.0,
            'indicators': []
        }
    
    # Use last row (most recent data)
    last_row = features.iloc[-1]
    
    indicators = model.get('indicators', INDICATORS)
    min_position = model.get('min_position_threshold', 0.1)
    
    signals = []
    weights = []
    indicator_details = []
    
    for ind in indicators:
        name = ind['name']
        if name not in features.columns:
            continue
        
        value = last_row.get(name, np.nan)
        if pd.isna(value):
            continue
        
        # Generate signal for this indicator
        upper = ind.get('upper_threshold', np.nan)
        lower = ind.get('lower_threshold', np.nan)
        direction = ind['direction']
        
        if pd.isna(upper) or pd.isna(lower):
            continue
        
        if direction == 1:
            if value > upper:
                sig = 1
                status = "🟢 LONG"
            elif value < lower:
                sig = -1
                status = "🔴 SHORT"
            else:
                sig = 0
                status = "⚪ FLAT"
        else:
            if value > upper:
                sig = -1
                status = "🔴 SHORT"
            elif value < lower:
                sig = 1
                status = "🟢 LONG"
            else:
                sig = 0
                status = "⚪ FLAT"
        
        signals.append(sig)
        weights.append(ind['weight'])
        
        short_name = ind.get('short_name', name[:20])
        indicator_details.append({
            'name': short_name,
            'signal': sig,
            'status': status,
            'weight': ind['weight'],
            'value': value
        })
    
    if len(signals) == 0:
        return {
            'signal': 0,
            'direction': 'NEUTRAL',
            'confidence': 'NONE',
            'position': 0.0,
            'indicators': []
        }
    
    # Calculate weighted position
    weights = np.array(weights)
    weights = weights / weights.sum()
    position = np.sum(np.array(signals) * weights)
    
    # Determine final signal
    if position > min_position:
        signal = 1
        direction = 'LONG'
    elif position < -min_position:
        signal = -1
        direction = 'SHORT'
    else:
        signal = 0
        direction = 'NEUTRAL'
    
    # Confidence based on position strength
    abs_pos = abs(position)
    if abs_pos >= 0.40:
        confidence = 'HIGH'
    elif abs_pos >= 0.25:
        confidence = 'MEDIUM'
    elif abs_pos >= min_position:
        confidence = 'LOW'
    else:
        confidence = 'NONE'
    
    return {
        'signal': signal,
        'direction': direction,
        'confidence': confidence,
        'position': position,
        'indicators': indicator_details
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_message(signal_data: dict, btc_price: float, change_24h: float) -> str:
    """Format signal as Telegram message."""
    
    # Direction emoji
    dir_emoji = {
        'LONG': '🟢',
        'SHORT': '🔴',
        'NEUTRAL': '⚪'
    }
    
    # Confidence emoji
    conf_emoji = {
        'HIGH': '🔥',
        'MEDIUM': '✅',
        'LOW': '⚡',
        'NONE': '😐'
    }
    
    emoji = dir_emoji.get(signal_data['direction'], '⚪')
    conf = conf_emoji.get(signal_data['confidence'], '😐')
    
    # Format indicators
    ind_text = ""
    for ind in signal_data['indicators']:
        ind_text += f"  {ind['status']} {ind['name']} ({ind['weight']*100:.0f}%)\n"
    
    # Position strength bar
    pos = signal_data['position']
    pos_pct = abs(pos) * 100
    pos_bar = "█" * int(pos_pct / 10) + "░" * (10 - int(pos_pct / 10))
    pos_sign = "+" if pos > 0 else "-" if pos < 0 else " "
    
    # Price change color
    change_emoji = "📈" if change_24h > 0 else "📉" if change_24h < 0 else "➡️"
    
    message = f"""
{emoji} <b>BTC SIGNAL: {signal_data['direction']}</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

<b>💰 BTC Price:</b> ${btc_price:,.0f}
{change_emoji} <b>24h Change:</b> {change_24h:+.2f}%

<b>📊 Signal Strength:</b>
  [{pos_bar}] {pos_sign}{pos_pct:.0f}%
  
<b>🎯 Confidence:</b> {conf} {signal_data['confidence']}

<b>📈 Indicators:</b>
{ind_text}
━━━━━━━━━━━━━━━━━━━━━━
<i>Next update in {SCAN_INTERVAL_MINUTES} min</i>
"""
    
    return message.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_once():
    """Run a single scan and send Telegram notification."""
    logger.info("=" * 50)
    logger.info(f"Scanning at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Get market data
    btc_price = get_btc_price()
    change_24h = get_24h_change()
    logger.info(f"BTC: ${btc_price:,.0f} ({change_24h:+.2f}%)")
    
    # Load model and generate signal
    model, features = load_model_and_data()
    signal_data = generate_signal(model, features)
    
    logger.info(f"Signal: {signal_data['direction']} (conf: {signal_data['confidence']})")
    
    # Always send or only when signal?
    if ALWAYS_SEND or signal_data['signal'] != 0:
        message = format_message(signal_data, btc_price, change_24h)
        send_telegram(message)
    else:
        logger.info("No signal - skipping notification")


def run_continuous():
    """Run scanner continuously every N minutes."""
    logger.info("=" * 70)
    logger.info("TELEGRAM SCANNER STARTED")
    logger.info("=" * 70)
    logger.info(f"Interval: {SCAN_INTERVAL_MINUTES} minutes")
    logger.info(f"Always send: {ALWAYS_SEND}")
    logger.info(f"Bot token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    logger.info(f"Chat ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")
    logger.info("")
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram not configured!")
        logger.error("Add to .env file:")
        logger.error("  TELEGRAM_BOT_TOKEN=your_bot_token")
        logger.error("  TELEGRAM_CHAT_ID=your_chat_id")
        return
    
    # Run immediately
    scan_once()
    
    # Then run on schedule
    logger.info(f"\nNext scan in {SCAN_INTERVAL_MINUTES} minutes...")
    logger.info("Press Ctrl+C to stop")
    
    while True:
        time.sleep(SCAN_INTERVAL_MINUTES * 60)
        try:
            scan_once()
        except Exception as e:
            logger.error(f"Scan error: {e}")
            # Send error notification
            send_telegram(f"⚠️ Scanner error: {str(e)[:100]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BTC Telegram Scanner')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--test', action='store_true', help='Send test message')
    parser.add_argument('--interval', type=int, default=10, help='Scan interval in minutes')
    
    args = parser.parse_args()
    
    global SCAN_INTERVAL_MINUTES
    SCAN_INTERVAL_MINUTES = args.interval
    
    if args.test:
        logger.info("Sending test message...")
        if test_telegram():
            logger.info("✅ Test successful!")
        else:
            logger.error("❌ Test failed - check your .env settings")
    elif args.once:
        scan_once()
    else:
        run_continuous()


if __name__ == "__main__":
    main()
