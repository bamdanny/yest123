"""
Quick test to verify Telegram bot setup

Usage:
    python test_telegram.py YOUR_CHAT_ID

Example:
    python test_telegram.py 123456789
"""

import sys
import requests

BOT_TOKEN = "8580722750:AAFSFgP3CZOTZL9N4hU6mxIFxpwl0_qG9Zw"

def test_telegram(chat_id: str):
    print("=" * 50)
    print("TELEGRAM CONNECTION TEST")
    print("=" * 50)
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    message = """
✅ <b>Connection Test Successful!</b>

Your scanner is properly configured.
You will receive alerts here when signals trigger.

Run: <code>python run_scanner.py</code>
"""
    
    payload = {
        "chat_id": chat_id,
        "text": message.strip(),
        "parse_mode": "HTML"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        
        if resp.status_code == 200:
            print("\n✓ SUCCESS! Check your Telegram for the test message.\n")
            print("Your chat_id is correct. Now:")
            print("1. Make sure config/scanner_config.json has this chat_id")
            print("2. Run: python run_scanner.py")
        else:
            print(f"\n✗ FAILED: {resp.text}")
            print("\nCommon issues:")
            print("- Wrong chat_id (run get_chat_id.py)")
            print("- Bot blocked (unblock it in Telegram)")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_telegram.py YOUR_CHAT_ID")
        print("\nTo find your chat_id:")
        print("1. Message your bot on Telegram")
        print("2. Run: python get_chat_id.py")
    else:
        test_telegram(sys.argv[1])
