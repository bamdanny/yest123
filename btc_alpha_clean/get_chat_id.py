"""
Get your Telegram Chat ID

INSTRUCTIONS:
1. First, message your bot on Telegram (say "hello")
2. Run this script: python get_chat_id.py
3. Copy the chat_id from the output
4. Add to your .env file: TELEGRAM_CHAT_ID=your_chat_id
"""

import requests

# Your bot token (hardcoded for convenience)
BOT_TOKEN = "8580722750:AAFSFgP3CZOTZL9N4hU6mxIFxpwl0_qG9Zw"

def get_chat_id():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print("=" * 60)
        print("TELEGRAM CHAT ID FINDER")
        print("=" * 60)
        
        if not data.get('ok'):
            print(f"ERROR: {data}")
            return
        
        results = data.get('result', [])
        
        if not results:
            print("\nNo messages found!")
            print("Make sure you:")
            print("1. Search for your bot on Telegram")
            print("2. Send it a message (just say 'hello')")
            print("3. Run this script again")
            return
        
        print(f"\nFound {len(results)} update(s):\n")
        
        chat_ids = set()
        for update in results:
            if 'message' in update:
                chat = update['message']['chat']
                chat_id = chat['id']
                chat_ids.add(chat_id)
                
                first_name = chat.get('first_name', 'Unknown')
                username = chat.get('username', 'N/A')
                text = update['message'].get('text', '')[:50]
                
                print(f"  Chat ID: {chat_id}")
                print(f"  Name: {first_name}")
                print(f"  Username: @{username}")
                print(f"  Message: {text}")
                print("-" * 40)
        
        if chat_ids:
            print(f"\n{'=' * 60}")
            print("YOUR CHAT ID(S):")
            for cid in chat_ids:
                print(f"  {cid}")
            print(f"{'=' * 60}")
            print("\nAdd this to your .env file:")
            print(f"  TELEGRAM_CHAT_ID={list(chat_ids)[0]}")
            print("\nThen run:")
            print("  python telegram_scanner.py --test")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_chat_id()
