# BTC Alpha Discovery

ML-based BTC direction prediction with Telegram alerts.

## Quick Start

```bash
# 1. Setup
pip install -r requirements.txt
copy .env.example .env  # Add your API keys

# 2. Phase 1: Find Alpha (run once)
python run_exhaustive_search.py --mode single --top-n 10

# 3. Phase 2: Train ML Model
python train_model.py

# 4. Phase 3: Run Scanner
python run_ml_scanner.py
```

## Telegram Setup

```bash
# 1. Message your bot on Telegram
# 2. Get your chat_id
python get_chat_id.py

# 3. Update config/ml_config.json with your chat_id
# 4. Test connection
python test_telegram.py YOUR_CHAT_ID
```

## API Keys (free tiers)

- **CoinGlass**: https://www.coinglass.com/pricing (required)
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html (required)
- **Polygon**: https://polygon.io/ (optional - VIX/DXY data)

## Files

```
run_exhaustive_search.py  - Phase 1: Find alpha indicators
train_model.py            - Phase 2: Train ML ensemble
run_ml_scanner.py         - Phase 3: Live scanner + alerts
```
