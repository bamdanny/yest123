import pickle
import pandas as pd
from pathlib import Path

# Check features index
cache_path = Path("data_cache/features_cache.pkl")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)
features = cache['features']
print(f"Features shape: {features.shape}")
print(f"Features index type: {type(features.index)}")
print(f"Features index[:5]: {features.index[:5].tolist()}")

# Check prices index
price_path = Path("data_cache/binance/price_4h_365d.parquet")
prices = pd.read_parquet(price_path)
print(f"\nPrices shape: {prices.shape}")
print(f"Prices columns: {prices.columns.tolist()}")
print(f"Prices index type: {type(prices.index)}")
print(f"Prices index[:5]: {prices.index[:5].tolist()}")

# Check if timestamp column exists
if 'timestamp' in prices.columns:
    print(f"\nPrices timestamp[:5]: {prices['timestamp'].iloc[:5].tolist()}")
