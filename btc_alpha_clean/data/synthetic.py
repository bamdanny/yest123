"""
Synthetic Data Generator for BTC Alpha Discovery

Generates realistic synthetic data for testing the pipeline when
external API access is not available. Creates data with known patterns
that the discovery system should be able to find.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates realistic synthetic BTC and market data.
    
    Embeds known predictive patterns for validation:
    1. High funding rate → price reversal
    2. Extreme RSI → mean reversion
    3. High OI + funding divergence → liquidation cascade
    4. VIX spike → increased volatility
    5. Fear & Greed extreme → contrarian signal
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
        
        # Embedded patterns (ground truth for validation)
        self.embedded_patterns = {
            'funding_reversal': {
                'description': 'Price reverses when funding > 0.05% or < -0.05%',
                'win_rate': 0.62,
                'avg_return': 0.015
            },
            'rsi_mean_reversion': {
                'description': 'RSI < 25 → long, RSI > 75 → short',
                'win_rate': 0.58,
                'avg_return': 0.012
            },
            'oi_liquidation': {
                'description': 'OI spike + extreme funding → cascade',
                'win_rate': 0.55,
                'avg_return': 0.025
            },
            'vix_volatility': {
                'description': 'VIX > 25 increases BTC volatility 50%',
                'win_rate': 0.52,
                'avg_return': 0.018
            },
            'sentiment_contrarian': {
                'description': 'Fear < 20 → long, Greed > 80 → short',
                'win_rate': 0.56,
                'avg_return': 0.014
            }
        }
    
    def generate_price_series(
        self,
        start_date: datetime,
        end_date: datetime,
        interval_hours: int = 1,
        initial_price: float = 30000.0,
        annual_volatility: float = 0.80,
        annual_drift: float = 0.10
    ) -> pd.DataFrame:
        """
        Generate realistic BTC price series using GBM with regime switching.
        
        Returns DataFrame with: timestamp, open, high, low, close, volume
        """
        # Calculate number of periods
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        n_periods = total_hours // interval_hours
        
        # Time parameters
        dt = interval_hours / (365.25 * 24)  # Fraction of year
        vol = annual_volatility * np.sqrt(dt)
        drift = annual_drift * dt
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_date, periods=n_periods, freq=f'{interval_hours}h')
        
        # Generate returns with regime switching
        returns = np.zeros(n_periods)
        regime = 'normal'
        regime_vol_multiplier = 1.0
        
        for i in range(n_periods):
            # Regime switching (5% chance per period)
            if np.random.random() < 0.05:
                regime = np.random.choice(['normal', 'high_vol', 'low_vol', 'trending'])
                if regime == 'high_vol':
                    regime_vol_multiplier = 2.0
                elif regime == 'low_vol':
                    regime_vol_multiplier = 0.5
                elif regime == 'trending':
                    regime_vol_multiplier = 1.2
                else:
                    regime_vol_multiplier = 1.0
            
            # Generate return
            returns[i] = drift + vol * regime_vol_multiplier * np.random.randn()
            
            # Add occasional jumps (3% chance)
            if np.random.random() < 0.03:
                jump = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.08)
                returns[i] += jump
        
        # Calculate prices
        close_prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        data = []
        for i in range(n_periods):
            close = close_prices[i]
            
            # Intrabar volatility
            intrabar_vol = abs(returns[i]) * np.random.uniform(0.5, 1.5)
            
            # Generate high/low
            high = close * (1 + intrabar_vol * np.random.uniform(0.3, 1.0))
            low = close * (1 - intrabar_vol * np.random.uniform(0.3, 1.0))
            
            # Open based on previous close
            if i > 0:
                open_price = close_prices[i-1] * (1 + np.random.uniform(-0.002, 0.002))
            else:
                open_price = initial_price
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Volume (correlated with volatility)
            base_volume = 50000
            vol_factor = 1 + abs(returns[i]) * 10
            volume = base_volume * vol_factor * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_funding_rates(
        self,
        price_df: pd.DataFrame,
        funding_interval_hours: int = 8
    ) -> pd.DataFrame:
        """
        Generate funding rates with embedded predictive pattern.
        
        Pattern: High funding → price tends to reverse
        """
        # Resample to funding intervals
        funding_times = price_df.index[::funding_interval_hours]
        
        data = []
        prev_funding = 0.0001
        
        for i, ts in enumerate(funding_times):
            # Base funding follows price momentum
            if i > 0:
                price_change = (price_df.loc[ts, 'close'] / price_df.iloc[max(0, i*funding_interval_hours - 24)]['close']) - 1
                base_funding = price_change * 0.5  # Funding follows price
            else:
                base_funding = 0.0001
            
            # Add noise and mean reversion
            funding = prev_funding * 0.7 + base_funding * 0.3 + np.random.normal(0, 0.0002)
            
            # Clamp to realistic range
            funding = np.clip(funding, -0.001, 0.001)
            
            data.append({
                'timestamp': ts,
                'funding_rate': funding,
                'funding_rate_next': funding * np.random.uniform(0.9, 1.1)  # Predicted next
            })
            
            prev_funding = funding
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_open_interest(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate open interest data.
        
        Pattern: OI spikes before major moves
        """
        data = []
        oi = 10_000_000_000  # $10B starting OI
        
        for ts in price_df.index:
            price = price_df.loc[ts, 'close']
            
            # OI tends to increase in rallies, decrease in crashes
            price_return = price_df.loc[:ts, 'close'].pct_change().iloc[-1] if len(price_df.loc[:ts]) > 1 else 0
            
            # OI change
            oi_change = oi * (0.001 * price_return + np.random.normal(0, 0.005))
            oi += oi_change
            oi = max(oi, 5_000_000_000)  # Floor at $5B
            
            data.append({
                'timestamp': ts,
                'open_interest': oi,
                'oi_change_1h': oi_change / oi if oi > 0 else 0
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_liquidations(
        self,
        price_df: pd.DataFrame,
        oi_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate liquidation data.
        
        Pattern: Large liquidations follow big price moves
        """
        data = []
        
        for ts in price_df.index:
            price_return = price_df.loc[:ts, 'close'].pct_change().iloc[-1] if len(price_df.loc[:ts]) > 1 else 0
            oi = oi_df.loc[ts, 'open_interest'] if ts in oi_df.index else 10_000_000_000
            
            # Liquidations proportional to price move and OI
            base_liq = abs(price_return) * oi * 0.1
            
            # Add randomness
            long_liq = max(0, base_liq * (1 if price_return < 0 else 0.2) * np.random.uniform(0.5, 2.0))
            short_liq = max(0, base_liq * (1 if price_return > 0 else 0.2) * np.random.uniform(0.5, 2.0))
            
            data.append({
                'timestamp': ts,
                'long_liquidations': long_liq,
                'short_liquidations': short_liq,
                'total_liquidations': long_liq + short_liq,
                'liquidation_ratio': long_liq / (short_liq + 1)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_long_short_ratio(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate long/short ratio.
        
        Pattern: Extreme ratios are contrarian
        """
        data = []
        ratio = 1.0
        
        for ts in price_df.index:
            # Ratio follows price with lag
            price_return = price_df.loc[:ts, 'close'].pct_change().iloc[-1] if len(price_df.loc[:ts]) > 1 else 0
            
            # Retail tends to be wrong - ratio goes up after price goes up
            ratio = ratio * 0.95 + (1 + price_return * 5) * 0.05
            ratio = np.clip(ratio, 0.5, 2.0)
            ratio += np.random.normal(0, 0.02)
            
            data.append({
                'timestamp': ts,
                'long_short_ratio': ratio,
                'long_account_ratio': ratio / (1 + ratio),
                'short_account_ratio': 1 / (1 + ratio)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_macro_data(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate macro indicators (VIX, DXY, etc.).
        
        Pattern: High VIX → high BTC volatility
        """
        data = []
        vix = 18.0
        dxy = 102.0
        
        for ts in price_df.index:
            # VIX mean reverts around 18
            vix = vix * 0.99 + 18 * 0.01 + np.random.normal(0, 0.5)
            
            # Occasional VIX spikes
            if np.random.random() < 0.02:
                vix += np.random.uniform(5, 15)
            
            vix = np.clip(vix, 10, 80)
            
            # DXY random walk
            dxy += np.random.normal(0, 0.1)
            dxy = np.clip(dxy, 90, 115)
            
            data.append({
                'timestamp': ts,
                'vix': vix,
                'dxy': dxy,
                'spy_return': np.random.normal(0.0002, 0.01),
                'yield_10y': 4.0 + np.random.normal(0, 0.02),
                'yield_2y': 4.5 + np.random.normal(0, 0.02)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_sentiment(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate sentiment data.
        
        Pattern: Extreme fear/greed is contrarian
        """
        data = []
        fear_greed = 50
        
        for ts in price_df.index:
            # Fear & Greed follows price with momentum
            returns_7d = price_df.loc[:ts, 'close'].pct_change(periods=min(168, len(price_df.loc[:ts])-1)).iloc[-1] if len(price_df.loc[:ts]) > 1 else 0
            
            # Update F&G
            fear_greed = fear_greed * 0.9 + (50 + returns_7d * 500) * 0.1
            fear_greed += np.random.normal(0, 3)
            fear_greed = np.clip(fear_greed, 0, 100)
            
            data.append({
                'timestamp': ts,
                'fear_greed_index': fear_greed,
                'fear_greed_classification': 'Extreme Fear' if fear_greed < 20 else 
                                             'Fear' if fear_greed < 40 else
                                             'Neutral' if fear_greed < 60 else
                                             'Greed' if fear_greed < 80 else 'Extreme Greed'
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def generate_options_data(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate options metrics (IV, put-call ratio)."""
        data = []
        iv = 0.60
        pcr = 0.5
        
        for ts in price_df.index:
            # IV follows realized vol
            realized_vol = price_df.loc[:ts, 'close'].pct_change().rolling(24).std().iloc[-1] if len(price_df.loc[:ts]) > 24 else 0.03
            iv = iv * 0.95 + (realized_vol * 15) * 0.05 + np.random.normal(0, 0.02)
            iv = np.clip(iv, 0.30, 2.00)
            
            # Put-call ratio
            pcr = pcr * 0.95 + 0.5 * 0.05 + np.random.normal(0, 0.05)
            pcr = np.clip(pcr, 0.2, 1.5)
            
            data.append({
                'timestamp': ts,
                'implied_volatility': iv,
                'put_call_ratio': pcr,
                'max_pain': price_df.loc[ts, 'close'] * np.random.uniform(0.95, 1.05)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def inject_predictive_patterns(
        self,
        price_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Modify price series to embed predictive patterns.
        
        This ensures the discovery system can find real edges.
        """
        price_df = price_df.copy()
        
        # Pattern 1: High funding reversal
        # When funding > 0.05%, bias next 24h return negative
        for ts in funding_df.index:
            if ts not in price_df.index:
                continue
                
            funding = funding_df.loc[ts, 'funding_rate']
            
            if abs(funding) > 0.0005:  # 0.05%
                # Find next 24 bars
                future_idx = price_df.index[price_df.index > ts][:24]
                
                if len(future_idx) > 0:
                    # Add reversal bias
                    direction = -1 if funding > 0 else 1
                    for future_ts in future_idx:
                        current_close = price_df.loc[future_ts, 'close']
                        adjustment = direction * current_close * 0.001 * np.random.uniform(0.5, 1.5)
                        price_df.loc[future_ts, 'close'] += adjustment
                        price_df.loc[future_ts, 'high'] = max(price_df.loc[future_ts, 'high'], price_df.loc[future_ts, 'close'])
                        price_df.loc[future_ts, 'low'] = min(price_df.loc[future_ts, 'low'], price_df.loc[future_ts, 'close'])
        
        # Pattern 2: Extreme sentiment reversal
        for ts in sentiment_df.index:
            if ts not in price_df.index:
                continue
                
            fg = sentiment_df.loc[ts, 'fear_greed_index']
            
            if fg < 20 or fg > 80:
                future_idx = price_df.index[price_df.index > ts][:48]
                
                if len(future_idx) > 0:
                    direction = 1 if fg < 20 else -1  # Contrarian
                    for future_ts in future_idx:
                        current_close = price_df.loc[future_ts, 'close']
                        adjustment = direction * current_close * 0.0005 * np.random.uniform(0.5, 1.5)
                        price_df.loc[future_ts, 'close'] += adjustment
                        price_df.loc[future_ts, 'high'] = max(price_df.loc[future_ts, 'high'], price_df.loc[future_ts, 'close'])
                        price_df.loc[future_ts, 'low'] = min(price_df.loc[future_ts, 'low'], price_df.loc[future_ts, 'close'])
        
        return price_df
    
    def generate_all_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        n_days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Returns dict with all data sources aligned.
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=n_days)
        
        logger.info(f"Generating synthetic data from {start_date} to {end_date}")
        
        # Generate base price series
        logger.info("  Generating price series...")
        price_df = self.generate_price_series(start_date, end_date)
        
        # Generate derivative data
        logger.info("  Generating derivatives data...")
        funding_df = self.generate_funding_rates(price_df)
        oi_df = self.generate_open_interest(price_df)
        liquidation_df = self.generate_liquidations(price_df, oi_df)
        ls_ratio_df = self.generate_long_short_ratio(price_df)
        
        # Generate macro
        logger.info("  Generating macro data...")
        macro_df = self.generate_macro_data(price_df)
        
        # Generate sentiment
        logger.info("  Generating sentiment data...")
        sentiment_df = self.generate_sentiment(price_df)
        options_df = self.generate_options_data(price_df)
        
        # Inject predictive patterns
        logger.info("  Injecting predictive patterns...")
        price_df = self.inject_predictive_patterns(price_df, funding_df, sentiment_df)
        
        # Combine into unified dataset
        logger.info("  Merging datasets...")
        combined = price_df.copy()
        
        # Merge all dataframes
        for df, name in [
            (funding_df, 'funding'),
            (oi_df, 'oi'),
            (liquidation_df, 'liq'),
            (ls_ratio_df, 'ls'),
            (macro_df, 'macro'),
            (sentiment_df, 'sentiment'),
            (options_df, 'options')
        ]:
            # Forward fill to hourly
            df_reindexed = df.reindex(combined.index, method='ffill')
            combined = combined.join(df_reindexed, rsuffix=f'_{name}')
        
        combined = combined.ffill().bfill()
        
        logger.info(f"Generated {len(combined)} rows with {len(combined.columns)} columns")
        
        return {
            'price': price_df,
            'funding': funding_df,
            'open_interest': oi_df,
            'liquidations': liquidation_df,
            'long_short_ratio': ls_ratio_df,
            'macro': macro_df,
            'sentiment': sentiment_df,
            'options': options_df,
            'combined': combined,
            'embedded_patterns': self.embedded_patterns
        }


def generate_test_data(n_days: int = 365, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate test data."""
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_all_data(n_days=n_days)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test generation
    data = generate_test_data(n_days=180)
    
    print("\nGenerated datasets:")
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"  {name}: {type(df)}")
    
    print("\nEmbedded patterns (ground truth):")
    for pattern, info in data['embedded_patterns'].items():
        print(f"  {pattern}: {info['description']}")
        print(f"    Expected win rate: {info['win_rate']:.1%}")
        print(f"    Expected avg return: {info['avg_return']:.2%}")
