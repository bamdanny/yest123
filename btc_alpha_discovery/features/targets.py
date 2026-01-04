"""
Target Variable Engineering for Alpha Discovery.

Generates comprehensive target variables for ML-based discovery:
- Forward returns at multiple horizons
- Risk-adjusted targets (Sharpe-based)
- Classification targets (direction, profitability)
- Exit-aware targets (with stop-loss/take-profit)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TargetGenerator:
    """
    Generates target variables for alpha discovery.
    
    Key principle: We don't know what the optimal holding period is,
    so we generate targets for multiple horizons and let the ML discover.
    """
    
    # Transaction costs (round-trip)
    COMMISSION_PCT = 0.04  # 0.04% per side = 0.08% round trip
    SLIPPAGE_PCT = 0.02    # 0.02% per side = 0.04% round trip
    TOTAL_COST_PCT = (COMMISSION_PCT + SLIPPAGE_PCT) * 2 / 100  # 0.0012
    
    # Holding periods to test (in CANDLES at 4h intervals)
    # For 4h timeframe: 1=4h, 3=12h, 6=24h(1d), 12=48h(2d), 18=72h(3d), 42=168h(1w), 84=336h(2w)
    # NOTE: Value 24 = 96h (4 days), NOT 24 hours!
    HORIZONS = [1, 3, 6, 12, 18, 42, 84]  # Removed misleading "24" (was 4d, not 24h)
    
    # Stop-loss / Take-profit levels to test
    STOP_LOSS_PCTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    TAKE_PROFIT_PCTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                Index should be datetime
        """
        self.df = df.copy()
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data."""
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
            
    def generate_all_targets(self) -> pd.DataFrame:
        """
        Generate comprehensive target variable set.
        
        Returns:
            DataFrame with all target variables
        """
        logger.info("Generating target variables...")
        
        targets = pd.DataFrame(index=self.df.index)
        
        # 1. Simple forward returns
        for horizon in self.HORIZONS:
            targets = self._add_forward_returns(targets, horizon)
            
        # 2. Risk-adjusted returns (Sharpe-style)
        for horizon in self.HORIZONS:
            targets = self._add_risk_adjusted_returns(targets, horizon)
            
        # 3. Classification targets
        for horizon in self.HORIZONS:
            targets = self._add_classification_targets(targets, horizon)
            
        # 4. Exit-aware targets (with stop-loss/take-profit)
        for horizon in [4, 8, 24, 48, 168]:  # Subset for efficiency
            for sl in [1.0, 2.0, 3.0]:
                for tp in [1.5, 2.0, 3.0, 5.0]:
                    targets = self._add_exit_aware_targets(targets, horizon, sl, tp)
                    
        # 5. Volatility-adjusted targets
        for horizon in self.HORIZONS:
            targets = self._add_volatility_adjusted_returns(targets, horizon)
            
        # 6. Maximum adverse excursion targets
        for horizon in [4, 8, 24, 48]:
            targets = self._add_mae_mfe_targets(targets, horizon)
            
        logger.info(f"Generated {len(targets.columns)} target variables")
        return targets
        
    def _add_forward_returns(self, targets: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add simple forward return targets."""
        # Log returns (more stable for ML)
        targets[f'return_log_{horizon}h'] = np.log(
            self.df['close'].shift(-horizon) / self.df['close']
        )
        
        # Simple returns
        targets[f'return_simple_{horizon}h'] = (
            self.df['close'].shift(-horizon) / self.df['close'] - 1
        )
        
        # Net returns (after costs)
        targets[f'return_net_{horizon}h'] = (
            targets[f'return_simple_{horizon}h'] - self.TOTAL_COST_PCT
        )
        
        return targets
        
    def _add_risk_adjusted_returns(self, targets: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add risk-adjusted return targets (Sharpe-style)."""
        # Calculate forward volatility
        future_returns = self.df['close'].pct_change().shift(-1)
        
        # Rolling volatility over the horizon
        vol = future_returns.rolling(window=horizon).std().shift(-horizon)
        
        # Sharpe-style ratio (return / volatility)
        ret = targets.get(f'return_simple_{horizon}h')
        if ret is not None:
            targets[f'sharpe_{horizon}h'] = ret / (vol * np.sqrt(horizon) + 1e-8)
            
        return targets
        
    def _add_classification_targets(self, targets: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add classification targets."""
        ret = targets.get(f'return_net_{horizon}h')
        
        if ret is not None:
            # Binary: profitable after costs
            targets[f'profitable_{horizon}h'] = (ret > 0).astype(int)
            
            # Ternary: strong long / neutral / strong short
            # Threshold based on typical volatility
            threshold = 0.005  # 0.5%
            targets[f'direction_{horizon}h'] = pd.cut(
                ret,
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=[-1, 0, 1]
            ).astype(float)
            
            # Multi-class: quintiles
            targets[f'quintile_{horizon}h'] = pd.qcut(
                ret.rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
            
        return targets
        
    def _add_exit_aware_targets(
        self,
        targets: pd.DataFrame,
        max_horizon: int,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> pd.DataFrame:
        """
        Add targets that account for stop-loss and take-profit exits.
        
        This simulates actual trading with exit conditions.
        """
        col_name = f'exit_sl{stop_loss_pct}_tp{take_profit_pct}_{max_horizon}h'
        
        returns = []
        exit_reasons = []
        holding_periods = []
        
        close = self.df['close'].values
        high = self.df['high'].values
        low = self.df['low'].values
        
        for i in range(len(close)):
            entry_price = close[i]
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            tp_price = entry_price * (1 + take_profit_pct / 100)
            
            # Look forward up to max_horizon bars
            exit_price = None
            exit_reason = None
            bars_held = max_horizon
            
            for j in range(1, min(max_horizon + 1, len(close) - i)):
                # Check if stop-loss hit (using low)
                if low[i + j] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    bars_held = j
                    break
                    
                # Check if take-profit hit (using high)
                if high[i + j] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    bars_held = j
                    break
                    
            # If no exit triggered, use close at max_horizon
            if exit_price is None:
                if i + max_horizon < len(close):
                    exit_price = close[i + max_horizon]
                    exit_reason = 'time_exit'
                else:
                    exit_price = np.nan
                    exit_reason = 'no_data'
                    bars_held = np.nan
                    
            # Calculate return
            if exit_price is not None and not np.isnan(exit_price):
                ret = (exit_price / entry_price - 1) - self.TOTAL_COST_PCT
            else:
                ret = np.nan
                
            returns.append(ret)
            exit_reasons.append(exit_reason)
            holding_periods.append(bars_held)
            
        targets[f'{col_name}_return'] = returns
        targets[f'{col_name}_bars_held'] = holding_periods
        
        # Encode exit reasons
        reason_map = {'stop_loss': -1, 'time_exit': 0, 'take_profit': 1, 'no_data': np.nan}
        targets[f'{col_name}_exit_type'] = [reason_map.get(r, np.nan) for r in exit_reasons]
        
        return targets
        
    def _add_volatility_adjusted_returns(
        self,
        targets: pd.DataFrame,
        horizon: int
    ) -> pd.DataFrame:
        """Add returns normalized by recent volatility."""
        # Recent volatility (lookback)
        lookback = max(24, horizon)
        vol = self.df['close'].pct_change().rolling(window=lookback).std()
        
        ret = targets.get(f'return_simple_{horizon}h')
        if ret is not None:
            targets[f'return_vol_adj_{horizon}h'] = ret / (vol + 1e-8)
            
        return targets
        
    def _add_mae_mfe_targets(
        self,
        targets: pd.DataFrame,
        horizon: int
    ) -> pd.DataFrame:
        """
        Add Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
        
        These are useful for discovering optimal stop-loss/take-profit levels.
        """
        close = self.df['close'].values
        high = self.df['high'].values
        low = self.df['low'].values
        
        mae_list = []
        mfe_list = []
        
        for i in range(len(close)):
            entry_price = close[i]
            
            # Look forward
            if i + horizon < len(close):
                future_highs = high[i+1:i+horizon+1]
                future_lows = low[i+1:i+horizon+1]
                
                # MFE: maximum gain (for long position)
                mfe = (np.max(future_highs) / entry_price - 1) * 100
                
                # MAE: maximum loss (for long position)
                mae = (np.min(future_lows) / entry_price - 1) * 100
            else:
                mfe = np.nan
                mae = np.nan
                
            mae_list.append(mae)
            mfe_list.append(mfe)
            
        targets[f'mae_{horizon}h'] = mae_list
        targets[f'mfe_{horizon}h'] = mfe_list
        targets[f'mfe_mae_ratio_{horizon}h'] = np.array(mfe_list) / (np.abs(mae_list) + 0.01)
        
        return targets


class MultiTimeframeTargetGenerator:
    """
    Generates targets considering multiple timeframes simultaneously.
    
    The idea: A good entry might be one where multiple timeframes agree.
    """
    
    TIMEFRAMES = ['1h', '4h', '1d']
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize with data at multiple timeframes.
        
        Args:
            data_dict: Dict mapping timeframe string to OHLCV DataFrame
        """
        self.data_dict = data_dict
        
    def generate_aligned_targets(self) -> pd.DataFrame:
        """
        Generate targets aligned across timeframes.
        
        Returns targets at the finest granularity (1h) with
        higher-timeframe agreement signals.
        """
        # Generate targets for each timeframe
        tf_targets = {}
        for tf, df in self.data_dict.items():
            gen = TargetGenerator(df)
            tf_targets[tf] = gen.generate_all_targets()
            
        # Use 1h as base, add higher-timeframe alignment
        base_tf = '1h'
        if base_tf not in tf_targets:
            base_tf = list(tf_targets.keys())[0]
            
        targets = tf_targets[base_tf].copy()
        
        # Add alignment signals
        for horizon in [24, 48, 168]:
            # Check if 4h and 1d agree with 1h
            alignments = []
            for tf in ['4h', '1d']:
                if tf in tf_targets and f'direction_{horizon}h' in tf_targets[tf].columns:
                    # Resample to 1h
                    resampled = tf_targets[tf][f'direction_{horizon}h'].reindex(
                        targets.index, method='ffill'
                    )
                    alignments.append(resampled)
                    
            if alignments:
                # Count how many timeframes agree
                alignment_df = pd.concat(alignments, axis=1)
                base_direction = targets.get(f'direction_{horizon}h', pd.Series(0, index=targets.index))
                
                # Agreement score: -1 to 1
                targets[f'tf_alignment_{horizon}h'] = alignment_df.mean(axis=1)
                
        return targets


def generate_entry_quality_score(
    returns: pd.Series,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    trade_count: int,
    period_days: int = 90  # Default 90-day period
) -> float:
    """
    Calculate entry quality score for optimization.
    
    This is what we're trying to maximize when discovering entry rules.
    
    Args:
        returns: Series of trade returns
        win_rate: Percentage of winning trades
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (negative)
        trade_count: Number of trades
        period_days: Number of days in the evaluation period
        
    Returns:
        Quality score (higher is better)
    """
    if trade_count < 30:
        return -np.inf  # Need statistical significance
        
    # Profit factor
    if avg_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = abs(avg_win / avg_loss) * (win_rate / (1 - win_rate + 1e-8))
        
    # Expectancy (expected return per trade)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Sharpe ratio using TRADE FREQUENCY (not bar frequency!)
    if returns.std() > 0 and period_days > 0:
        trades_per_year = (trade_count / period_days) * 365
        per_trade_sharpe = returns.mean() / returns.std()
        sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
    else:
        sharpe = 0
        
    # Penalty for low trade count (prefer more trades for robustness)
    count_factor = min(1.0, trade_count / 100)
    
    # Combined score
    score = (
        0.3 * np.clip(profit_factor, 0, 10) / 10 +
        0.3 * np.clip(sharpe, -3, 3) / 3 +
        0.2 * np.clip(expectancy * 100, -10, 10) / 10 +
        0.2 * count_factor
    )
    
    return score
