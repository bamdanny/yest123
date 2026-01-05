#!/usr/bin/env python3
"""
SIMPLE RULE-BASED TRADING SYSTEM

Why this instead of ML:
- Single indicators get OOS Sharpe 6-8
- ML model with 50 features gets Sharpe -1.74
- ML destroys the alpha that simple rules capture

This system uses ONLY the indicators that passed OOS validation in Phase 1.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TOP 5 OOS-VALIDATED INDICATORS (from Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════
# These are the ONLY indicators that matter - proven in out-of-sample testing

OOS_VALIDATED_RULES = {
    'oi_change': {
        'feature': 'deriv_feat_cg_oi_aggregated_oi_close_change_1h',
        'oos_sharpe': 8.59,
        'direction': 'higher_is_bullish',  # High OI change = bullish
        'threshold_type': 'percentile',
        'long_threshold': 0.67,  # Top 33%
        'short_threshold': 0.33,  # Bottom 33%
    },
    'oi_accel': {
        'feature': 'deriv_feat_cg_oi_aggregated_oi_close_accel',
        'oos_sharpe': 7.36,
        'direction': 'higher_is_bullish',
        'threshold_type': 'percentile',
        'long_threshold': 0.67,
        'short_threshold': 0.33,
    },
    'taker_ratio': {
        'feature': 'taker_buy_ratio',
        'oos_sharpe': 6.44,
        'direction': 'higher_is_bullish',  # More buying = bullish
        'threshold_type': 'absolute',
        'long_threshold': 0.52,  # Buyers > 52%
        'short_threshold': 0.48,  # Buyers < 48%
    },
    'rsi_lag': {
        'feature': 'price_rsi_14_lag_48h',
        'oos_sharpe': 6.08,
        'direction': 'mean_reversion',  # Oversold = bullish, overbought = bearish
        'threshold_type': 'absolute',
        'long_threshold': 35,  # RSI < 35 = oversold = buy
        'short_threshold': 65,  # RSI > 65 = overbought = sell
    },
    'bb_width': {
        'feature': 'price_bb_width_50',
        'oos_sharpe': 4.95,
        'direction': 'lower_is_bullish',  # Narrow bands = breakout coming
        'threshold_type': 'percentile',
        'long_threshold': 0.33,  # Bottom 33% (narrow)
        'short_threshold': 0.67,  # Top 33% (wide)
    },
}


class SimpleRuleSystem:
    """
    Simple voting-based trading system using OOS-validated rules.
    
    Each rule votes +1 (bullish), -1 (bearish), or 0 (neutral).
    Final signal based on vote count.
    """
    
    def __init__(self, min_votes: int = 2):
        """
        Args:
            min_votes: Minimum votes needed to generate signal (default 2)
        """
        self.min_votes = min_votes
        self.rules = OOS_VALIDATED_RULES
        self.percentile_cache = {}
        
    def fit(self, df: pd.DataFrame):
        """
        Calculate percentile thresholds from training data.
        
        Args:
            df: Training dataframe with features
        """
        logger.info("Fitting simple rule system...")
        
        for rule_name, rule_config in self.rules.items():
            feature = rule_config['feature']
            
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found for rule {rule_name}")
                continue
            
            if rule_config['threshold_type'] == 'percentile':
                # Cache the actual values at these percentiles
                self.percentile_cache[rule_name] = {
                    'long': df[feature].quantile(rule_config['long_threshold']),
                    'short': df[feature].quantile(rule_config['short_threshold']),
                }
                logger.info(f"  {rule_name}: long > {self.percentile_cache[rule_name]['long']:.4f}, "
                           f"short < {self.percentile_cache[rule_name]['short']:.4f}")
        
        logger.info(f"Fitted {len(self.percentile_cache)} percentile-based rules")
        
    def predict_single(self, features: Dict[str, float]) -> Tuple[str, int, Dict[str, int]]:
        """
        Generate signal for a single observation.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Tuple of (signal, vote_count, individual_votes)
        """
        votes = {}
        
        for rule_name, rule_config in self.rules.items():
            feature = rule_config['feature']
            
            if feature not in features:
                votes[rule_name] = 0
                continue
            
            value = features[feature]
            
            if pd.isna(value):
                votes[rule_name] = 0
                continue
            
            # Get thresholds
            if rule_config['threshold_type'] == 'percentile':
                if rule_name not in self.percentile_cache:
                    votes[rule_name] = 0
                    continue
                long_thresh = self.percentile_cache[rule_name]['long']
                short_thresh = self.percentile_cache[rule_name]['short']
            else:
                long_thresh = rule_config['long_threshold']
                short_thresh = rule_config['short_threshold']
            
            # Generate vote based on rule direction
            direction = rule_config['direction']
            
            if direction == 'higher_is_bullish':
                if value > long_thresh:
                    votes[rule_name] = 1  # Bullish
                elif value < short_thresh:
                    votes[rule_name] = -1  # Bearish
                else:
                    votes[rule_name] = 0
                    
            elif direction == 'lower_is_bullish':
                if value < long_thresh:
                    votes[rule_name] = 1  # Bullish (low value)
                elif value > short_thresh:
                    votes[rule_name] = -1  # Bearish (high value)
                else:
                    votes[rule_name] = 0
                    
            elif direction == 'mean_reversion':
                # For RSI: low = oversold = bullish, high = overbought = bearish
                if value < long_thresh:
                    votes[rule_name] = 1  # Oversold = buy
                elif value > short_thresh:
                    votes[rule_name] = -1  # Overbought = sell
                else:
                    votes[rule_name] = 0
            else:
                votes[rule_name] = 0
        
        # Sum votes
        total_votes = sum(votes.values())
        
        # Generate signal
        if total_votes >= self.min_votes:
            signal = 'LONG'
        elif total_votes <= -self.min_votes:
            signal = 'SHORT'
        else:
            signal = 'NO_TRADE'
        
        return signal, total_votes, votes
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire dataframe.
        
        Args:
            df: Dataframe with features
            
        Returns:
            Dataframe with signals and votes
        """
        results = []
        
        for idx in df.index:
            features = df.loc[idx].to_dict()
            signal, total_votes, individual_votes = self.predict_single(features)
            
            result = {
                'index': idx,
                'signal': signal,
                'total_votes': total_votes,
            }
            result.update({f'vote_{k}': v for k, v in individual_votes.items()})
            results.append(result)
        
        return pd.DataFrame(results).set_index('index')
    
    def backtest(self, df: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Backtest the simple rule system.
        
        Args:
            df: Feature dataframe
            target: Target variable (1 = up, 0 = down)
            
        Returns:
            Dictionary of backtest results
        """
        logger.info("\n" + "="*60)
        logger.info("SIMPLE RULES BACKTEST")
        logger.info("="*60)
        
        # Generate predictions
        predictions = self.predict(df)
        
        # Align with target
        common_idx = predictions.index.intersection(target.index)
        predictions = predictions.loc[common_idx]
        target = target.loc[common_idx]
        
        # Calculate returns
        # Win: +0.8% - 0.12% cost = +0.68%
        # Lose: -0.8% - 0.12% cost = -0.92%
        base_return = 0.008
        cost = 0.0012
        
        returns = []
        for idx in common_idx:
            signal = predictions.loc[idx, 'signal']
            actual = target.loc[idx]
            
            if signal == 'NO_TRADE':
                returns.append(0.0)
            elif signal == 'LONG':
                if actual == 1:
                    returns.append(base_return - cost)
                else:
                    returns.append(-base_return - cost)
            elif signal == 'SHORT':
                if actual == 0:
                    returns.append(base_return - cost)
                else:
                    returns.append(-base_return - cost)
        
        returns = np.array(returns)
        
        # Calculate metrics
        trades = predictions[predictions['signal'] != 'NO_TRADE']
        n_trades = len(trades)
        n_long = len(trades[trades['signal'] == 'LONG'])
        n_short = len(trades[trades['signal'] == 'SHORT'])
        
        trade_returns = returns[returns != 0]
        
        if len(trade_returns) > 0:
            total_return = trade_returns.sum()
            win_rate = (trade_returns > 0).mean()
            
            # Sharpe (proper calculation)
            if len(trade_returns) > 1 and trade_returns.std() > 0:
                per_trade_sharpe = trade_returns.mean() / trade_returns.std()
                # Annualize: sqrt(trades per year)
                # Assuming we trade ~30% of bars, and there are 2190 bars/year
                trades_per_year = 2190 * (n_trades / len(df))
                sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumsum = np.cumsum(trade_returns)
            running_max = np.maximum.accumulate(cumsum)
            drawdowns = cumsum - running_max
            max_dd = drawdowns.min()
            
            # Profit factor
            gains = trade_returns[trade_returns > 0].sum()
            losses = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = gains / losses if losses > 0 else float('inf')
        else:
            total_return = 0.0
            win_rate = 0.0
            sharpe = 0.0
            max_dd = 0.0
            profit_factor = 0.0
        
        results = {
            'n_samples': len(df),
            'n_trades': n_trades,
            'n_long': n_long,
            'n_short': n_short,
            'trade_pct': n_trades / len(df),
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
        }
        
        # Log results
        logger.info(f"\nResults:")
        logger.info(f"  Samples: {results['n_samples']}")
        logger.info(f"  Trades: {results['n_trades']} ({results['trade_pct']*100:.1f}% of bars)")
        logger.info(f"    LONG:  {results['n_long']}")
        logger.info(f"    SHORT: {results['n_short']}")
        logger.info(f"  Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"  Win Rate: {results['win_rate']*100:.1f}%")
        logger.info(f"  Sharpe: {results['sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
        
        # Vote breakdown
        logger.info(f"\nVote Breakdown:")
        for rule_name in self.rules.keys():
            vote_col = f'vote_{rule_name}'
            if vote_col in predictions.columns:
                bullish = (predictions[vote_col] == 1).sum()
                bearish = (predictions[vote_col] == -1).sum()
                neutral = (predictions[vote_col] == 0).sum()
                logger.info(f"  {rule_name}: +{bullish} / -{bearish} / 0:{neutral}")
        
        return results
    
    def save(self, path: str = 'models/simple_rules.pkl'):
        """Save the fitted system."""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'percentile_cache': self.percentile_cache,
                'min_votes': self.min_votes,
            }, f)
        logger.info(f"Saved to {path}")
    
    @classmethod
    def load(cls, path: str = 'models/simple_rules.pkl') -> 'SimpleRuleSystem':
        """Load a saved system."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        system = cls(min_votes=data['min_votes'])
        system.rules = data['rules']
        system.percentile_cache = data['percentile_cache']
        return system


def main():
    """Train and backtest the simple rule system."""
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("SIMPLE RULE-BASED SYSTEM")
    logger.info("="*60)
    logger.info("\nWhy this approach:")
    logger.info("  - Single OI change rule: OOS Sharpe 8.59")
    logger.info("  - ML with 50 features: OOS Sharpe -1.74")
    logger.info("  - Simple rules WORK. ML destroys alpha.")
    logger.info("="*60)
    
    # Load data
    from ml.data_loader import load_data
    
    try:
        features, target = load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("\nRun this first:")
        logger.info("  python run_exhaustive_search.py --mode single --top-n 10")
        return
    
    logger.info(f"\nLoaded: {len(features)} samples, {len(features.columns)} features")
    
    # Split: 60% train, 40% test (same as Phase 1)
    split_idx = int(len(features) * 0.6)
    
    train_features = features.iloc[:split_idx]
    train_target = target.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    test_target = target.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_features)} samples")
    logger.info(f"Test:  {len(test_features)} samples")
    
    # Create and fit system
    system = SimpleRuleSystem(min_votes=2)
    system.fit(train_features)
    
    # Backtest on TRAIN (in-sample)
    logger.info("\n" + "="*60)
    logger.info("IN-SAMPLE RESULTS (train set)")
    logger.info("="*60)
    is_results = system.backtest(train_features, train_target)
    
    # Backtest on TEST (out-of-sample)
    logger.info("\n" + "="*60)
    logger.info("OUT-OF-SAMPLE RESULTS (test set)")
    logger.info("="*60)
    oos_results = system.backtest(test_features, test_target)
    
    # Save model
    system.save()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: Simple Rules vs ML")
    logger.info("="*60)
    logger.info(f"\n{'Metric':<20} {'Simple Rules':<15} {'ML Model':<15}")
    logger.info("-" * 50)
    logger.info(f"{'OOS Sharpe':<20} {oos_results['sharpe']:<15.2f} {-1.74:<15.2f}")
    logger.info(f"{'OOS Win Rate':<20} {oos_results['win_rate']*100:<15.1f}% {'54.0':<15}%")
    logger.info(f"{'OOS Return':<20} {oos_results['total_return']*100:<15.2f}% {'-2.8':<15}%")
    logger.info(f"{'Trades':<20} {oos_results['n_trades']:<15} {'50':<15}")
    
    if oos_results['sharpe'] > 0:
        logger.info("\n✅ Simple rules outperform ML!")
    else:
        logger.info("\n⚠️ Results below expectations - may need threshold tuning")
    
    logger.info("\n" + "="*60)
    logger.info("Model saved to models/simple_rules.pkl")
    logger.info("To use for live trading: python run_simple_scanner.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
