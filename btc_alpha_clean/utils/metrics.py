"""
Correct metrics calculations for trading backtests.

Fixes the bugs in the original implementation:
1. Sharpe uses proper annualization
2. Returns are compounded (multiplicative)
3. Win rate counts actual wins vs losses
4. Max drawdown uses running peak
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


# Constants
BARS_PER_DAY = 6  # 4-hour bars
BARS_PER_YEAR = 365 * BARS_PER_DAY  # 2190


def calculate_sharpe(returns: np.ndarray, 
                     annualize: bool = True,
                     risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio correctly.
    
    Args:
        returns: Array of per-bar returns
        annualize: Whether to annualize (multiply by sqrt of periods)
        risk_free_rate: Annual risk-free rate (default 0)
        
    Returns:
        Sharpe ratio (annualized if requested)
    """
    if len(returns) < 2:
        return 0.0
    
    # Remove NaN
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns) - (risk_free_rate / BARS_PER_YEAR)
    std_return = np.std(returns, ddof=1)  # Sample std
    
    if std_return < 1e-10:  # Avoid division by zero
        return 0.0
    
    sharpe = mean_return / std_return
    
    if annualize:
        sharpe *= np.sqrt(BARS_PER_YEAR)
    
    return float(sharpe)


def calculate_returns(signals: np.ndarray, 
                      price_returns: np.ndarray,
                      shift: int = 1) -> np.ndarray:
    """
    Calculate trading returns correctly.
    
    Args:
        signals: Array of positions (1=long, -1=short, 0=flat)
        price_returns: Array of price returns
        shift: Bars to shift signal (1 = trade on next bar)
        
    Returns:
        Array of trading returns per bar
    """
    if len(signals) != len(price_returns):
        raise ValueError(f"Signal length {len(signals)} != returns length {len(price_returns)}")
    
    # Shift signals to avoid lookahead
    shifted_signals = np.zeros(len(signals))
    shifted_signals[shift:] = signals[:-shift]
    
    # Trading return = position * price return
    trade_returns = shifted_signals * price_returns
    
    return trade_returns


def calculate_cumulative_return(returns: np.ndarray) -> float:
    """
    Calculate cumulative return correctly (compound, not sum).
    
    Args:
        returns: Array of per-bar returns
        
    Returns:
        Total return as decimal (0.50 = 50% gain)
    """
    # Filter NaN
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Compound returns: (1+r1) * (1+r2) * ... - 1
    cumulative = np.prod(1 + returns) - 1
    
    return float(cumulative)


def calculate_win_rate(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate win rate correctly.
    
    Args:
        returns: Array of trade returns
        
    Returns:
        (win_rate, n_wins, n_losses)
    """
    # Only count non-zero returns (actual trades)
    nonzero = returns[returns != 0]
    nonzero = nonzero[~np.isnan(nonzero)]
    
    if len(nonzero) == 0:
        return 0.0, 0, 0
    
    n_wins = int(np.sum(nonzero > 0))
    n_losses = int(np.sum(nonzero < 0))
    
    total = n_wins + n_losses
    win_rate = n_wins / total if total > 0 else 0.0
    
    return float(win_rate), n_wins, n_losses


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown correctly.
    
    Args:
        returns: Array of per-bar returns
        
    Returns:
        Maximum drawdown as negative decimal (-0.20 = 20% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    # Cumulative wealth
    cumulative = np.cumprod(1 + returns)
    
    # Running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Drawdown at each point
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown (most negative)
    max_dd = np.min(drawdown)
    
    return float(max_dd)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: Array of trade returns
        
    Returns:
        Profit factor (>1 is profitable)
    """
    nonzero = returns[returns != 0]
    nonzero = nonzero[~np.isnan(nonzero)]
    
    if len(nonzero) == 0:
        return 0.0
    
    gross_profit = np.sum(nonzero[nonzero > 0])
    gross_loss = abs(np.sum(nonzero[nonzero < 0]))
    
    if gross_loss < 1e-10:
        return np.inf if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def calculate_all_metrics(signals: np.ndarray,
                          price_returns: np.ndarray,
                          prices: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate all trading metrics.
    
    Args:
        signals: Position signals (1, -1, 0)
        price_returns: Per-bar price returns
        prices: Raw prices (optional, for additional metrics)
        
    Returns:
        Dictionary of all metrics
    """
    # Calculate trade returns
    trade_returns = calculate_returns(signals, price_returns)
    
    # Basic counts
    n_long = int(np.sum(signals == 1))
    n_short = int(np.sum(signals == -1))
    n_flat = int(np.sum(signals == 0))
    
    # Metrics
    total_return = calculate_cumulative_return(trade_returns)
    sharpe = calculate_sharpe(trade_returns)
    win_rate, n_wins, n_losses = calculate_win_rate(trade_returns)
    max_dd = calculate_max_drawdown(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)
    
    # Average return per trade
    nonzero_returns = trade_returns[trade_returns != 0]
    avg_return = float(np.mean(nonzero_returns)) if len(nonzero_returns) > 0 else 0.0
    
    return {
        'n_bars': len(signals),
        'n_trades': n_long + n_short,
        'n_long': n_long,
        'n_short': n_short,
        'n_flat': n_flat,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'trade_returns': trade_returns  # Include for debugging
    }


def validate_metrics(metrics: Dict) -> Tuple[bool, list]:
    """
    Validate that metrics are reasonable (sanity checks).
    
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Sharpe should be reasonable
    if abs(metrics.get('sharpe', 0)) > 50:
        errors.append(f"Sharpe {metrics['sharpe']:.2f} is unrealistic (>50 or <-50)")
    
    # Win rate should be between 0 and 1
    wr = metrics.get('win_rate', 0)
    if wr < 0 or wr > 1:
        errors.append(f"Win rate {wr:.2%} is outside [0, 1]")
    
    # Win rate of exactly 0% with trades is suspicious
    if wr == 0 and metrics.get('n_trades', 0) > 10:
        errors.append(f"Win rate 0% with {metrics['n_trades']} trades is suspicious")
    
    # Total return should not exceed -100% without leverage
    tr = metrics.get('total_return', 0)
    if tr < -1.0:
        errors.append(f"Return {tr:.1%} < -100% is impossible without leverage")
    
    # Should have both long and short signals (usually)
    if metrics.get('n_trades', 0) > 50:
        if metrics.get('n_long', 0) == 0:
            errors.append("No long trades - possible bug in signal generation")
        if metrics.get('n_short', 0) == 0:
            errors.append("No short trades - possible bug in signal generation")
    
    return len(errors) == 0, errors


# Test
if __name__ == "__main__":
    # Simple test case
    np.random.seed(42)
    
    signals = np.array([0, 1, 1, -1, -1, 0, 1, -1, 1, 0])
    price_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.02, 0.005, -0.01, 0.03, -0.005, 0.01])
    
    metrics = calculate_all_metrics(signals, price_returns)
    
    print("Test Results:")
    print(f"  Trades: {metrics['n_trades']} (L:{metrics['n_long']}, S:{metrics['n_short']})")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    valid, errors = validate_metrics(metrics)
    print(f"\nValidation: {'PASSED' if valid else 'FAILED'}")
    for e in errors:
        print(f"  - {e}")
