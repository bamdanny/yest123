"""
Centralized metrics calculations.
All Sharpe ratio calculations MUST use these functions.

CRITICAL: Sharpe must be annualized by TRADE FREQUENCY, not BAR FREQUENCY.

Wrong:  sharpe = (mean/std) * sqrt(2190)  # Assumes trading every 4h bar
Right:  sharpe = (mean/std) * sqrt(trades_per_year)  # Based on actual trade count

Example:
  - 37 trades in 60 days
  - trades_per_year = (37/60) * 365 = 225
  - per_trade_sharpe = 0.18% / 0.48% = 0.376
  - annualized = 0.376 * sqrt(225) = 5.64
"""

import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# Constants
DAYS_PER_YEAR = 365
WARN_THRESHOLD = 5.0
ERROR_THRESHOLD = 10.0


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, 'pd.Series'],
    n_trades: int,
    period_days: int,
    risk_free_rate: float = 0.0,
    warn_threshold: float = WARN_THRESHOLD,
    error_threshold: float = ERROR_THRESHOLD,
    context: str = ""
) -> float:
    """
    Calculate annualized Sharpe ratio based on TRADE FREQUENCY.
    
    This is the ONLY correct way to calculate Sharpe for a trading strategy.
    DO NOT annualize by bar frequency (2190) — that's mathematically wrong.
    
    Args:
        returns: Array of per-trade returns (not per-bar!)
        n_trades: Number of trades in the period
        period_days: Number of days in the backtest period
        risk_free_rate: Annual risk-free rate (default 0 for crypto)
        warn_threshold: Log WARNING if Sharpe exceeds this
        error_threshold: Log ERROR if Sharpe exceeds this
        context: Description for logging (e.g., "walk-forward period 1")
    
    Returns:
        Annualized Sharpe ratio
    
    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> sharpe = calculate_sharpe_ratio(returns, n_trades=5, period_days=30)
    """
    # Convert pandas Series to numpy if needed
    if hasattr(returns, 'values'):
        returns = returns.values
    
    returns = np.asarray(returns)
    
    # Input validation
    if len(returns) < 2:
        logger.debug(f"{context}: Insufficient returns for Sharpe calculation ({len(returns)} returns)")
        return 0.0
    
    std_return = np.std(returns, ddof=1)  # Use sample std (ddof=1)
    if std_return == 0 or np.isnan(std_return):
        logger.debug(f"{context}: Zero or NaN volatility — cannot calculate Sharpe")
        return 0.0
    
    if n_trades <= 0 or period_days <= 0:
        logger.warning(f"{context}: Invalid inputs: n_trades={n_trades}, period_days={period_days}")
        return 0.0
    
    # Calculate per-trade Sharpe
    mean_return = np.mean(returns)
    
    # Annualize by TRADE frequency, not BAR frequency
    trades_per_year = (n_trades / period_days) * DAYS_PER_YEAR
    
    # Adjust for risk-free rate (convert annual to per-trade)
    if trades_per_year > 0 and risk_free_rate > 0:
        risk_free_per_trade = risk_free_rate / trades_per_year
        excess_return = mean_return - risk_free_per_trade
    else:
        excess_return = mean_return
    
    per_trade_sharpe = excess_return / std_return
    
    # Annualize
    annualized_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
    
    # Sanity checks
    if abs(annualized_sharpe) > error_threshold:
        logger.error(
            f"SHARPE SANITY CHECK FAILED {context}: {annualized_sharpe:.2f} > {error_threshold}\n"
            f"  mean_return={mean_return:.6f}, std_return={std_return:.6f}\n"
            f"  n_trades={n_trades}, period_days={period_days}\n"
            f"  trades_per_year={trades_per_year:.1f}\n"
            f"  This likely indicates a calculation error upstream."
        )
    elif abs(annualized_sharpe) > warn_threshold:
        logger.warning(
            f"SHARPE WARNING {context}: {annualized_sharpe:.2f} > {warn_threshold} — verify this is correct"
        )
    
    return round(annualized_sharpe, 4)


def calculate_sharpe_from_equity(
    equity_curve: Union[np.ndarray, 'pd.Series'],
    period_days: int,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe from an equity curve (daily or bar-level).
    Use this when you have an equity curve, not individual trade returns.
    
    For equity curves, we DO use observation frequency since we're
    measuring day-to-day or bar-to-bar volatility.
    
    Args:
        equity_curve: Array of equity values over time
        period_days: Number of days in the period
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Annualized Sharpe ratio
    """
    if hasattr(equity_curve, 'values'):
        equity_curve = equity_curve.values
    
    equity_curve = np.asarray(equity_curve)
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate returns from equity curve
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    
    # For equity curve, annualize by observation frequency
    observations_per_year = (len(returns) / period_days) * DAYS_PER_YEAR
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    if observations_per_year > 0 and risk_free_rate > 0:
        risk_free_per_obs = risk_free_rate / observations_per_year
        excess_return = mean_return - risk_free_per_obs
    else:
        excess_return = mean_return
    
    sharpe = (excess_return / std_return) * np.sqrt(observations_per_year)
    
    return round(sharpe, 4)


def calculate_sharpe_simple(
    returns: Union[np.ndarray, 'pd.Series'],
    period_days: int = 90,
    context: str = ""
) -> float:
    """
    Simplified Sharpe calculation that infers trade count from returns length.
    
    Use this when you only have the returns array and period length.
    Assumes each return represents one trade.
    
    Args:
        returns: Array of per-trade returns
        period_days: Number of days in the backtest period (default 90)
        context: Description for logging
    
    Returns:
        Annualized Sharpe ratio
    """
    if hasattr(returns, 'values'):
        returns = returns.values
    
    returns = np.asarray(returns)
    n_trades = len(returns)
    
    return calculate_sharpe_ratio(
        returns=returns,
        n_trades=n_trades,
        period_days=period_days,
        context=context
    )
