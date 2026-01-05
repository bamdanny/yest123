"""
Correct Backtester

Fixes all the math bugs from the original implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    n_bars: int
    n_trades: int
    n_long: int
    n_short: int
    n_wins: int
    n_losses: int
    win_rate: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    signals: np.ndarray
    returns: np.ndarray
    equity_curve: np.ndarray


class Backtester:
    """
    Correct backtesting implementation.
    
    Key differences from broken version:
    1. Compound returns (multiplicative)
    2. Proper Sharpe annualization
    3. Correct win rate calculation
    4. Signals shifted to avoid lookahead
    """
    
    def __init__(self, 
                 bars_per_year: int = 2190,  # 4h bars
                 commission: float = 0.0,
                 slippage: float = 0.0):
        self.bars_per_year = bars_per_year
        self.commission = commission
        self.slippage = slippage
    
    def run(self, 
            signals: np.ndarray,
            prices: np.ndarray,
            returns: np.ndarray = None) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            signals: Position signals (1=long, -1=short, 0=flat)
            prices: Price series
            returns: Pre-computed returns (optional)
            
        Returns:
            BacktestResult with all metrics
        """
        n = len(signals)
        
        if returns is None:
            returns = np.zeros(n)
            returns[1:] = np.diff(prices) / prices[:-1]
        
        # Shift signals by 1 to avoid lookahead
        # Signal at bar t affects return at bar t+1
        shifted_signals = np.zeros(n)
        shifted_signals[1:] = signals[:-1]
        
        # Calculate trade returns
        trade_returns = shifted_signals * returns
        
        # Apply costs
        entries = np.abs(np.diff(np.concatenate([[0], shifted_signals]))) > 0
        trade_returns[entries] -= (self.commission + self.slippage)
        
        # Calculate metrics
        n_long = int(np.sum(shifted_signals == 1))
        n_short = int(np.sum(shifted_signals == -1))
        n_trades = n_long + n_short
        
        # Win/loss analysis (only on actual trades)
        trade_mask = shifted_signals != 0
        trade_only_returns = trade_returns[trade_mask]
        
        n_wins = int(np.sum(trade_only_returns > 0))
        n_losses = int(np.sum(trade_only_returns < 0))
        win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
        
        # Average win/loss
        wins = trade_only_returns[trade_only_returns > 0]
        losses = trade_only_returns[trade_only_returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
        
        # Total return (compound)
        total_return = float(np.prod(1 + trade_returns) - 1)
        
        # Sharpe ratio
        if len(trade_returns) > 1 and np.std(trade_returns) > 1e-10:
            sharpe = float((np.mean(trade_returns) / np.std(trade_returns)) * 
                          np.sqrt(self.bars_per_year))
        else:
            sharpe = 0.0
        
        # Equity curve and max drawdown
        equity_curve = np.cumprod(1 + trade_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))
        
        # Profit factor
        gross_profit = np.sum(trade_only_returns[trade_only_returns > 0])
        gross_loss = abs(np.sum(trade_only_returns[trade_only_returns < 0]))
        if gross_loss > 1e-10:
            profit_factor = float(gross_profit / gross_loss)
        else:
            profit_factor = np.inf if gross_profit > 0 else 0
        
        return BacktestResult(
            n_bars=n,
            n_trades=n_trades,
            n_long=n_long,
            n_short=n_short,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=win_rate,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            signals=shifted_signals,
            returns=trade_returns,
            equity_curve=equity_curve
        )
    
    def validate_result(self, result: BacktestResult) -> Tuple[bool, List[str]]:
        """
        Validate backtest results are reasonable.
        
        Returns:
            (is_valid, list of errors)
        """
        errors = []
        
        # Sharpe sanity
        if abs(result.sharpe) > 50:
            errors.append(f"Sharpe {result.sharpe:.2f} is unrealistic")
        
        # Win rate sanity
        if result.win_rate < 0 or result.win_rate > 1:
            errors.append(f"Win rate {result.win_rate:.1%} outside [0,1]")
        
        if result.win_rate == 0 and result.n_trades > 20:
            errors.append(f"Win rate 0% with {result.n_trades} trades is suspicious")
        
        # Return sanity
        if result.total_return < -1:
            errors.append(f"Return {result.total_return:.1%} < -100% is impossible")
        
        # Trade balance
        if result.n_trades > 50:
            if result.n_long == 0:
                errors.append("No long trades generated")
            if result.n_short == 0:
                errors.append("No short trades generated")
        
        return len(errors) == 0, errors
    
    def print_result(self, result: BacktestResult, name: str = "Backtest"):
        """Print formatted results."""
        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"{name} Results")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Bars: {result.n_bars}")
        logger.info(f"  Trades: {result.n_trades} (L:{result.n_long}, S:{result.n_short})")
        logger.info(f"  Wins/Losses: {result.n_wins}W / {result.n_losses}L")
        logger.info(f"  Win Rate: {result.win_rate:.1%}")
        logger.info(f"  Total Return: {result.total_return:.1%}")
        logger.info(f"  Sharpe Ratio: {result.sharpe:.2f}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.1%}")
        logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  Avg Win: {result.avg_win:.2%}")
        logger.info(f"  Avg Loss: {result.avg_loss:.2%}")
        
        # Validation
        valid, errors = self.validate_result(result)
        if not valid:
            logger.warning("  Validation FAILED:")
            for e in errors:
                logger.warning(f"    - {e}")
        else:
            logger.info(f"  Validation: PASSED")


def compare_results(results: Dict[str, BacktestResult]):
    """Print comparison table."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Approach':<30} {'Sharpe':>10} {'Win Rate':>10} {'Return':>10} {'Trades':>10}")
    logger.info("-" * 80)
    
    for name, result in results.items():
        logger.info(
            f"{name:<30} {result.sharpe:>10.2f} {result.win_rate:>10.1%} "
            f"{result.total_return:>10.1%} {result.n_trades:>10}"
        )
