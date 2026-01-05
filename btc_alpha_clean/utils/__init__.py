"""Utility functions with CORRECTED implementations."""

from .metrics import (
    calculate_sharpe,
    calculate_returns,
    calculate_cumulative_return,
    calculate_win_rate,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_all_metrics,
    validate_metrics,
    BARS_PER_YEAR,
    BARS_PER_DAY
)

__all__ = [
    'calculate_sharpe',
    'calculate_returns', 
    'calculate_cumulative_return',
    'calculate_win_rate',
    'calculate_max_drawdown',
    'calculate_profit_factor',
    'calculate_all_metrics',
    'validate_metrics',
    'BARS_PER_YEAR',
    'BARS_PER_DAY'
]
