"""ML module with CORRECTED implementations."""

from .anchored_ensemble import AnchoredEnsemble
from .backtester import Backtester, BacktestResult

# Re-export existing components that work
from .data_loader import DataLoader, load_data
from .feature_selector import FeatureSelector

__all__ = [
    'AnchoredEnsemble',
    'Backtester', 
    'BacktestResult',
    'DataLoader',
    'load_data',
    'FeatureSelector'
]
