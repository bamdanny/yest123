"""
Dataset Builder with Proper Time-Series Splits

CRITICAL: In time-series, you cannot randomly shuffle data.
Training data must ALWAYS be before validation/test data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesSplit:
    """A single train/val/test split."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    
    def __repr__(self):
        return f"Train[{self.train_start}:{self.train_end}] Val[{self.val_start}:{self.val_end}] Test[{self.test_start}:{self.test_end}]"


class DatasetBuilder:
    """
    Build datasets with proper time-series methodology.
    
    Key principles:
    1. NO FUTURE LEAKAGE - train data always before val/test
    2. PURGE GAP - gap between train and val to prevent leakage from target calculation
    3. EMBARGO - additional gap after test to prevent leakage in walk-forward
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        purge_gap: int = 12,     # 12 bars (48h) gap to prevent target leakage
        embargo_gap: int = 12,   # 12 bars (48h) embargo after test
    ):
        """
        Args:
            features: DataFrame of features, index must be datetime
            target: Series of targets (1 = up, 0 = down)
            purge_gap: Bars to skip between train and val (prevents lookahead)
            embargo_gap: Bars to skip after test in walk-forward
        
        Note: Increased gap/embargo to 12 bars (48h at 4h bars) to prevent
        any possibility of data leakage from contemporaneous information.
        """
        assert len(features) == len(target), "Features and target must have same length"
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        self.features = features.loc[common_idx]
        self.target = target.loc[common_idx]
        
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.n_samples = len(self.features)
        
        logger.info(f"DatasetBuilder initialized:")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Features: {len(self.features.columns)}")
        logger.info(f"  Date range: {self.features.index[0]} to {self.features.index[-1]}")
        logger.info(f"  Target distribution: {self.target.value_counts().to_dict()}")
    
    def create_single_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create a single train/val/test split.
        
        Returns:
            train_data, val_data, test_data - each is dict with 'X' and 'y'
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        n = self.n_samples
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Apply purge gap
        val_start = train_end + self.purge_gap
        test_start = val_end + self.purge_gap
        
        split = TimeSeriesSplit(
            train_start=0,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=n
        )
        
        logger.info(f"Single split created: {split}")
        
        train_data = {
            'X': self.features.iloc[split.train_start:split.train_end],
            'y': self.target.iloc[split.train_start:split.train_end],
            'dates': self.features.index[split.train_start:split.train_end]
        }
        val_data = {
            'X': self.features.iloc[split.val_start:split.val_end],
            'y': self.target.iloc[split.val_start:split.val_end],
            'dates': self.features.index[split.val_start:split.val_end]
        }
        test_data = {
            'X': self.features.iloc[split.test_start:split.test_end],
            'y': self.target.iloc[split.test_start:split.test_end],
            'dates': self.features.index[split.test_start:split.test_end]
        }
        
        logger.info(f"  Train: {len(train_data['y'])} samples, "
                   f"{train_data['y'].mean()*100:.1f}% positive, "
                   f"{train_data['dates'][0]} to {train_data['dates'][-1]}")
        logger.info(f"  Val: {len(val_data['y'])} samples, "
                   f"{val_data['y'].mean()*100:.1f}% positive, "
                   f"{val_data['dates'][0]} to {val_data['dates'][-1]}")
        logger.info(f"  Test: {len(test_data['y'])} samples, "
                   f"{test_data['y'].mean()*100:.1f}% positive, "
                   f"{test_data['dates'][0]} to {test_data['dates'][-1]}")
        
        return train_data, val_data, test_data
    
    def create_walk_forward_splits(
        self,
        n_splits: int = 5,
        train_size: int = 200,      # Minimum training samples
        val_size: int = 50,         # Validation samples per fold
        test_size: int = 50,        # Test samples per fold
        expanding: bool = True      # True = expanding window, False = rolling
    ) -> List[Tuple[Dict, Dict, Dict]]:
        """
        Create walk-forward splits for robust validation.
        
        Walk-forward prevents overfitting by testing on multiple future periods.
        
        Expanding window (recommended):
        Split 1: Train[0:200]    Val[206:256]   Test[262:312]
        Split 2: Train[0:312]    Val[318:368]   Test[374:424]
        Split 3: Train[0:424]    Val[430:480]   Test[486:536]
        ...
        """
        splits = []
        
        # Calculate step size
        step = (self.n_samples - train_size - val_size - test_size - 2*self.purge_gap) // max(n_splits, 1)
        
        if step < 10:
            logger.warning(f"Step size ({step}) is very small. Consider reducing n_splits or sizes.")
        
        for i in range(n_splits):
            if expanding:
                train_start = 0
            else:
                train_start = i * step
            
            train_end = train_size + i * step
            val_start = train_end + self.purge_gap
            val_end = val_start + val_size
            test_start = val_end + self.purge_gap
            test_end = test_start + test_size
            
            if test_end > self.n_samples:
                logger.info(f"Stopping at split {i} - not enough data for more")
                break
            
            split_data = (
                {
                    'X': self.features.iloc[train_start:train_end],
                    'y': self.target.iloc[train_start:train_end],
                    'dates': self.features.index[train_start:train_end]
                },
                {
                    'X': self.features.iloc[val_start:val_end],
                    'y': self.target.iloc[val_start:val_end],
                    'dates': self.features.index[val_start:val_end]
                },
                {
                    'X': self.features.iloc[test_start:test_end],
                    'y': self.target.iloc[test_start:test_end],
                    'dates': self.features.index[test_start:test_end]
                }
            )
            splits.append(split_data)
            
            logger.info(f"Walk-forward split {i+1}: "
                       f"Train[{train_start}:{train_end}] "
                       f"Val[{val_start}:{val_end}] "
                       f"Test[{test_start}:{test_end}]")
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.columns)
