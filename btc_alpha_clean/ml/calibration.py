"""
Probability Calibration

Raw model probabilities are often not well-calibrated.
A model saying "70% confident" should be right 70% of the time.

We use Platt scaling (logistic regression on probabilities) to calibrate.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate model probabilities so they reflect true confidence.
    """
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation set.
        
        Args:
            probabilities: Raw model probabilities (for class 1)
            y_true: True labels
        """
        logger.info("Fitting probability calibrator...")
        
        # Ensure proper shape
        if len(probabilities.shape) == 1:
            probs = probabilities.reshape(-1, 1)
        else:
            probs = probabilities.reshape(-1, 1)
        
        # Use logistic regression (Platt scaling)
        self.calibrator = LogisticRegression(max_iter=1000)
        self.calibrator.fit(probs, y_true)
        
        self.is_fitted = True
        
        # Report calibration quality
        calibrated = self.transform(probabilities)
        bins = np.linspace(0, 1, 11)
        
        logger.info("Calibration check (predicted vs actual):")
        for i in range(len(bins) - 1):
            mask = (calibrated >= bins[i]) & (calibrated < bins[i+1])
            if mask.sum() > 5:  # Only report bins with enough samples
                actual = y_true[mask].mean()
                predicted = calibrated[mask].mean()
                logger.info(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: "
                           f"predicted={predicted:.2f}, actual={actual:.2f}, n={mask.sum()}")
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        probs = probabilities.reshape(-1, 1)
        return self.calibrator.predict_proba(probs)[:, 1]
    
    def fit_transform(self, probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probabilities, y_true)
        return self.transform(probabilities)


def get_confidence_level(probability: float) -> str:
    """
    Convert probability to confidence level.
    
    Used for position sizing and filtering.
    
    Args:
        probability: Calibrated probability (0-1)
        
    Returns:
        Confidence level string
    """
    edge = abs(probability - 0.5)
    
    if edge >= 0.20:  # >70% or <30%
        return "HIGH"
    elif edge >= 0.10:  # >60% or <40%
        return "MEDIUM"
    elif edge >= 0.05:  # >55% or <45%
        return "LOW"
    else:
        return "NEUTRAL"  # 45-55% = no signal


def calculate_kelly_fraction(probability: float, win_loss_ratio: float = 2.0) -> float:
    """
    Calculate Kelly criterion position size.
    
    Kelly = (p * b - q) / b
    Where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (how much you win vs lose)
    
    Args:
        probability: Probability of winning
        win_loss_ratio: Average win / average loss
        
    Returns:
        Fraction of capital to risk (0-1)
    """
    if probability <= 0.5:
        # No edge or negative edge
        return 0.0
    
    p = probability
    q = 1 - p
    b = win_loss_ratio
    
    kelly = (p * b - q) / b
    
    # Use half-Kelly for safety
    half_kelly = kelly / 2
    
    # Clamp to reasonable range
    return max(0, min(0.25, half_kelly))  # Max 25% of capital
