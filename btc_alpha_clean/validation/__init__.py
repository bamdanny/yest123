"""
Validation module for BTC Alpha Discovery.

Provides:
- OOSValidator: Out-of-sample validation for indicator testing
- DataCompletenessAuditor: Data quality checks
"""

from .oos_validator import (
    OOSValidator,
    ValidationReport,
    ValidationMetrics,
    Verdict,
    VALIDATION_THRESHOLDS,
    calculate_sharpe_ratio,
    calculate_metrics,
    run_exhaustive_search_with_oos
)

from .data_audit import DataCompletenessAuditor, audit_data_completeness

__all__ = [
    'OOSValidator',
    'ValidationReport',
    'ValidationMetrics',
    'Verdict',
    'VALIDATION_THRESHOLDS',
    'calculate_sharpe_ratio',
    'calculate_metrics',
    'run_exhaustive_search_with_oos',
    'DataCompletenessAuditor',
    'audit_data_completeness'
]
