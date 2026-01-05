"""
Data Completeness Auditor
=========================

Validates that all required and high-value data sources are present
before running the discovery pipeline.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataCompletenessAuditor:
    """
    Audits data completeness before discovery.
    
    Required sources must pass for valid results.
    High-value sources improve accuracy but aren't mandatory.
    """
    
    # Required data sources (must have)
    REQUIRED_SOURCES = {
        'price': 2000,           # Price OHLCV
        'oi_history': 300,       # Open interest history
        'funding_history': 300,  # Funding rate history
        'ls_history': 300,       # Long/short ratio
        'liquidation_history': 100,  # Liquidation data
    }
    
    # High-value optional sources (nice to have)
    HIGH_VALUE_SOURCES = {
        'top_ls_history': 300,       # Top trader L/S
        'top_position_history': 300,  # Top trader positions
        'taker_history': 300,         # Taker buy/sell
        'funding_oi_weighted': 300,   # OI-weighted funding
        'funding_vol_weighted': 300,  # Vol-weighted funding
        'oi_aggregated': 300,         # Aggregated OI
        'liquidation_aggregated': 100,  # Aggregated liquidations
        'taker_aggregated': 300,      # Aggregated taker
        'yield_curve': 200,           # FRED yield curve
        'fin_conditions': 200,        # Financial conditions
        'fear_greed': 200,            # Fear & Greed index
        'etf_flows': 30,              # ETF flows
    }
    
    # Snapshot sources (just need to exist)
    SNAPSHOT_SOURCES = [
        'futures_basis',
        'coinbase_premium',
        'options_max_pain',
        'options_oi',
        'liq_heatmap_model1',
        'liq_heatmap_model2',
        'liq_heatmap_model3',
    ]
    
    def __init__(self):
        self.results = {
            'required': {},
            'high_value': {},
            'snapshots': {},
            'warnings': [],
            'errors': [],
        }
    
    def audit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete data audit.
        
        Args:
            data: Dictionary of all fetched data
            
        Returns:
            Audit results with pass/fail status
        """
        logger.info("=" * 60)
        logger.info("DATA COMPLETENESS AUDIT")
        logger.info("=" * 60)
        
        # Reset results
        self.results = {
            'required': {},
            'high_value': {},
            'snapshots': {},
            'warnings': [],
            'errors': [],
        }
        
        # Check required sources
        required_pass = True
        for source, min_rows in self.REQUIRED_SOURCES.items():
            result = self._check_source(data, source, min_rows, required=True)
            self.results['required'][source] = result
            if not result['pass']:
                required_pass = False
        
        # Check high-value sources
        for source, min_rows in self.HIGH_VALUE_SOURCES.items():
            result = self._check_source(data, source, min_rows, required=False)
            self.results['high_value'][source] = result
        
        # Check snapshot sources
        for source in self.SNAPSHOT_SOURCES:
            exists = self._find_source(data, source) is not None
            self.results['snapshots'][source] = {'exists': exists}
        
        # Calculate scores
        required_passed = sum(1 for r in self.results['required'].values() if r['pass'])
        required_total = len(self.REQUIRED_SOURCES)
        required_score = required_passed / required_total if required_total > 0 else 0
        
        high_value_passed = sum(1 for r in self.results['high_value'].values() if r['pass'])
        high_value_total = len(self.HIGH_VALUE_SOURCES)
        high_value_score = high_value_passed / high_value_total if high_value_total > 0 else 0
        
        snapshot_passed = sum(1 for r in self.results['snapshots'].values() if r['exists'])
        snapshot_total = len(self.SNAPSHOT_SOURCES)
        snapshot_score = snapshot_passed / snapshot_total if snapshot_total > 0 else 0
        
        # Overall score (weighted)
        overall_score = (required_score * 0.5) + (high_value_score * 0.35) + (snapshot_score * 0.15)
        
        # Log results
        self._log_results(required_score, high_value_score, snapshot_score, overall_score)
        
        return {
            'pass': required_pass,
            'required_score': required_score,
            'high_value_score': high_value_score,
            'snapshot_score': snapshot_score,
            'overall_score': overall_score,
            'utilization_pct': overall_score * 100,
            'warnings': self.results['warnings'],
            'errors': self.results['errors'],
            'details': self.results,
        }
    
    def _check_source(self, data: Dict, source: str, min_rows: int, required: bool) -> Dict:
        """Check a single data source."""
        value = self._find_source(data, source)
        
        if value is None:
            msg = f"{source}: NOT FOUND"
            if required:
                self.results['errors'].append(msg)
            else:
                self.results['warnings'].append(msg)
            return {'pass': False, 'rows': 0, 'required': min_rows}
        
        # Check row count
        if isinstance(value, pd.DataFrame):
            rows = len(value)
        elif isinstance(value, dict):
            rows = 1  # Snapshot data
        elif isinstance(value, list):
            rows = len(value)
        else:
            rows = 1
        
        passed = rows >= min_rows
        
        if not passed:
            msg = f"{source}: {rows} rows (need {min_rows}+)"
            if required:
                self.results['errors'].append(msg)
            else:
                self.results['warnings'].append(msg)
        
        return {'pass': passed, 'rows': rows, 'required': min_rows}
    
    def _find_source(self, data: Dict, source: str) -> Any:
        """Find a source in potentially nested data structure."""
        # Direct lookup
        if source in data:
            return data[source]
        
        # Check in 'derivatives' sub-dict
        if 'derivatives' in data and isinstance(data['derivatives'], dict):
            if source in data['derivatives']:
                return data['derivatives'][source]
            # Also check with 'cg_' prefix
            cg_source = f"cg_{source}"
            if cg_source in data['derivatives']:
                return data['derivatives'][cg_source]
        
        # Check with 'cg_' prefix at top level
        cg_source = f"cg_{source}"
        if cg_source in data:
            return data[cg_source]
        
        # Check in 'macro' sub-dict
        if 'macro' in data and isinstance(data['macro'], dict):
            if source in data['macro']:
                return data['macro'][source]
        
        # Check in 'sentiment' sub-dict
        if 'sentiment' in data and isinstance(data['sentiment'], dict):
            if source in data['sentiment']:
                return data['sentiment'][source]
        
        return None
    
    def _log_results(self, required_score: float, high_value_score: float, 
                     snapshot_score: float, overall_score: float):
        """Log audit results."""
        logger.info("\nREQUIRED DATA SOURCES:")
        logger.info("-" * 40)
        for source, result in self.results['required'].items():
            status = "[OK]" if result['pass'] else "[FAIL]"
            logger.info(f"  {status} {source}: {result['rows']}/{result['required']} rows")
        
        logger.info(f"\nRequired Score: {required_score*100:.1f}%")
        
        logger.info("\nHIGH-VALUE DATA SOURCES:")
        logger.info("-" * 40)
        for source, result in self.results['high_value'].items():
            status = "[OK]" if result['pass'] else "[--]"
            logger.info(f"  {status} {source}: {result['rows']}/{result['required']} rows")
        
        logger.info(f"\nHigh-Value Score: {high_value_score*100:.1f}%")
        
        logger.info("\nSNAPSHOT DATA SOURCES:")
        logger.info("-" * 40)
        for source, result in self.results['snapshots'].items():
            status = "[OK]" if result['exists'] else "[--]"
            logger.info(f"  {status} {source}")
        
        logger.info(f"\nSnapshot Score: {snapshot_score*100:.1f}%")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"OVERALL DATA UTILIZATION: {overall_score*100:.1f}%")
        logger.info("=" * 60)
        
        if overall_score < 0.5:
            logger.warning("[!] LOW DATA UTILIZATION - Results may be incomplete!")
        elif overall_score < 0.8:
            logger.info("[*] ACCEPTABLE - Some high-value data missing")
        else:
            logger.info("[OK] EXCELLENT - Using most available data")
        
        if self.results['errors']:
            logger.error("\nCRITICAL ERRORS:")
            for error in self.results['errors']:
                logger.error(f"  [X] {error}")
        
        if self.results['warnings']:
            logger.warning("\nWARNINGS:")
            for warning in self.results['warnings'][:10]:  # Limit to 10
                logger.warning(f"  [!] {warning}")
            if len(self.results['warnings']) > 10:
                logger.warning(f"  ... and {len(self.results['warnings']) - 10} more warnings")


def audit_data_completeness(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to audit data completeness."""
    auditor = DataCompletenessAuditor()
    return auditor.audit(data)
