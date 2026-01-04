"""
Data Validation Gate
====================

CRITICAL: Validates data BEFORE any analysis.
If minimum requirements not met, ABORT with detailed error report.

This prevents garbage-in-garbage-out analysis.
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataRequirement:
    """Data requirement specification"""
    name: str
    min_rows: int
    critical: bool  # If True, abort if not met
    description: str


# Minimum data requirements
# For 4h timeframe: 365 days = ~2190 candles
DATA_REQUIREMENTS = [
    # Price data - CRITICAL (365 days @ 4h = 2190 candles)
    DataRequirement("price_4h", 2000, True, "4-hour price data for 365 days"),
    
    # CoinGlass derivatives data - Preferred source (may not work on hobbyist)
    DataRequirement("funding_history", 300, False, "Funding rate history"),
    DataRequirement("oi_history", 300, False, "Open interest history"),
    DataRequirement("ls_history", 300, False, "Long/short ratio history"),
    DataRequirement("liquidation_history", 100, False, "Liquidation history (THE EDGE)"),
    DataRequirement("taker_history", 100, False, "Taker buy/sell history"),
    
    # Binance derivatives removed - CoinGlass provides same data aggregated across all exchanges
    # Using CoinGlass for all derivatives data instead
    
    # Macro data - Important but not critical
    DataRequirement("yield_curve", 200, False, "Treasury yield curve data"),
    DataRequirement("rates", 200, False, "Interest rates data"),
    
    # Sentiment data
    DataRequirement("fear_greed", 200, False, "Fear & Greed index"),
    DataRequirement("historical_vol", 200, False, "Historical volatility from Deribit"),
]


class DataValidationGate:
    """
    Validates fetched data meets minimum requirements.
    
    If critical requirements not met, raises DataValidationError.
    """
    
    def __init__(self, requirements: List[DataRequirement] = None):
        self.requirements = requirements or DATA_REQUIREMENTS
        
    def validate(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all data against requirements.
        
        Args:
            all_data: Dict containing all fetched data
            
        Returns:
            Validation report dict
            
        Raises:
            DataValidationError if critical requirements not met
        """
        report = {
            "passed": True,
            "critical_failures": [],
            "warnings": [],
            "row_counts": {},
            "coverage": {}
        }
        
        # Flatten nested data structure
        flat_data = self._flatten_data(all_data)
        
        for req in self.requirements:
            df = flat_data.get(req.name)
            
            if df is None:
                rows = 0
            elif isinstance(df, pd.DataFrame):
                rows = len(df)
            else:
                rows = 0  # Not a DataFrame
            
            report["row_counts"][req.name] = rows
            report["coverage"][req.name] = rows / req.min_rows if req.min_rows > 0 else 1.0
            
            if rows < req.min_rows:
                msg = f"{req.name}: {rows} rows (need {req.min_rows}+) - {req.description}"
                
                if req.critical:
                    report["critical_failures"].append(msg)
                    report["passed"] = False
                else:
                    report["warnings"].append(msg)
        
        return report
    
    def _flatten_data(self, all_data: Dict) -> Dict[str, pd.DataFrame]:
        """Flatten nested data structure for validation"""
        flat = {}
        
        # Price data (check both 4h and 1h)
        if "price" in all_data and all_data["price"] is not None:
            flat["price_4h"] = all_data["price"]  # Default to 4h
        
        # Derivatives data (may be nested)
        derivatives = all_data.get("derivatives", {})
        if isinstance(derivatives, dict):
            for key, value in derivatives.items():
                if isinstance(value, pd.DataFrame):
                    # Map various names to standard names
                    if "funding" in key.lower():
                        if "history" in key.lower() or len(value) > 100:
                            flat["funding_history"] = value
                    elif "oi" in key.lower() or "open_interest" in key.lower():
                        if "history" in key.lower() or "aggregated" in key.lower() or len(value) > 100:
                            flat["oi_history"] = value
                    elif "ls" in key.lower() or "long_short" in key.lower():
                        if "history" in key.lower() or len(value) > 100:
                            flat["ls_history"] = value
                    elif "liquidation" in key.lower():
                        flat["liquidation_history"] = value
                    elif "taker" in key.lower():
                        if "history" in key.lower() or len(value) > 100:
                            flat["taker_history"] = value
                    else:
                        # Keep original name
                        flat[key] = value
        
        # Macro data
        macro = all_data.get("macro", {})
        if isinstance(macro, dict):
            for key, value in macro.items():
                if isinstance(value, pd.DataFrame):
                    if "yield" in key.lower():
                        flat["yield_curve"] = value
                    elif "rate" in key.lower():
                        flat["rates"] = value
                    else:
                        flat[key] = value
        
        # Sentiment data
        sentiment = all_data.get("sentiment", {})
        if isinstance(sentiment, dict):
            for key, value in sentiment.items():
                if isinstance(value, pd.DataFrame):
                    if "fear" in key.lower() or "greed" in key.lower():
                        flat["fear_greed"] = value
                    elif "vol" in key.lower():
                        flat["historical_vol"] = value
                    else:
                        flat[key] = value
        
        return flat
    
    def print_report(self, report: Dict) -> None:
        """Print formatted validation report"""
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)
        
        # Overall status
        status = "PASSED" if report["passed"] else "FAILED - CANNOT PROCEED"
        color = "\033[92m" if report["passed"] else "\033[91m"  # Green/Red
        print(f"\nStatus: {color}{status}\033[0m")
        
        # Critical failures
        if report["critical_failures"]:
            print(f"\n\033[91mCRITICAL FAILURES ({len(report['critical_failures'])}):\033[0m")
            for failure in report["critical_failures"]:
                print(f"  [X] {failure}")
        
        # Warnings
        if report["warnings"]:
            print(f"\n\033[93mWARNINGS ({len(report['warnings'])}):\033[0m")
            for warning in report["warnings"]:
                print(f"  [!] {warning}")
        
        # Row counts
        print("\nDATA COVERAGE:")
        print("-"*70)
        
        for req in self.requirements:
            rows = report["row_counts"].get(req.name, 0)
            coverage = report["coverage"].get(req.name, 0)
            
            if rows >= req.min_rows:
                status_char = "[OK]"
                bar = "#" * int(min(coverage, 1.0) * 20)
            elif rows > 0:
                status_char = "[LOW]"
                bar = "=" * int(coverage * 20)
            else:
                status_char = "[FAIL]"
                bar = ""
            
            critical_marker = "*" if req.critical else " "
            print(f"  {critical_marker}{status_char} {req.name:25s}: {rows:6d} / {req.min_rows:5d} {bar}")
        
        print("\n  * = critical requirement")
        print("="*70 + "\n")
    
    def abort_if_failed(self, report: Dict) -> None:
        """Raise error if validation failed"""
        if not report["passed"]:
            self.print_report(report)
            raise DataValidationError(
                f"Data validation failed with {len(report['critical_failures'])} critical failures. "
                "Cannot proceed with analysis. Fix data acquisition first."
            )


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


def generate_data_audit_report(all_data: Dict) -> str:
    """
    Generate comprehensive data audit report.
    
    Returns formatted string for logging/display.
    """
    lines = []
    lines.append("="*70)
    lines.append("DATA ACQUISITION AUDIT")
    lines.append("="*70)
    
    def count_rows(obj):
        if isinstance(obj, pd.DataFrame):
            return len(obj)
        elif isinstance(obj, dict):
            return sum(count_rows(v) for v in obj.values())
        return 0
    
    def print_section(name, data, indent=0):
        prefix = "  " * indent
        if isinstance(data, pd.DataFrame):
            rows = len(data)
            status = "[OK]" if rows >= 100 else "[LOW]" if rows > 0 else "[FAIL]"
            lines.append(f"{prefix}{status} {name}: {rows} rows")
            if rows > 0:
                lines.append(f"{prefix}     Columns: {list(data.columns)[:5]}...")
                lines.append(f"{prefix}     Date range: {data.index.min() if hasattr(data.index, 'min') else 'N/A'} to {data.index.max() if hasattr(data.index, 'max') else 'N/A'}")
        elif isinstance(data, dict):
            total = count_rows(data)
            lines.append(f"{prefix}{name}: {len(data)} datasets, {total} total rows")
            for key, value in data.items():
                print_section(key, value, indent + 1)
        elif data is not None:
            lines.append(f"{prefix}[OK] {name}: data available")
        else:
            lines.append(f"{prefix}[FAIL] {name}: NO DATA")
    
    # Price
    lines.append("\nPRICE DATA (Binance):")
    lines.append("-"*50)
    print_section("price", all_data.get("price"), 1)
    
    # Derivatives
    lines.append("\nDERIVATIVES DATA:")
    lines.append("-"*50)
    print_section("derivatives", all_data.get("derivatives", {}), 1)
    
    # Macro
    lines.append("\nMACRO DATA:")
    lines.append("-"*50)
    print_section("macro", all_data.get("macro", {}), 1)
    
    # Sentiment
    lines.append("\nSENTIMENT DATA:")
    lines.append("-"*50)
    print_section("sentiment", all_data.get("sentiment", {}), 1)
    
    # Summary
    total_rows = count_rows(all_data)
    lines.append("\n" + "="*70)
    lines.append(f"TOTAL ROWS: {total_rows}")
    lines.append("="*70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "price": pd.DataFrame({"close": range(8000)}),
        "derivatives": {
            "funding_history": pd.DataFrame({"rate": range(500)}),
            "oi_history": pd.DataFrame({"oi": range(300)}),  # Below threshold
            "ls_history": None,  # Missing
            "liquidation_history": pd.DataFrame({"liq": range(600)}),
        },
        "macro": {
            "yield_curve": pd.DataFrame({"spread": range(250)}),
        },
        "sentiment": {
            "fear_greed": pd.DataFrame({"value": range(365)}),
        }
    }
    
    validator = DataValidationGate()
    report = validator.validate(sample_data)
    validator.print_report(report)
    
    print("\nData Audit Report:")
    print(generate_data_audit_report(sample_data))
