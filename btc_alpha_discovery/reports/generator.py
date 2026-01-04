"""
Report Generator for Alpha Discovery.

Generates comprehensive HTML and text reports summarizing:
- Data acquisition results
- Feature importance findings
- Optimization results
- Discovered strategies
- Validation results
- Actionable recommendations

Output files:
- _report.txt: Human readable text report
- _report.html: Human readable styled HTML report
- _raw.json: AI-readable structured JSON for machine parsing

FIXES APPLIED (v1.1):
- Fixed _obj_to_dict to handle namedtuples, attrs, pydantic models
- Added robust logging throughout extraction functions
- Fixed bare except clauses that swallowed errors
- Added validation checks for extracted data
- Improved type handling in _get_attr
"""

import logging
import os
import json
import math
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE TYPE CONVERSION HELPERS
# =============================================================================

def _safe_float(value: Any, default: float = None) -> Optional[float]:
    """Safely convert value to float, handling numpy types and edge cases."""
    if value is None:
        return default
    try:
        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()
        result = float(value)
        # Handle NaN and Inf
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError, AttributeError):
        return default


def _safe_int(value: Any, default: int = None) -> Optional[int]:
    """Safely convert value to int, handling numpy types."""
    if value is None:
        return default
    try:
        if hasattr(value, 'item'):
            value = value.item()
        return int(value)
    except (TypeError, ValueError, AttributeError):
        return default


def _safe_str(value: Any, default: str = '') -> str:
    """Safely convert value to string."""
    if value is None:
        return default
    try:
        return str(value)
    except (TypeError, ValueError, AttributeError):
        return default


def _safe_list(value: Any) -> List:
    """Safely convert value to list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            return []
    try:
        if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
            return list(value)
        return [value]
    except (TypeError, ValueError):
        return []


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool."""
    if value is None:
        return default
    try:
        return bool(value)
    except (TypeError, ValueError):
        return default


def _get_attr(obj: Any, *keys, default=None):
    """
    Get attribute from object or dict, trying multiple key names.
    
    Args:
        obj: Object or dict to get attribute from
        *keys: Multiple possible key/attribute names to try
        default: Default value if none found
    
    Returns:
        The first found value, or default
    """
    if obj is None:
        return default
    
    for key in keys:
        # Try dict access first
        if isinstance(obj, dict):
            if key in obj:
                val = obj[key]
                # Return even if val is falsy (empty list, 0, etc.) - just not None
                if val is not None:
                    return val
        
        # Try attribute access
        try:
            if hasattr(obj, key):
                val = getattr(obj, key, None)
                if val is not None:
                    return val
        except (TypeError, AttributeError):
            pass
        
        # Try _asdict() for namedtuples
        if hasattr(obj, '_asdict'):
            try:
                d = obj._asdict()
                if key in d and d[key] is not None:
                    return d[key]
            except (TypeError, ValueError):
                pass
    
    return default


def _obj_to_dict(obj: Any) -> Dict:
    """
    Convert object to dict, handling dataclasses, namedtuples, attrs, pydantic, and regular objects.
    
    FIXED: Now handles namedtuples, attrs classes, pydantic models, and other common patterns.
    
    Returns a dict with all accessible attributes.
    """
    if obj is None:
        return {}
    
    # Already a dict
    if isinstance(obj, dict):
        return obj.copy()
    
    # Namedtuple (has _asdict method)
    if hasattr(obj, '_asdict'):
        try:
            return dict(obj._asdict())
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to convert namedtuple: {e}")
    
    # Pydantic v2 model (has model_dump)
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed pydantic v2 model_dump: {e}")
    
    # Pydantic v1 model (has dict method but NOT a regular dict)
    if hasattr(obj, 'dict') and callable(getattr(obj, 'dict', None)) and hasattr(obj, '__fields__'):
        try:
            return obj.dict()
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed pydantic v1 dict: {e}")
    
    # Attrs class (has __attrs_attrs__)
    if hasattr(obj, '__attrs_attrs__'):
        try:
            import attr
            return attr.asdict(obj)
        except (ImportError, TypeError, ValueError) as e:
            # Fallback if attrs not available or fails
            try:
                return {a.name: getattr(obj, a.name, None) for a in obj.__attrs_attrs__}
            except (TypeError, AttributeError) as e2:
                logger.debug(f"Failed attrs conversion: {e}, {e2}")
    
    # Dataclass (has __dataclass_fields__)
    if hasattr(obj, '__dataclass_fields__'):
        try:
            return {k: getattr(obj, k, None) for k in obj.__dataclass_fields__}
        except (TypeError, AttributeError) as e:
            logger.debug(f"Failed dataclass conversion: {e}")
    
    # Object with __dict__
    if hasattr(obj, '__dict__'):
        try:
            d = obj.__dict__
            if isinstance(d, dict):
                return d.copy()
        except (TypeError, AttributeError) as e:
            logger.debug(f"Failed __dict__ access: {e}")
    
    # Object with __slots__
    if hasattr(obj, '__slots__'):
        try:
            return {k: getattr(obj, k, None) for k in obj.__slots__ if hasattr(obj, k)}
        except (TypeError, AttributeError) as e:
            logger.debug(f"Failed __slots__ conversion: {e}")
    
    # Last resort: try vars()
    try:
        return dict(vars(obj))
    except (TypeError, ValueError):
        pass
    
    # If object is iterable of key-value pairs
    if hasattr(obj, 'items') and callable(obj.items):
        try:
            return dict(obj.items())
        except (TypeError, ValueError):
            pass
    
    logger.warning(f"Could not convert object of type {type(obj).__name__} to dict")
    return {}


# =============================================================================
# RULE EXTRACTION FUNCTIONS
# =============================================================================

def _extract_entry_rules(results: Dict[str, Any]) -> Dict:
    """
    Extract all entry rules with full details.
    
    Handles both:
    - Dict-based rules (from JSON/config)
    - TradingRule dataclass objects (from discovery module)
    
    TradingRule attributes: rule_id, conditions, logic, direction, confidence,
                           support, win_rate, avg_return, sharpe, max_drawdown
    
    FIXED: Now properly handles namedtuples and other object types.
    """
    strategy = results.get('strategy', {})
    critic = results.get('critic', {})
    
    # Convert strategy to dict if needed
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    if not isinstance(critic, dict):
        critic = _obj_to_dict(critic)
    
    rules = []
    entry_rules_raw = _get_attr(strategy, 'entry_rules', default=[])
    critic_results = _get_attr(critic, 'results', default=[])
    
    # Ensure we have lists
    entry_rules_raw = _safe_list(entry_rules_raw)
    critic_results = _safe_list(critic_results)
    
    logger.info(f"Extracting entry rules: found {len(entry_rules_raw)} rules to process")
    
    # Create lookup for critic verdicts by rule name/id
    critic_lookup = {}
    for cr in critic_results:
        cr_dict = _obj_to_dict(cr)
        # Try multiple possible name fields
        rule_name = cr_dict.get('rule_name') or cr_dict.get('name') or cr_dict.get('rule_id')
        if rule_name:
            critic_lookup[str(rule_name)] = cr_dict
    
    logger.debug(f"Built critic lookup with {len(critic_lookup)} entries")
    
    # Process each entry rule
    for i, rule in enumerate(entry_rules_raw):
        try:
            # Convert to dict - this is the critical fix
            rule_dict = _obj_to_dict(rule)
            
            if not rule_dict:
                # Try direct attribute access as fallback
                logger.warning(f"Could not convert entry rule {i} to dict (type: {type(rule).__name__})")
                # Attempt to create a minimal dict from common attributes
                rule_dict = {}
                for attr in ['name', 'rule_id', 'direction', 'feature', 'win_rate', 'sharpe', 'n_trades', 'threshold', 'operator']:
                    val = getattr(rule, attr, None)
                    if val is not None:
                        rule_dict[attr] = val
                
                if not rule_dict:
                    logger.error(f"Completely failed to extract rule {i}")
                    continue
            
            # Get rule name/id (TradingRule uses rule_id, dicts might use name)
            rule_name = _get_attr(rule_dict, 'name', 'rule_id', default=f'rule_{i}')
            
            # Get direction (TradingRule uses int 1/-1, convert to string)
            direction_raw = _get_attr(rule_dict, 'direction', 'side', default=1)
            if isinstance(direction_raw, (int, float)):
                direction = 'LONG' if direction_raw >= 0 else 'SHORT'
            else:
                direction = str(direction_raw).upper() if direction_raw else 'LONG'
            
            # Get feature (might be single or from conditions list)
            feature = _get_attr(rule_dict, 'feature', 'feature_name', default='')
            conditions = _get_attr(rule_dict, 'conditions', default=[])
            conditions = _safe_list(conditions)
            
            # If no single feature but has conditions, extract from first condition
            if not feature and conditions:
                first_cond = conditions[0] if conditions else {}
                first_cond_dict = _obj_to_dict(first_cond) if not isinstance(first_cond, dict) else first_cond
                feature = first_cond_dict.get('feature', '')
            
            # Extract conditions list
            extracted_conditions = []
            for cond in conditions:
                cond_dict = _obj_to_dict(cond) if not isinstance(cond, dict) else cond
                extracted_conditions.append({
                    "feature": _safe_str(cond_dict.get('feature')),
                    "operator": _safe_str(cond_dict.get('operator', '>')),
                    "threshold": _safe_float(cond_dict.get('threshold')),
                })
            
            # Build condition string
            condition_str = _get_attr(rule_dict, 'condition_str', 'description', default='')
            if not condition_str and extracted_conditions:
                parts = [f"{c['feature']} {c['operator']} {c['threshold']}" for c in extracted_conditions if c.get('feature')]
                logic = _get_attr(rule_dict, 'logic', default='AND')
                condition_str = f" {logic} ".join(parts)
            
            # Get n_trades (TradingRule uses support)
            n_trades = _safe_int(_get_attr(rule_dict, 'n_trades', 'trades', 'support'))
            
            # Get critic info for this rule
            critic_info = critic_lookup.get(str(rule_name), {})
            
            rule_data = {
                "id": i,
                "name": _safe_str(rule_name),
                "direction": direction,
                "feature": _safe_str(feature),
                "operator": _safe_str(_get_attr(rule_dict, 'operator', 'op', default='>')),
                "threshold": _safe_float(_get_attr(rule_dict, 'threshold', 'value')),
                "percentile": _safe_float(_get_attr(rule_dict, 'percentile')),
                "logic": _safe_str(_get_attr(rule_dict, 'logic', default='AND')),
                "conditions": extracted_conditions,
                "condition_str": condition_str,
                "metrics": {
                    "win_rate": _safe_float(_get_attr(rule_dict, 'win_rate', 'entry_win_rate')),
                    "sharpe": _safe_float(_get_attr(rule_dict, 'sharpe', 'entry_sharpe')),
                    "n_trades": n_trades,
                    "avg_return": _safe_float(_get_attr(rule_dict, 'avg_return', 'mean_return')),
                    "max_drawdown": _safe_float(_get_attr(rule_dict, 'max_drawdown')),
                    "profit_factor": _safe_float(_get_attr(rule_dict, 'profit_factor')),
                    "confidence": _safe_float(_get_attr(rule_dict, 'confidence')),
                },
                "critic": {
                    "verdict": _safe_str(critic_info.get('verdict')),
                    "tests_passed": _safe_int(critic_info.get('tests_passed'), 0),
                    "tests_total": _safe_int(critic_info.get('tests_total'), 10),
                    "oos_sharpe": _safe_float(critic_info.get('oos_sharpe')),
                    "oos_retention": _safe_float(critic_info.get('oos_retention')),
                    "is_sharpe": _safe_float(critic_info.get('is_sharpe')),
                }
            }
            
            # FIX: Extract threshold from conditions if not at top level
            if rule_data["threshold"] is None and extracted_conditions:
                rule_data["threshold"] = extracted_conditions[0].get("threshold")
            
            rules.append(rule_data)
            logger.debug(f"Successfully extracted entry rule {i}: {rule_name}")
            
        except Exception as e:
            logger.error(f"Error extracting entry rule {i}: {e}", exc_info=True)
            continue
    
    # Get performance metrics
    perf = _get_attr(strategy, 'performance', default={})
    if not isinstance(perf, dict):
        perf = _obj_to_dict(perf)
    
    n_entry_rules = _safe_int(_get_attr(strategy, 'n_entry_rules'), len(rules))
    
    logger.info(f"Extracted {len(rules)} entry rules (expected {n_entry_rules})")
    
    # Warn if mismatch
    if len(rules) != n_entry_rules and n_entry_rules > 0:
        logger.warning(f"Entry rule count mismatch: extracted {len(rules)}, expected {n_entry_rules}")
    
    return {
        "total_discovered": n_entry_rules,
        "best_win_rate": _safe_float(_get_attr(perf, 'entry_win_rate', 'win_rate')),
        "best_sharpe": _safe_float(_get_attr(perf, 'entry_sharpe', 'sharpe')),
        "best_n_trades": _safe_int(_get_attr(perf, 'n_trades', 'trades')),
        "rules": rules
    }


def _extract_exit_rules(results: Dict[str, Any]) -> Dict:
    """
    Extract all exit rules with parameters.
    
    Handles both:
    - Dict-based rules (from JSON/config)
    - ExitRule dataclass objects (from discovery module)
    
    ExitRule attributes: rule_id, exit_type, conditions, priority,
                        avg_bars_held, avg_exit_return, hit_rate
    
    FIXED: Now properly handles namedtuples and other object types.
    """
    strategy = results.get('strategy', {})
    
    # Convert strategy to dict if needed
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    
    rules = []
    exit_rules_raw = _get_attr(strategy, 'exit_rules', default=[])
    
    # Ensure we have a list
    exit_rules_raw = _safe_list(exit_rules_raw)
    
    logger.info(f"Extracting exit rules: found {len(exit_rules_raw)} rules to process")
    
    for i, rule in enumerate(exit_rules_raw):
        try:
            # Convert to dict
            rule_dict = _obj_to_dict(rule)
            
            if not rule_dict:
                logger.warning(f"Could not convert exit rule {i} to dict (type: {type(rule).__name__})")
                # Try direct attribute access as fallback
                rule_dict = {}
                for attr in ['type', 'exit_type', 'name', 'rule_id', 'bars_held', 'stop_loss', 'take_profit']:
                    val = getattr(rule, attr, None)
                    if val is not None:
                        rule_dict[attr] = val
                
                if not rule_dict:
                    logger.error(f"Completely failed to extract exit rule {i}")
                    continue
            
            # Get exit type (ExitRule uses exit_type, dicts might use type)
            exit_type = _safe_str(_get_attr(rule_dict, 'type', 'exit_type', default='unknown')).upper()
            
            # Get rule name/id
            rule_name = _safe_str(_get_attr(rule_dict, 'name', 'rule_id', default=f'exit_{i}'))
            
            rule_data = {
                "id": i,
                "type": exit_type,
                "name": rule_name,
                "priority": _safe_int(_get_attr(rule_dict, 'priority'), i),
                "parameters": {},
                "metrics": {
                    "avg_bars_held": _safe_float(_get_attr(rule_dict, 'avg_bars_held')),
                    "avg_exit_return": _safe_float(_get_attr(rule_dict, 'avg_exit_return')),
                    "hit_rate": _safe_float(_get_attr(rule_dict, 'hit_rate')),
                }
            }
            
            # Extract type-specific parameters
            if exit_type == "TIME" or 'bars_held' in rule_dict or rule_name.startswith('time_'):
                # Try to get bars_held from dict first
                bars = _safe_int(_get_attr(rule_dict, 'bars_held', 'value'))
                # If not found, try avg_bars_held
                if bars is None:
                    bars = _safe_int(rule_data["metrics"].get("avg_bars_held"))
                # If still not found, parse from name like "time_48"
                if bars is None and rule_name.startswith('time_'):
                    try:
                        bars = int(rule_name.split('_')[1])
                    except (IndexError, ValueError):
                        pass
                rule_data["parameters"]["bars_held"] = bars
            
            if exit_type in ["STOP_LOSS", "TAKE_PROFIT", "STOP_LOSS_TAKE_PROFIT"] or 'stop_loss' in rule_dict or 'take_profit' in rule_dict or rule_name.startswith('sl_tp_'):
                sl = _get_attr(rule_dict, 'stop_loss')
                tp = _get_attr(rule_dict, 'take_profit')
                
                # If not in dict, parse from name like "sl_tp_5.0_10.0"
                if (sl is None or tp is None) and rule_name.startswith('sl_tp_'):
                    try:
                        parts = rule_name.split('_')
                        if len(parts) >= 4:
                            sl = float(parts[2]) if sl is None else sl
                            tp = float(parts[3]) if tp is None else tp
                    except (IndexError, ValueError):
                        pass
                
                if sl is not None:
                    rule_data["parameters"]["stop_loss"] = _safe_float(sl)
                if tp is not None:
                    rule_data["parameters"]["take_profit"] = _safe_float(tp)
            
            if exit_type == "TRAILING_STOP" or 'trail_pct' in rule_dict:
                rule_data["parameters"]["trail_pct"] = _safe_float(_get_attr(rule_dict, 'trail_pct'))
            
            # Extract conditions if present
            conditions = _get_attr(rule_dict, 'conditions', default=[])
            conditions = _safe_list(conditions)
            
            if conditions:
                extracted_conditions = []
                for cond in conditions:
                    cond_dict = _obj_to_dict(cond) if not isinstance(cond, dict) else cond
                    extracted_conditions.append({
                        "feature": _safe_str(cond_dict.get('feature')),
                        "operator": _safe_str(cond_dict.get('operator', '>')),
                        "threshold": _safe_float(cond_dict.get('threshold')),
                    })
                rule_data["conditions"] = extracted_conditions
                
                # For SIGNAL/CONDITION type, also put first condition in parameters
                if exit_type in ["SIGNAL", "CONDITION"] and extracted_conditions:
                    first = extracted_conditions[0]
                    rule_data["parameters"]["feature"] = first.get("feature")
                    rule_data["parameters"]["operator"] = first.get("operator")
                    rule_data["parameters"]["threshold"] = first.get("threshold")
            
            # Legacy: direct feature/operator/threshold in rule dict
            if _get_attr(rule_dict, 'feature') and 'conditions' not in rule_data:
                rule_data["parameters"]["feature"] = _safe_str(rule_dict.get('feature'))
                rule_data["parameters"]["operator"] = _safe_str(rule_dict.get('operator'))
                rule_data["parameters"]["threshold"] = _safe_float(rule_dict.get('threshold'))
            
            rules.append(rule_data)
            logger.debug(f"Successfully extracted exit rule {i}: {rule_name} ({exit_type})")
            
        except Exception as e:
            logger.error(f"Error extracting exit rule {i}: {e}", exc_info=True)
            continue
    
    n_exit_rules = _safe_int(_get_attr(strategy, 'n_exit_rules'), len(rules))
    
    logger.info(f"Extracted {len(rules)} exit rules (expected {n_exit_rules})")
    
    # Warn if mismatch
    if len(rules) != n_exit_rules and n_exit_rules > 0:
        logger.warning(f"Exit rule count mismatch: extracted {len(rules)}, expected {n_exit_rules}")
    
    return {
        "total_discovered": n_exit_rules,
        "rules": rules
    }


def _extract_critic_analysis(results: Dict[str, Any]) -> Dict:
    """
    Extract Devil's Advocate critic analysis with detailed test results.
    
    FIXED: Now properly handles namedtuples and other object types.
    """
    critic = results.get('critic', {})
    
    if not isinstance(critic, dict):
        critic = _obj_to_dict(critic)
    
    verdicts = critic.get('verdicts', {})
    if not isinstance(verdicts, dict):
        verdicts = _obj_to_dict(verdicts)
    
    summary = {
        "total_tested": _safe_int(critic.get('rules_tested'), 0),
        "credible": _safe_int(verdicts.get('CREDIBLE'), 0),
        "suspicious": _safe_int(verdicts.get('SUSPICIOUS'), 0),
        "debunked": _safe_int(verdicts.get('DEBUNKED'), 0),
        "survival_rate": _safe_float(critic.get('survival_rate'), 0)
    }
    
    critic_results_raw = critic.get('results', [])
    critic_results_raw = _safe_list(critic_results_raw)
    
    logger.info(f"Extracting critic results: found {len(critic_results_raw)} results to process")
    
    rules = []
    for i, result in enumerate(critic_results_raw):
        try:
            result_dict = _obj_to_dict(result) if not isinstance(result, dict) else result
            
            if not result_dict:
                logger.warning(f"Could not convert critic result {i} to dict (type: {type(result).__name__})")
                continue
            
            # Extract test results
            test_results = {}
            raw_tests = result_dict.get('test_results', result_dict.get('tests', {}))
            
            if isinstance(raw_tests, dict):
                for test_name, test_data in raw_tests.items():
                    try:
                        if isinstance(test_data, dict):
                            test_results[test_name] = {
                                "passed": _safe_bool(test_data.get('passed')),
                                "value": _safe_float(test_data.get('value')),
                                "details": _safe_str(test_data.get('details', test_data.get('reason', '')))
                            }
                            # Include any additional fields
                            for k, v in test_data.items():
                                if k not in ['passed', 'value', 'details', 'reason']:
                                    test_results[test_name][k] = v
                        elif isinstance(test_data, bool):
                            test_results[test_name] = {"passed": test_data}
                        else:
                            # Try to convert test_data
                            td = _obj_to_dict(test_data)
                            if td:
                                test_results[test_name] = {
                                    "passed": _safe_bool(td.get('passed')),
                                    "value": _safe_float(td.get('value')),
                                    "details": _safe_str(td.get('details', ''))
                                }
                    except Exception as e:
                        logger.debug(f"Error extracting test result {test_name}: {e}")
            
            rules.append({
                "rule_name": _safe_str(result_dict.get('rule_name', result_dict.get('name', result_dict.get('rule_id')))),
                "verdict": _safe_str(result_dict.get('verdict')),
                "tests_passed": _safe_int(result_dict.get('tests_passed'), 0),
                "tests_total": _safe_int(result_dict.get('tests_total'), 10),
                "test_results": test_results,
                "oos_sharpe": _safe_float(result_dict.get('oos_sharpe')),
                "oos_retention": _safe_float(result_dict.get('oos_retention')),
                "is_sharpe": _safe_float(result_dict.get('is_sharpe')),
                "recommendation": _safe_str(result_dict.get('recommendation', ''))
            })
            logger.debug(f"Successfully extracted critic result {i}")
            
        except Exception as e:
            logger.error(f"Error extracting critic result {i}: {e}", exc_info=True)
            continue
    
    logger.info(f"Extracted {len(rules)} critic results")
    
    return {"summary": summary, "rules": rules}


def _extract_validation(results: Dict[str, Any]) -> Dict:
    """Extract comprehensive validation results."""
    validation = results.get('validation', {})
    
    if not isinstance(validation, dict):
        validation = _obj_to_dict(validation)
    
    # Extract walk-forward results
    walk_forward = []
    wf_raw = validation.get('walk_forward', [])
    wf_raw = _safe_list(wf_raw)
    
    for i, wf in enumerate(wf_raw):
        try:
            wf_dict = _obj_to_dict(wf) if not isinstance(wf, dict) else wf
            
            # Handle nested format (from run_discovery.py)
            in_sample = wf_dict.get('in_sample', {})
            out_sample = wf_dict.get('out_sample', {})
            if isinstance(in_sample, dict) and isinstance(out_sample, dict):
                # New nested format
                walk_forward.append({
                    "period": wf_dict.get('period_id', i),
                    "train_start": _safe_str(wf_dict.get('train_start')),
                    "train_end": _safe_str(wf_dict.get('train_end')),
                    "test_start": _safe_str(wf_dict.get('test_start')),
                    "test_end": _safe_str(wf_dict.get('test_end')),
                    "is_degraded": wf_dict.get('is_degraded', False),
                    "train_sharpe": _safe_float(in_sample.get('sharpe')),
                    "train_win_rate": _safe_float(in_sample.get('win_rate')),
                    "train_n_trades": _safe_int(in_sample.get('n_trades')),
                    "test_sharpe": _safe_float(out_sample.get('sharpe')),
                    "test_win_rate": _safe_float(out_sample.get('win_rate')),
                    "test_n_trades": _safe_int(out_sample.get('n_trades')),
                })
            else:
                # Legacy flat format
                walk_forward.append({
                    "period": wf_dict.get('period', i),
                    "train_start": _safe_str(wf_dict.get('train_start')),
                    "train_end": _safe_str(wf_dict.get('train_end')),
                    "test_start": _safe_str(wf_dict.get('test_start')),
                    "test_end": _safe_str(wf_dict.get('test_end')),
                    "train_signals": _safe_int(wf_dict.get('train_signals')),
                    "test_signals": _safe_int(wf_dict.get('test_signals')),
                    "test_sharpe": _safe_float(wf_dict.get('test_sharpe', wf_dict.get('sharpe'))),
                    "test_win_rate": _safe_float(wf_dict.get('test_win_rate', wf_dict.get('win_rate'))),
                    "test_return": _safe_float(wf_dict.get('test_return', wf_dict.get('return'))),
                })
        except Exception as e:
            logger.debug(f"Error extracting walk-forward period {i}: {e}")
    
    # Extract regime performance if present
    regime_performance = []
    regime_raw = validation.get('regime_performance', validation.get('regimes', []))
    regime_raw = _safe_list(regime_raw)
    
    for regime in regime_raw:
        try:
            r_dict = _obj_to_dict(regime) if not isinstance(regime, dict) else regime
            regime_performance.append({
                "regime": _safe_str(r_dict.get('regime', r_dict.get('name'))),
                "n_trades": _safe_int(r_dict.get('n_trades')),
                "win_rate": _safe_float(r_dict.get('win_rate')),
                "sharpe": _safe_float(r_dict.get('sharpe')),
                "avg_return": _safe_float(r_dict.get('avg_return')),
            })
        except Exception as e:
            logger.debug(f"Error extracting regime: {e}")
    
    return {
        "grade": _safe_str(validation.get('grade')),
        "recommendation": _safe_str(validation.get('recommendation')),
        "statistically_significant": _safe_bool(validation.get('statistically_significant')),
        "oos_metrics": {
            "sharpe": _safe_float(validation.get('oos_sharpe')),
            "win_rate": _safe_float(validation.get('oos_win_rate')),
            "max_drawdown": _safe_float(validation.get('oos_max_drawdown', validation.get('max_drawdown'))),
            "n_trades": _safe_int(validation.get('oos_n_trades', validation.get('n_trades'))),
            "profit_factor": _safe_float(validation.get('oos_profit_factor', validation.get('profit_factor'))),
            "avg_trade_return": _safe_float(validation.get('oos_avg_trade_return', validation.get('avg_trade_return'))),
            "total_return": _safe_float(validation.get('oos_total_return', validation.get('total_return'))),
        },
        "in_sample_metrics": {
            "sharpe": _safe_float(validation.get('is_sharpe')),
            "win_rate": _safe_float(validation.get('is_win_rate')),
            "n_trades": _safe_int(validation.get('is_n_trades')),
        },
        "walk_forward": walk_forward,
        "regime_performance": regime_performance,
        "warnings": _safe_list(validation.get('warnings', [])),
    }


def _extract_features(results: Dict[str, Any]) -> Dict:
    """Extract feature information."""
    features = results.get('features', {})
    fi = results.get('feature_importance', {})
    
    if not isinstance(features, dict):
        features = _obj_to_dict(features)
    if not isinstance(fi, dict):
        fi = _obj_to_dict(fi)
    
    # Get importance scores
    importance_scores = fi.get('importance_scores', fi.get('scores', {}))
    if not isinstance(importance_scores, dict):
        importance_scores = _obj_to_dict(importance_scores)
    
    # Build top features list with scores
    top_features = []
    raw_top_features = fi.get('top_features', [])
    raw_top_features = _safe_list(raw_top_features)
    
    for i, f in enumerate(raw_top_features[:50]):
        top_features.append({
            "rank": i + 1,
            "name": _safe_str(f),
            "importance": _safe_float(importance_scores.get(str(f)))
        })
    
    return {
        "total_count": _safe_int(features.get('n_features'), 0),
        "sample_size": _safe_int(features.get('n_samples'), 0),
        "date_range": {
            "start": _safe_str(features.get('date_start', features.get('start_date'))),
            "end": _safe_str(features.get('date_end', features.get('end_date'))),
        },
        "category_importance": {
            k: _safe_float(v) for k, v in fi.get('category_importance', {}).items()
        },
        "top_features": top_features
    }


def _extract_data_acquisition(results: Dict[str, Any]) -> Dict:
    """Extract data acquisition results."""
    data = results.get('data', {})
    
    if not isinstance(data, dict):
        data = _obj_to_dict(data)
    
    sources = {}
    total_rows = 0
    
    for key, value in data.items():
        if isinstance(value, int):
            sources[key] = {"rows": value}
            total_rows += value
        elif isinstance(value, dict):
            rows = _safe_int(value.get('rows', value.get('n_rows')), 0)
            sources[key] = {
                "rows": rows,
                "date_start": _safe_str(value.get('date_start')),
                "date_end": _safe_str(value.get('date_end')),
            }
            total_rows += rows
        else:
            # Try to get length
            try:
                rows = len(value) if hasattr(value, '__len__') else 0
                sources[key] = {"rows": rows}
                total_rows += rows
            except (TypeError, AttributeError):
                sources[key] = {"rows": 0}
    
    return {
        "sources": sources,
        "total_rows": total_rows,
        "n_sources": len(sources)
    }


def _extract_antipatterns(results: Dict[str, Any]) -> Dict:
    """Extract antipattern information."""
    ap = results.get('antipatterns', {})
    
    if not isinstance(ap, dict):
        ap = _obj_to_dict(ap)
    
    patterns = []
    raw_patterns = ap.get('patterns', [])
    raw_patterns = _safe_list(raw_patterns)
    
    for p in raw_patterns:
        try:
            p_dict = _obj_to_dict(p) if not isinstance(p, dict) else p
            patterns.append({
                "name": _safe_str(p_dict.get('name')),
                "description": _safe_str(p_dict.get('description')),
                "frequency": _safe_float(p_dict.get('frequency')),
                "avg_loss": _safe_float(p_dict.get('avg_loss')),
            })
        except Exception as e:
            logger.debug(f"Error extracting antipattern: {e}")
    
    return {
        "count": _safe_int(ap.get('n_patterns'), 0),
        "regimes": _safe_int(ap.get('n_regimes'), 0),
        "patterns": patterns
    }


def _extract_pillar_analysis(results: Dict[str, Any]) -> Dict:
    """Extract pillar validation analysis."""
    pv = results.get('pillar_validation', {})
    
    if not isinstance(pv, dict):
        pv = _obj_to_dict(pv)
    
    deviations = {}
    pillar_deviations = pv.get('pillar_deviations', {})
    if not isinstance(pillar_deviations, dict):
        pillar_deviations = _obj_to_dict(pillar_deviations)
    
    for k, v in pillar_deviations.items():
        if isinstance(v, dict):
            deviations[k] = {
                "assumed": _safe_float(v.get('assumed')),
                "actual": _safe_float(v.get('actual')),
                "deviation": _safe_float(v.get('deviation')),
            }
        else:
            deviations[k] = {"deviation": _safe_float(v)}
    
    return {
        "recommendation": _safe_str(pv.get('recommendation')),
        "deviations": deviations,
        "total_deviation": _safe_float(pv.get('total_deviation')),
    }


# =============================================================================
# WEEKLY FOLDER ORGANIZATION
# =============================================================================

def get_weekly_report_paths(base_dir: str = 'reports') -> tuple:
    """
    Generate organized file paths based on current date.
    
    Structure: reports/YYYY/week_WW_mmmDD_mmmDD/
    
    Returns:
        tuple: (text_path, html_path, raw_json_path, folder_path)
    """
    now = datetime.now()
    year = now.year
    
    week_num = now.isocalendar()[1]
    week_start = now - timedelta(days=now.weekday())
    week_end = week_start + timedelta(days=6)
    
    week_folder = f"week_{week_num:02d}_{week_start.strftime('%b%d').lower()}_{week_end.strftime('%b%d').lower()}"
    
    folder_path = os.path.join(base_dir, str(year), week_folder)
    os.makedirs(folder_path, exist_ok=True)
    
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    return (
        os.path.join(folder_path, f"discovery_{timestamp}_report.txt"),
        os.path.join(folder_path, f"discovery_{timestamp}_report.html"),
        os.path.join(folder_path, f"discovery_{timestamp}_raw.json"),
        folder_path
    )


# =============================================================================
# AI RAW DATA GENERATION
# =============================================================================

def _generate_ai_raw_data(results: Dict[str, Any], path: Path):
    """
    Generate comprehensive raw data export for AI consumption.
    
    This is the main machine-readable output optimized for:
    - Comparing runs over time
    - Extracting trading rules programmatically
    - Auditing critic decisions
    - Retraining models
    - Automated analysis
    """
    now = datetime.now()
    
    logger.info("Generating AI raw data export...")
    
    # Extract all data with proper error handling
    try:
        entry_rules = _extract_entry_rules(results)
    except Exception as e:
        logger.error(f"Failed to extract entry rules: {e}", exc_info=True)
        entry_rules = {"total_discovered": 0, "rules": []}
    
    try:
        exit_rules = _extract_exit_rules(results)
    except Exception as e:
        logger.error(f"Failed to extract exit rules: {e}", exc_info=True)
        exit_rules = {"total_discovered": 0, "rules": []}
    
    try:
        critic_analysis = _extract_critic_analysis(results)
    except Exception as e:
        logger.error(f"Failed to extract critic analysis: {e}", exc_info=True)
        critic_analysis = {"summary": {}, "rules": []}
    
    raw_export = {
        "_metadata": {
            "format": "btc_alpha_discovery_raw_v1.1",  # Bumped version for fixes
            "generated_at": now.isoformat(),
            "timestamp_unix": int(now.timestamp()),
            "purpose": "AI consumption - structured data for machine parsing"
        },
        
        "config": {
            "target": _safe_str(results.get('config', {}).get('target', 'profitable_24h')),
            "lookback_days": _safe_int(results.get('config', {}).get('lookback_days'), 90),
            "timeframe": _safe_str(results.get('config', {}).get('timeframe', '4h')),
        },
        
        "data_acquisition": _extract_data_acquisition(results),
        "features": _extract_features(results),
        "entry_rules": entry_rules,
        "exit_rules": exit_rules,
        "critic_analysis": critic_analysis,
        "validation": _extract_validation(results),
        "antipatterns": _extract_antipatterns(results),
        "pillar_analysis": _extract_pillar_analysis(results),
        
        "optimization": {
            "recommendations": _safe_list(results.get('optimization', {}).get('recommendations', []))
        },
        
        # Include strategy metadata
        "strategy": {
            "name": _safe_str(_get_attr(results.get('strategy', {}), 'name')),
            "n_entry_rules": _safe_int(_get_attr(results.get('strategy', {}), 'n_entry_rules')),
            "n_exit_rules": _safe_int(_get_attr(results.get('strategy', {}), 'n_exit_rules')),
        }
    }
    
    # Log extraction results
    logger.info(f"  Entry rules: {len(raw_export['entry_rules']['rules'])} extracted")
    logger.info(f"  Exit rules: {len(raw_export['exit_rules']['rules'])} extracted")
    logger.info(f"  Critic results: {len(raw_export['critic_analysis']['rules'])} extracted")
    
    # Validate extraction success
    strategy = results.get('strategy', {})
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    
    expected_entry = _safe_int(_get_attr(strategy, 'n_entry_rules'), 0)
    expected_exit = _safe_int(_get_attr(strategy, 'n_exit_rules'), 0)
    
    if expected_entry > 0 and len(raw_export['entry_rules']['rules']) == 0:
        logger.error(f"CRITICAL: Expected {expected_entry} entry rules but extracted 0!")
    if expected_exit > 0 and len(raw_export['exit_rules']['rules']) == 0:
        logger.error(f"CRITICAL: Expected {expected_exit} exit rules but extracted 0!")
    
    # Custom encoder that handles any edge cases
    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle numpy types
            if hasattr(obj, 'item'):
                return obj.item()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            # Handle NaN/Inf
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
            # Handle datetime
            if isinstance(obj, datetime):
                return obj.isoformat()
            # Handle pandas Timestamp
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            # Fallback to string
            try:
                return str(obj)
            except (TypeError, ValueError):
                return None
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw_export, f, indent=2, cls=SafeEncoder)
        logger.info(f"AI raw data saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save AI raw data: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_discovery_report(
    results: Dict[str, Any],
    output_dir: str = 'reports',
    format: str = 'both',
    use_weekly_folders: bool = True
) -> Dict[str, str]:
    """
    Generate comprehensive discovery report.
    
    Args:
        results: Dictionary with all discovery results
        output_dir: Output directory for reports
        format: 'html', 'text', or 'both'
        use_weekly_folders: If True, organize into weekly subfolders
        
    Returns:
        Dict with paths: {'text': ..., 'html': ..., 'raw': ..., 'folder': ...}
    """
    output_paths = {}
    
    if use_weekly_folders:
        text_path, html_path, raw_path, folder_path = get_weekly_report_paths(output_dir)
        logger.info(f"Using weekly folder: {folder_path}")
        output_paths['folder'] = folder_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_path = str(output_path / f"discovery_{timestamp}_report.txt")
        html_path = str(output_path / f"discovery_{timestamp}_report.html")
        raw_path = str(output_path / f"discovery_{timestamp}_raw.json")
        output_paths['folder'] = str(output_path)
    
    if format in ['text', 'both']:
        try:
            _generate_text_report(results, Path(text_path))
            output_paths['text'] = text_path
            logger.info(f"Text report saved: {text_path}")
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
            import traceback
            traceback.print_exc()
        
    if format in ['html', 'both']:
        try:
            _generate_html_report(results, Path(html_path))
            output_paths['html'] = html_path
            logger.info(f"HTML report saved: {html_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            import traceback
            traceback.print_exc()
    
    # ALWAYS generate AI raw data
    _generate_ai_raw_data(results, Path(raw_path))
    output_paths['raw'] = raw_path
    
    return output_paths


def generate_discovery_report_to_paths(
    results: Dict[str, Any],
    text_path: str,
    html_path: str
) -> None:
    """Generate discovery reports to specific file paths."""
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    _generate_text_report(results, Path(text_path))
    _generate_html_report(results, Path(html_path))


# =============================================================================
# TEXT REPORT GENERATION
# =============================================================================

def _generate_text_report(results: Dict[str, Any], path: Path):
    """Generate plain text report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("BTC ALPHA DISCOVERY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # Get validation (handle both dict and object)
    validation = results.get('validation', {})
    if not isinstance(validation, dict):
        validation = _obj_to_dict(validation)
    
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    if validation:
        lines.append(f"Overall Grade: {validation.get('grade', 'N/A')}")
        lines.append(f"Recommendation: {validation.get('recommendation', 'N/A')}")
        oos_sharpe = _safe_float(validation.get('oos_sharpe'), 0)
        oos_win_rate = _safe_float(validation.get('oos_win_rate'), 0)
        lines.append(f"Out-of-Sample Sharpe: {oos_sharpe:.2f}" if oos_sharpe else "Out-of-Sample Sharpe: N/A")
        lines.append(f"Out-of-Sample Win Rate: {oos_win_rate:.1%}" if oos_win_rate else "Out-of-Sample Win Rate: N/A")
    else:
        lines.append("Validation not completed")
    lines.append("")
    
    lines.append("DATA ACQUISITION")
    lines.append("-" * 40)
    data = results.get('data', {})
    if not isinstance(data, dict):
        data = _obj_to_dict(data)
    for source, count in data.items():
        if isinstance(count, dict):
            count = count.get('rows', count.get('n_rows', 0))
        elif hasattr(count, '__len__'):
            try:
                count = len(count)
            except (TypeError, AttributeError):
                count = 0
        lines.append(f"  {source}: {count} rows")
    lines.append("")
    
    lines.append("FEATURE ENGINEERING")
    lines.append("-" * 40)
    features = results.get('features', {})
    if not isinstance(features, dict):
        features = _obj_to_dict(features)
    lines.append(f"  Total Features: {features.get('n_features', 0)}")
    lines.append(f"  Sample Size: {features.get('n_samples', 0)}")
    lines.append("")
    
    lines.append("TOP FEATURES (by importance)")
    lines.append("-" * 40)
    fi = results.get('feature_importance', {})
    if not isinstance(fi, dict):
        fi = _obj_to_dict(fi)
    top_features = fi.get('top_features', [])
    top_features = _safe_list(top_features)
    for i, feat in enumerate(top_features[:15], 1):
        lines.append(f"  {i:2}. {feat}")
    lines.append("")
    
    lines.append("CATEGORY IMPORTANCE")
    lines.append("-" * 40)
    cat_importance = fi.get('category_importance', {})
    if not isinstance(cat_importance, dict):
        cat_importance = _obj_to_dict(cat_importance)
    for cat, imp in cat_importance.items():
        imp_val = _safe_float(imp, 0)
        bar = "#" * int(imp_val * 50)
        lines.append(f"  {cat:15}: {bar} {imp_val:.1%}")
    lines.append("")
    
    # Strategy section
    strategy = results.get('strategy', {})
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    
    if strategy:
        lines.append("DISCOVERED STRATEGY")
        lines.append("-" * 40)
        lines.append(f"  Name: {strategy.get('name', 'N/A')}")
        lines.append(f"  Entry Rules: {strategy.get('n_entry_rules', 0)}")
        lines.append(f"  Exit Rules: {strategy.get('n_exit_rules', 0)}")
        perf = strategy.get('performance', {})
        if not isinstance(perf, dict):
            perf = _obj_to_dict(perf)
        if perf:
            wr = _safe_float(perf.get('entry_win_rate'), 0)
            sr = _safe_float(perf.get('entry_sharpe'), 0)
            dd = _safe_float(perf.get('max_drawdown'), 0)
            lines.append(f"  Win Rate: {wr:.1%}")
            lines.append(f"  Sharpe: {sr:.2f}")
            lines.append(f"  Max Drawdown: {dd:.1%}")
    lines.append("")
    
    # Critic section
    critic = results.get('critic', {})
    if not isinstance(critic, dict):
        critic = _obj_to_dict(critic)
    
    if critic:
        lines.append("DEVIL'S ADVOCATE (CRITIC PASS)")
        lines.append("-" * 40)
        verdicts = critic.get('verdicts', {})
        if not isinstance(verdicts, dict):
            verdicts = _obj_to_dict(verdicts)
        lines.append(f"  Rules Tested: {critic.get('rules_tested', 0)}")
        lines.append(f"  Credible: {verdicts.get('CREDIBLE', 0)}")
        lines.append(f"  Suspicious: {verdicts.get('SUSPICIOUS', 0)}")
        lines.append(f"  Debunked: {verdicts.get('DEBUNKED', 0)}")
        sr = _safe_float(critic.get('survival_rate'), 0)
        lines.append(f"  Survival Rate: {sr:.1%}")
    lines.append("")
    
    # Warnings
    if validation.get('warnings'):
        lines.append("WARNINGS")
        lines.append("-" * 40)
        warnings = _safe_list(validation['warnings'])
        for w in warnings:
            lines.append(f"  [!] {w}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Text report saved: {path}")


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def _generate_html_report(results: Dict[str, Any], path: Path):
    """Generate HTML report with styling."""
    
    validation = results.get('validation', {})
    if not isinstance(validation, dict):
        validation = _obj_to_dict(validation)
    
    grade = validation.get('grade', 'N/A')
    grade_color = {'A': '#22c55e', 'B': '#84cc16', 'C': '#eab308', 'D': '#f97316', 'F': '#ef4444'}.get(grade, '#6b7280')
    
    oos_sharpe = _safe_float(validation.get('oos_sharpe'), 0)
    oos_win_rate = _safe_float(validation.get('oos_win_rate'), 0)
    
    features = results.get('features', {})
    if not isinstance(features, dict):
        features = _obj_to_dict(features)
    
    strategy = results.get('strategy', {})
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BTC Alpha Discovery Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #f8fafc; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
        h2 {{ color: #94a3b8; margin-top: 30px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .grade {{ font-size: 72px; font-weight: bold; color: {grade_color}; text-align: center; }}
        .metric {{ display: inline-block; background: #334155; padding: 10px 15px; border-radius: 6px; margin: 5px; }}
        .metric-label {{ color: #94a3b8; font-size: 12px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #f8fafc; }}
        .bar {{ height: 20px; background: #3b82f6; border-radius: 4px; margin: 5px 0; }}
        .warning {{ background: #422006; border-left: 4px solid #f97316; padding: 10px 15px; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>BTC Alpha Discovery Report</h1>
    <p style="color: #64748b;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="card">
        <h2 style="margin-top: 0;">Overall Grade</h2>
        <div class="grade">{grade}</div>
        <p style="text-align: center; color: #94a3b8;">{validation.get('recommendation', 'Validation pending')}</p>
    </div>
    
    <div class="card">
        <h2 style="margin-top: 0;">Key Metrics</h2>
        <div class="metric"><div class="metric-label">OOS Sharpe</div><div class="metric-value">{oos_sharpe:.2f}</div></div>
        <div class="metric"><div class="metric-label">Win Rate</div><div class="metric-value">{oos_win_rate:.1%}</div></div>
        <div class="metric"><div class="metric-label">Features</div><div class="metric-value">{features.get('n_features', 0)}</div></div>
        <div class="metric"><div class="metric-label">Entry Rules</div><div class="metric-value">{strategy.get('n_entry_rules', 0)}</div></div>
    </div>
"""
    
    fi = results.get('feature_importance', {})
    if not isinstance(fi, dict):
        fi = _obj_to_dict(fi)
    
    top_features = fi.get('top_features', [])
    top_features = _safe_list(top_features)
    
    if top_features:
        html += '<div class="card"><h2 style="margin-top: 0;">Top Features</h2><table style="width:100%">'
        for i, feat in enumerate(top_features[:10], 1):
            html += f'<tr><td>{i}</td><td>{feat}</td></tr>'
        html += '</table></div>'
    
    warnings = validation.get('warnings', [])
    warnings = _safe_list(warnings)
    
    if warnings:
        html += '<div class="card"><h2 style="margin-top: 0;">Warnings</h2>'
        for w in warnings:
            html += f'<div class="warning">{w}</div>'
        html += '</div>'
    
    html += '</body></html>'
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML report saved: {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_strategy_card(strategy: Dict[str, Any]) -> str:
    """Generate a trading card summary for a strategy."""
    if not isinstance(strategy, dict):
        strategy = _obj_to_dict(strategy)
    
    name = strategy.get('name', 'Unknown Strategy')
    performance = strategy.get('performance', {})
    if not isinstance(performance, dict):
        performance = _obj_to_dict(performance)
    
    wr = _safe_float(performance.get('entry_win_rate'), 0)
    sr = _safe_float(performance.get('entry_sharpe'), 0)
    dd = _safe_float(performance.get('max_drawdown'), 0)
    nt = _safe_int(performance.get('n_trades'), 0)
    
    return f"""
+------------------------------------------------------------------+
|  STRATEGY CARD: {name:^47} |
+------------------------------------------------------------------+
|  Win Rate:    {wr:>6.1%}  |  Sharpe:    {sr:>6.2f}       |
|  Max DD:      {dd:>6.1%}  |  Trades:    {nt:>6}       |
+------------------------------------------------------------------+
"""
