# BTC ALPHA DISCOVERY PIPELINE AUDIT
## Run: 2026-01-01 22:14:32

═══════════════════════════════════════════════════════════════
# PHASE 1: BUG DETECTION
═══════════════════════════════════════════════════════════════

## BUG 1: Threshold Values Not Exported to JSON
- **Severity**: 🔴 CRITICAL
- **Evidence**: All 20 entry rules show `"threshold": null` in JSON, but logs show actual values:
  ```
  Rule: LONG when (deriv_feat_cg_oi_aggregated_oi_close_change_1h > 0.0121)
  ```
  Yet JSON shows: `"threshold": None`
- **Impact**: Trading system cannot reconstruct rules without thresholds. Rules are USELESS.
- **Root Cause**: `run_discovery.py` stores TradingRule objects, but the threshold is embedded in `conditions` list, not as a top-level attribute
- **Fix**: In `reports/generator.py` `_extract_entry_rules()`, add:
  ```python
  # After line ~220, extract threshold from conditions if not present
  threshold = _get_attr(rule_dict, 'threshold', 'value')
  if threshold is None and extracted_conditions:
      threshold = extracted_conditions[0].get('threshold')
  rule_data["threshold"] = _safe_float(threshold)
  ```

## BUG 2: Exit Rule Parameters Empty
- **Severity**: 🔴 CRITICAL  
- **Evidence**: 
  ```json
  "sl_tp_3.0_1.5": params={}, metrics={'hit_rate': 0.0}
  "time_1": params={'bars_held': None}
  ```
- **Impact**: Cannot execute exit logic without SL/TP values or time thresholds
- **Root Cause**: Parameter extraction looks for `bars_held` attribute but ExitRule uses `avg_bars_held`. Name parsing was added but `bars_held` in output is still None
- **Fix**: Already partially fixed. Need to ensure `avg_bars_held` is converted to int and stored as `bars_held`:
  ```python
  if exit_type == "TIME":
      bars = _safe_int(rule_dict.get('avg_bars_held'))
      rule_data["parameters"]["bars_held"] = bars
  ```

## BUG 3: Critic test_results Objects All Empty
- **Severity**: 🟡 HIGH
- **Evidence**: All 20 critic rules show `"test_results": {}`
  But logs show detailed test results:
  ```
  [X] Sample Size: 54 trades (need 100+)
  [OK] Time Period Bias: Profitable in 4/4 quarters
  ```
- **Impact**: Cannot programmatically analyze WHY rules passed/failed
- **Root Cause**: `critic.py` stores test results as objects, `run_discovery.py` doesn't serialize them
- **Fix**: In `run_discovery.py` around line 978, add test_results extraction:
  ```python
  'results': [
      {
          'rule_name': r.rule_name,
          'verdict': r.verdict,
          'tests_passed': r.tests_passed,
          'test_results': {k: {'passed': v.passed, 'value': v.value, 'details': v.details} 
                          for k, v in r.test_results.items()}
      } for r in critic_reports
  ]
  ```

## BUG 4: OOS/IS Sharpe Not Exported
- **Severity**: 🟡 HIGH
- **Evidence**: All rules show `oos_sharpe: null, oos_retention: null, is_sharpe: null`
  But logs show: `IS Sharpe: 81.17, OOS Sharpe: 61.98, Retention: 76.4%`
- **Impact**: Cannot validate OOS performance degradation programmatically
- **Root Cause**: Critic results have these fields but they're not being extracted to the stored dict
- **Fix**: In `run_discovery.py`, ensure critic report serialization includes:
  ```python
  'oos_sharpe': r.oos_sharpe,
  'is_sharpe': r.is_sharpe,
  'oos_retention': r.oos_retention
  ```

## BUG 5: Walk-Forward Results Empty
- **Severity**: 🟡 HIGH
- **Evidence**: `"walk_forward": []` but logs show:
  ```
  Period 0: train signals=26, test signals=8
  Period 1: train signals=30, test signals=12
  Period 2: train signals=24, test signals=8
  ```
- **Impact**: Cannot verify strategy stability across time
- **Root Cause**: `validation/framework.py` computes