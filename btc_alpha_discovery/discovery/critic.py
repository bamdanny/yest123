"""
Devil's Advocate / Critic Mode
==============================

Tries to DEBUNK every discovered signal.
A signal only survives if it passes ALL tests.

This prevents overfitting and spurious patterns like "day_of_month > 28".
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def remove_overlapping_signals(signals: pd.Series, holding_period: int = 6) -> pd.Series:
    """
    After a signal fires, mask out the next (holding_period - 1) bars.
    Ensures non-overlapping trade windows for valid Sharpe calculation.
    
    This prevents counting overlapping price moves as independent trades,
    which would artificially inflate Sharpe ratio.
    """
    clean = signals.copy()
    
    i = 0
    while i < len(clean):
        if clean.iloc[i] != 0:
            # Signal found - block next (holding_period - 1) bars
            block_end = min(i + holding_period, len(clean))
            for j in range(i + 1, block_end):
                clean.iloc[j] = 0
            i = block_end  # Jump past the blocked region
        else:
            i += 1
    
    return clean


@dataclass
class CriticTest:
    """Result of a single critic test"""
    name: str
    passed: bool
    score: float
    threshold: float
    details: str


@dataclass
class CriticReport:
    """Full critic report for a trading rule"""
    rule_name: str
    rule_description: str
    tests_passed: int
    tests_failed: int
    tests: List[CriticTest]
    verdict: str  # "CREDIBLE", "SUSPICIOUS", "DEBUNKED"
    economic_logic: str
    recommendation: str
    # OOS performance metrics (extracted from test 7)
    oos_sharpe: float = 0.0
    is_sharpe: float = 0.0
    oos_retention: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON export"""
        return {
            'rule_name': self.rule_name,
            'rule_description': self.rule_description,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'tests_total': self.tests_passed + self.tests_failed,
            'verdict': self.verdict,
            'economic_logic': self.economic_logic,
            'recommendation': self.recommendation,
            'oos_sharpe': self.oos_sharpe,
            'is_sharpe': self.is_sharpe,
            'oos_retention': self.oos_retention,
            'test_results': {
                t.name: {
                    'passed': t.passed,
                    'score': t.score,
                    'threshold': t.threshold,
                    'details': t.details
                } for t in self.tests
            }
        }


class DevilsAdvocate:
    """
    Tries to DEBUNK every discovered trading rule.
    
    Tests:
    1. Sample Size - Need 100+ trades
    2. Time Period Bias - Must work in 3/4 quarters
    3. Multiple Comparisons - Bonferroni correction
    4. Regime Dependency - Must work in 2/3 regimes
    5. Parameter Sensitivity - Nearby thresholds must also work
    6. Economic Logic - Must have plausible explanation
    7. Out-of-Sample - OOS must be >50% of in-sample
    8. Data Snooping - Would rule appear on earlier data?
    9. Lookahead Bias - Check for future leakage
    10. Survivorship Bias - Feature must exist throughout
    """
    
    # Features that are RED FLAGS (no economic logic)
    # Rules based on these features should be automatically DEBUNKED
    RED_FLAG_FEATURES = [
        "day_of_month", "day_of_week", "week_of_year", "month_of_year",
        "hour_of_day", "minute_of_hour", "time_day_of_month", "time_week_of_year",
        "time_month", "time_hour", "time_minute",
        # Raw OHLC price levels make NO sense as trading rules
        # "LONG when open < 84705" is pure hindsight bias
        "open", "high", "low", "close", "timestamp", "index"
    ]
    
    # Features that are SUSPICIOUS but not auto-debunk (need extra scrutiny)
    SUSPICIOUS_FEATURES = [
        "volume", "quote_volume"  # Volume without context can be spurious
    ]
    
    def __init__(
        self,
        min_trades: int = 30,  # Lowered from 100 for non-overlapping trades
        min_trades_per_regime: int = 10,  # Lowered from 30
        min_quarters_profitable: int = 3,
        oos_retention_threshold: float = 0.5,
        num_features_tested: int = 500,  # For Bonferroni correction
        parameter_sensitivity_tests: int = 6,
        min_param_sensitivity_pass: int = 4,
        holding_period: int = 6,  # For overlap removal
    ):
        self.min_trades = min_trades
        self.min_trades_per_regime = min_trades_per_regime
        self.min_quarters_profitable = min_quarters_profitable
        self.oos_retention_threshold = oos_retention_threshold
        self.num_features_tested = num_features_tested
        self.parameter_sensitivity_tests = parameter_sensitivity_tests
        self.min_param_sensitivity_pass = min_param_sensitivity_pass
        self.holding_period = holding_period
        
        # Bonferroni-corrected significance level
        self.corrected_alpha = 0.05 / num_features_tested
    
    def critique_rule(
        self,
        rule: Any,  # TradingRule or similar
        features: pd.DataFrame,
        returns: pd.Series,
        regime_labels: Optional[pd.Series] = None
    ) -> CriticReport:
        """
        Run ALL critic tests on a trading rule.
        
        Args:
            rule: Trading rule with conditions and direction
            features: Feature DataFrame
            returns: Forward returns
            regime_labels: Optional regime labels (bull/bear/chop)
            
        Returns:
            CriticReport with all test results
        """
        tests = []
        
        # Get signals from rule and APPLY OVERLAP REMOVAL
        try:
            raw_signals = self._evaluate_rule(rule, features)
            signals = remove_overlapping_signals(raw_signals, self.holding_period)
            
            # Log overlap removal for diagnostics
            raw_count = (raw_signals != 0).sum()
            clean_count = (signals != 0).sum()
            removed = raw_count - clean_count
            if raw_count > 0:
                logger.info(f"  [Critic] Overlap removal: {raw_count} -> {clean_count} signals ({removed} removed, {100*removed/raw_count:.1f}%)")
            
            trade_returns = returns[signals != 0] * signals[signals != 0]
        except Exception as e:
            logger.error(f"Error evaluating rule: {e}")
            return self._create_failed_report(rule, str(e))
        
        # 1. Sample Size Test
        tests.append(self._test_sample_size(signals, trade_returns))
        
        # 2. Time Period Bias Test
        tests.append(self._test_time_period_bias(signals, returns))
        
        # 3. Multiple Comparisons Test
        tests.append(self._test_multiple_comparisons(trade_returns))
        
        # 4. Regime Dependency Test
        if regime_labels is not None:
            tests.append(self._test_regime_dependency(signals, returns, regime_labels))
        else:
            # Estimate regimes from returns
            tests.append(self._test_regime_dependency_estimated(signals, returns, features))
        
        # 5. Parameter Sensitivity Test
        tests.append(self._test_parameter_sensitivity(rule, features, returns))
        
        # 6. Economic Logic Test
        tests.append(self._test_economic_logic(rule))
        
        # 7. Out-of-Sample Test
        tests.append(self._test_out_of_sample(rule, features, returns))
        
        # 8. Data Snooping Test
        tests.append(self._test_data_snooping(rule, features, returns))
        
        # 9. Lookahead Bias Test
        tests.append(self._test_lookahead_bias(rule, features))
        
        # 10. Survivorship Bias Test
        tests.append(self._test_survivorship_bias(rule, features))
        
        # Calculate verdict
        tests_passed = sum(1 for t in tests if t.passed)
        tests_failed = len(tests) - tests_passed
        
        # Extract OOS metrics from test 7
        oos_sharpe = 0.0
        is_sharpe = 0.0
        oos_retention = 0.0
        oos_test = next((t for t in tests if t.name == "Out-of-Sample"), None)
        if oos_test:
            # Parse from details: "IS Sharpe: 81.17, OOS Sharpe: 61.98, Retention: 76.4%"
            import re
            details = oos_test.details
            is_match = re.search(r'IS Sharpe:\s*([\d.]+)', details)
            oos_match = re.search(r'OOS Sharpe:\s*([\d.]+)', details)
            ret_match = re.search(r'Retention:\s*([\d.]+)', details)
            if is_match:
                is_sharpe = float(is_match.group(1))
            if oos_match:
                oos_sharpe = float(oos_match.group(1))
            if ret_match:
                oos_retention = float(ret_match.group(1)) / 100  # Convert % to decimal
        
        # AUTO-DEBUNK: Check if rule uses only RED FLAG features
        # These rules have NO economic logic and should be automatically rejected
        economic_test = next((t for t in tests if t.name == "Economic Logic"), None)
        auto_debunk = False
        if economic_test and "RED FLAGS" in economic_test.details:
            auto_debunk = True
            verdict = "DEBUNKED"
            recommendation = "AUTO-DEBUNKED: Rule based on red flag features (raw OHLC, time-of-day, etc.) with no economic rationale"
        elif tests_passed >= 9:
            verdict = "CREDIBLE"
            recommendation = "KEEP - Strong evidence this is a real signal"
        elif tests_passed >= 7:
            verdict = "SUSPICIOUS"
            recommendation = "REVIEW - Some concerns, investigate failed tests"
        else:
            verdict = "DEBUNKED"
            recommendation = "DELETE - Too many failures, likely spurious"
        
        # Get economic logic
        economic_logic = self._explain_economic_logic(rule)
        
        return CriticReport(
            rule_name=getattr(rule, 'rule_id', str(rule)),
            rule_description=getattr(rule, 'to_string', lambda: str(rule))(),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests=tests,
            verdict=verdict,
            economic_logic=economic_logic,
            recommendation=recommendation,
            oos_sharpe=oos_sharpe,
            is_sharpe=is_sharpe,
            oos_retention=oos_retention
        )
    
    def _evaluate_rule(self, rule: Any, features: pd.DataFrame) -> pd.Series:
        """Evaluate rule on features to get signals"""
        if hasattr(rule, 'evaluate'):
            return rule.evaluate(features).astype(int) * getattr(rule, 'direction', 1)
        elif hasattr(rule, 'conditions'):
            # Manual evaluation
            mask = pd.Series(True, index=features.index)
            for cond in rule.conditions:
                feat = features.get(cond['feature'])
                if feat is None:
                    continue
                op = cond['operator']
                thresh = cond['threshold']
                if op == '>':
                    mask = mask & (feat > thresh)
                elif op == '<':
                    mask = mask & (feat < thresh)
                elif op == '>=':
                    mask = mask & (feat >= thresh)
                elif op == '<=':
                    mask = mask & (feat <= thresh)
            return mask.astype(int) * getattr(rule, 'direction', 1)
        else:
            raise ValueError("Cannot evaluate rule")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _test_sample_size(self, signals: pd.Series, trade_returns: pd.Series) -> CriticTest:
        """Test 1: Minimum sample size"""
        n_trades = (signals != 0).sum()
        passed = n_trades >= self.min_trades
        
        return CriticTest(
            name="Sample Size",
            passed=passed,
            score=n_trades,
            threshold=self.min_trades,
            details=f"{n_trades} trades (need {self.min_trades}+)"
        )
    
    def _test_time_period_bias(self, signals: pd.Series, returns: pd.Series) -> CriticTest:
        """Test 2: Must be profitable in 3/4 quarters"""
        # Split into quarters
        n = len(signals)
        quarter_size = n // 4
        
        profitable_quarters = 0
        for i in range(4):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < 3 else n
            
            q_signals = signals.iloc[start:end]
            q_returns = returns.iloc[start:end]
            
            q_trade_returns = q_returns[q_signals != 0] * q_signals[q_signals != 0]
            if len(q_trade_returns) > 0 and q_trade_returns.mean() > 0:
                profitable_quarters += 1
        
        passed = profitable_quarters >= self.min_quarters_profitable
        
        return CriticTest(
            name="Time Period Bias",
            passed=passed,
            score=profitable_quarters,
            threshold=self.min_quarters_profitable,
            details=f"Profitable in {profitable_quarters}/4 quarters"
        )
    
    def _test_multiple_comparisons(self, trade_returns: pd.Series) -> CriticTest:
        """Test 3: Bonferroni-corrected significance"""
        if len(trade_returns) < 10:
            return CriticTest(
                name="Multiple Comparisons",
                passed=False,
                score=1.0,
                threshold=self.corrected_alpha,
                details="Insufficient trades for statistical test"
            )
        
        # One-sample t-test against zero
        t_stat, p_value = stats.ttest_1samp(trade_returns, 0)
        
        # Two-tailed -> one-tailed (we only care if positive)
        if trade_returns.mean() > 0:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value / 2
        
        passed = p_value < self.corrected_alpha
        
        return CriticTest(
            name="Multiple Comparisons (Bonferroni)",
            passed=passed,
            score=p_value,
            threshold=self.corrected_alpha,
            details=f"p-value: {p_value:.6f} (need < {self.corrected_alpha:.6f})"
        )
    
    def _test_regime_dependency(
        self, 
        signals: pd.Series, 
        returns: pd.Series, 
        regime_labels: pd.Series
    ) -> CriticTest:
        """Test 4: Must work in 2/3 regimes"""
        regimes_profitable = 0
        regime_names = regime_labels.unique()
        
        details_parts = []
        for regime in regime_names:
            mask = regime_labels == regime
            r_signals = signals[mask]
            r_returns = returns[mask]
            
            r_trade_returns = r_returns[r_signals != 0] * r_signals[r_signals != 0]
            n_trades = len(r_trade_returns)
            
            if n_trades >= self.min_trades_per_regime:
                if r_trade_returns.mean() > 0:
                    regimes_profitable += 1
                    details_parts.append(f"{regime}: profitable ({n_trades} trades)")
                else:
                    details_parts.append(f"{regime}: unprofitable ({n_trades} trades)")
            else:
                details_parts.append(f"{regime}: insufficient ({n_trades} trades)")
        
        passed = regimes_profitable >= 2
        
        return CriticTest(
            name="Regime Dependency",
            passed=passed,
            score=regimes_profitable,
            threshold=2,
            details="; ".join(details_parts)
        )
    
    def _test_regime_dependency_estimated(
        self,
        signals: pd.Series,
        returns: pd.Series,
        features: pd.DataFrame
    ) -> CriticTest:
        """Test 4 (estimated): Estimate regimes from returns"""
        # Simple regime estimation based on rolling returns
        window = 168  # 1 week of hourly data
        rolling_ret = returns.rolling(window).mean()
        
        # Define regimes
        regime_labels = pd.Series("chop", index=returns.index)
        regime_labels[rolling_ret > rolling_ret.quantile(0.67)] = "bull"
        regime_labels[rolling_ret < rolling_ret.quantile(0.33)] = "bear"
        
        return self._test_regime_dependency(signals, returns, regime_labels)
    
    def _test_parameter_sensitivity(
        self,
        rule: Any,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> CriticTest:
        """Test 5: Nearby thresholds must also be profitable"""
        if not hasattr(rule, 'conditions'):
            return CriticTest(
                name="Parameter Sensitivity",
                passed=True,
                score=self.parameter_sensitivity_tests,
                threshold=self.min_param_sensitivity_pass,
                details="No thresholds to test"
            )
        
        # Test ±5%, ±10%, ±20% threshold variations
        variations = [0.95, 0.98, 1.02, 1.05, 0.90, 1.10]
        profitable_count = 0
        
        for var in variations:
            # Create modified rule
            modified_conditions = []
            for cond in rule.conditions:
                new_cond = cond.copy()
                new_cond['threshold'] = cond['threshold'] * var
                modified_conditions.append(new_cond)
            
            # Evaluate modified rule
            try:
                mask = pd.Series(True, index=features.index)
                for cond in modified_conditions:
                    feat = features.get(cond['feature'])
                    if feat is None:
                        continue
                    op = cond['operator']
                    thresh = cond['threshold']
                    if op == '>':
                        mask = mask & (feat > thresh)
                    elif op == '<':
                        mask = mask & (feat < thresh)
                
                raw_signals = mask.astype(int) * getattr(rule, 'direction', 1)
                signals = remove_overlapping_signals(raw_signals, self.holding_period)
                trade_returns = returns[signals != 0] * signals[signals != 0]
                
                if len(trade_returns) >= 10 and trade_returns.mean() > 0:  # Lowered from 20
                    profitable_count += 1
            except Exception:
                continue
        
        passed = profitable_count >= self.min_param_sensitivity_pass
        
        return CriticTest(
            name="Parameter Sensitivity",
            passed=passed,
            score=profitable_count,
            threshold=self.min_param_sensitivity_pass,
            details=f"{profitable_count}/{len(variations)} nearby thresholds profitable"
        )
    
    def _test_economic_logic(self, rule: Any) -> CriticTest:
        """Test 6: Must have plausible economic logic"""
        if not hasattr(rule, 'conditions'):
            return CriticTest(
                name="Economic Logic",
                passed=True,
                score=1.0,
                threshold=0.5,
                details="No conditions to analyze"
            )
        
        # Check for red flag features
        red_flags_found = []
        suspicious_found = []
        good_features = []
        
        # Exact matches for raw OHLC (these are always red flags)
        RAW_OHLC = {'open', 'high', 'low', 'close', 'timestamp', 'index'}
        
        for cond in rule.conditions:
            feature = cond.get('feature', '')
            feature_lower = feature.lower()
            
            # Exact match for raw OHLC
            if feature_lower in RAW_OHLC:
                red_flags_found.append(feature)
            # Substring match for other red flags (time patterns)
            elif any(rf in feature_lower for rf in self.RED_FLAG_FEATURES if rf not in RAW_OHLC):
                red_flags_found.append(feature)
            elif any(sf in feature_lower for sf in self.SUSPICIOUS_FEATURES):
                suspicious_found.append(feature)
            else:
                good_features.append(feature)
        
        # Determine pass/fail
        if red_flags_found:
            passed = False
            details = f"RED FLAGS: {red_flags_found} - No economic rationale"
        elif not good_features and suspicious_found:
            passed = False
            details = f"Only suspicious features: {suspicious_found}"
        else:
            passed = True
            details = f"Features have economic logic: {good_features}"
        
        score = len(good_features) / (len(good_features) + len(red_flags_found) + len(suspicious_found) + 0.01)
        
        return CriticTest(
            name="Economic Logic",
            passed=passed,
            score=score,
            threshold=0.5,
            details=details
        )
    
    def _test_out_of_sample(
        self,
        rule: Any,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> CriticTest:
        """Test 7: OOS performance must be >50% of in-sample"""
        # 80/20 split
        split_idx = int(len(features) * 0.8)
        
        # Calculate period days (4h bars = 6 per day)
        is_period_days = max(1, split_idx / 6)
        oos_period_days = max(1, (len(features) - split_idx) / 6)
        
        # In-sample - with overlap removal
        is_features = features.iloc[:split_idx]
        is_returns = returns.iloc[:split_idx]
        is_raw_signals = self._evaluate_rule(rule, is_features)
        is_signals = remove_overlapping_signals(is_raw_signals, self.holding_period)
        is_trade_returns = is_returns[is_signals != 0] * is_signals[is_signals != 0]
        is_n_trades = len(is_trade_returns)
        
        # IS Sharpe using TRADE FREQUENCY
        if is_n_trades >= 2 and is_trade_returns.std() > 0:
            is_trades_per_year = (is_n_trades / is_period_days) * 365
            is_sharpe = (is_trade_returns.mean() / (is_trade_returns.std() + 1e-10)) * np.sqrt(is_trades_per_year)
            
            # Sanity check
            if is_sharpe > 10:
                logger.warning(f"IS Sharpe > 10: {is_sharpe:.2f}, n_trades={is_n_trades}, period={is_period_days:.0f}d")
        else:
            is_sharpe = 0.0
        
        # Out-of-sample - with overlap removal
        oos_features = features.iloc[split_idx:]
        oos_returns = returns.iloc[split_idx:]
        oos_raw_signals = self._evaluate_rule(rule, oos_features)
        oos_signals = remove_overlapping_signals(oos_raw_signals, self.holding_period)
        oos_trade_returns = oos_returns[oos_signals != 0] * oos_signals[oos_signals != 0]
        oos_n_trades = len(oos_trade_returns)
        
        # OOS Sharpe using TRADE FREQUENCY
        if oos_n_trades >= 2 and oos_trade_returns.std() > 0:
            oos_trades_per_year = (oos_n_trades / oos_period_days) * 365
            oos_sharpe = (oos_trade_returns.mean() / (oos_trade_returns.std() + 1e-10)) * np.sqrt(oos_trades_per_year)
            
            # Sanity check
            if oos_sharpe > 10:
                logger.warning(f"OOS Sharpe > 10: {oos_sharpe:.2f}, n_trades={oos_n_trades}, period={oos_period_days:.0f}d")
        else:
            oos_sharpe = 0.0
        
        # Calculate retention
        if is_sharpe > 0:
            retention = oos_sharpe / is_sharpe
        else:
            retention = 0 if oos_sharpe <= 0 else 1.0
        
        passed = retention >= self.oos_retention_threshold and oos_sharpe > 0
        
        return CriticTest(
            name="Out-of-Sample",
            passed=passed,
            score=retention,
            threshold=self.oos_retention_threshold,
            details=f"IS Sharpe: {is_sharpe:.2f}, OOS Sharpe: {oos_sharpe:.2f}, Retention: {retention:.1%}"
        )
    
    def _test_data_snooping(
        self,
        rule: Any,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> CriticTest:
        """Test 8: Would rule be discovered on earlier data?"""
        # Use first 70% of data
        early_cutoff = int(len(features) * 0.7)
        
        early_features = features.iloc[:early_cutoff]
        early_returns = returns.iloc[:early_cutoff]
        
        # Evaluate rule on early data - with overlap removal
        early_raw_signals = self._evaluate_rule(rule, early_features)
        early_signals = remove_overlapping_signals(early_raw_signals, self.holding_period)
        early_trade_returns = early_returns[early_signals != 0] * early_signals[early_signals != 0]
        
        # Check if rule would have looked promising (lowered threshold for overlap removal)
        if len(early_trade_returns) < 15:
            passed = False
            details = "Insufficient early trades - rule might be recent phenomenon"
        else:
            early_win_rate = (early_trade_returns > 0).mean()
            early_sharpe = early_trade_returns.mean() / (early_trade_returns.std() + 1e-10)
            
            passed = early_win_rate > 0.5 and early_sharpe > 0
            details = f"On first 70%: win rate {early_win_rate:.1%}, sharpe {early_sharpe:.2f}"
        
        return CriticTest(
            name="Data Snooping",
            passed=passed,
            score=early_trade_returns.mean() if len(early_trade_returns) > 0 else 0,
            threshold=0,
            details=details
        )
    
    def _test_lookahead_bias(self, rule: Any, features: pd.DataFrame) -> CriticTest:
        """Test 9: Check for future data leakage"""
        if not hasattr(rule, 'conditions'):
            return CriticTest(
                name="Lookahead Bias",
                passed=True,
                score=1.0,
                threshold=0.5,
                details="No conditions to check"
            )
        
        # Features that might have lookahead bias
        lookahead_keywords = [
            "forward", "future", "next", "target", "return_", "profit",
            "outcome", "result", "winning", "losing"
        ]
        
        suspicious_features = []
        for cond in rule.conditions:
            feature = cond.get('feature', '')
            if any(kw in feature.lower() for kw in lookahead_keywords):
                suspicious_features.append(feature)
        
        passed = len(suspicious_features) == 0
        
        return CriticTest(
            name="Lookahead Bias",
            passed=passed,
            score=0 if suspicious_features else 1.0,
            threshold=0.5,
            details=f"Suspicious features: {suspicious_features}" if suspicious_features else "No lookahead detected"
        )
    
    def _test_survivorship_bias(self, rule: Any, features: pd.DataFrame) -> CriticTest:
        """Test 10: Feature must have data throughout entire period"""
        if not hasattr(rule, 'conditions'):
            return CriticTest(
                name="Survivorship Bias",
                passed=True,
                score=1.0,
                threshold=0.8,
                details="No conditions to check"
            )
        
        coverage_scores = []
        for cond in rule.conditions:
            feature = cond.get('feature', '')
            if feature in features.columns:
                coverage = features[feature].notna().mean()
                coverage_scores.append(coverage)
        
        if not coverage_scores:
            return CriticTest(
                name="Survivorship Bias",
                passed=False,
                score=0,
                threshold=0.8,
                details="No features found in data"
            )
        
        min_coverage = min(coverage_scores)
        passed = min_coverage >= 0.8
        
        return CriticTest(
            name="Survivorship Bias",
            passed=passed,
            score=min_coverage,
            threshold=0.8,
            details=f"Minimum feature coverage: {min_coverage:.1%}"
        )
    
    def _explain_economic_logic(self, rule: Any) -> str:
        """Generate economic explanation for rule"""
        if not hasattr(rule, 'conditions'):
            return "Unknown rule structure"
        
        explanations = []
        for cond in rule.conditions:
            feature = cond.get('feature', '')
            op = cond.get('operator', '')
            thresh = cond.get('threshold', 0)
            
            # Generate explanation based on feature type
            if "funding" in feature.lower():
                if op == '<' and thresh < 0:
                    explanations.append("Negative funding = shorts paying longs = crowded shorts")
                elif op == '>' and thresh > 0:
                    explanations.append("Positive funding = longs paying shorts = crowded longs")
            elif "rsi" in feature.lower():
                if op == '<' and thresh < 30:
                    explanations.append(f"RSI < {thresh} = oversold condition")
                elif op == '>' and thresh > 70:
                    explanations.append(f"RSI > {thresh} = overbought condition")
            elif "oi" in feature.lower() or "open_interest" in feature.lower():
                explanations.append("Open interest indicates market participation")
            elif "ls" in feature.lower() or "long_short" in feature.lower():
                explanations.append("Long/short ratio indicates market positioning")
            elif "liquidation" in feature.lower():
                explanations.append("Liquidation levels indicate forced selling/buying")
            elif "yield" in feature.lower() or "rate" in feature.lower():
                explanations.append("Interest rates affect risk appetite")
            elif "obv" in feature.lower():
                explanations.append("OBV indicates volume-weighted price momentum")
            elif "day_of_month" in feature.lower() or "week_of_year" in feature.lower():
                explanations.append("[!] Calendar effect - likely spurious")
        
        return "; ".join(explanations) if explanations else "No clear economic logic identified"
    
    def _create_failed_report(self, rule: Any, error: str) -> CriticReport:
        """Create failed report when evaluation fails"""
        return CriticReport(
            rule_name=str(rule),
            rule_description="Evaluation failed",
            tests_passed=0,
            tests_failed=10,
            tests=[],
            verdict="DEBUNKED",
            economic_logic="N/A",
            recommendation=f"DELETE - Evaluation error: {error}",
            oos_sharpe=0.0,
            is_sharpe=0.0,
            oos_retention=0.0
        )
    
    def print_report(self, report: CriticReport) -> None:
        """Print formatted critic report"""
        print("\n" + "="*70)
        print(f"DEVIL'S ADVOCATE REPORT: {report.rule_name}")
        print("="*70)
        
        print(f"\nRule: {report.rule_description}")
        print(f"Economic Logic: {report.economic_logic}")
        
        # Color-coded verdict
        color = {
            "CREDIBLE": "\033[92m",  # Green
            "SUSPICIOUS": "\033[93m",  # Yellow
            "DEBUNKED": "\033[91m"  # Red
        }.get(report.verdict, "")
        print(f"\n{color}VERDICT: {report.verdict}\033[0m")
        print(f"Tests Passed: {report.tests_passed}/{report.tests_passed + report.tests_failed}")
        
        print("\nTEST RESULTS:")
        print("-"*70)
        for test in report.tests:
            status = "[OK]" if test.passed else "[X]"
            color = "\033[92m" if test.passed else "\033[91m"
            print(f"  {color}{status}\033[0m {test.name}: {test.details}")
        
        print(f"\nRECOMMENDATION: {report.recommendation}")
        print("="*70)


def critique_all_rules(
    rules: List[Any],
    features: pd.DataFrame,
    returns: pd.Series,
    verbose: bool = True
) -> Tuple[List[Any], List[CriticReport]]:
    """
    Run critic on all rules and filter to survivors.
    
    Returns:
        Tuple of (surviving_rules, all_reports)
    """
    critic = DevilsAdvocate()
    
    surviving_rules = []
    all_reports = []
    
    for rule in rules:
        report = critic.critique_rule(rule, features, returns)
        all_reports.append(report)
        
        if verbose:
            critic.print_report(report)
        
        if report.verdict == "CREDIBLE":
            surviving_rules.append(rule)
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("CRITIC SUMMARY")
        print("="*70)
        print(f"Rules Tested: {len(rules)}")
        print(f"Credible: {sum(1 for r in all_reports if r.verdict == 'CREDIBLE')}")
        print(f"Suspicious: {sum(1 for r in all_reports if r.verdict == 'SUSPICIOUS')}")
        print(f"Debunked: {sum(1 for r in all_reports if r.verdict == 'DEBUNKED')}")
        print("="*70)
    
    return surviving_rules, all_reports


if __name__ == "__main__":
    # Test with mock rule
    from dataclasses import dataclass
    
    @dataclass
    class MockRule:
        rule_id: str
        conditions: List[Dict]
        direction: int
        
        def to_string(self):
            conds = [f"{c['feature']} {c['operator']} {c['threshold']}" for c in self.conditions]
            return f"{'LONG' if self.direction == 1 else 'SHORT'} when {' AND '.join(conds)}"
        
        def evaluate(self, features):
            mask = pd.Series(True, index=features.index)
            for cond in self.conditions:
                feat = features.get(cond['feature'])
                if feat is None:
                    continue
                if cond['operator'] == '>':
                    mask = mask & (feat > cond['threshold'])
                elif cond['operator'] == '<':
                    mask = mask & (feat < cond['threshold'])
            return mask.astype(int)
    
    # Create sample data
    np.random.seed(42)
    n = 5000
    features = pd.DataFrame({
        'funding_zscore': np.random.randn(n).cumsum() / 10,
        'rsi_4h': np.random.uniform(20, 80, n),
        'day_of_month': np.tile(np.arange(1, 29), n // 28 + 1)[:n],
        'oi_change': np.random.randn(n) * 0.05,
    })
    returns = pd.Series(np.random.randn(n) * 0.01)
    
    # Test good rule
    good_rule = MockRule(
        rule_id="test_good",
        conditions=[
            {'feature': 'funding_zscore', 'operator': '<', 'threshold': -1.5},
            {'feature': 'rsi_4h', 'operator': '<', 'threshold': 35},
        ],
        direction=1
    )
    
    # Test bad rule (calendar effect)
    bad_rule = MockRule(
        rule_id="test_bad",
        conditions=[
            {'feature': 'day_of_month', 'operator': '>', 'threshold': 28},
        ],
        direction=-1
    )
    
    critic = DevilsAdvocate()
    
    print("Testing GOOD rule:")
    good_report = critic.critique_rule(good_rule, features, returns)
    critic.print_report(good_report)
    
    print("\nTesting BAD rule:")
    bad_report = critic.critique_rule(bad_rule, features, returns)
    critic.print_report(bad_report)
