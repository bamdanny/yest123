#!/usr/bin/env python3
"""
Test script to verify the corrected implementations.

Run this first to make sure the math is correct before using real data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.metrics import (
    calculate_sharpe,
    calculate_cumulative_return,
    calculate_win_rate,
    calculate_max_drawdown,
    validate_metrics,
    calculate_all_metrics
)
from ml.backtester import Backtester


def test_sharpe():
    """Test Sharpe ratio calculation."""
    print("\n=== Testing Sharpe Calculation ===")
    
    # Known case: constant positive returns
    returns = np.array([0.01] * 100)  # 1% every bar
    sharpe = calculate_sharpe(returns, annualize=True)
    
    # With constant returns, std = 0, should return 0 (not inf)
    assert sharpe == 0 or np.isclose(sharpe, 0), f"Constant returns should give 0 Sharpe, got {sharpe}"
    print(f"  ✓ Constant returns: Sharpe = {sharpe}")
    
    # Random returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    sharpe = calculate_sharpe(returns, annualize=True)
    
    assert -10 < sharpe < 10, f"Random returns Sharpe should be reasonable, got {sharpe}"
    print(f"  ✓ Random returns: Sharpe = {sharpe:.2f}")
    
    # Edge case: empty
    sharpe = calculate_sharpe(np.array([]))
    assert sharpe == 0, f"Empty returns should give 0 Sharpe, got {sharpe}"
    print(f"  ✓ Empty returns: Sharpe = {sharpe}")
    
    print("  ✓ Sharpe tests PASSED")


def test_cumulative_return():
    """Test cumulative return calculation."""
    print("\n=== Testing Cumulative Return ===")
    
    # 10% then -10% should not be 0 (compound)
    returns = np.array([0.10, -0.10])
    cum_return = calculate_cumulative_return(returns)
    expected = (1.10 * 0.90) - 1  # -0.01 = -1%
    
    assert np.isclose(cum_return, expected), f"Expected {expected}, got {cum_return}"
    print(f"  ✓ 10% then -10%: Return = {cum_return:.2%} (compound, not 0%)")
    
    # All positive
    returns = np.array([0.01, 0.02, 0.03])
    cum_return = calculate_cumulative_return(returns)
    expected = (1.01 * 1.02 * 1.03) - 1
    
    assert np.isclose(cum_return, expected), f"Expected {expected}, got {cum_return}"
    print(f"  ✓ Positive returns: {cum_return:.2%}")
    
    # Should never go below -100% without leverage
    returns = np.array([-0.50, -0.50])  # Two 50% losses
    cum_return = calculate_cumulative_return(returns)
    
    assert cum_return > -1, f"Return {cum_return} should be > -100%"
    print(f"  ✓ Two -50% losses: {cum_return:.2%} (not -100%)")
    
    print("  ✓ Cumulative return tests PASSED")


def test_win_rate():
    """Test win rate calculation."""
    print("\n=== Testing Win Rate ===")
    
    # 7 wins, 3 losses
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, -0.01, 0.02])
    win_rate, n_wins, n_losses = calculate_win_rate(returns)
    
    assert n_wins == 7, f"Expected 7 wins, got {n_wins}"
    assert n_losses == 3, f"Expected 3 losses, got {n_losses}"
    assert np.isclose(win_rate, 0.7), f"Expected 70%, got {win_rate:.1%}"
    print(f"  ✓ 7W/3L: Win rate = {win_rate:.1%}")
    
    # With zeros (no trade)
    returns = np.array([0.01, 0, -0.01, 0, 0.02])
    win_rate, n_wins, n_losses = calculate_win_rate(returns)
    
    assert n_wins == 2, f"Expected 2 wins, got {n_wins}"
    assert n_losses == 1, f"Expected 1 loss, got {n_losses}"
    print(f"  ✓ With zeros: Win rate = {win_rate:.1%} ({n_wins}W/{n_losses}L)")
    
    # Edge case: all zeros
    returns = np.array([0, 0, 0])
    win_rate, n_wins, n_losses = calculate_win_rate(returns)
    
    assert win_rate == 0, f"All zeros should give 0%, got {win_rate}"
    print(f"  ✓ All zeros: Win rate = {win_rate:.1%}")
    
    print("  ✓ Win rate tests PASSED")


def test_max_drawdown():
    """Test max drawdown calculation."""
    print("\n=== Testing Max Drawdown ===")
    
    # Known drawdown
    returns = np.array([0.10, -0.20, 0.05])  # Up 10%, down 20%, up 5%
    max_dd = calculate_max_drawdown(returns)
    
    # Peak after bar 1: 1.10
    # Trough after bar 2: 1.10 * 0.80 = 0.88
    # DD = (0.88 - 1.10) / 1.10 = -0.20
    expected = -0.20
    
    assert np.isclose(max_dd, expected, atol=0.01), f"Expected ~{expected}, got {max_dd}"
    print(f"  ✓ Known case: Max DD = {max_dd:.1%}")
    
    # All positive (no drawdown)
    returns = np.array([0.01, 0.02, 0.03])
    max_dd = calculate_max_drawdown(returns)
    
    assert max_dd == 0, f"All positive should have 0 drawdown, got {max_dd}"
    print(f"  ✓ All positive: Max DD = {max_dd:.1%}")
    
    print("  ✓ Max drawdown tests PASSED")


def test_backtester():
    """Test full backtester."""
    print("\n=== Testing Backtester ===")
    
    # Simulate 100 bars
    np.random.seed(42)
    n = 100
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    
    # Generate some signals (alternating with gaps)
    signals = np.zeros(n)
    signals[10:20] = 1   # Long for 10 bars
    signals[30:40] = -1  # Short for 10 bars
    signals[50:60] = 1   # Long again
    signals[70:80] = -1  # Short again
    
    backtester = Backtester()
    result = backtester.run(signals, prices)
    
    # Validate
    valid, errors = backtester.validate_result(result)
    
    print(f"  Trades: {result.n_trades} (L:{result.n_long}, S:{result.n_short})")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Return: {result.total_return:.2%}")
    print(f"  Sharpe: {result.sharpe:.2f}")
    print(f"  Max DD: {result.max_drawdown:.1%}")
    
    if valid:
        print("  ✓ Validation PASSED")
    else:
        print("  ✗ Validation FAILED:")
        for e in errors:
            print(f"    - {e}")
    
    # Basic sanity
    assert result.n_long == 20, f"Expected 20 long bars, got {result.n_long}"
    assert result.n_short == 20, f"Expected 20 short bars, got {result.n_short}"
    assert 0 <= result.win_rate <= 1, f"Win rate {result.win_rate} outside [0,1]"
    assert result.total_return > -1, f"Return {result.total_return} < -100%"
    
    print("  ✓ Backtester tests PASSED")


def test_validation():
    """Test metric validation catches bugs."""
    print("\n=== Testing Validation ===")
    
    # Good metrics
    good_metrics = {
        'sharpe': 2.5,
        'win_rate': 0.60,
        'total_return': 0.25,
        'n_trades': 50,
        'n_long': 25,
        'n_short': 25
    }
    valid, errors = validate_metrics(good_metrics)
    assert valid, f"Good metrics should pass: {errors}"
    print("  ✓ Good metrics pass validation")
    
    # Bad Sharpe
    bad_metrics = good_metrics.copy()
    bad_metrics['sharpe'] = 1000
    valid, errors = validate_metrics(bad_metrics)
    assert not valid, "Sharpe 1000 should fail"
    print(f"  ✓ Catches impossible Sharpe: {errors[0]}")
    
    # Bad return
    bad_metrics = good_metrics.copy()
    bad_metrics['total_return'] = -1.5  # -150%
    valid, errors = validate_metrics(bad_metrics)
    assert not valid, "Return -150% should fail"
    print(f"  ✓ Catches impossible return: {errors[0]}")
    
    # 0% win rate with trades
    bad_metrics = good_metrics.copy()
    bad_metrics['win_rate'] = 0
    bad_metrics['n_trades'] = 100
    valid, errors = validate_metrics(bad_metrics)
    assert not valid, "0% win rate with 100 trades should fail"
    print(f"  ✓ Catches suspicious win rate: {errors[0]}")
    
    print("  ✓ Validation tests PASSED")


def main():
    print("=" * 60)
    print("TESTING CORRECTED IMPLEMENTATIONS")
    print("=" * 60)
    
    test_sharpe()
    test_cumulative_return()
    test_win_rate()
    test_max_drawdown()
    test_backtester()
    test_validation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nThe math is correct. You can now use the implementations.")
    print("Run: python simple_rules_correct.py")


if __name__ == "__main__":
    main()
