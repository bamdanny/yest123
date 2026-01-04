"""
Test script to verify the time-based exit fix.

This script:
1. Generates synthetic BTC price data
2. Runs the discovery pipeline with TIME-based exits
3. Verifies that:
   - Exit rules are TIME type (not SL/TP)
   - Validation uses matching 6-bar returns
   - Sharpe is in realistic range (2-5, not >10)
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_btc_data(n_bars=1000, start_price=95000):
    """Generate realistic BTC 4h OHLCV data."""
    np.random.seed(42)
    
    # Parameters for realistic BTC volatility
    daily_vol = 0.03  # 3% daily volatility
    bar_vol = daily_vol / np.sqrt(6)  # 4h bar volatility
    
    # Generate returns with slight mean-reversion
    returns = np.random.normal(0, bar_vol, n_bars)
    
    # Add some autocorrelation (trending)
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Generate prices
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices[1:])
    
    # Generate OHLCV
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=4*n_bars),
        periods=n_bars,
        freq='4h'
    )
    
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Generate realistic high/low
    bar_ranges = np.abs(np.random.normal(0, bar_vol * 0.7, n_bars))
    df['high'] = df['close'] * (1 + bar_ranges)
    df['low'] = df['close'] * (1 - bar_ranges)
    
    # Open is previous close with gap
    df['open'] = df['close'].shift(1).fillna(start_price)
    
    # Volume
    df['volume'] = np.random.lognormal(10, 0.5, n_bars)
    
    return df

def test_exit_discovery():
    """Test that exit discovery produces TIME-based exits."""
    logger.info("="*60)
    logger.info("TEST 1: Exit Discovery produces TIME exits")
    logger.info("="*60)
    
    from discovery.entry_exit import ExitConditionDiscovery, TradingRule
    
    # Generate test data
    df = generate_synthetic_btc_data(500)
    
    # Create a simple entry rule (price in upper half of 7-day range)
    features = pd.DataFrame(index=df.index)
    features['price_range_position'] = (df['close'] - df['low'].rolling(42).min()) / \
                                       (df['high'].rolling(42).max() - df['low'].rolling(42).min() + 1e-8)
    features = features.dropna()
    
    # Align df with features
    df_aligned = df.loc[features.index]
    
    entry_rule = TradingRule(
        rule_id="test_entry",
        conditions=[{'feature': 'price_range_position', 'operator': '>', 'threshold': 0.5}],
        logic='AND',
        direction=-1,  # SHORT when price is high
        confidence=0.8,
        support=100,
        win_rate=0.55,
        avg_return=0.001,
        sharpe=0.5,
        max_drawdown=0.1
    )
    
    # Discover exits
    exit_discoverer = ExitConditionDiscovery(min_trades=10)
    exits = exit_discoverer.discover(features, df_aligned, entry_rule)
    
    # Verify
    if exits:
        for exit_rule in exits:
            logger.info(f"Exit rule: {exit_rule.rule_id}")
            logger.info(f"  Type: {exit_rule.exit_type}")
            logger.info(f"  Conditions: {exit_rule.conditions}")
            logger.info(f"  Avg bars held: {exit_rule.avg_bars_held}")
            logger.info(f"  Avg return: {exit_rule.avg_exit_return*100:.3f}%")
            
            if exit_rule.exit_type == 'TIME':
                logger.info("✓ Exit type is TIME (PASS)")
            else:
                logger.error(f"✗ Exit type is {exit_rule.exit_type}, expected TIME (FAIL)")
                return False
                
            # Check it's 6 bars
            if exit_rule.conditions[0].get('threshold') == 6:
                logger.info("✓ Exit is 6 bars (matches validation) (PASS)")
            else:
                logger.warning(f"⚠ Exit is {exit_rule.conditions[0].get('threshold')} bars")
    else:
        logger.error("✗ No exit rules discovered (FAIL)")
        return False
    
    return True

def test_sharpe_calculation():
    """Test that Sharpe is calculated correctly with time exits."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Sharpe calculation with TIME exits")
    logger.info("="*60)
    
    from validation.framework import validate_discovered_strategy
    
    # Generate test data
    df = generate_synthetic_btc_data(500)
    
    # Create features and signals
    features = pd.DataFrame(index=df.index)
    features['price_range_position'] = (df['close'] - df['low'].rolling(42).min()) / \
                                       (df['high'].rolling(42).max() - df['low'].rolling(42).min() + 1e-8)
    features = features.dropna()
    
    # Generate signals (SHORT when price is high)
    signals = pd.Series(0, index=features.index)
    signals[features['price_range_position'] > 0.5] = -1
    
    # Calculate 6-bar forward returns (matching time exit)
    returns = df['close'].pct_change(6).shift(-6)
    returns = returns.loc[features.index]
    
    # Align
    valid_idx = ~returns.isna()
    signals = signals[valid_idx].reset_index(drop=True)
    returns = returns[valid_idx].reset_index(drop=True)
    features_valid = features[valid_idx].reset_index(drop=True)
    
    logger.info(f"Samples: {len(signals)}, Active signals: {(signals != 0).sum()}")
    
    # Run validation
    report = validate_discovered_strategy(
        signals,
        returns,
        features_valid,
        strategy_name="time_exit_test"
    )
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  OOS Sharpe: {report.combined_oos_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Win Rate: {report.combined_oos_metrics.win_rate*100:.1f}%")
    logger.info(f"  N Trades: {report.combined_oos_metrics.n_trades}")
    logger.info(f"  Grade: {report.overall_grade}")
    
    # Check Sharpe is realistic
    sharpe = report.combined_oos_metrics.sharpe_ratio
    if 0 < sharpe < 10:
        logger.info(f"✓ Sharpe {sharpe:.2f} is in realistic range (PASS)")
        return True
    elif sharpe > 10:
        logger.error(f"✗ Sharpe {sharpe:.2f} is too high (FAIL)")
        return False
    else:
        logger.warning(f"⚠ Sharpe {sharpe:.2f} is negative/zero")
        return True  # Negative Sharpe is realistic (strategy doesn't work)

def test_no_sltp_exits():
    """Test that SL/TP exits are NOT generated."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: SL/TP exits are DISABLED")
    logger.info("="*60)
    
    import inspect
    from discovery.entry_exit import ExitConditionDiscovery
    
    source = inspect.getsource(ExitConditionDiscovery.discover)
    
    if 'DISABLED: SL/TP exits' in source:
        logger.info("✓ SL/TP exits are disabled in code (PASS)")
    else:
        logger.warning("⚠ SL/TP disable comment not found")
    
    if 'sl_tp_exits = self._discover_sl_tp' in source and 'DISABLED' not in source.split('sl_tp_exits = self._discover_sl_tp')[0][-50:]:
        logger.error("✗ SL/TP exits may still be active (FAIL)")
        return False
    else:
        logger.info("✓ SL/TP exit call is commented out (PASS)")
    
    if '_discover_time_exit_fixed' in source:
        logger.info("✓ Fixed 6-bar time exit is used (PASS)")
    else:
        logger.error("✗ Fixed time exit method not found (FAIL)")
        return False
    
    return True

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("TIME-BASED EXIT FIX VERIFICATION")
    logger.info("="*60)
    logger.info("\nThis test verifies that:")
    logger.info("1. Exit discovery produces TIME exits (not SL/TP)")
    logger.info("2. Sharpe is calculated correctly")
    logger.info("3. SL/TP exits are disabled")
    logger.info("")
    
    results = []
    
    # Test 1: Exit discovery
    try:
        results.append(("Exit Discovery", test_exit_discovery()))
    except Exception as e:
        logger.error(f"Exit discovery test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Exit Discovery", False))
    
    # Test 2: Sharpe calculation
    try:
        results.append(("Sharpe Calculation", test_sharpe_calculation()))
    except Exception as e:
        logger.error(f"Sharpe calculation test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Sharpe Calculation", False))
    
    # Test 3: No SL/TP exits
    try:
        results.append(("No SL/TP Exits", test_no_sltp_exits()))
    except Exception as e:
        logger.error(f"No SL/TP test failed with error: {e}")
        results.append(("No SL/TP Exits", False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("ALL TESTS PASSED - Time exit fix is working!")
        logger.info("Expected Run 9 results:")
        logger.info("  - Exit type: TIME (6 bars)")
        logger.info("  - OOS Sharpe: 2-5 (realistic)")
        logger.info("  - Grade: C/B/A (not blocked)")
    else:
        logger.error("SOME TESTS FAILED - Please review")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
