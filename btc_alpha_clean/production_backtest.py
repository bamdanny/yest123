#!/usr/bin/env python3
"""
PRODUCTION BACKTEST - Realistic Costs & Risk Analysis

Uses EXACT same logic as simple_ensemble.py (validated Sharpe 12.18).
Adds realistic transaction costs to determine if alpha is tradeable.

Cost Model (Binance Futures):
- Taker fee: 0.04% per side (0.08% round trip)
- Slippage: 0.02% estimated
- Funding: 0.01% per 8h average

Decision Criteria:
- Net Sharpe > 3.0: TRADEABLE
- Net Sharpe 2.0-3.0: MARGINAL
- Net Sharpe < 2.0: NOT VIABLE
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """Transaction costs for BTC perpetual futures."""
    taker_fee: float = 0.0004      # 0.04% per side
    slippage: float = 0.0002       # 0.02% estimated
    funding_rate: float = 0.0001   # 0.01% per 8h average
    hold_periods: float = 0.75     # 6h hold = 0.75 × 8h funding periods
    
    @property
    def entry_cost(self) -> float:
        """Cost to enter position."""
        return self.taker_fee + self.slippage
    
    @property
    def exit_cost(self) -> float:
        """Cost to exit position."""
        return self.taker_fee + self.slippage
    
    @property
    def funding_cost(self) -> float:
        """Funding cost for hold period."""
        return self.funding_rate * self.hold_periods
    
    @property
    def total_round_trip(self) -> float:
        """Total cost for round-trip trade."""
        return self.entry_cost + self.exit_cost + self.funding_cost
    
    def __str__(self):
        return (f"CostModel(entry={self.entry_cost*100:.3f}%, "
                f"exit={self.exit_cost*100:.3f}%, "
                f"funding={self.funding_cost*100:.3f}%, "
                f"total={self.total_round_trip*100:.3f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CONFIGURATIONS (Same as simple_ensemble.py)
# ═══════════════════════════════════════════════════════════════════════════════

PHASE1_RULES = [
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_change_1h",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 10.15,
        "weight": 0.30
    },
    {
        "name": "deriv_feat_cg_oi_aggregated_oi_close_accel",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 80,
        "oos_sharpe": 8.41,
        "weight": 0.25
    },
    {
        "name": "price_rsi_14_lag_48h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 1.5,
        "oos_sharpe": 6.33,
        "weight": 0.20
    },
    {
        "name": "deriv_feat_cg_funding_vol_weighted_funding_vol_close_cumul_168h",
        "direction": -1,
        "threshold_type": "zscore",
        "threshold_value": 2.0,
        "oos_sharpe": 4.98,
        "weight": 0.15
    },
    {
        "name": "sent_feat_fg_zscore_90d",
        "direction": 1,
        "threshold_type": "percentile",
        "threshold_value": 90,
        "oos_sharpe": 5.62,
        "weight": 0.10
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (Same as simple_ensemble.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load features AND targets from cache."""
    cache_path = Path("data_cache/features_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError("No features cache. Run: python run_exhaustive_search.py first")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    features = cache['features']
    targets = cache['targets']
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features = features[numeric_cols].copy()
    
    # Get return_simple_6h (same as Phase 1)
    if 'return_simple_6h' in targets.columns:
        returns = targets['return_simple_6h'].copy()
    else:
        ret_cols = [c for c in targets.columns if 'return_simple' in c]
        returns = targets[ret_cols[0]].copy() if ret_cols else None
    
    if returns is None:
        raise ValueError("No return columns in targets")
    
    # Align
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len].reset_index(drop=True)
    returns = returns.iloc[:min_len].reset_index(drop=True)
    
    valid_idx = returns.dropna().index
    features = features.loc[valid_idx].reset_index(drop=True)
    returns = returns.loc[valid_idx].reset_index(drop=True)
    
    return features, returns


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION (EXACT same as simple_ensemble.py)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(feature_values: np.ndarray, rule: dict, train_mask: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Generate +1/-1/0 signals using Phase 1 logic."""
    train_values = feature_values[train_mask]
    
    if rule['threshold_type'] == 'percentile':
        pct_upper = rule['threshold_value']
        pct_lower = 100 - pct_upper
        
        upper = np.nanpercentile(train_values, pct_upper)
        lower = np.nanpercentile(train_values, pct_lower)
        
        signals = np.zeros(len(feature_values))
        
        if rule['direction'] == 1:
            signals[feature_values > upper] = 1
            signals[feature_values < lower] = -1
        else:
            signals[feature_values > upper] = -1
            signals[feature_values < lower] = 1
        
        return signals, upper, lower
    
    else:  # zscore
        mean = np.nanmean(train_values)
        std = np.nanstd(train_values)
        
        if std < 1e-10:
            return np.zeros(len(feature_values)), np.nan, np.nan
        
        zscore = (feature_values - mean) / std
        z_upper = rule['threshold_value']
        z_lower = -rule['threshold_value']
        
        signals = np.zeros(len(feature_values))
        
        if rule['direction'] == 1:
            signals[zscore > z_upper] = 1
            signals[zscore < z_lower] = -1
        else:
            signals[zscore > z_upper] = -1
            signals[zscore < z_lower] = 1
        
        return signals, mean + z_upper * std, mean + z_lower * std


def generate_ensemble_signals(features: pd.DataFrame, rules: list, train_mask: np.ndarray, 
                              min_position: float = 0.1) -> np.ndarray:
    """Generate ensemble signals using position sizing."""
    n = len(features)
    
    all_signals = []
    weights = []
    
    for rule in rules:
        if rule['name'] not in features.columns:
            continue
        
        feature_values = features[rule['name']].values
        signals, upper, lower = generate_signals(feature_values, rule, train_mask)
        
        if np.isnan(upper):
            continue
        
        all_signals.append(signals)
        weights.append(rule['weight'])
    
    if len(all_signals) == 0:
        return np.zeros(n)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Combine: position = weighted average
    all_signals = np.array(all_signals)
    combined_position = np.sum(all_signals * weights[:, np.newaxis], axis=0)
    
    # Generate final signal
    final_signal = np.zeros(n)
    final_signal[combined_position > min_position] = 1
    final_signal[combined_position < -min_position] = -1
    
    return final_signal


# ═══════════════════════════════════════════════════════════════════════════════
# SHARPE CALCULATION (Same as simple_ensemble.py)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_sharpe(trade_returns: np.ndarray, period_days: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(trade_returns) < 5 or period_days <= 0:
        return 0.0
    
    total_return = np.prod(1 + trade_returns) - 1
    daily_return = (1 + total_return) ** (1 / period_days) - 1
    
    trade_std = np.std(trade_returns)
    trades_per_day = len(trade_returns) / period_days
    daily_std = trade_std * np.sqrt(trades_per_day)
    
    if daily_std < 1e-10:
        return 0.0
    
    return (daily_return / daily_std) * np.sqrt(365)


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION BACKTEST WITH COSTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Single trade record."""
    idx: int
    direction: int      # 1 = long, -1 = short
    gross_return: float
    entry_cost: float
    exit_cost: float
    funding_cost: float
    net_return: float
    cumulative_gross: float
    cumulative_net: float


def run_production_backtest(
    features: pd.DataFrame,
    returns: pd.Series,
    rules: list,
    cost_model: CostModel,
    train_ratio: float = 0.6,
    min_position: float = 0.1
) -> Dict:
    """
    Run production backtest with realistic transaction costs.
    
    Returns comprehensive results including gross/net metrics and trade log.
    """
    n = len(features)
    n_train = int(n * train_ratio)
    
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    
    return_values = returns.values
    
    # Generate ensemble signals (EXACT same logic as simple_ensemble.py)
    signals = generate_ensemble_signals(features, rules, train_mask, min_position)
    
    # OOS only
    oos_signals = signals[n_train:]
    oos_returns = return_values[n_train:]
    n_oos = len(oos_signals)
    
    # Track trades
    trade_log: List[TradeRecord] = []
    
    # Gross tracking
    gross_equity = 1.0
    gross_returns_list = []
    gross_equity_curve = [1.0]
    
    # Net tracking
    net_equity = 1.0
    net_returns_list = []
    net_equity_curve = [1.0]
    
    # Cost tracking
    total_entry_costs = 0.0
    total_exit_costs = 0.0
    total_funding_costs = 0.0
    
    # Process each bar
    for i in range(n_oos):
        signal = oos_signals[i]
        
        if signal == 0:
            gross_equity_curve.append(gross_equity)
            net_equity_curve.append(net_equity)
            continue
        
        # Gross return (no costs)
        gross_ret = signal * oos_returns[i]
        gross_returns_list.append(gross_ret)
        gross_equity *= (1 + gross_ret)
        
        # Calculate costs
        entry_cost = cost_model.entry_cost
        exit_cost = cost_model.exit_cost
        funding_cost = cost_model.funding_cost
        total_cost = entry_cost + exit_cost + funding_cost
        
        total_entry_costs += entry_cost
        total_exit_costs += exit_cost
        total_funding_costs += funding_cost
        
        # Net return (with costs)
        net_ret = gross_ret - total_cost
        net_returns_list.append(net_ret)
        net_equity *= (1 + net_ret)
        
        # Log trade
        trade_log.append(TradeRecord(
            idx=n_train + i,
            direction=int(signal),
            gross_return=gross_ret,
            entry_cost=entry_cost,
            exit_cost=exit_cost,
            funding_cost=funding_cost,
            net_return=net_ret,
            cumulative_gross=gross_equity,
            cumulative_net=net_equity
        ))
        
        gross_equity_curve.append(gross_equity)
        net_equity_curve.append(net_equity)
    
    # Calculate metrics
    n_trades = len(trade_log)
    period_days = n_oos / 6  # 6 bars per day at 4h timeframe
    
    # Gross metrics
    gross_returns_arr = np.array(gross_returns_list) if gross_returns_list else np.array([0])
    gross_total_return = np.prod(1 + gross_returns_arr) - 1
    gross_sharpe = calc_sharpe(gross_returns_arr, period_days)
    gross_wins = np.sum(gross_returns_arr > 0)
    gross_win_rate = gross_wins / n_trades if n_trades > 0 else 0
    
    # Net metrics
    net_returns_arr = np.array(net_returns_list) if net_returns_list else np.array([0])
    net_total_return = np.prod(1 + net_returns_arr) - 1
    net_sharpe = calc_sharpe(net_returns_arr, period_days)
    net_wins = np.sum(net_returns_arr > 0)
    net_win_rate = net_wins / n_trades if n_trades > 0 else 0
    
    # Drawdown calculations
    def calc_max_drawdown(equity_curve):
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return np.min(drawdown)
    
    gross_max_dd = calc_max_drawdown(gross_equity_curve)
    net_max_dd = calc_max_drawdown(net_equity_curve)
    
    # Profit factor
    def calc_profit_factor(returns):
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) == 0 or losses.sum() == 0:
            return np.inf
        return abs(wins.sum() / losses.sum())
    
    gross_pf = calc_profit_factor(gross_returns_arr)
    net_pf = calc_profit_factor(net_returns_arr)
    
    # Count longs/shorts
    n_long = sum(1 for t in trade_log if t.direction == 1)
    n_short = sum(1 for t in trade_log if t.direction == -1)
    
    # Total costs
    total_costs = total_entry_costs + total_exit_costs + total_funding_costs
    
    return {
        'n_samples': n,
        'n_train': n_train,
        'n_oos': n_oos,
        'period_days': period_days,
        'n_trades': n_trades,
        'n_long': n_long,
        'n_short': n_short,
        
        # Gross metrics
        'gross_total_return': gross_total_return,
        'gross_sharpe': gross_sharpe,
        'gross_win_rate': gross_win_rate,
        'gross_max_drawdown': gross_max_dd,
        'gross_profit_factor': gross_pf,
        
        # Net metrics
        'net_total_return': net_total_return,
        'net_sharpe': net_sharpe,
        'net_win_rate': net_win_rate,
        'net_max_drawdown': net_max_dd,
        'net_profit_factor': net_pf,
        
        # Costs
        'total_entry_costs': total_entry_costs,
        'total_exit_costs': total_exit_costs,
        'total_funding_costs': total_funding_costs,
        'total_costs': total_costs,
        
        # Curves and logs
        'gross_equity_curve': gross_equity_curve,
        'net_equity_curve': net_equity_curve,
        'trade_log': trade_log,
        'gross_returns': gross_returns_arr,
        'net_returns': net_returns_arr,
        
        # Cost model
        'cost_model': cost_model
    }


def run_sensitivity_analysis(features, returns, rules, train_ratio=0.6, min_position=0.1):
    """Run backtest under different cost scenarios."""
    
    scenarios = [
        {'name': 'Optimistic (Maker)', 'fee': 0.0002, 'slippage': 0.0001, 'funding': 0.00005},
        {'name': 'Base Case (Taker)', 'fee': 0.0004, 'slippage': 0.0002, 'funding': 0.0001},
        {'name': 'Pessimistic', 'fee': 0.0005, 'slippage': 0.0004, 'funding': 0.00015},
        {'name': 'High Volume', 'fee': 0.0003, 'slippage': 0.0003, 'funding': 0.0001},
    ]
    
    results = []
    
    for scenario in scenarios:
        cost_model = CostModel(
            taker_fee=scenario['fee'],
            slippage=scenario['slippage'],
            funding_rate=scenario['funding']
        )
        
        result = run_production_backtest(
            features=features,
            returns=returns,
            rules=rules,
            cost_model=cost_model,
            train_ratio=train_ratio,
            min_position=min_position
        )
        
        results.append({
            'name': scenario['name'],
            'cost_per_trade': cost_model.total_round_trip,
            'gross_sharpe': result['gross_sharpe'],
            'net_sharpe': result['net_sharpe'],
            'net_return': result['net_total_return'],
            'sharpe_decay': (result['gross_sharpe'] - result['net_sharpe']) / result['gross_sharpe'] * 100 if result['gross_sharpe'] > 0 else 0
        })
    
    return results


def save_charts(results: Dict, output_dir: Path):
    """Save equity curves and charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Equity curves
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gross vs Net equity
        ax1 = axes[0]
        ax1.plot(results['gross_equity_curve'], label='Gross (No Costs)', color='blue', alpha=0.8)
        ax1.plot(results['net_equity_curve'], label='Net (With Costs)', color='green', alpha=0.8)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curves: Gross vs Net')
        ax1.set_ylabel('Equity Multiple')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        gross_eq = np.array(results['gross_equity_curve'])
        net_eq = np.array(results['net_equity_curve'])
        
        gross_peak = np.maximum.accumulate(gross_eq)
        net_peak = np.maximum.accumulate(net_eq)
        
        gross_dd = (gross_eq - gross_peak) / gross_peak * 100
        net_dd = (net_eq - net_peak) / net_peak * 100
        
        ax2.fill_between(range(len(gross_dd)), gross_dd, 0, alpha=0.3, color='blue', label='Gross DD')
        ax2.fill_between(range(len(net_dd)), net_dd, 0, alpha=0.3, color='green', label='Net DD')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'equity_curves.png', dpi=150)
        plt.close()
        
        logger.info(f"Charts saved to {output_dir}")
        
    except ImportError:
        logger.warning("matplotlib not available - skipping charts")
    except Exception as e:
        logger.warning(f"Could not save charts: {e}")


def save_trade_log(results: Dict, output_dir: Path):
    """Save trade log to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trades_data = []
    for t in results['trade_log']:
        trades_data.append({
            'index': t.idx,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'gross_return': t.gross_return,
            'entry_cost': t.entry_cost,
            'exit_cost': t.exit_cost,
            'funding_cost': t.funding_cost,
            'total_cost': t.entry_cost + t.exit_cost + t.funding_cost,
            'net_return': t.net_return,
            'cumulative_gross': t.cumulative_gross,
            'cumulative_net': t.cumulative_net
        })
    
    df = pd.DataFrame(trades_data)
    df.to_csv(output_dir / 'trade_log.csv', index=False)
    logger.info(f"Trade log saved to {output_dir / 'trade_log.csv'}")


def print_results(results: Dict, sensitivity: List[Dict]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("PRODUCTION BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'DATA:':<20}")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Train: {results['n_train']} ({results['n_train']/results['n_samples']*100:.0f}%)")
    print(f"  OOS: {results['n_oos']} ({results['n_oos']/results['n_samples']*100:.0f}%)")
    print(f"  OOS Period: {results['period_days']:.1f} days")
    print(f"  Trades: {results['n_trades']} ({results['n_long']}L + {results['n_short']}S)")
    print(f"  Trades/Day: {results['n_trades']/results['period_days']:.1f}")
    
    print(f"\n{'GROSS (No Costs):':<20}")
    print(f"  Total Return: {results['gross_total_return']*100:.1f}%")
    print(f"  Sharpe Ratio: {results['gross_sharpe']:.2f}")
    print(f"  Win Rate: {results['gross_win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {results['gross_max_drawdown']*100:.1f}%")
    print(f"  Profit Factor: {results['gross_profit_factor']:.2f}")
    
    print(f"\n{'NET (With Costs):':<20}")
    print(f"  Total Return: {results['net_total_return']*100:.1f}%")
    print(f"  Sharpe Ratio: {results['net_sharpe']:.2f}")
    print(f"  Win Rate: {results['net_win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {results['net_max_drawdown']*100:.1f}%")
    print(f"  Profit Factor: {results['net_profit_factor']:.2f}")
    
    print(f"\n{'COST BREAKDOWN:':<20}")
    cost_model = results['cost_model']
    print(f"  Per-Trade Costs:")
    print(f"    Entry: {cost_model.entry_cost*100:.3f}%")
    print(f"    Exit: {cost_model.exit_cost*100:.3f}%")
    print(f"    Funding: {cost_model.funding_cost*100:.3f}%")
    print(f"    Total: {cost_model.total_round_trip*100:.3f}%")
    print(f"  Cumulative Costs ({results['n_trades']} trades):")
    print(f"    Entry Fees: {results['total_entry_costs']*100:.2f}%")
    print(f"    Exit Fees: {results['total_exit_costs']*100:.2f}%")
    print(f"    Funding: {results['total_funding_costs']*100:.2f}%")
    print(f"    Total Paid: {results['total_costs']*100:.2f}%")
    
    print(f"\n{'COMPARISON:':<20}")
    print(f"  Gross Sharpe: {results['gross_sharpe']:.2f}")
    print(f"  Net Sharpe: {results['net_sharpe']:.2f}")
    sharpe_decay = (results['gross_sharpe'] - results['net_sharpe']) / results['gross_sharpe'] * 100 if results['gross_sharpe'] > 0 else 0
    print(f"  Sharpe Decay: {sharpe_decay:.1f}%")
    print(f"  Return Lost to Costs: {(results['gross_total_return'] - results['net_total_return'])*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\n{'Scenario':<25} {'Cost/Trade':>12} {'Net Sharpe':>12} {'Net Return':>12} {'Decay':>8}")
    print("-" * 70)
    for s in sensitivity:
        print(f"{s['name']:<25} {s['cost_per_trade']*100:>11.2f}% {s['net_sharpe']:>12.2f} {s['net_return']*100:>11.1f}% {s['sharpe_decay']:>7.1f}%")
    
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    
    net_sharpe = results['net_sharpe']
    if net_sharpe >= 3.0:
        verdict = "✅ TRADEABLE"
        explanation = "Strong alpha survives costs. Proceed to paper trading."
    elif net_sharpe >= 2.0:
        verdict = "⚠️  MARGINAL"
        explanation = "Alpha exists but margins are tight. Consider cost optimization."
    else:
        verdict = "❌ NOT VIABLE"
        explanation = "Costs consume too much alpha. Strategy not profitable."
    
    print(f"\n  Net Sharpe: {net_sharpe:.2f}")
    print(f"  {verdict}")
    print(f"  {explanation}")
    
    # Additional insights
    print(f"\n{'INSIGHTS:':<20}")
    avg_gross_ret = np.mean(results['gross_returns']) * 100
    avg_net_ret = np.mean(results['net_returns']) * 100
    print(f"  Avg Gross Return/Trade: {avg_gross_ret:.2f}%")
    print(f"  Avg Net Return/Trade: {avg_net_ret:.2f}%")
    print(f"  Avg Cost/Trade: {(avg_gross_ret - avg_net_ret):.2f}%")
    
    # Break-even analysis
    avg_gross = avg_gross_ret / 100
    if avg_gross > 0:
        max_cost = avg_gross
        print(f"  Break-even Cost: {max_cost*100:.2f}% per trade")
        print(f"  Cost Headroom: {(max_cost - cost_model.total_round_trip)*100:.2f}%")


def main():
    logger.info("=" * 70)
    logger.info("PRODUCTION BACKTEST - Cost Analysis")
    logger.info("=" * 70)
    
    # Load data
    features, returns = load_data()
    logger.info(f"Loaded {len(features)} samples")
    
    # Base case cost model
    cost_model = CostModel(
        taker_fee=0.0004,    # 0.04% taker fee
        slippage=0.0002,     # 0.02% slippage
        funding_rate=0.0001  # 0.01% per 8h
    )
    logger.info(f"Cost model: {cost_model}")
    
    # Run main backtest
    logger.info("\nRunning production backtest...")
    results = run_production_backtest(
        features=features,
        returns=returns,
        rules=PHASE1_RULES,
        cost_model=cost_model,
        train_ratio=0.6,
        min_position=0.1
    )
    
    # Run sensitivity analysis
    logger.info("Running sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(
        features=features,
        returns=returns,
        rules=PHASE1_RULES,
        train_ratio=0.6,
        min_position=0.1
    )
    
    # Print results
    print_results(results, sensitivity)
    
    # Save outputs
    output_dir = Path("reports/production_backtest")
    save_charts(results, output_dir)
    save_trade_log(results, output_dir)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_trades': results['n_trades'],
        'period_days': results['period_days'],
        'gross_sharpe': results['gross_sharpe'],
        'net_sharpe': results['net_sharpe'],
        'gross_return': results['gross_total_return'],
        'net_return': results['net_total_return'],
        'total_costs': results['total_costs'],
        'verdict': 'TRADEABLE' if results['net_sharpe'] >= 3.0 else 'MARGINAL' if results['net_sharpe'] >= 2.0 else 'NOT_VIABLE'
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    # Validation check
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION CHECK")
    logger.info("=" * 70)
    
    expected_gross_sharpe = 12.18
    if abs(results['gross_sharpe'] - expected_gross_sharpe) > 1.0:
        logger.warning(f"⚠️  Gross Sharpe ({results['gross_sharpe']:.2f}) differs from expected ({expected_gross_sharpe})")
        logger.warning("   There may be a bug - please verify signal generation")
    else:
        logger.info(f"✅ Gross Sharpe ({results['gross_sharpe']:.2f}) matches expected ({expected_gross_sharpe})")


if __name__ == "__main__":
    main()
