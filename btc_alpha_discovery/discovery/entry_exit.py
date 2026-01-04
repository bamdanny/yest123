"""
Entry and Exit Condition Discovery.

Phase 7: Discovers optimal entry conditions from data
Phase 8: Discovers optimal exit conditions from data

Key principle: Don't assume we know the right entry/exit rules.
Let the data tell us what combinations of features predict profitable trades.

Methods:
- Decision tree rule extraction
- Association rule mining
- Genetic programming for rule evolution
- Conditional probability analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


def remove_overlapping_signals(signals: pd.Series, holding_period: int = 6) -> pd.Series:
    """
    After a signal fires, mask out the next (holding_period - 1) bars.
    Ensures non-overlapping trade windows for valid Sharpe calculation.
    
    This is critical for accurate rule evaluation - without it, clustered
    signals during trends get counted as multiple independent trades,
    artificially inflating Sharpe ratios.
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


# Optional imports
try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.tree import export_text
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN_TREE = True
except ImportError:
    HAS_SKLEARN_TREE = False

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    logger.warning("DEAP not available. Genetic programming disabled.")


@dataclass
class TradingRule:
    """A discovered trading rule."""
    rule_id: str
    conditions: List[Dict[str, Any]]  # List of {feature, operator, threshold}
    logic: str  # 'AND' or 'OR'
    direction: int  # 1 for long, -1 for short
    confidence: float
    support: int  # Number of trades
    win_rate: float
    avg_return: float
    sharpe: float
    max_drawdown: float
    
    def to_string(self) -> str:
        """Human-readable rule description."""
        cond_strs = []
        for c in self.conditions:
            cond_strs.append(f"{c['feature']} {c['operator']} {c['threshold']:.4f}")
        return f"{'LONG' if self.direction == 1 else 'SHORT'} when ({f' {self.logic} '.join(cond_strs)})"
    
    def evaluate(self, features: pd.DataFrame) -> pd.Series:
        """Evaluate rule on features, return boolean mask."""
        masks = []
        for cond in self.conditions:
            feat = features.get(cond['feature'])
            if feat is None:
                masks.append(pd.Series(False, index=features.index))
                continue
                
            op = cond['operator']
            thresh = cond['threshold']
            
            if op == '>':
                masks.append(feat > thresh)
            elif op == '<':
                masks.append(feat < thresh)
            elif op == '>=':
                masks.append(feat >= thresh)
            elif op == '<=':
                masks.append(feat <= thresh)
            elif op == '==':
                masks.append(feat == thresh)
            else:
                masks.append(pd.Series(True, index=features.index))
                
        if not masks:
            return pd.Series(False, index=features.index)
            
        if self.logic == 'AND':
            result = masks[0]
            for m in masks[1:]:
                result = result & m
        else:  # OR
            result = masks[0]
            for m in masks[1:]:
                result = result | m
                
        return result


@dataclass
class ExitRule:
    """A discovered exit rule."""
    rule_id: str
    exit_type: str  # 'stop_loss', 'take_profit', 'time', 'signal'
    conditions: List[Dict[str, Any]]
    priority: int  # Order of evaluation
    avg_bars_held: float
    avg_exit_return: float
    hit_rate: float  # % of trades hitting this exit
    
    def to_string(self) -> str:
        cond_strs = [f"{c['feature']} {c['operator']} {c['threshold']:.4f}" for c in self.conditions]
        return f"{self.exit_type.upper()}: {' AND '.join(cond_strs)}"


@dataclass
class DiscoveredStrategy:
    """Complete strategy with entry and exit rules."""
    name: str
    entry_rules: List[TradingRule]
    exit_rules: List[ExitRule]
    performance: Dict[str, float]
    feature_importance: Dict[str, float]


class EntryConditionDiscovery:
    """
    Discovers optimal entry conditions from data.
    
    Uses multiple approaches:
    1. Decision tree rule extraction
    2. Conditional probability analysis
    3. Genetic programming (if available)
    4. Threshold search
    """
    
    def __init__(
        self,
        min_trades: int = 20,  # Lowered from 50 due to overlap removal
        min_win_rate: float = 0.52,
        min_sharpe: float = 0.5,
        max_conditions: int = 5,
        random_state: int = 42
    ):
        """
        Initialize discovery.
        
        Args:
            min_trades: Minimum trades for a rule to be valid
            min_win_rate: Minimum win rate threshold
            min_sharpe: Minimum Sharpe ratio
            max_conditions: Maximum conditions in a rule
            random_state: Random seed
        """
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.max_conditions = max_conditions
        self.random_state = random_state
        
    def discover(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        top_features: Optional[List[str]] = None,
        n_rules: int = 20
    ) -> List[TradingRule]:
        """
        Discover entry rules.
        
        Args:
            features: Feature DataFrame
            forward_returns: Forward returns for target
            top_features: Optional list of features to consider
            n_rules: Maximum number of rules to return
            
        Returns:
            List of discovered TradingRule objects
        """
        logger.info("Starting entry condition discovery...")
        
        # Only keep numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features_numeric = features[numeric_cols].copy()
        
        logger.info(f"Numeric features: {len(features_numeric.columns)}, samples: {len(features_numeric)}")
        
        # Clean data - fill NaN instead of dropping to preserve data
        features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)
        features_numeric = features_numeric.ffill().bfill().fillna(0)
        
        # Align with forward returns
        valid_idx = ~forward_returns.isna()
        features_clean = features_numeric[valid_idx].reset_index(drop=True)
        returns_clean = forward_returns[valid_idx].reset_index(drop=True)
        
        logger.info(f"After cleaning: {len(features_clean)} samples with valid returns")
        
        if len(features_clean) < self.min_trades * 2:
            logger.warning(f"Insufficient data for discovery (need {self.min_trades * 2}, have {len(features_clean)})")
            return []
        
        # CRITICAL: Exclude raw OHLC features - rules based on these are NONSENSE
        # "LONG when open < 84705" is pure hindsight bias
        RAW_OHLC_FEATURES = ['open', 'high', 'low', 'close', 'timestamp']
        features_clean = features_clean.drop(columns=[c for c in RAW_OHLC_FEATURES if c in features_clean.columns], errors='ignore')
        
        X = features_clean
        y = returns_clean
        
        # Select features to analyze
        if top_features:
            # Also filter raw OHLC from top_features
            filtered_top = [f for f in top_features if f not in RAW_OHLC_FEATURES]
            available = [f for f in filtered_top if f in X.columns]
            X = X[available[:50]]  # Limit for efficiency
            logger.info(f"Using {len(X.columns)} top features")
            
        all_rules = []
        
        # Method 1: Decision tree rule extraction
        logger.info("Extracting rules from decision trees...")
        dt_rules = self._extract_decision_tree_rules(X, y)
        all_rules.extend(dt_rules)
        
        # Method 2: Threshold search on individual features
        logger.info("Searching single-feature thresholds...")
        threshold_rules = self._search_thresholds(X, y)
        all_rules.extend(threshold_rules)
        
        # Method 3: Condition combinations
        logger.info("Testing condition combinations...")
        combo_rules = self._search_combinations(X, y, dt_rules + threshold_rules)
        all_rules.extend(combo_rules)
        
        # Method 4: Genetic programming (if available)
        if HAS_DEAP and len(X.columns) > 0:
            logger.info("Running genetic programming...")
            gp_rules = self._genetic_programming(X, y)
            all_rules.extend(gp_rules)
            
        # Validate and rank rules
        valid_rules = []
        for rule in all_rules:
            validated = self._validate_rule(rule, X, y)
            if validated is not None:
                valid_rules.append(validated)
                
        # Remove duplicates (similar rules)
        unique_rules = self._deduplicate_rules(valid_rules)
        
        # Sort by Sharpe ratio
        unique_rules.sort(key=lambda r: -r.sharpe)
        
        logger.info(f"Discovered {len(unique_rules)} valid entry rules")
        return unique_rules[:n_rules]
        
    def _extract_decision_tree_rules(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[TradingRule]:
        """Extract rules from decision trees."""
        if not HAS_SKLEARN_TREE:
            return []
            
        rules = []
        
        # Create binary target for classification
        y_binary = (y > 0).astype(int)
        
        for max_depth in [2, 3, 4, 5]:
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=max(self.min_trades // 2, 20),
                random_state=self.random_state
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree.fit(X, y_binary)
                
            # Extract rules from tree paths
            tree_rules = self._parse_tree_rules(tree, X.columns.tolist())
            rules.extend(tree_rules)
            
        return rules
        
    def _parse_tree_rules(
        self,
        tree: 'DecisionTreeClassifier',
        feature_names: List[str]
    ) -> List[TradingRule]:
        """Parse decision tree into rules."""
        rules = []
        
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, conditions):
            if tree_.feature[node] != -2:  # Not a leaf
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left branch (<=)
                left_conditions = conditions + [{
                    'feature': name,
                    'operator': '<=',
                    'threshold': threshold
                }]
                recurse(tree_.children_left[node], left_conditions)
                
                # Right branch (>)
                right_conditions = conditions + [{
                    'feature': name,
                    'operator': '>',
                    'threshold': threshold
                }]
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf node
                values = tree_.value[node][0]
                total = sum(values)
                if total > 0:
                    prob_positive = values[1] / total if len(values) > 1 else 0
                    n_samples = int(total)
                    
                    if n_samples >= self.min_trades // 2:
                        direction = 1 if prob_positive > 0.5 else -1
                        
                        rule = TradingRule(
                            rule_id=f"dt_{len(rules)}",
                            conditions=conditions,
                            logic='AND',
                            direction=direction,
                            confidence=prob_positive if direction == 1 else (1 - prob_positive),
                            support=n_samples,
                            win_rate=0.0,  # Will be calculated in validation
                            avg_return=0.0,
                            sharpe=0.0,
                            max_drawdown=0.0
                        )
                        rules.append(rule)
                        
        recurse(0, [])
        return rules
        
    def _search_thresholds(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[TradingRule]:
        """Search for optimal thresholds on individual features."""
        rules = []
        
        for col in X.columns:
            feat = X[col]
            
            # Test percentile thresholds
            for pct in [10, 20, 25, 30, 70, 75, 80, 90]:
                threshold = np.percentile(feat.dropna(), pct)
                
                for op in ['>', '<']:
                    if op == '>':
                        raw_mask = feat > threshold
                    else:
                        raw_mask = feat < threshold
                    
                    # Convert mask to signals and remove overlaps
                    raw_signals = raw_mask.astype(int)
                    clean_signals = remove_overlapping_signals(raw_signals, holding_period=6)
                    mask = clean_signals > 0
                        
                    if mask.sum() >= self.min_trades:
                        returns = y[mask]
                        
                        if len(returns) > 0:
                            win_rate = (returns > 0).mean()
                            
                            # Only create rule if promising
                            if win_rate >= self.min_win_rate or win_rate <= (1 - self.min_win_rate):
                                direction = 1 if win_rate >= 0.5 else -1
                                
                                rule = TradingRule(
                                    rule_id=f"thresh_{col}_{op}_{pct}",
                                    conditions=[{
                                        'feature': col,
                                        'operator': op,
                                        'threshold': threshold
                                    }],
                                    logic='AND',
                                    direction=direction,
                                    confidence=win_rate if direction == 1 else (1 - win_rate),
                                    support=int(mask.sum()),
                                    win_rate=0.0,
                                    avg_return=0.0,
                                    sharpe=0.0,
                                    max_drawdown=0.0
                                )
                                rules.append(rule)
                                
        return rules
        
    def _search_combinations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_rules: List[TradingRule]
    ) -> List[TradingRule]:
        """Search for combinations of conditions."""
        rules = []
        
        # Get unique conditions from base rules
        all_conditions = []
        for rule in base_rules[:20]:  # Limit for efficiency
            for cond in rule.conditions:
                if cond not in all_conditions:
                    all_conditions.append(cond)
                    
        # Test pairs
        for cond1, cond2 in combinations(all_conditions[:30], 2):
            # Skip if same feature
            if cond1['feature'] == cond2['feature']:
                continue
                
            # Evaluate AND combination
            mask1 = self._evaluate_condition(X, cond1)
            mask2 = self._evaluate_condition(X, cond2)
            raw_mask = mask1 & mask2
            
            # Remove overlapping signals for accurate trade count
            raw_signals = raw_mask.astype(int)
            clean_signals = remove_overlapping_signals(raw_signals, holding_period=6)
            combined_mask = clean_signals > 0
            
            if combined_mask.sum() >= self.min_trades:
                returns = y[combined_mask]
                win_rate = (returns > 0).mean()
                
                if win_rate >= self.min_win_rate or win_rate <= (1 - self.min_win_rate):
                    direction = 1 if win_rate >= 0.5 else -1
                    
                    rule = TradingRule(
                        rule_id=f"combo_{len(rules)}",
                        conditions=[cond1, cond2],
                        logic='AND',
                        direction=direction,
                        confidence=win_rate if direction == 1 else (1 - win_rate),
                        support=int(combined_mask.sum()),
                        win_rate=0.0,
                        avg_return=0.0,
                        sharpe=0.0,
                        max_drawdown=0.0
                    )
                    rules.append(rule)
                    
        return rules
        
    def _evaluate_condition(self, X: pd.DataFrame, cond: Dict) -> pd.Series:
        """Evaluate a single condition."""
        feat = X.get(cond['feature'])
        if feat is None:
            return pd.Series(False, index=X.index)
            
        op = cond['operator']
        thresh = cond['threshold']
        
        if op == '>':
            return feat > thresh
        elif op == '<':
            return feat < thresh
        elif op == '>=':
            return feat >= thresh
        elif op == '<=':
            return feat <= thresh
        else:
            return pd.Series(True, index=X.index)
            
    def _genetic_programming(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[TradingRule]:
        """Use genetic programming to evolve rules."""
        if not HAS_DEAP:
            logger.info("DEAP not available, skipping GP")
            return []
        
        # Validate inputs
        if X is None or len(X) < 50:
            logger.warning(f"GP: Insufficient data ({len(X) if X is not None else 0} rows)")
            return []
            
        rules = []
        
        try:
            # Ensure max_conditions is valid (at least 2)
            max_conds = max(2, self.max_conditions)
            
            # Setup DEAP
            if not hasattr(creator, 'FitnessMax'):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, 'Individual'):
                creator.create("Individual", list, fitness=creator.FitnessMax)
                
            toolbox = base.Toolbox()
            
            # Gene: (feature_idx, operator_idx, threshold_percentile)
            feature_names = list(X.columns)
            n_features = len(feature_names)
            
            if n_features < 2:
                logger.warning(f"GP: Too few features ({n_features})")
                return []
            
            def create_gene():
                return (
                    np.random.randint(0, max(1, n_features)),
                    np.random.randint(0, 2),  # 0: >, 1: <
                    np.random.randint(10, 91)  # percentile
                )
                
            def create_individual():
                # Need at least 2 genes for cxTwoPoint crossover to work
                n_genes = np.random.randint(2, max_conds + 1)  # Min 2, max max_conds
                return creator.Individual([create_gene() for _ in range(n_genes)])
                
            toolbox.register("individual", create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                conditions = []
                for gene in individual:
                    feat_idx, op_idx, pct = gene
                    if feat_idx >= len(feature_names):
                        continue
                    feat_name = feature_names[feat_idx]
                    feat_vals = X[feat_name].dropna()
                    if len(feat_vals) == 0:
                        continue
                    threshold = np.percentile(feat_vals, pct)
                    conditions.append({
                        'feature': feat_name,
                        'operator': '>' if op_idx == 0 else '<',
                        'threshold': threshold
                    })
                    
                if not conditions:
                    return (-np.inf,)
                    
                # Evaluate
                raw_mask = pd.Series(True, index=X.index)
                for cond in conditions:
                    raw_mask = raw_mask & self._evaluate_condition(X, cond)
                
                # Remove overlapping signals for accurate trade count
                raw_signals = raw_mask.astype(int)
                clean_signals = remove_overlapping_signals(raw_signals, holding_period=6)
                mask = clean_signals > 0
                    
                if mask.sum() < self.min_trades:
                    return (-np.inf,)
                    
                returns = y[mask]
                n_trades = len(returns)
                if n_trades < 10:
                    return (-np.inf,)
                    
                # Fitness: Sharpe using TRADE FREQUENCY
                mean_ret = returns.mean()
                std_ret = returns.std() + 1e-8
                
                # Use actual period from data length (4h bars = 6 per day)
                period_days = max(1, len(y) / 6)
                trades_per_year = (n_trades / period_days) * 365
                per_trade_sharpe = mean_ret / std_ret
                sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
                
                return (sharpe,)
                
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            
            def mutate(individual):
                if len(individual) > 0:
                    idx = np.random.randint(len(individual))
                    individual[idx] = create_gene()
                return individual,
                
            toolbox.register("mutate", mutate)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Run evolution
            pop = toolbox.population(n=50)
            
            for gen in range(20):
                offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
                fits = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fits):
                    ind.fitness.values = fit
                pop = toolbox.select(offspring, k=len(pop))
                
            # Extract best individuals as rules
            for ind in sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)[:10]:
                if ind.fitness.values[0] > 0:
                    conditions = []
                    for gene in ind:
                        feat_idx, op_idx, pct = gene
                        if feat_idx < len(feature_names):
                            feat_name = feature_names[feat_idx]
                            feat_vals = X[feat_name].dropna()
                            if len(feat_vals) > 0:
                                threshold = np.percentile(feat_vals, pct)
                                conditions.append({
                                    'feature': feat_name,
                                    'operator': '>' if op_idx == 0 else '<',
                                    'threshold': threshold
                                })
                                
                    if conditions:
                        rule = TradingRule(
                            rule_id=f"gp_{len(rules)}",
                            conditions=conditions,
                            logic='AND',
                            direction=1,
                            confidence=0.0,
                            support=0,
                            win_rate=0.0,
                            avg_return=0.0,
                            sharpe=ind.fitness.values[0],
                            max_drawdown=0.0
                        )
                        rules.append(rule)
                        
        except Exception as e:
            logger.warning(f"Genetic programming failed: {e}")
            
        return rules
        
    def _validate_rule(
        self,
        rule: TradingRule,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Optional[TradingRule]:
        """Validate and calculate full statistics for a rule."""
        mask = rule.evaluate(X)
        n_trades = mask.sum()
        
        if n_trades < self.min_trades:
            return None
            
        returns = y[mask] * rule.direction
        
        if len(returns) == 0:
            return None
        
        n_trades = len(returns)
        win_rate = (returns > 0).mean()
        avg_return = returns.mean()
        std_return = returns.std() + 1e-8
        
        # Sharpe using TRADE FREQUENCY (not bar frequency!)
        # Estimate period_days from data length (4h bars = 6 per day)
        period_days = max(1, len(y) / 6)
        trades_per_year = (n_trades / period_days) * 365
        per_trade_sharpe = avg_return / std_return
        sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
        
        # SANITY CHECK - DO NOT REMOVE
        if sharpe > 10:
            logger.warning(
                f"SHARPE > 10 DETECTED in rule validation: {sharpe:.2f}\n"
                f"  mean={avg_return:.6f}, std={std_return:.6f}\n"
                f"  n_trades={n_trades}, period_days={period_days:.1f}\n"
                f"  trades_per_year={trades_per_year:.1f}"
            )
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Update rule
        rule.support = int(n_trades)
        rule.win_rate = win_rate
        rule.avg_return = avg_return
        rule.sharpe = sharpe
        rule.max_drawdown = max_dd
        rule.confidence = win_rate
        
        # Filter by minimum quality
        if win_rate >= self.min_win_rate and sharpe >= self.min_sharpe:
            return rule
        elif win_rate <= (1 - self.min_win_rate) and sharpe >= self.min_sharpe:
            # Reverse direction
            rule.direction = -rule.direction
            rule.win_rate = 1 - win_rate
            return rule
            
        return None
        
    def _deduplicate_rules(
        self,
        rules: List[TradingRule]
    ) -> List[TradingRule]:
        """Remove very similar rules."""
        if not rules:
            return []
            
        unique = [rules[0]]
        
        for rule in rules[1:]:
            is_duplicate = False
            for existing in unique:
                # Check similarity
                if len(rule.conditions) == len(existing.conditions):
                    same_features = set(c['feature'] for c in rule.conditions)
                    existing_features = set(c['feature'] for c in existing.conditions)
                    if same_features == existing_features:
                        # Similar thresholds?
                        similar = True
                        for rc in rule.conditions:
                            for ec in existing.conditions:
                                if rc['feature'] == ec['feature']:
                                    if abs(rc['threshold'] - ec['threshold']) / (abs(ec['threshold']) + 1e-8) > 0.1:
                                        similar = False
                        if similar:
                            is_duplicate = True
                            break
                            
            if not is_duplicate:
                unique.append(rule)
                
        return unique


class ExitConditionDiscovery:
    """
    Discovers optimal exit conditions from data.
    
    Key insight: The best exit might not be a fixed stop-loss/take-profit.
    It might depend on market conditions.
    """
    
    def __init__(
        self,
        min_trades: int = 20,  # Lowered from 50 due to overlap removal
        random_state: int = 42
    ):
        self.min_trades = min_trades
        self.random_state = random_state
        
    def discover(
        self,
        features: pd.DataFrame,
        price_data: pd.DataFrame,
        entry_rule: TradingRule,
        max_bars: int = 48
    ) -> List[ExitRule]:
        """
        Discover optimal exit conditions for an entry rule.
        
        CRITICAL FIX (Run 9): Use ONLY time-based exits that match validation.
        
        The validation framework uses pre-computed 6-bar forward returns.
        SL/TP exits hold for 10-15 bars on average, creating a mismatch.
        This mismatch caused inflated Sharpe (10.94 vs expected 2-5).
        
        Args:
            features: Feature DataFrame
            price_data: DataFrame with 'close', 'high', 'low'
            entry_rule: The entry rule to find exits for
            max_bars: Maximum holding period to consider
            
        Returns:
            List of ExitRule objects
        """
        logger.info(f"Discovering exits for rule: {entry_rule.rule_id}")
        logger.info("="*60)
        logger.info("EXIT DISCOVERY: Using TIME-BASED exits only (6 bars)")
        logger.info("This matches the 6-bar forward returns used in validation")
        logger.info("="*60)
        
        # Get entry signals
        entries = entry_rule.evaluate(features)
        entry_indices = entries[entries].index.tolist()
        
        if len(entry_indices) < self.min_trades:
            logger.warning("Insufficient entries for exit discovery")
            return []
            
        exits = []
        
        # CRITICAL: Use ONLY 6-bar time exit to match validation
        # DO NOT use SL/TP exits - they create return mismatch
        time_exit_6bar = self._discover_time_exit_fixed(
            price_data, entry_indices, entry_rule.direction, bars=6
        )
        if time_exit_6bar:
            exits.append(time_exit_6bar)
            logger.info(f"Created 6-bar time exit: avg return = {time_exit_6bar.avg_exit_return*100:.3f}%")
        
        # DISABLED: SL/TP exits cause validation mismatch
        # sl_tp_exits = self._discover_sl_tp(...)
        # exits.extend(sl_tp_exits)
        
        # DISABLED: Variable time exits - only 6-bar matches validation
        # time_exits = self._discover_time_exits(...)
        # exits.extend(time_exits)
        
        # DISABLED: Conditional exits - complex and don't match validation
        # cond_exits = self._discover_conditional_exits(...)
        # exits.extend(cond_exits)
        
        logger.info(f"Exit discovery complete: {len(exits)} exit rules (all TIME-based)")
        
        return exits
    
    def _discover_time_exit_fixed(
        self,
        price_data: pd.DataFrame,
        entry_indices: List,
        direction: int,
        bars: int = 6
    ) -> ExitRule:
        """
        Create a fixed time-based exit that matches validation horizon.
        
        Args:
            price_data: DataFrame with 'close' prices
            entry_indices: List of entry bar indices
            direction: 1 for LONG, -1 for SHORT
            bars: Number of bars to hold (default 6 = 24h on 4h timeframe)
            
        Returns:
            ExitRule for time-based exit
        """
        returns = []
        
        for entry_idx in entry_indices:
            try:
                idx_pos = price_data.index.get_loc(entry_idx)
                
                if idx_pos + bars < len(price_data):
                    entry_price = price_data['close'].iloc[idx_pos]
                    exit_price = price_data['close'].iloc[idx_pos + bars]
                    
                    # Calculate return based on direction
                    raw_return = (exit_price / entry_price - 1)
                    trade_return = raw_return * direction
                    returns.append(trade_return)
            except Exception:
                continue
        
        if len(returns) < self.min_trades // 2:
            logger.warning(f"Insufficient trades for {bars}-bar exit: {len(returns)}")
            return None
        
        avg_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 0.01
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        logger.info(f"Time exit ({bars} bars):")
        logger.info(f"  N trades: {len(returns)}")
        logger.info(f"  Mean return: {avg_ret*100:.3f}%")
        logger.info(f"  Std return: {std_ret*100:.3f}%")
        logger.info(f"  Win rate: {win_rate*100:.1f}%")
        
        return ExitRule(
            rule_id=f"time_{bars}_fixed",
            exit_type='TIME',
            conditions=[
                {'feature': 'bars_held', 'operator': '>=', 'threshold': bars}
            ],
            priority=1,
            avg_bars_held=float(bars),
            avg_exit_return=avg_ret,
            hit_rate=1.0  # Time exits always trigger
        )
        
    def _discover_sl_tp(
        self,
        price_data: pd.DataFrame,
        entry_indices: List,
        direction: int,
        max_bars: int
    ) -> List[ExitRule]:
        """Find optimal stop-loss and take-profit levels."""
        exits = []
        
        sl_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        tp_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
        
        best_returns = {}
        
        for sl in sl_levels:
            for tp in tp_levels:
                returns = []
                bars_held = []
                hits = {'sl': 0, 'tp': 0, 'time': 0}
                
                for entry_idx in entry_indices:
                    try:
                        idx_pos = price_data.index.get_loc(entry_idx)
                        entry_price = price_data['close'].iloc[idx_pos]
                        
                        if direction == 1:  # Long
                            sl_price = entry_price * (1 - sl / 100)
                            tp_price = entry_price * (1 + tp / 100)
                        else:  # Short
                            sl_price = entry_price * (1 + sl / 100)
                            tp_price = entry_price * (1 - tp / 100)
                            
                        exit_price = None
                        exit_bar = max_bars
                        
                        for j in range(1, min(max_bars + 1, len(price_data) - idx_pos)):
                            bar_high = price_data['high'].iloc[idx_pos + j]
                            bar_low = price_data['low'].iloc[idx_pos + j]
                            
                            if direction == 1:
                                if bar_low <= sl_price:
                                    exit_price = sl_price
                                    exit_bar = j
                                    hits['sl'] += 1
                                    break
                                if bar_high >= tp_price:
                                    exit_price = tp_price
                                    exit_bar = j
                                    hits['tp'] += 1
                                    break
                            else:
                                if bar_high >= sl_price:
                                    exit_price = sl_price
                                    exit_bar = j
                                    hits['sl'] += 1
                                    break
                                if bar_low <= tp_price:
                                    exit_price = tp_price
                                    exit_bar = j
                                    hits['tp'] += 1
                                    break
                                    
                        if exit_price is None:
                            if idx_pos + max_bars < len(price_data):
                                exit_price = price_data['close'].iloc[idx_pos + max_bars]
                                hits['time'] += 1
                            else:
                                continue
                                
                        ret = (exit_price / entry_price - 1) * direction
                        returns.append(ret)
                        bars_held.append(exit_bar)
                        
                        # Store trade details for diagnostics
                        if sl == 4.0 and tp == 5.0 and len(returns) <= 5:
                            logger.info("="*70)
                            logger.info(f"TRADE {len(returns)} DIAGNOSTICS (SL={sl}%, TP={tp}%)")
                            logger.info("="*70)
                            logger.info(f"  Entry bar idx:   {idx_pos}")
                            logger.info(f"  Exit bar:        {exit_bar} (after {exit_bar} bars = {exit_bar * 4}h)")
                            logger.info(f"  Bars held:       {exit_bar}")
                            logger.info(f"  Entry price:     ${entry_price:.2f}")
                            logger.info(f"  Exit price:      ${exit_price:.2f}")
                            logger.info(f"  Direction:       {'LONG' if direction == 1 else 'SHORT'}")
                            logger.info(f"  SL price:        ${sl_price:.2f}")
                            logger.info(f"  TP price:        ${tp_price:.2f}")
                            expected_ret = (exit_price - entry_price) / entry_price
                            logger.info(f"  Expected return: {expected_ret*100:.4f}% (raw price change)")
                            logger.info(f"  Actual return:   {ret*100:.4f}% (direction-adjusted)")
                            if direction == -1:
                                logger.info(f"  SHORT math: -1 * ({exit_price:.2f}/{entry_price:.2f} - 1) = {ret*100:.4f}%")
                            logger.info("="*70)
                        
                    except Exception:
                        continue
                        
                if len(returns) >= self.min_trades // 2:
                    avg_ret = np.mean(returns)
                    best_returns[(sl, tp)] = avg_ret
                    
        # Get top combinations
        if best_returns:
            sorted_combos = sorted(best_returns.items(), key=lambda x: -x[1])
            
            for (sl, tp), avg_ret in sorted_combos[:5]:
                exits.append(ExitRule(
                    rule_id=f"sl_tp_{sl}_{tp}",
                    exit_type='stop_loss_take_profit',
                    conditions=[
                        {'feature': 'stop_loss', 'operator': '=', 'threshold': sl},
                        {'feature': 'take_profit', 'operator': '=', 'threshold': tp}
                    ],
                    priority=1,
                    avg_bars_held=0.0,
                    avg_exit_return=avg_ret,
                    hit_rate=0.0
                ))
                
        return exits
        
    def _discover_time_exits(
        self,
        price_data: pd.DataFrame,
        entry_indices: List,
        direction: int,
        max_bars: int
    ) -> List[ExitRule]:
        """Find optimal time-based exit."""
        exits = []
        
        # CRITICAL FIX: Minimum hold period should be 6 bars (24 hours on 4h timeframe)
        # Holding for only 1-2 bars captures noise, not signal
        # Original bug: [1, 2, 4, 8, 12, 24, 48] - first exit was after just 4 hours!
        time_periods = [6, 12, 18, 24, 48] if max_bars >= 48 else [max(6, p) for p in range(6, max_bars + 1, max(1, max_bars // 5))]
        
        for period in time_periods:
            returns = []
            
            for entry_idx in entry_indices:
                try:
                    idx_pos = price_data.index.get_loc(entry_idx)
                    
                    if idx_pos + period < len(price_data):
                        entry_price = price_data['close'].iloc[idx_pos]
                        exit_price = price_data['close'].iloc[idx_pos + period]
                        ret = (exit_price / entry_price - 1) * direction
                        returns.append(ret)
                except Exception:
                    continue
                    
            if len(returns) >= self.min_trades // 2:
                avg_ret = np.mean(returns)
                exits.append(ExitRule(
                    rule_id=f"time_{period}",
                    exit_type='time',
                    conditions=[
                        {'feature': 'bars_held', 'operator': '>=', 'threshold': period}
                    ],
                    priority=2,
                    avg_bars_held=float(period),
                    avg_exit_return=avg_ret,
                    hit_rate=1.0
                ))
                
        return exits
        
    def _discover_conditional_exits(
        self,
        features: pd.DataFrame,
        price_data: pd.DataFrame,
        entry_indices: List,
        direction: int,
        max_bars: int
    ) -> List[ExitRule]:
        """Find condition-based exits (e.g., RSI cross)."""
        exits = []
        
        # Test common exit signals
        exit_signals = [
            ('rsi_14', '>', 70, 'overbought'),
            ('rsi_14', '<', 30, 'oversold'),
            ('macd_histogram', '<', 0, 'macd_bearish'),
            ('macd_histogram', '>', 0, 'macd_bullish'),
            ('bb_position', '>', 1, 'bb_upper'),
            ('bb_position', '<', 0, 'bb_lower'),
        ]
        
        for feat, op, thresh, name in exit_signals:
            if feat not in features.columns:
                continue
                
            returns = []
            bars_held_list = []
            hits = 0
            
            for entry_idx in entry_indices:
                try:
                    idx_pos = features.index.get_loc(entry_idx)
                    entry_price = price_data['close'].iloc[idx_pos]
                    
                    exit_price = None
                    exit_bar = 0
                    
                    for j in range(1, min(max_bars + 1, len(features) - idx_pos)):
                        feat_val = features[feat].iloc[idx_pos + j]
                        
                        triggered = False
                        if op == '>' and feat_val > thresh:
                            triggered = True
                        elif op == '<' and feat_val < thresh:
                            triggered = True
                            
                        if triggered:
                            exit_price = price_data['close'].iloc[idx_pos + j]
                            exit_bar = j
                            hits += 1
                            break
                            
                    if exit_price is None:
                        if idx_pos + max_bars < len(price_data):
                            exit_price = price_data['close'].iloc[idx_pos + max_bars]
                            exit_bar = max_bars
                        else:
                            continue
                            
                    ret = (exit_price / entry_price - 1) * direction
                    returns.append(ret)
                    bars_held_list.append(exit_bar)
                    
                except Exception:
                    continue
                    
            if len(returns) >= self.min_trades // 2:
                avg_ret = np.mean(returns)
                avg_bars = np.mean(bars_held_list)
                hit_rate = hits / len(returns) if returns else 0
                
                exits.append(ExitRule(
                    rule_id=f"cond_{name}",
                    exit_type='signal',
                    conditions=[
                        {'feature': feat, 'operator': op, 'threshold': thresh}
                    ],
                    priority=3,
                    avg_bars_held=avg_bars,
                    avg_exit_return=avg_ret,
                    hit_rate=hit_rate
                ))
                
        return exits


def discover_complete_strategy(
    features: pd.DataFrame,
    price_data: pd.DataFrame,
    forward_returns: pd.Series,
    top_features: List[str],
    min_trades: int = 20  # Lowered from 50 due to overlap removal
) -> DiscoveredStrategy:
    """
    Discover complete trading strategy (entries + exits).
    
    Args:
        features: Feature DataFrame
        price_data: OHLCV data
        forward_returns: Target returns
        top_features: List of important features
        min_trades: Minimum trades for validity
        
    Returns:
        DiscoveredStrategy with entry and exit rules
    """
    logger.info("Discovering complete strategy...")
    
    # Discover entries
    entry_discovery = EntryConditionDiscovery(min_trades=min_trades)
    entry_rules = entry_discovery.discover(
        features, forward_returns, top_features=top_features
    )
    
    if not entry_rules:
        logger.warning("No valid entry rules discovered")
        return DiscoveredStrategy(
            name="No Strategy",
            entry_rules=[],
            exit_rules=[],
            performance={},
            feature_importance={}
        )
        
    # Discover exits for best entry rule
    exit_discovery = ExitConditionDiscovery(min_trades=min_trades)
    exit_rules = exit_discovery.discover(
        features, price_data, entry_rules[0]
    )
    
    # Calculate strategy performance
    best_entry = entry_rules[0]
    performance = {
        'entry_win_rate': best_entry.win_rate,
        'entry_sharpe': best_entry.sharpe,
        'n_trades': best_entry.support,
        'max_drawdown': best_entry.max_drawdown,
        'avg_return': best_entry.avg_return,
    }
    
    if exit_rules:
        performance['best_exit_return'] = exit_rules[0].avg_exit_return
        
    # Feature importance from rules
    feature_counts = defaultdict(int)
    for rule in entry_rules[:10]:
        for cond in rule.conditions:
            feature_counts[cond['feature']] += 1
            
    total_counts = sum(feature_counts.values()) or 1
    feature_importance = {f: c / total_counts for f, c in feature_counts.items()}
    
    return DiscoveredStrategy(
        name=f"Discovered_{best_entry.rule_id}",
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        performance=performance,
        feature_importance=feature_importance
    )
