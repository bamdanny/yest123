#!/usr/bin/env python3
"""
Verification script: Check if lagged liquidation features exist AND survive selection.

Usage:
    python check_liq_features.py
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    print("\n" + "="*70)
    print("LIQUIDATION FEATURE VERIFICATION")
    print("="*70)
    
    # Check feature_scores.csv (selected features)
    scores_path = None
    for path in ['reports/feature_scores.csv', 'feature_scores.csv']:
        if Path(path).exists():
            scores_path = path
            break
    
    if scores_path:
        df = pd.read_csv(scores_path)
        print(f"\n=== SELECTED FEATURES ({len(df)} total) ===")
        print(f"Loaded: {scores_path}")
        
        # Find all features containing 'liq'
        liq_features = df[df['feature'].str.contains('liq', case=False, na=False)]
        print(f"\nLiquidation features in TOP 50 SELECTED: {len(liq_features)}")
        
        if len(liq_features) > 0:
            print("\nSelected liquidation features:")
            for _, row in liq_features.iterrows():
                print(f"  - {row['feature']}: {row['importance']:.4f}")
    else:
        print("\n⚠️ feature_scores.csv not found - run train_model.py first")
        df = None
    
    # Also check the cache to see ALL generated features
    cache_path = 'data_cache/features_cache.pkl'
    if Path(cache_path).exists():
        import pickle
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        all_features = cache['features'].columns.tolist()
        all_liq = [c for c in all_features if 'liq' in c.lower()]
        
        print(f"\n=== ALL GENERATED FEATURES ===")
        print(f"Total features in cache: {len(all_features)}")
        print(f"Total liquidation features: {len(all_liq)}")
        
        # Check patterns
        patterns = {
            'liq_feat_.*_lag_': 'Lagged values',
            'liq_feat_.*_sum_': 'Rolling sums',
            'liq_feat_.*_mean_': 'Rolling means',
            'liq_feat_.*_max_': 'Rolling max',
            'liq_feat_.*_zscore_': 'Z-scores',
            'liq_feat_.*_mom_': 'Momentum',
        }
        
        import re
        print(f"\nLagged liquidation feature patterns:")
        total_lagged = 0
        for pattern, desc in patterns.items():
            matches = [c for c in all_liq if re.match(pattern, c)]
            total_lagged += len(matches)
            status = "✅" if len(matches) > 0 else "❌"
            print(f"  {status} {pattern}: {len(matches)} ({desc})")
        
        # Check same-bar (leaky) features
        same_bar_patterns = [
            'deriv_cg_liquidation_history_liq_ratio',
            'deriv_cg_liquidation_aggregated_liq_ratio',
            'deriv_cg_liquidation_history_long_liq_usd',
            'deriv_cg_liquidation_history_short_liq_usd',
            'deriv_cg_liquidation_history_total_liq_usd',
            'deriv_cg_liquidation_aggregated_long_liq_usd',
            'deriv_cg_liquidation_aggregated_short_liq_usd',
            'deriv_cg_liquidation_aggregated_total_liq_usd',
        ]
        same_bar_count = len([c for c in all_features if c in same_bar_patterns])
        
        print(f"\n=== SUMMARY ===")
        print(f"Same-bar liquidation (LEAKAGE): {same_bar_count}")
        print(f"Lagged liquidation (LEGITIMATE): {total_lagged}")
        
        if df is not None:
            selected_liq = len(liq_features)
            print(f"Liquidation in top 50 selected: {selected_liq}")
            
            if selected_liq >= 5:
                print("\n✅ PASS: Sufficient liquidation features in model")
            elif selected_liq > 0:
                print("\n⚠️ PARTIAL: Some liquidation features, but could be more")
            else:
                print("\n❌ FAIL: No liquidation features survived selection")
        
        if total_lagged >= 100:
            print("✅ PASS: Lagged features being generated correctly")
        else:
            print("❌ FAIL: Not enough lagged features generated")
            
    else:
        print(f"\n⚠️ Cache not found at {cache_path}")
        print("Run: python run_exhaustive_search.py --mode single --top-n 10")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
