"""
Feature Engineering - Create additional features for ML model

Adds interaction features and transformations based on Phase 1 insights.

Key Phase 1 findings:
- OI changes are strongest predictors
- 1h timeframe better than 24h
- Acceleration (2nd derivative) adds value
- Liquidation ratio is strong signal
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create additional features based on Phase 1 insights.
    
    Focuses on:
    1. OI × Liquidation interactions (both strong signals)
    2. Momentum of key indicators
    3. Regime features
    4. Cross-feature z-scores
    """
    
    def __init__(self):
        self.created_features: List[str] = []
    
    def engineer_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features.
        
        Args:
            features: Original feature DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info(f"Engineering features from {len(features.columns)} base features")
        
        df = features.copy()
        initial_cols = len(df.columns)
        
        # 1. OI × Liquidation interaction (both are strong signals from Phase 1)
        df = self._add_oi_liq_interactions(df)
        
        # 2. Momentum features (rate of change of key indicators)
        df = self._add_momentum_features(df)
        
        # 3. Regime indicators
        df = self._add_regime_features(df)
        
        # 4. Cross-feature z-scores for Tier 1 features
        df = self._add_cross_zscores(df)
        
        # 5. Funding × OI interactions
        df = self._add_funding_interactions(df)
        
        new_cols = len(df.columns) - initial_cols
        logger.info(f"After engineering: {len(df.columns)} features (+{new_cols} new)")
        logger.info(f"  Created features: {len(self.created_features)}")
        
        return df
    
    def _add_oi_liq_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OI × Liquidation interaction features."""
        
        # Find OI and liquidation columns
        oi_cols = [c for c in df.columns if 
                   ('oi_close_change' in c.lower() or 'oi_close_accel' in c.lower())
                   and 'history' not in c.lower()]  # Prefer aggregated over history
        liq_cols = [c for c in df.columns if 'liq_ratio' in c.lower()]
        
        interactions_created = 0
        for oi_col in oi_cols[:3]:  # Limit to top 3
            for liq_col in liq_cols[:2]:  # Limit to top 2
                if oi_col in df.columns and liq_col in df.columns:
                    # Multiplicative interaction
                    new_col = f"interact_oi_liq_{interactions_created}"
                    df[new_col] = df[oi_col] * df[liq_col]
                    self.created_features.append(new_col)
                    interactions_created += 1
        
        if interactions_created > 0:
            logger.info(f"  Added {interactions_created} OI×Liquidation interactions")
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum (rate of change) of key features."""
        
        # Key Tier 1 and Tier 2 features from Phase 1
        key_features = [
            'deriv_feat_cg_oi_aggregated_oi_close_change_1h',
            'deriv_cg_liquidation_aggregated_liq_ratio',
            'deriv_feat_cg_oi_aggregated_oi_close_accel',
            'price_bb_width_50'
        ]
        
        momentum_created = 0
        for feat in key_features:
            if feat in df.columns:
                # 1-bar momentum (first derivative)
                new_col = f"mom1_{feat[:30]}"
                df[new_col] = df[feat].diff(1)
                self.created_features.append(new_col)
                momentum_created += 1
                
                # 3-bar momentum
                new_col = f"mom3_{feat[:30]}"
                df[new_col] = df[feat].diff(3)
                self.created_features.append(new_col)
                momentum_created += 1
                
                # Momentum of momentum (acceleration change)
                new_col = f"mom_acc_{feat[:30]}"
                df[new_col] = df[feat].diff(1).diff(1)
                self.created_features.append(new_col)
                momentum_created += 1
        
        if momentum_created > 0:
            logger.info(f"  Added {momentum_created} momentum features")
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        
        regime_created = 0
        
        # Volatility regime (if parkinson vol exists)
        vol_cols = [c for c in df.columns if 'parkinson' in c.lower()]
        if vol_cols:
            vol_col = vol_cols[0]
            vol_percentile = df[vol_col].rolling(50, min_periods=10).rank(pct=True)
            
            # High volatility regime
            df['regime_high_vol'] = (vol_percentile > 0.8).astype(int)
            self.created_features.append('regime_high_vol')
            
            # Low volatility regime
            df['regime_low_vol'] = (vol_percentile < 0.2).astype(int)
            self.created_features.append('regime_low_vol')
            
            regime_created += 2
        
        # Funding regime (if funding exists)
        funding_cols = [c for c in df.columns if 'funding' in c.lower() and 'high' in c.lower()]
        if funding_cols:
            funding_col = funding_cols[0]
            funding_pct = df[funding_col].rolling(50, min_periods=10).rank(pct=True)
            
            # Positive funding regime (overleveraged longs)
            df['regime_high_funding'] = (funding_pct > 0.8).astype(int)
            self.created_features.append('regime_high_funding')
            
            # Negative funding regime (overleveraged shorts)
            df['regime_low_funding'] = (funding_pct < 0.2).astype(int)
            self.created_features.append('regime_low_funding')
            
            regime_created += 2
        
        # OI regime
        oi_cols = [c for c in df.columns if 'oi_close_change_1h' in c.lower()]
        if oi_cols:
            oi_col = oi_cols[0]
            oi_pct = df[oi_col].rolling(50, min_periods=10).rank(pct=True)
            
            # High OI change regime
            df['regime_high_oi'] = (oi_pct > 0.9).astype(int)
            self.created_features.append('regime_high_oi')
            
            regime_created += 1
        
        if regime_created > 0:
            logger.info(f"  Added {regime_created} regime features")
        
        return df
    
    def _add_cross_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-scores for Tier 1 features."""
        
        tier1_features = [
            'deriv_feat_cg_oi_aggregated_oi_close_change_1h',
            'deriv_cg_liquidation_aggregated_liq_ratio',
            'deriv_feat_cg_oi_aggregated_oi_close_accel'
        ]
        
        zscore_created = 0
        for feat in tier1_features:
            if feat in df.columns:
                # 20-bar rolling z-score
                rolling_mean = df[feat].rolling(20, min_periods=5).mean()
                rolling_std = df[feat].rolling(20, min_periods=5).std()
                
                new_col = f"zscore20_{feat[:25]}"
                df[new_col] = (df[feat] - rolling_mean) / (rolling_std + 1e-10)
                self.created_features.append(new_col)
                zscore_created += 1
                
                # 50-bar rolling z-score (longer lookback)
                rolling_mean = df[feat].rolling(50, min_periods=10).mean()
                rolling_std = df[feat].rolling(50, min_periods=10).std()
                
                new_col = f"zscore50_{feat[:25]}"
                df[new_col] = (df[feat] - rolling_mean) / (rolling_std + 1e-10)
                self.created_features.append(new_col)
                zscore_created += 1
        
        if zscore_created > 0:
            logger.info(f"  Added {zscore_created} z-score features")
        
        return df
    
    def _add_funding_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Funding × OI interactions (crowded trade detection)."""
        
        # Find funding and OI columns
        funding_cols = [c for c in df.columns if 'funding' in c.lower() and 
                       ('cumul' in c.lower() or 'zscore' in c.lower())]
        oi_cols = [c for c in df.columns if 'oi_close_change' in c.lower()]
        
        interactions_created = 0
        for fund_col in funding_cols[:2]:  # Limit
            for oi_col in oi_cols[:2]:
                if fund_col in df.columns and oi_col in df.columns:
                    # Funding × OI = crowded trade indicator
                    new_col = f"crowded_{interactions_created}"
                    df[new_col] = df[fund_col] * df[oi_col]
                    self.created_features.append(new_col)
                    interactions_created += 1
        
        if interactions_created > 0:
            logger.info(f"  Added {interactions_created} funding×OI interactions")
        
        return df
    
    def get_created_features(self) -> List[str]:
        """Return list of features created by this engineer."""
        return self.created_features
