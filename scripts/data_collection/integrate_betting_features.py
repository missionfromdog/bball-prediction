#!/usr/bin/env python3
"""
Integrate Vegas Betting Lines with Game Data

Merges betting lines with engineered features and creates new betting-based features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATAPATH = Path('data')
BETTING_PATH = DATAPATH / 'betting'


def integrate_betting_lines():
    """
    Integrate betting lines with game data
    """
    print("="*70)
    print("🎰 INTEGRATING BETTING LINES WITH GAME DATA")
    print("="*70)
    
    # Load game data
    print("\n📊 Step 1: Loading game data...")
    games_df = pd.read_csv(DATAPATH / 'games_with_injuries.csv')
    print(f"   Games: {len(games_df):,}")
    
    # Load betting lines
    print("\n📊 Step 2: Loading betting lines...")
    betting_df = pd.read_csv(BETTING_PATH / 'nba_betting_lines_historical.csv')
    print(f"   Betting lines: {len(betting_df):,}")
    
    # Merge on GAME_ID
    print("\n🔗 Step 3: Merging data...")
    
    if 'GAME_ID' in games_df.columns and 'GAME_ID' in betting_df.columns:
        merged = games_df.merge(betting_df, on='GAME_ID', how='left', suffixes=('', '_betting'))
        print(f"   ✅ Merged on GAME_ID")
    else:
        print("   ⚠️  No GAME_ID, merging on date and teams...")
        # Ensure date columns are aligned
        games_df['GAME_DATE_EST'] = pd.to_datetime(games_df['GAME_DATE_EST']).dt.tz_localize(None)
        betting_df['GAME_DATE_EST'] = pd.to_datetime(betting_df['GAME_DATE_EST'])
        
        # Merge on date and teams
        merged = games_df.merge(
            betting_df,
            left_on=['GAME_DATE_EST', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION'],
            right_on=['GAME_DATE_EST', 'HOME_TEAM', 'VISITOR_TEAM'],
            how='left'
        )
    
    print(f"   Merged data: {len(merged):,} rows")
    print(f"   Betting features populated: {merged['spread'].notna().sum():,} ({merged['spread'].notna().mean()*100:.1f}%)")
    
    # Create additional betting features
    print("\n🔧 Step 4: Creating derived betting features...")
    
    # 1. Spread category
    merged['spread_category'] = pd.cut(
        merged['spread'],
        bins=[-np.inf, -7, -3, 0, 3, 7, np.inf],
        labels=['heavy_visitor_fav', 'visitor_fav', 'visitor_slight', 'home_slight', 'home_fav', 'heavy_home_fav']
    )
    merged['spread_category_encoded'] = merged['spread_category'].cat.codes
    
    # 2. Is it a close game? (spread < 3)
    merged['betting_close_game'] = (merged['spread'].abs() < 3).astype(int)
    
    # 3. Is it a blowout prediction? (spread > 10)
    merged['betting_blowout'] = (merged['spread'].abs() > 10).astype(int)
    
    # 4. Total category (high/low scoring game)
    merged['total_category'] = pd.cut(
        merged['total'],
        bins=[0, 200, 210, 220, np.inf],
        labels=['low_scoring', 'normal', 'high_scoring', 'very_high_scoring']
    )
    merged['total_category_encoded'] = merged['total_category'].cat.codes
    
    # 5. Home team strength from betting (normalized spread)
    merged['betting_home_strength'] = merged['spread'] / 15  # Normalize -15 to +15 → -1 to +1
    
    # 6. Confidence from betting market (how far from pick'em)
    merged['betting_confidence'] = merged['spread'].abs() / 15
    merged['betting_confidence'] = merged['betting_confidence'].clip(0, 1)
    
    # 7. Expected scoring (from total)
    merged['expected_home_pts'] = (merged['total'] / 2) + (merged['spread'] / 2)
    merged['expected_visitor_pts'] = (merged['total'] / 2) - (merged['spread'] / 2)
    
    # Fill NaN values for any missing betting lines
    betting_cols = ['spread', 'total', 'home_ml', 'visitor_ml', 'home_win_prob_implied',
                    'betting_edge_exists', 'spread_category_encoded', 'betting_close_game',
                    'betting_blowout', 'total_category_encoded', 'betting_home_strength',
                    'betting_confidence', 'expected_home_pts', 'expected_visitor_pts']
    
    for col in betting_cols:
        if col in merged.columns:
            if merged[col].isna().any():
                # Fill with neutral values
                if col in ['spread', 'betting_home_strength']:
                    merged[col].fillna(0, inplace=True)
                elif col in ['total', 'expected_home_pts', 'expected_visitor_pts']:
                    merged[col].fillna(210, inplace=True)
                elif col == 'home_win_prob_implied':
                    merged[col].fillna(0.5, inplace=True)
                else:
                    merged[col].fillna(0, inplace=True)
    
    print(f"   ✅ Created {len(betting_cols)} betting-derived features")
    
    # Summary statistics
    print("\n📊 Step 5: Betting Line Statistics...")
    print(f"\n   Spread distribution:")
    print(f"      Home favorites (spread > 0): {(merged['spread'] > 0).sum():,} ({(merged['spread'] > 0).mean()*100:.1f}%)")
    print(f"      Pick'em (spread ≈ 0):        {(merged['spread'].abs() < 1).sum():,} ({(merged['spread'].abs() < 1).mean()*100:.1f}%)")
    print(f"      Visitor favorites (< 0):     {(merged['spread'] < 0).sum():,} ({(merged['spread'] < 0).mean()*100:.1f}%)")
    
    print(f"\n   Game types:")
    print(f"      Close games (spread < 3):    {merged['betting_close_game'].sum():,} ({merged['betting_close_game'].mean()*100:.1f}%)")
    print(f"      Blowouts (spread > 10):      {merged['betting_blowout'].sum():,} ({merged['betting_blowout'].mean()*100:.1f}%)")
    
    print(f"\n   Scoring:")
    print(f"      Average total: {merged['total'].mean():.1f}")
    print(f"      High scoring games (>220): {(merged['total'] > 220).sum():,}")
    print(f"      Low scoring games (<200):  {(merged['total'] < 200).sum():,}")
    
    # Correlation with outcome
    if 'TARGET' in merged.columns or 'HOME_TEAM_WINS' in merged.columns:
        target_col = 'TARGET' if 'TARGET' in merged.columns else 'HOME_TEAM_WINS'
        
        print(f"\n📈 Correlation with {target_col}:")
        betting_features_corr = [
            'home_win_prob_implied', 'betting_home_strength', 'spread',
            'betting_confidence', 'betting_close_game', 'betting_blowout'
        ]
        
        correlations = []
        for feat in betting_features_corr:
            if feat in merged.columns:
                corr = merged[feat].corr(merged[target_col])
                correlations.append((feat, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feat, corr in correlations:
            print(f"      {feat:30s}: {corr:+.4f}")
    
    # Save
    print("\n💾 Step 6: Saving merged data...")
    output_file = DATAPATH / 'games_with_betting.csv'
    merged.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   Total features: {len(merged.columns)}")
    
    betting_feature_names = [col for col in merged.columns if 'betting' in col.lower() or col in ['spread', 'total', 'home_ml', 'visitor_ml', 'home_win_prob_implied']]
    print(f"   Betting features: {len(betting_feature_names)}")
    
    print("\n" + "="*70)
    print("✅ BETTING LINES INTEGRATED!")
    print("="*70)
    print("\n🎯 Next step:")
    print("   Run: python train_with_betting.py")
    print("\n💡 Expected impact:")
    print("   Betting lines typically provide +3-5% AUC improvement")
    print("   They aggregate ALL available information into one signal")
    
    return merged


if __name__ == "__main__":
    result = integrate_betting_lines()

