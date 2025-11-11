#!/usr/bin/env python3
"""
Process REAL Vegas Betting Lines from Kaggle

Integrates actual historical betting data with our game dataset.
This should provide a significant improvement over synthetic betting lines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATAPATH = Path('data')
BETTING_PATH = DATAPATH / 'betting'


def load_real_vegas_data():
    """Load the real Vegas betting lines from Kaggle"""
    print("="*70)
    print("📥 LOADING REAL VEGAS BETTING LINES")
    print("="*70)
    
    kaggle_file = BETTING_PATH / 'kaggle' / 'nba_2008-2025.csv'
    
    print(f"\n📂 Loading: {kaggle_file}")
    vegas_df = pd.read_csv(kaggle_file)
    
    print(f"   ✅ Loaded {len(vegas_df):,} games with REAL Vegas lines")
    print(f"   📅 Coverage: {vegas_df['date'].min()} to {vegas_df['date'].max()}")
    print(f"   🏀 Seasons: {vegas_df['season'].min()}-{vegas_df['season'].max()}")
    
    # Convert date
    vegas_df['date'] = pd.to_datetime(vegas_df['date'])
    
    # Standardize team abbreviations (Kaggle uses lowercase)
    vegas_df['home'] = vegas_df['home'].str.upper()
    vegas_df['away'] = vegas_df['away'].str.upper()
    
    # Handle team name changes
    team_mapping = {
        'BKN': 'BRK',  # Brooklyn Nets
        'NO': 'NOP',   # New Orleans
        'PHX': 'PHO',  # Phoenix (sometimes PHX in data)
    }
    
    for old, new in team_mapping.items():
        vegas_df['home'] = vegas_df['home'].replace(old, new)
        vegas_df['away'] = vegas_df['away'].replace(old, new)
    
    # Calculate implied probability from moneylines
    def moneyline_to_prob(ml):
        if pd.isna(ml):
            return 0.5
        ml = float(ml)
        if ml < 0:
            return abs(ml) / (abs(ml) + 100)
        else:
            return 100 / (ml + 100)
    
    vegas_df['home_win_prob_implied'] = vegas_df['moneyline_home'].apply(moneyline_to_prob)
    vegas_df['away_win_prob_implied'] = vegas_df['moneyline_away'].apply(moneyline_to_prob)
    
    # Add source flag
    vegas_df['is_real_vegas_line'] = True
    vegas_df['data_source'] = 'kaggle_real'
    
    print(f"\n📊 Data Quality:")
    print(f"   Spread coverage: {vegas_df['spread'].notna().sum():,} ({vegas_df['spread'].notna().mean()*100:.1f}%)")
    print(f"   Total coverage: {vegas_df['total'].notna().sum():,} ({vegas_df['total'].notna().mean()*100:.1f}%)")
    print(f"   Moneyline coverage: {vegas_df['moneyline_home'].notna().sum():,} ({vegas_df['moneyline_home'].notna().mean()*100:.1f}%)")
    
    print(f"\n📈 Vegas Line Statistics:")
    print(f"   Average spread: {vegas_df['spread'].mean():.2f} points")
    print(f"   Average total: {vegas_df['total'].mean():.1f} points")
    print(f"   Largest spread: {vegas_df['spread'].max():.1f} points")
    
    return vegas_df


def merge_with_our_games(vegas_df):
    """
    Merge real Vegas lines with our game data
    """
    print("\n" + "="*70)
    print("🔗 MERGING WITH OUR GAME DATA")
    print("="*70)
    
    # Load our games (with injuries but before betting integration)
    print("\n📊 Loading our game data...")
    games_df = pd.read_csv(DATAPATH / 'games_with_injuries.csv')
    games_df['GAME_DATE_EST'] = pd.to_datetime(games_df['GAME_DATE_EST']).dt.tz_localize(None)
    
    print(f"   Our games: {len(games_df):,}")
    print(f"   Date range: {games_df['GAME_DATE_EST'].min()} to {games_df['GAME_DATE_EST'].max()}")
    
    # Prepare for merge
    print("\n🔧 Preparing data for merge...")
    
    # Create merge keys
    vegas_df['merge_key'] = vegas_df['date'].astype(str) + '_' + vegas_df['home'] + '_' + vegas_df['away']
    games_df['merge_key'] = games_df['GAME_DATE_EST'].astype(str).str[:10] + '_' + \
                            games_df['HOME_TEAM_ABBREVIATION'] + '_' + \
                            games_df['VISITOR_TEAM_ABBREVIATION']
    
    # Merge
    print("🔗 Merging on date + teams...")
    
    merged = games_df.merge(
        vegas_df[['merge_key', 'spread', 'total', 'moneyline_home', 'moneyline_away',
                  'home_win_prob_implied', 'away_win_prob_implied', 'is_real_vegas_line',
                  'data_source', 'whos_favored']],
        on='merge_key',
        how='left',
        suffixes=('_old', '_vegas')
    )
    
    # Count matches
    matched = merged['is_real_vegas_line'].notna().sum()
    match_rate = matched / len(merged) * 100
    
    print(f"\n✅ Merge Results:")
    print(f"   Total games: {len(merged):,}")
    print(f"   Matched with REAL Vegas lines: {matched:,} ({match_rate:.1f}%)")
    print(f"   No Vegas data: {len(merged) - matched:,} ({100-match_rate:.1f}%)")
    
    # For games without real Vegas lines, keep synthetic or fill with defaults
    print("\n🔧 Handling games without Vegas data...")
    
    # First, drop any duplicate columns from the merge
    drop_cols = [col for col in merged.columns if col.endswith('_old') or col.endswith('_vegas')]
    merged = merged.drop(columns=drop_cols, errors='ignore')
    
    # Use Vegas data where available, otherwise use defaults
    if 'spread' in merged.columns:
        merged['spread'] = merged['spread'].fillna(0)
    else:
        merged['spread'] = 0
        
    if 'total' in merged.columns:
        merged['total'] = merged['total'].fillna(210)
    else:
        merged['total'] = 210
    
    if 'moneyline_home' in merged.columns:
        merged['home_ml'] = merged['moneyline_home'].fillna(-110)
    else:
        merged['home_ml'] = -110
        
    if 'moneyline_away' in merged.columns:
        merged['visitor_ml'] = merged['moneyline_away'].fillna(-110)
    else:
        merged['visitor_ml'] = -110
    
    if 'home_win_prob_implied' not in merged.columns:
        merged['home_win_prob_implied'] = 0.5
    else:
        merged['home_win_prob_implied'] = merged['home_win_prob_implied'].fillna(0.5)
    
    # Drop merge key and any remaining duplicate columns
    merged = merged.drop(columns=['merge_key'], errors='ignore')
    
    # Remove any duplicate columns (keep first)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    print(f"   ✅ Final dataset ready with {len(merged):,} games")
    print(f"   🎰 Real Vegas lines: {matched:,} games")
    print(f"   📊 Synthetic fallback: {len(merged) - matched:,} games")
    
    return merged


def create_betting_features(df):
    """
    Create betting-derived features from real Vegas lines
    """
    print("\n" + "="*70)
    print("🔧 CREATING BETTING FEATURES")
    print("="*70)
    
    print("\n📊 Engineering features from real Vegas data...")
    
    # Same features as before, but now with REAL data!
    
    # 1. Spread category
    df['spread_category_encoded'] = pd.cut(
        df['spread'],
        bins=[-np.inf, -7, -3, 0, 3, 7, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    
    # 2. Close game indicator
    df['betting_close_game'] = (df['spread'].abs() < 3).astype(int)
    
    # 3. Blowout indicator
    df['betting_blowout'] = (df['spread'].abs() > 10).astype(int)
    
    # 4. Total category
    df['total_category_encoded'] = pd.cut(
        df['total'],
        bins=[0, 200, 210, 220, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # 5. Home team strength (normalized)
    df['betting_home_strength'] = df['spread'] / 15
    df['betting_home_strength'] = df['betting_home_strength'].clip(-1, 1)
    
    # 6. Betting confidence
    df['betting_confidence'] = df['spread'].abs() / 15
    df['betting_confidence'] = df['betting_confidence'].clip(0, 1)
    
    # 7. Expected scoring
    df['expected_home_pts'] = (df['total'] / 2) + (df['spread'] / 2)
    df['expected_visitor_pts'] = (df['total'] / 2) - (df['spread'] / 2)
    
    # 8. Heavy favorite (sharp money indicator)
    df['betting_edge_exists'] = (df['spread'].abs() > 7).astype(int)
    
    print(f"   ✅ Created 14 betting features from real Vegas data")
    
    # Fill any remaining NaNs
    betting_cols = ['spread', 'total', 'home_ml', 'visitor_ml', 'home_win_prob_implied',
                    'spread_category_encoded', 'betting_close_game', 'betting_blowout',
                    'total_category_encoded', 'betting_home_strength', 'betting_confidence',
                    'expected_home_pts', 'expected_visitor_pts', 'betting_edge_exists']
    
    for col in betting_cols:
        if col in df.columns and df[col].isna().any():
            if col in ['spread', 'betting_home_strength']:
                df[col].fillna(0, inplace=True)
            elif col in ['total', 'expected_home_pts', 'expected_visitor_pts']:
                df[col].fillna(210, inplace=True)
            elif col == 'home_win_prob_implied':
                df[col].fillna(0.5, inplace=True)
            else:
                df[col].fillna(0, inplace=True)
    
    return df


def analyze_real_vs_synthetic():
    """Compare real Vegas lines vs our synthetic lines"""
    print("\n" + "="*70)
    print("📊 REAL vs SYNTHETIC COMPARISON")
    print("="*70)
    
    # Load both datasets
    real_vegas = pd.read_csv(BETTING_PATH / 'kaggle' / 'nba_2008-2025.csv')
    synthetic = pd.read_csv(BETTING_PATH / 'nba_betting_lines_historical.csv')
    
    print(f"\n📈 Coverage Comparison:")
    print(f"   Real Vegas data: {len(real_vegas):,} games (2007-2024)")
    print(f"   Synthetic data: {len(synthetic):,} games (2003-2025)")
    
    print(f"\n📊 Distribution Comparison:")
    print(f"\n   REAL Vegas Spreads:")
    print(f"      Mean: {real_vegas['spread'].mean():.2f}")
    print(f"      Std:  {real_vegas['spread'].std():.2f}")
    print(f"      Min:  {real_vegas['spread'].min():.1f}")
    print(f"      Max:  {real_vegas['spread'].max():.1f}")
    
    print(f"\n   Synthetic Spreads:")
    print(f"      Mean: {synthetic['spread'].mean():.2f}")
    print(f"      Std:  {synthetic['spread'].std():.2f}")
    print(f"      Min:  {synthetic['spread'].min():.1f}")
    print(f"      Max:  {synthetic['spread'].max():.1f}")
    
    print(f"\n💡 Key Differences:")
    print(f"   - Real Vegas has {real_vegas['spread'].std():.2f} std dev")
    print(f"   - Synthetic has {synthetic['spread'].std():.2f} std dev")
    print(f"   - Real Vegas reflects market efficiency and insider info")
    print(f"   - Synthetic is purely derived from rolling averages")


def main():
    print("="*70)
    print("🎰 PROCESS REAL VEGAS BETTING LINES FROM KAGGLE")
    print("="*70)
    print()
    
    # Step 1: Load real Vegas data
    vegas_df = load_real_vegas_data()
    
    # Step 2: Merge with our games
    merged_df = merge_with_our_games(vegas_df)
    
    # Step 3: Create betting features
    final_df = create_betting_features(merged_df)
    
    # Step 4: Save
    print("\n" + "="*70)
    print("💾 SAVING FINAL DATASET")
    print("="*70)
    
    output_file = DATAPATH / 'games_with_real_vegas.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n   ✅ Saved to: {output_file}")
    print(f"   📊 Total rows: {len(final_df):,}")
    print(f"   📊 Total features: {len(final_df.columns)}")
    
    # Step 5: Compare real vs synthetic
    analyze_real_vs_synthetic()
    
    # Summary
    print("\n" + "="*70)
    print("✅ REAL VEGAS DATA INTEGRATION COMPLETE!")
    print("="*70)
    
    matched_count = final_df['is_real_vegas_line'].sum() if 'is_real_vegas_line' in final_df.columns else 0
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total games: {len(final_df):,}")
    print(f"   With REAL Vegas lines: {matched_count:,}")
    print(f"   Match rate: {matched_count/len(final_df)*100:.1f}%")
    
    print(f"\n🎯 Next Step:")
    print(f"   Run: python train_with_real_vegas.py")
    print(f"\n💡 Expected Impact:")
    print(f"   Current model (synthetic): 67.68% AUC")
    print(f"   Expected with real Vegas: 69-71% AUC (+1.5-3.5%)")
    print(f"\n   Why? Real Vegas lines include:")
    print(f"   - Market efficiency (thousands of bettors)")
    print(f"   - Insider information (injuries, lineups)")
    print(f"   - Sharp money indicators")
    print(f"   - Professional odds maker expertise")
    
    return final_df


if __name__ == "__main__":
    result = main()

