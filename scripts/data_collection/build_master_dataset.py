"""
Build Master Dataset - One-Time Historical Feature Engineering

This script creates a comprehensive master dataset by:
1. Loading raw historical games (2003-2022)
2. Integrating injury data (synthetic for historical periods)
3. Integrating Vegas betting data (real data from Kaggle)
4. Running full feature engineering
5. Combining with already-engineered recent games (2021-2025)
6. Saving as master dataset for training

This only needs to run ONCE, then daily updates will be incremental.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Paths
PROJECTPATH = Path(__file__).resolve().parents[2]
DATAPATH = PROJECTPATH / 'data'
sys.path.insert(0, str(PROJECTPATH))

def load_raw_historical_games():
    """Load raw games from 2003-2022"""
    print("📊 Loading raw historical games (2003-2022)...")
    df = pd.read_csv(DATAPATH / 'original' / 'games.csv')
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], format='mixed', errors='coerce')
    
    # Filter to only completed games (not future)
    df = df[df['PTS_home'] > 0]
    
    print(f"   Loaded {len(df):,} completed games")
    print(f"   Date range: {df['GAME_DATE_EST'].min().date()} to {df['GAME_DATE_EST'].max().date()}")
    print(f"   Columns: {len(df.columns)}")
    return df


def load_already_engineered_games():
    """Load games that already have features engineered (2021-2025)"""
    print("\n📊 Loading already-engineered recent games...")
    df = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv', low_memory=False)
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
    
    print(f"   Loaded {len(df):,} games with features")
    print(f"   Date range: {df['GAME_DATE_EST'].min().date()} to {df['GAME_DATE_EST'].max().date()}")
    print(f"   Columns: {len(df.columns)} (includes injury + Vegas features)")
    return df


def integrate_vegas_data(df):
    """Integrate real Vegas betting data from Kaggle"""
    print("\n🎲 Integrating Vegas betting lines...")
    
    # Check if we have the processed Vegas data
    vegas_file = DATAPATH / 'betting' / 'nba_odds_2007_2020.csv'
    if not vegas_file.exists():
        print("   ⚠️  Real Vegas data not found, using placeholders")
        # Add placeholder columns
        df['spread'] = 0.0
        df['total'] = 0.0
        df['moneyline_home'] = 0.0
        df['moneyline_away'] = 0.0
        df['data_source'] = 'synthetic'
        df['is_real_vegas_line'] = False
        return df
    
    # Load Vegas data
    vegas_df = pd.read_csv(vegas_file, low_memory=False)
    print(f"   Loaded {len(vegas_df):,} Vegas lines")
    
    # Merge Vegas data with games
    # This is a simplified merge - adjust keys based on actual Vegas data structure
    df = df.merge(vegas_df, how='left', on='GAME_ID')
    
    # Fill missing Vegas data with 0
    vegas_cols = ['spread', 'total', 'moneyline_home', 'moneyline_away']
    for col in vegas_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    df['data_source'] = df.get('data_source', 'kaggle_historical')
    df['is_real_vegas_line'] = df['spread'].notna()
    
    print(f"   ✅ Vegas data integrated")
    return df


def integrate_injury_data(df):
    """Integrate injury data (synthetic for historical periods)"""
    print("\n🏥 Integrating injury data...")
    
    # For historical data, we'll use synthetic/estimated injury data
    # This provides baseline injury features even for old games
    
    # Add injury columns with zeros (will be filled by feature engineering)
    injury_cols = [
        'HOME_injuries_active', 'HOME_star_injuries', 'HOME_injury_impact',
        'HOME_injuries_severity', 'HOME_injuries_major', 'HOME_injuries_recent_7d',
        'HOME_days_since_injury',
        'VISITOR_injuries_active', 'VISITOR_star_injuries', 'VISITOR_injury_impact',
        'VISITOR_injuries_severity', 'VISITOR_injuries_major', 'VISITOR_injuries_recent_7d',
        'VISITOR_days_since_injury'
    ]
    
    for col in injury_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    print(f"   ✅ Added {len(injury_cols)} injury feature columns")
    return df


def run_feature_engineering(df):
    """Run full feature engineering pipeline"""
    print("\n🔧 Running feature engineering on historical data...")
    print(f"   Input: {len(df):,} games, {len(df.columns)} columns")
    
    # Import feature engineering
    from src.feature_engineering import process_features
    
    print("   This will take 5-10 minutes for 25k games...")
    print("   Progress indicators:")
    
    try:
        df_engineered = process_features(df)
        print(f"\n   ✅ Feature engineering complete!")
        print(f"   Output: {len(df_engineered):,} games, {len(df_engineered.columns)} columns")
        return df_engineered
    except Exception as e:
        print(f"\n   ❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None


def combine_with_recent_data(df_historical, df_recent):
    """Combine historical engineered data with recent data"""
    print("\n🔗 Combining historical and recent data...")
    
    # Find date overlap
    hist_max = df_historical['GAME_DATE_EST'].max()
    recent_min = df_recent['GAME_DATE_EST'].min()
    
    print(f"   Historical ends: {hist_max.date()}")
    print(f"   Recent starts: {recent_min.date()}")
    
    # Remove overlap from historical data (keep recent as source of truth)
    df_historical_clean = df_historical[df_historical['GAME_DATE_EST'] < recent_min]
    
    print(f"   Historical (clean): {len(df_historical_clean):,} games")
    print(f"   Recent: {len(df_recent):,} games")
    
    # Align columns
    common_cols = list(set(df_historical_clean.columns) & set(df_recent.columns))
    print(f"   Common columns: {len(common_cols)}")
    
    # Combine
    df_combined = pd.concat([
        df_historical_clean[common_cols],
        df_recent[common_cols]
    ], ignore_index=True)
    
    # Sort by date
    df_combined = df_combined.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    print(f"   ✅ Combined: {len(df_combined):,} games")
    print(f"   Date range: {df_combined['GAME_DATE_EST'].min().date()} to {df_combined['GAME_DATE_EST'].max().date()}")
    
    return df_combined


def save_master_dataset(df):
    """Save the master dataset"""
    print("\n💾 Saving master dataset...")
    
    output_file = DATAPATH / 'games_master_engineered.csv'
    df.to_csv(output_file, index=False)
    
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"   ✅ Saved to: {output_file}")
    print(f"   📏 Size: {size_mb:.1f} MB")
    print(f"   📊 Games: {len(df):,}")
    print(f"   📊 Columns: {len(df.columns)}")
    
    # Verify features
    injury_cols = [c for c in df.columns if 'injury' in c.lower() or 'injuries' in c.lower()]
    vegas_cols = [c for c in df.columns if 'spread' in c.lower() or 'moneyline' in c.lower()]
    
    print(f"\n   Feature verification:")
    print(f"   🏥 Injury features: {len(injury_cols)}")
    print(f"   🎲 Vegas features: {len(vegas_cols)}")
    
    return output_file


def main():
    """Main execution"""
    print("=" * 80)
    print("🏗️  BUILDING MASTER DATASET")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Load raw historical data
    df_historical = load_raw_historical_games()
    
    # Step 2: Load already-engineered recent data
    df_recent = load_already_engineered_games()
    
    # Step 3: Integrate Vegas data into historical
    df_historical = integrate_vegas_data(df_historical)
    
    # Step 4: Integrate injury data into historical
    df_historical = integrate_injury_data(df_historical)
    
    # Step 5: Run feature engineering on historical data
    df_historical_engineered = run_feature_engineering(df_historical)
    
    if df_historical_engineered is None:
        print("\n❌ Feature engineering failed. Exiting.")
        return False
    
    # Step 6: Combine with recent data
    df_master = combine_with_recent_data(df_historical_engineered, df_recent)
    
    # Step 7: Save master dataset
    output_file = save_master_dataset(df_master)
    
    print("\n" + "=" * 80)
    print("✅ MASTER DATASET BUILD COMPLETE!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📝 Next steps:")
    print("   1. Update scripts/predictions/setup_model.py to use games_master_engineered.csv")
    print("   2. Retrain model with full historical data")
    print("   3. Test predictions")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

