"""
Run feature engineering on the full dataset

This script:
1. Loads the dataset (with any new games from schedule fetch)
2. Runs the full feature engineering pipeline
3. Saves the updated dataset with calculated features

This allows us to predict on new games by calculating their features
from each team's historical performance.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
PROJECTPATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECTPATH / 'src'))

from feature_engineering import process_features
from constants import LONG_INTEGER_FIELDS, SHORT_INTEGER_FIELDS, DATE_FIELDS

DATAPATH = PROJECTPATH / 'data'

def main():
    print("="*80)
    print("🔧 NBA FEATURE ENGINEERING")
    print("="*80)
    print()
    
    # Load dataset
    print("📊 Loading dataset...")
    data_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file, low_memory=False)
    initial_count = len(df)
    print(f"   Loaded {initial_count:,} games")
    
    # Leave GAME_ID as-is (don't convert - some IDs are too large even for int64)
    
    # Add PLAYOFF column if it doesn't exist (feature engineering expects it)
    if 'PLAYOFF' not in df.columns:
        df['PLAYOFF'] = 0  # All games are regular season by default
    
    # Check for unengineered games (PTS_home=0 and no rolling features)
    unplayed = df[df['PTS_home'] == 0]
    print(f"   Found {len(unplayed)} unplayed games")
    
    if 'HOME_TEAM_WINS_AVG_LAST_10_HOME' in df.columns:
        unengineered = unplayed[unplayed['HOME_TEAM_WINS_AVG_LAST_10_HOME'] == 0]
        print(f"   {len(unengineered)} need feature engineering")
        
        if len(unengineered) == 0:
            print("\n✅ All games already have features - nothing to do!")
            return
    
    # Run feature engineering
    print("\n🔧 Running feature engineering pipeline...")
    print("   (This may take 2-3 minutes for large datasets)")
    print()
    
    try:
        df_engineered = process_features(df)
        print(f"\n✅ Feature engineering complete!")
        print(f"   Processed {len(df_engineered):,} games")
        print(f"   Features: {len(df_engineered.columns)}")
    except Exception as e:
        print(f"\n❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save updated dataset
    print("\n💾 Saving updated dataset...")
    df_engineered.to_csv(data_file, index=False)
    print(f"   Saved to: {data_file}")
    
    # Verify unplayed games now have features
    if 'HOME_TEAM_WINS_AVG_LAST_10_HOME' in df_engineered.columns:
        unplayed_after = df_engineered[df_engineered['PTS_home'] == 0]
        with_features = unplayed_after[unplayed_after['HOME_TEAM_WINS_AVG_LAST_10_HOME'] != 0]
        print(f"\n✅ {len(with_features)} unplayed games now have features")
        
        if len(with_features) > 0:
            print("\n📋 Games ready for prediction:")
            for _, game in with_features.iterrows():
                game_date = pd.to_datetime(game['GAME_DATE_EST']).strftime('%Y-%m-%d')
                print(f"   • {game['VISITOR_TEAM_ABBREVIATION']} @ {game['HOME_TEAM_ABBREVIATION']} on {game_date}")
    
    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

