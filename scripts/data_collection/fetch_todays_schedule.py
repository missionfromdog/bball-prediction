#!/usr/bin/env python3
"""
Fetch today's NBA schedule and append to data file with engineered features.
Uses incremental updates (change data capture) to keep file sizes manageable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path for imports
PROJECTPATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECTPATH / 'src'))

from webscraping import get_todays_matchups, get_game_details
from feature_engineering import process_features

DATAPATH = PROJECTPATH / 'data'


def load_teams():
    """Load team data for ID mappings"""
    teams_df = pd.read_csv(DATAPATH / 'original' / 'teams.csv')
    return teams_df


def fetch_todays_games():
    """
    Fetch today's NBA schedule from NBA.com
    Returns DataFrame with basic game info
    """
    print("="*80)
    print("📅 FETCHING TODAY'S NBA SCHEDULE")
    print("="*80)
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Get today's matchups
        matchups, game_ids = get_todays_matchups(api_key="", driver=driver)
        driver.quit()
        
        if matchups is None or len(matchups) == 0:
            print("ℹ️  No games scheduled for today")
            return None
        
        print(f"✅ Found {len(matchups)} games scheduled for today")
        
        # Create DataFrame with game info
        games = []
        today = datetime.now().strftime('%Y-%m-%d')
        current_season = datetime.now().year
        if datetime.now().month < 10:
            current_season -= 1
        
        for i, (matchup, game_id) in enumerate(zip(matchups, game_ids)):
            visitor_id, home_id = matchup
            games.append({
                'GAME_ID': game_id,
                'GAME_DATE_EST': today,
                'SEASON': current_season,
                'HOME_TEAM_ID': int(home_id),
                'VISITOR_TEAM_ID': int(visitor_id),
                # Placeholder values for unplayed games
                'PTS_home': 0,
                'PTS_away': 0,
                'FG_PCT_home': 0,
                'FT_PCT_home': 0,
                'FG3_PCT_home': 0,
                'AST_home': 0,
                'REB_home': 0,
                'FG_PCT_away': 0,
                'FT_PCT_away': 0,
                'FG3_PCT_away': 0,
                'AST_away': 0,
                'REB_away': 0,
                'HOME_TEAM_WINS': 0,
                'TARGET': 0,
            })
        
        df_today = pd.DataFrame(games)
        
        # Add team abbreviations
        teams_df = load_teams()
        df_today = df_today.merge(
            teams_df[['TEAM_ID', 'ABBREVIATION']].rename(
                columns={'TEAM_ID': 'HOME_TEAM_ID', 'ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}
            ),
            on='HOME_TEAM_ID',
            how='left'
        )
        df_today = df_today.merge(
            teams_df[['TEAM_ID', 'ABBREVIATION']].rename(
                columns={'TEAM_ID': 'VISITOR_TEAM_ID', 'ABBREVIATION': 'VISITOR_TEAM_ABBREVIATION'}
            ),
            on='VISITOR_TEAM_ID',
            how='left'
        )
        
        # Create MATCHUP
        df_today['MATCHUP'] = df_today['VISITOR_TEAM_ABBREVIATION'] + ' @ ' + df_today['HOME_TEAM_ABBREVIATION']
        
        print(f"   Games: {df_today['MATCHUP'].tolist()}")
        return df_today
    
    except Exception as e:
        print(f"❌ Error fetching schedule: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_existing_data():
    """Load existing game data"""
    print("\n📂 Loading existing data...")
    
    # Try different data files
    for filename in ['games_with_real_vegas.csv', 'games_engineered.csv', 'games.csv']:
        filepath = DATAPATH / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            try:
                df = pd.read_csv(filepath, low_memory=False)
                print(f"   ✅ Loaded {len(df):,} games from {filename}")
                return df, filename
            except Exception as e:
                print(f"   ⚠️  Could not load {filename}: {e}")
                continue
    
    print("   ⚠️  No existing data found, starting fresh")
    return None, None


def engineer_features_for_new_games(df_new, df_existing):
    """
    Engineer features for new games based on existing historical data.
    This is a simplified version - full feature engineering requires all historical context.
    """
    print("\n🔧 Engineering features for new games...")
    
    if df_existing is None:
        print("   ⚠️  No historical data available for feature engineering")
        print("   ℹ️  New games will have minimal features")
        return df_new
    
    # Combine new and existing for feature engineering
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Run feature engineering (this will calculate rolling stats for new games)
    try:
        df_engineered = process_features(df_combined)
        
        # Extract only the new games
        new_game_ids = df_new['GAME_ID'].tolist()
        df_new_engineered = df_engineered[df_engineered['GAME_ID'].isin(new_game_ids)]
        
        print(f"   ✅ Engineered {len(df_new_engineered.columns)} features")
        return df_new_engineered
    
    except Exception as e:
        print(f"   ⚠️  Feature engineering failed: {e}")
        print("   ℹ️  Returning games with basic features only")
        return df_new


def append_and_save(df_new, df_existing, filename):
    """
    Append new games to existing data and save.
    Implements change data capture - only adds new/updated records.
    """
    print("\n💾 Saving updated data...")
    
    if df_existing is None:
        # No existing data, just save new
        df_final = df_new
    else:
        # Remove any existing records for today's games (in case of re-run)
        today = datetime.now().strftime('%Y-%m-%d')
        df_existing_filtered = df_existing[
            pd.to_datetime(df_existing['GAME_DATE_EST']).dt.strftime('%Y-%m-%d') != today
        ]
        
        # Append new games
        df_final = pd.concat([df_existing_filtered, df_new], ignore_index=True)
        
        print(f"   📊 Previous: {len(df_existing):,} games")
        print(f"   ➕ Added: {len(df_new):,} new games")
        print(f"   📊 Total: {len(df_final):,} games")
    
    # Save
    output_file = DATAPATH / filename if filename else DATAPATH / 'games_with_real_vegas.csv'
    df_final.to_csv(output_file, index=False)
    
    # Also save to workflow dataset (smaller, last 5K games)
    df_workflow = df_final.tail(5000)
    workflow_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    df_workflow.to_csv(workflow_file, index=False)
    
    print(f"   ✅ Saved to {output_file.name}")
    print(f"   ✅ Saved workflow dataset: {len(df_workflow):,} games")
    
    return df_final


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("NBA SCHEDULE UPDATER - Incremental Data Capture")
    print("="*80)
    
    # Step 1: Fetch today's schedule
    df_new = fetch_todays_games()
    if df_new is None or len(df_new) == 0:
        print("\n✅ No games to add today")
        return
    
    # Step 2: Load existing data
    df_existing, filename = load_existing_data()
    
    # Step 3: Engineer features for new games
    df_new_engineered = engineer_features_for_new_games(df_new, df_existing)
    
    # Step 4: Append and save
    df_final = append_and_save(df_new_engineered, df_existing, filename)
    
    print("\n" + "="*80)
    print(f"✅ SCHEDULE UPDATE COMPLETE")
    print(f"   Total games in database: {len(df_final):,}")
    print(f"   Added today: {len(df_new):,} games")
    print("="*80)


if __name__ == "__main__":
    main()

