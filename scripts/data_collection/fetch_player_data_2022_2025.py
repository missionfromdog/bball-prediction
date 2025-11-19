"""
Fetch player-level box score data from 2022-2025 using nba_api
and merge with existing games_details.csv

Uses:
- V2 (boxscoretraditionalv2) for 2022-2024 (exact column match)
- V3 (boxscoretraditionalv3) for 2025+ (with column mapping)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoretraditionalv3
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams
except ImportError:
    print("❌ nba_api not installed. Run: pip install nba_api")
    sys.exit(1)

DATAPATH = PROJECT_ROOT / 'data'
ORIGINAL_PATH = DATAPATH / 'original'
OUTPUT_FILE = ORIGINAL_PATH / 'games_details_updated.csv'
BACKUP_FILE = ORIGINAL_PATH / 'games_details_backup.csv'

# Date ranges
# Full dataset: 2003-2025
FULL_START = datetime(2003, 10, 1)  # Start of dataset
FULL_END = datetime.now()            # Current date

# V2/V3 periods (for reference)
V2_START = datetime(2022, 3, 13)  # After existing data ends
V2_END = datetime(2024, 6, 30)    # End of 2023-24 season
V3_START = datetime(2024, 10, 1)  # Start of 2024-25 season
V3_END = datetime.now()            # Current date


def map_v3_to_v2_format(df_v3):
    """
    Map V3 column names to match V2/existing format
    """
    mapping = {
        'gameId': 'GAME_ID',
        'teamId': 'TEAM_ID',
        'teamTricode': 'TEAM_ABBREVIATION',
        'teamCity': 'TEAM_CITY',
        'personId': 'PLAYER_ID',
        'nameI': 'PLAYER_NAME',  # Full name
        'position': 'START_POSITION',
        'comment': 'COMMENT',
        'minutes': 'MIN',
        'fieldGoalsMade': 'FGM',
        'fieldGoalsAttempted': 'FGA',
        'fieldGoalsPercentage': 'FG_PCT',
        'threePointersMade': 'FG3M',
        'threePointersAttempted': 'FG3A',
        'threePointersPercentage': 'FG3_PCT',
        'freeThrowsMade': 'FTM',
        'freeThrowsAttempted': 'FTA',
        'freeThrowsPercentage': 'FT_PCT',
        'reboundsOffensive': 'OREB',
        'reboundsDefensive': 'DREB',
        'reboundsTotal': 'REB',
        'assists': 'AST',
        'steals': 'STL',
        'blocks': 'BLK',
        'turnovers': 'TO',
        'foulsPersonal': 'PF',
        'points': 'PTS',
        'plusMinusPoints': 'PLUS_MINUS'
    }
    
    # Create new dataframe with mapped columns
    df_mapped = pd.DataFrame()
    
    for v3_col, v2_col in mapping.items():
        if v3_col in df_v3.columns:
            df_mapped[v2_col] = df_v3[v3_col]
        else:
            df_mapped[v2_col] = None
    
    # Handle NICKNAME - V3 doesn't have it, set to empty
    df_mapped['NICKNAME'] = ''
    
    # Ensure MIN format matches (should be "MM:SS" string)
    if 'MIN' in df_mapped.columns:
        # Convert numeric minutes to "MM:SS" format if needed
        def format_minutes(val):
            if pd.isna(val) or val == '':
                return '0:00'
            if isinstance(val, str) and ':' in val:
                return val
            try:
                # If numeric, convert to MM:SS
                total_seconds = int(float(val) * 60)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes}:{seconds:02d}"
            except:
                return '0:00'
        
        df_mapped['MIN'] = df_mapped['MIN'].apply(format_minutes)
    
    return df_mapped


def get_game_ids_from_existing_data(start_date, end_date):
    """
    Get game IDs from existing games.csv file (faster than API calls)
    """
    games_file = DATAPATH / 'games.csv'
    workflow_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    
    # Try to load from workflow file first (more complete)
    if workflow_file.exists():
        print(f"📂 Loading game IDs from {workflow_file.name}...")
        df = pd.read_csv(workflow_file, low_memory=False)
    elif games_file.exists():
        print(f"📂 Loading game IDs from {games_file.name}...")
        df = pd.read_csv(games_file, low_memory=False)
    else:
        print("⚠️  No games.csv found, will use API method")
        return None
    
    # Parse dates and remove timezone if present
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
    # Remove timezone info if present (convert timezone-aware to naive)
    if df['GAME_DATE_EST'].dt.tz is not None:
        # Convert to UTC first, then remove timezone
        df['GAME_DATE_EST'] = df['GAME_DATE_EST'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Filter by date range (both sides are now naive datetime)
    mask = (df['GAME_DATE_EST'] >= pd.Timestamp(start_date)) & (df['GAME_DATE_EST'] <= pd.Timestamp(end_date))
    filtered = df[mask]
    
    if 'GAME_ID' in filtered.columns:
        game_ids = filtered['GAME_ID'].unique().tolist()
        
        # Validate game IDs - NBA game IDs should be 10-digit strings starting with '00'
        # or at least positive integers > 1000
        valid_game_ids = []
        for gid in game_ids:
            if pd.notna(gid):
                gid_str = str(gid)
                # Check if it looks like a valid NBA game ID
                # Valid IDs are typically 10 digits (e.g., '0022400123') or at least > 1000
                if (len(gid_str) == 10 and gid_str.startswith('00')) or \
                   (gid_str.isdigit() and int(gid_str) > 1000):
                    valid_game_ids.append(gid_str)
        
        if len(valid_game_ids) > 0:
            print(f"✅ Found {len(valid_game_ids)} valid game IDs in date range")
            return sorted(valid_game_ids)
        else:
            print(f"⚠️  Found {len(game_ids)} game IDs but none are valid (likely sequential numbers)")
            print("   Will use API method instead")
            return None
    else:
        print("⚠️  GAME_ID column not found, will use API method")
        return None


def get_game_ids_for_date_range(start_date, end_date, use_existing=True):
    """
    Get all game IDs for a date range
    First tries existing data files, falls back to API
    """
    # Try existing data first (much faster)
    if use_existing:
        game_ids = get_game_ids_from_existing_data(start_date, end_date)
        if game_ids:
            return game_ids
    
    # Fallback to API method
    print(f"📅 Getting game IDs from API: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    game_ids = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Format date for scoreboard
            date_str = current_date.strftime('%m/%d/%Y')
            
            # Get scoreboard for this date
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                league_id='00'  # NBA
            )
            
            # Extract game IDs
            games = scoreboard.get_data_frames()[0]
            
            if len(games) > 0:
                date_game_ids = games['GAME_ID'].unique().tolist()
                game_ids.extend([str(gid) for gid in date_game_ids])
                print(f"   ✅ {current_date.strftime('%Y-%m-%d')}: {len(date_game_ids)} games")
            
            # Rate limiting
            time.sleep(0.6)  # NBA API rate limit: ~60 requests/minute
            
        except Exception as e:
            print(f"   ⚠️  {current_date.strftime('%Y-%m-%d')}: Error - {str(e)[:50]}")
            time.sleep(1)
        
        current_date += timedelta(days=1)
        
        # Progress update every 30 days
        if (current_date - start_date).days % 30 == 0:
            print(f"   📊 Progress: {len(game_ids)} games found so far...")
    
    print(f"✅ Found {len(game_ids)} total games")
    return sorted(set(game_ids))  # Remove duplicates and sort


def fetch_box_score_v2(game_id):
    """Fetch box score using V2 API"""
    try:
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box_score.get_data_frames()[0]
        return player_stats
    except Exception as e:
        print(f"   ⚠️  Error fetching V2 box score for {game_id}: {str(e)[:50]}")
        return None


def fetch_box_score_v3(game_id):
    """Fetch box score using V3 API and map to V2 format"""
    try:
        box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        player_stats = box_score.get_data_frames()[0]
        
        if len(player_stats) == 0:
            return None
        
        # Map to V2 format
        mapped_stats = map_v3_to_v2_format(player_stats)
        return mapped_stats
    except Exception as e:
        print(f"   ⚠️  Error fetching V3 box score for {game_id}: {str(e)[:50]}")
        return None


def fetch_all_player_data(start_date=None, end_date=None, use_v3_after=None):
    """
    Main function to fetch all player data for a date range
    
    Args:
        start_date: Start date (default: V2_START)
        end_date: End date (default: V3_END)
        use_v3_after: Date to switch from V2 to V3 API (default: V3_START)
    """
    if start_date is None:
        start_date = FULL_START  # Full dataset: 2003-2025
    if end_date is None:
        end_date = FULL_END
    if use_v3_after is None:
        use_v3_after = V3_START  # Switch to V3 API for 2024-25 season
    
    print("=" * 60)
    print(f"FETCHING PLAYER-LEVEL DATA ({start_date.date()} to {end_date.date()})")
    print("=" * 60)
    print("")
    
    all_player_data = []
    
    # Step 1: Get all game IDs for the date range
    print(f"📊 STEP 1: Getting game IDs from {start_date.date()} to {end_date.date()}...")
    print("   (This will use NBA API - takes ~1 minute per 60 dates)")
    all_game_ids = get_game_ids_for_date_range(start_date, end_date, use_existing=False)  # Force API lookup
    print(f"✅ Found {len(all_game_ids)} total games")
    print("")
    
    if len(all_game_ids) == 0:
        print("❌ No game IDs found. Exiting.")
        return None
    
    # Step 2: Fetch box scores for all games
    print("📊 STEP 2: Fetching box scores for all games...")
    print(f"   (This will take ~{len(all_game_ids) * 0.6 / 60:.1f} minutes)")
    print("")
    
    success_count = 0
    failed_count = 0
    
    for game_id in tqdm(all_game_ids, desc="Box Scores"):
        # Determine which API to use based on date
        # Game IDs starting with '00224' are 2024-25 season (V3)
        # Game IDs starting with '00223' or earlier are V2
        if game_id.startswith('00224') or game_id.startswith('00225'):
            # Use V3 API for 2024-25 season and later
            stats = fetch_box_score_v3(game_id)
        else:
            # Use V2 API for earlier seasons
            stats = fetch_box_score_v2(game_id)
        
        if stats is not None and len(stats) > 0:
            all_player_data.append(stats)
            success_count += 1
        else:
            failed_count += 1
        
        # Rate limiting
        time.sleep(0.6)
    
    print(f"✅ Success: {success_count}, Failed: {failed_count}")
    print("")
    
    # Step 5: Combine all data
    if len(all_player_data) == 0:
        print("❌ No player data fetched!")
        return None
    
    print("📊 STEP 5: Combining all fetched data...")
    new_data = pd.concat(all_player_data, ignore_index=True)
    print(f"✅ Combined {len(new_data):,} player-game records")
    print("")
    
    return new_data


def merge_with_existing(new_data):
    """
    Merge new data with existing games_details.csv
    """
    print("=" * 60)
    print("MERGING WITH EXISTING DATA")
    print("=" * 60)
    print("")
    
    # Load existing data
    existing_file = ORIGINAL_PATH / 'games_details.csv'
    
    if not existing_file.exists():
        print("⚠️  Existing games_details.csv not found. Creating new file...")
        existing_data = pd.DataFrame()
    else:
        print(f"📂 Loading existing data from {existing_file}...")
        existing_data = pd.read_csv(existing_file, low_memory=False)
        print(f"✅ Loaded {len(existing_data):,} existing records")
        
        # Backup existing file
        print(f"💾 Creating backup: {BACKUP_FILE}")
        existing_data.to_csv(BACKUP_FILE, index=False)
        print("✅ Backup created")
        print("")
    
    # Find games that are already in existing data
    # Note: GAME_ID formats may differ, so we'll check by date + teams instead
    if len(existing_data) > 0:
        # Convert GAME_IDs to strings for comparison
        existing_game_ids = set(str(gid) for gid in existing_data['GAME_ID'].unique() if pd.notna(gid))
        new_game_ids = set(str(gid) for gid in new_data['GAME_ID'].unique() if pd.notna(gid))
        
        # Only keep new games
        games_to_add = new_game_ids - existing_game_ids
        new_data_filtered = new_data[new_data['GAME_ID'].astype(str).isin(games_to_add)]
        
        print(f"📊 Existing games: {len(existing_game_ids):,}")
        print(f"📊 New games fetched: {len(new_game_ids):,}")
        print(f"📊 Games to add: {len(games_to_add):,}")
        print("")
        
        if len(new_data_filtered) == 0:
            print("ℹ️  No new games to add (all already in dataset)")
            return existing_data
    else:
        new_data_filtered = new_data
        print(f"📊 Adding all {len(new_data_filtered):,} records (no existing data)")
        print("")
    
    # Ensure column order matches
    if len(existing_data) > 0:
        column_order = existing_data.columns.tolist()
        new_data_filtered = new_data_filtered[column_order]
    
    # Combine
    print("🔗 Merging datasets...")
    merged_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
    
    # Sort by GAME_ID and PLAYER_NAME
    merged_data = merged_data.sort_values(['GAME_ID', 'PLAYER_NAME']).reset_index(drop=True)
    
    print(f"✅ Merged dataset: {len(merged_data):,} total records")
    print("")
    
    return merged_data


def main():
    """Main execution"""
    print("=" * 60)
    print("NBA PLAYER DATA FETCHER (2022-2025)")
    print("=" * 60)
    print("")
    print("This script will:")
    print("  1. Fetch ALL game IDs from NBA API (2003-2025)")
    print("  2. Fetch ALL player box scores for all games")
    print("  3. Use V2 API for 2003-2024 (exact column match)")
    print("  4. Use V3 API for 2024-2025 (with column mapping)")
    print("  5. Merge with existing games_details.csv")
    print("  6. Save updated dataset")
    print("")
    print("⚠️  This will take a while (rate limited to ~60 requests/minute)")
    print("    Estimated time:")
    print("    - Full dataset (2003-2025): ~5.8 hours")
    print("    - Game IDs: ~49 minutes")
    print("    - Box scores: ~5 hours")
    print("")
    
    # Allow auto-confirm via environment variable
    import os
    auto_confirm = os.getenv('AUTO_CONFIRM', '').lower() == 'true'
    
    if not auto_confirm:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    else:
        print("Auto-confirmed (AUTO_CONFIRM=true)")
    
    print("")
    
    # Fetch new data for FULL DATASET (2003-2025)
    print("🎯 Fetching data for FULL DATASET (2003-2025)")
    print("   This will update game IDs and fetch all box scores")
    print("")
    new_data = fetch_all_player_data(start_date=FULL_START, end_date=FULL_END)
    
    if new_data is None or len(new_data) == 0:
        print("❌ No data fetched. Exiting.")
        return
    
    # Merge with existing
    merged_data = merge_with_existing(new_data)
    
    # Save
    print("=" * 60)
    print("SAVING UPDATED DATASET")
    print("=" * 60)
    print("")
    # Ensure directory exists
    ORIGINAL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving to {OUTPUT_FILE}...")
    merged_data.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(merged_data):,} records")
    print("")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(merged_data):,}")
    print(f"Unique games: {merged_data['GAME_ID'].nunique():,}")
    print(f"Unique players: {merged_data['PLAYER_NAME'].nunique():,}")
    print("")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"💾 Backup file: {BACKUP_FILE}")
    print("")
    print("✅ Done! Review the output file and replace games_details.csv if satisfied.")
    print("")


if __name__ == '__main__':
    main()

