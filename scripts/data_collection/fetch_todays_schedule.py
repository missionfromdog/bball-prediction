#!/usr/bin/env python3
"""
Fetch today's NBA schedule from ESPN and append to data file.
Uses incremental updates (change data capture) to keep file sizes manageable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
from bs4 import BeautifulSoup
import requests

PROJECTPATH = Path(__file__).resolve().parents[2]
DATAPATH = PROJECTPATH / 'data'

# Team abbreviation mapping (ESPN to our format)
ESPN_TO_STANDARD = {
    'GS': 'GSW',
    'NY': 'NYK',
    'SA': 'SAS',
    'NO': 'NOP',
    'PHX': 'PHO',
    # Add more if needed
}


def load_teams():
    """
    Get team ID mappings - hardcoded to avoid LFS issues in workflows
    Returns a DataFrame with TEAM_ID and ABBREVIATION columns
    """
    # Hardcoded team mappings (from NBA official data)
    team_data = {
        'TEAM_ID': [
            1610612737, 1610612738, 1610612751, 1610612766, 1610612741, 1610612739,
            1610612742, 1610612743, 1610612765, 1610612744, 1610612745, 1610612754,
            1610612746, 1610612747, 1610612763, 1610612748, 1610612749, 1610612750,
            1610612740, 1610612752, 1610612753, 1610612755, 1610612756, 1610612757,
            1610612758, 1610612759, 1610612760, 1610612761, 1610612762, 1610612764
        ],
        'ABBREVIATION': [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE',
            'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND',
            'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
            'NOP', 'NYK', 'OKC', 'PHI', 'PHX', 'POR',
            'SAC', 'SAS', 'TOR', 'UTA', 'WAS', 'ORL'
        ]
    }
    
    return pd.DataFrame(team_data)


def fetch_todays_games_espn():
    """
    Fetch today's NBA schedule from ESPN
    Returns list of games with visitor/home team info
    """
    import re
    
    print("="*80, flush=True)
    print("📅 FETCHING TODAY'S NBA SCHEDULE FROM ESPN", flush=True)
    print("="*80, flush=True)
    sys.stdout.flush()
    
    # Team name to abbreviation mapping
    TEAM_NAME_MAP = {
        'Toronto': 'TOR', 'Cleveland': 'CLE', 'Indiana': 'IND', 'Phoenix': 'PHX',
        'Miami': 'MIA', 'New York': 'NYK', 'Brooklyn': 'BKN', 'Orlando': 'ORL',
        'Philadelphia': 'PHI', 'Memphis': 'MEM', 'Golden State': 'GSW', 'Oklahoma City': 'OKC',
        'Denver': 'DEN', 'Sacramento': 'SAC', 'Boston': 'BOS', 'Chicago': 'CHI',
        'Milwaukee': 'MIL', 'LA Lakers': 'LAL', 'LA Clippers': 'LAC', 'Los Angeles': 'LAL',  # Default LA to Lakers
        'LA': 'LAC',  # Fallback
        'Portland': 'POR', 'Utah': 'UTA', 'San Antonio': 'SAS', 'Dallas': 'DAL', 
        'Houston': 'HOU', 'New Orleans': 'NOP', 'Washington': 'WAS', 'Charlotte': 'CHA', 
        'Atlanta': 'ATL', 'Detroit': 'DET', 'Minnesota': 'MIN'
    }
    
    # Try today and tomorrow (in case of timezone differences)
    dates_to_try = [
        datetime.now(),
        datetime.now() + timedelta(days=1)
    ]
    
    for date_obj in dates_to_try:
        date_str = date_obj.strftime('%Y%m%d')
        url = f"https://www.espn.com/nba/schedule/_/date/{date_str}"
        
        print(f"\n🔍 Trying date: {date_obj.strftime('%Y-%m-%d')} ({date_str})", flush=True)
        print(f"   URL: {url}", flush=True)
        sys.stdout.flush()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            html = response.text
            soup_obj = BeautifulSoup(html, 'html.parser')
            
            games = []
            
            # Find the date header for today
            date_header_text = date_obj.strftime('%A, %B %-d, %Y')  # e.g., "Thursday, November 13, 2025"
            # Try with and without leading zero
            date_patterns = [
                date_obj.strftime('%A, %B %-d, %Y'),  # "Thursday, November 13, 2025"
                date_obj.strftime('%A, %B %d, %Y'),   # "Thursday, November 13, 2025" (with zero)
            ]
            
            date_div = None
            for pattern in date_patterns:
                date_div = soup_obj.find('div', string=lambda x: x and pattern in str(x))
                if date_div:
                    print(f"   ✅ Found date header: {pattern}", flush=True)
                    break
            
            if not date_div:
                print(f"   ⚠️  Could not find date header for {date_obj.strftime('%Y-%m-%d')}", flush=True)
                continue
            
            # Find the next table after the date header
            table = date_div.find_next('tbody', class_='Table__TBODY')
            
            if not table:
                print(f"   ⚠️  No table found after date header", flush=True)
                continue
            
            rows = table.find_all('tr', class_='Table__TR')
            print(f"   Found {len(rows)} games for this date", flush=True)
            
            for row in rows:
                # Find all team links (pattern: /nba/team/_/name/)
                team_links = row.find_all('a', href=re.compile(r'/nba/team/_/name/'))
                
                # Filter to only links with text (ignore logo links)
                team_links_with_text = [l for l in team_links if l.text.strip()]
                
                if len(team_links_with_text) >= 2:
                    away_name = team_links_with_text[0].text.strip()
                    home_name = team_links_with_text[1].text.strip()
                    
                    # Convert to abbreviations
                    away_abbr = TEAM_NAME_MAP.get(away_name)
                    home_abbr = TEAM_NAME_MAP.get(home_name)
                    
                    if away_abbr and home_abbr:
                        games.append({
                            'visitor': away_abbr,
                            'home': home_abbr,
                            'date': date_obj.strftime('%Y-%m-%d')
                        })
                    else:
                        print(f"   ⚠️  Unknown teams: {away_name} @ {home_name}", flush=True)
            
            if games:
                print(f"\n✅ Found {len(games)} games for {date_obj.strftime('%Y-%m-%d')}:", flush=True)
                for g in games:
                    print(f"   • {g['visitor']} @ {g['home']}", flush=True)
                sys.stdout.flush()
                return games
            else:
                print(f"   ⚠️  No games found for {date_obj.strftime('%Y-%m-%d')}", flush=True)
                sys.stdout.flush()
        
        except Exception as e:
            print(f"   ❌ Error fetching {date_str}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            continue
    
    print("\n⚠️  No games found for today or tomorrow", flush=True)
    sys.stdout.flush()
    return []


def create_game_rows(games, teams_df):
    """
    Convert games list to DataFrame rows with proper IDs
    """
    print("\n🔧 Creating game rows with team IDs...", flush=True)
    sys.stdout.flush()
    
    rows = []
    # Format date to match existing dataset format: 'YYYY-MM-DD HH:MM:SS+00:00'
    # CRITICAL: Must include +00:00 timezone to match existing data format
    # Otherwise pandas can't parse mixed timezone/non-timezone dates
    today = datetime.now().strftime('%Y-%m-%d 00:00:00+00:00')
    
    # Create team lookup
    team_lookup = {}
    for _, team in teams_df.iterrows():
        team_lookup[team['ABBREVIATION']] = team['TEAM_ID']
    
    for game in games:
        visitor_abbr = game['visitor'].upper()
        home_abbr = game['home'].upper()
        
        # Try to find team IDs
        visitor_id = team_lookup.get(visitor_abbr)
        home_id = team_lookup.get(home_abbr)
        
        if not visitor_id or not home_id:
            print(f"   ⚠️  Skipping {visitor_abbr} @ {home_abbr} - team not found in database", flush=True)
            continue
        
        # Create a game ID (use hash to keep it small enough for int32)
        # Format: YYYYMMDD + last 3 digits of visitor_id + last 3 digits of home_id
        date_part = int(datetime.now().strftime('%Y%m%d'))
        visitor_part = visitor_id % 1000  # Last 3 digits
        home_part = home_id % 1000  # Last 3 digits
        game_id = date_part * 1000000 + visitor_part * 1000 + home_part
        
        # Calculate season (same logic as prediction script)
        # NBA season year is the year in which it STARTS (Oct-Sep)
        season_year = datetime.now().year
        if datetime.now().month < 10:  # Jan-Sep uses previous year
            season_year -= 1
        
        row = {
            'GAME_ID': game_id,
            'GAME_DATE_EST': today,
            'HOME_TEAM_ID': home_id,
            'VISITOR_TEAM_ID': visitor_id,
            'HOME_TEAM_ABBREVIATION': home_abbr,
            'VISITOR_TEAM_ABBREVIATION': visitor_abbr,
            'SEASON': season_year,  # Current season (2024 for 2024-25 season)
            'PTS_home': 0,  # Unplayed
            'PTS_away': 0,  # Unplayed
            'HOME_TEAM_WINS': 0,  # Placeholder
            'MATCHUP': f"{visitor_abbr} @ {home_abbr}"
        }
        
        rows.append(row)
    
    print(f"   ✅ Created {len(rows)} game rows", flush=True)
    sys.stdout.flush()
    
    return pd.DataFrame(rows)


def append_to_data_file(new_games_df):
    """
    Append new games to the workflow data file
    Only adds games that don't already exist (by GAME_ID)
    """
    print("\n💾 Appending to data file...", flush=True)
    sys.stdout.flush()
    
    data_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    
    if not data_file.exists():
        print(f"   ❌ Data file not found: {data_file}", flush=True)
        return False
    
    # Load existing data
    existing_df = pd.read_csv(data_file)
    print(f"   Existing data: {len(existing_df):,} rows", flush=True)
    
    # Filter out games that already exist
    new_game_ids = set(new_games_df['GAME_ID'].values)
    existing_game_ids = set(existing_df['GAME_ID'].values)
    
    duplicate_ids = new_game_ids & existing_game_ids
    if duplicate_ids:
        print(f"   ⚠️  Skipping {len(duplicate_ids)} games that already exist", flush=True)
        new_games_df = new_games_df[~new_games_df['GAME_ID'].isin(duplicate_ids)]
    
    if len(new_games_df) == 0:
        print("   ℹ️  No new games to add", flush=True)
        return True
    
    # Ensure new games have all the same columns as existing data
    # Fill missing columns with 0 or NaN as appropriate
    for col in existing_df.columns:
        if col not in new_games_df.columns:
            if existing_df[col].dtype in ['int64', 'float64']:
                new_games_df[col] = 0
            else:
                new_games_df[col] = ''
    
    # Reorder columns to match existing data
    new_games_df = new_games_df[existing_df.columns]
    
    # Append
    combined_df = pd.concat([existing_df, new_games_df], ignore_index=True)
    
    # Save
    combined_df.to_csv(data_file, index=False)
    
    print(f"   ✅ Added {len(new_games_df)} new games", flush=True)
    print(f"   📊 Total games: {len(combined_df):,}", flush=True)
    sys.stdout.flush()
    
    return True


def main():
    """Main execution"""
    print("\n" + "="*80, flush=True)
    print("🏀 NBA SCHEDULE FETCH - ESPN EDITION", flush=True)
    print("="*80 + "\n", flush=True)
    sys.stdout.flush()
    
    # Load teams
    teams_df = load_teams()
    print(f"✅ Loaded {len(teams_df)} teams\n", flush=True)
    
    # Fetch today's games
    games = fetch_todays_games_espn()
    
    if not games:
        print("\n⚠️  No games found - nothing to update", flush=True)
        sys.exit(0)
    
    # Create game rows
    new_games_df = create_game_rows(games, teams_df)
    
    if len(new_games_df) == 0:
        print("\n⚠️  Could not create valid game rows", flush=True)
        sys.exit(1)
    
    # Append to data file
    success = append_to_data_file(new_games_df)
    
    if success:
        print("\n" + "="*80, flush=True)
        print("✅ SCHEDULE FETCH COMPLETE", flush=True)
        print("="*80, flush=True)
        sys.stdout.flush()
    else:
        print("\n❌ Failed to update data file", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
