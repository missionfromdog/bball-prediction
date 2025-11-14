"""
Update completed game scores from ESPN.com

This script:
1. Scrapes completed NBA game scores from ESPN scoreboard
2. Updates games in the dataset that have PTS_home == 0 but are now complete
3. Updates both home and away scores
4. Saves the updated dataset

Run daily to keep dataset current with completed games.
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ESPN team name to ID mapping (same as in constants.py)
TEAM_NAME_TO_ID = {
    'Hawks': 1610612737, 'Celtics': 1610612738, 'Cavaliers': 1610612739,
    'Pelicans': 1610612740, 'Bulls': 1610612741, 'Mavericks': 1610612742,
    'Nuggets': 1610612743, 'Warriors': 1610612744, 'Rockets': 1610612745,
    'Clippers': 1610612746, 'Pacers': 1610612754, 'Lakers': 1610612747,
    'Grizzlies': 1610612763, 'Heat': 1610612748, 'Bucks': 1610612749,
    'Timberwolves': 1610612750, 'Nets': 1610612751, 'Knicks': 1610612752,
    'Magic': 1610612753, '76ers': 1610612755, 'Suns': 1610612756,
    'Trail Blazers': 1610612757, 'Kings': 1610612758, 'Spurs': 1610612759,
    'Thunder': 1610612760, 'Raptors': 1610612761, 'Jazz': 1610612762,
    'Wizards': 1610612764, 'Pistons': 1610612765, 'Hornets': 1610612766
}

# Alternate names (some teams use city name on ESPN)
TEAM_ALT_NAMES = {
    'Atlanta': 'Hawks', 'Boston': 'Celtics', 'Cleveland': 'Cavaliers',
    'New Orleans': 'Pelicans', 'Chicago': 'Bulls', 'Dallas': 'Mavericks',
    'Denver': 'Nuggets', 'Golden State': 'Warriors', 'Houston': 'Rockets',
    'LA': 'Clippers', 'Indiana': 'Pacers', 'Los Angeles': 'Lakers',
    'Memphis': 'Grizzlies', 'Miami': 'Heat', 'Milwaukee': 'Bucks',
    'Minnesota': 'Timberwolves', 'Brooklyn': 'Nets', 'New York': 'Knicks',
    'Orlando': 'Magic', 'Philadelphia': '76ers', 'Phoenix': 'Suns',
    'Portland': 'Trail Blazers', 'Sacramento': 'Kings', 'San Antonio': 'Spurs',
    'Oklahoma City': 'Thunder', 'Toronto': 'Raptors', 'Utah': 'Jazz',
    'Washington': 'Wizards', 'Detroit': 'Pistons', 'Charlotte': 'Hornets'
}


def get_team_id(team_name):
    """Convert ESPN team name to team ID"""
    # Remove city prefixes
    if team_name in TEAM_NAME_TO_ID:
        return TEAM_NAME_TO_ID[team_name]
    
    # Try alternate names
    if team_name in TEAM_ALT_NAMES:
        return TEAM_NAME_TO_ID[TEAM_ALT_NAMES[team_name]]
    
    # Try partial match
    for key, value in TEAM_NAME_TO_ID.items():
        if key in team_name or team_name in key:
            return value
    
    print(f"⚠️  Warning: Could not find team ID for '{team_name}'")
    return None


def scrape_completed_games(date):
    """
    Scrape completed games from ESPN for a specific date
    
    Returns: List of dicts with game info
    """
    date_str = date.strftime('%Y%m%d')
    url = f'https://www.espn.com/nba/scoreboard/_/date/{date_str}'
    
    print(f"🔍 Scraping ESPN for completed games on {date.strftime('%Y-%m-%d')}...")
    print(f"   URL: {url}")
    
    # Add headers to avoid 403 Forbidden
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        completed_games = []
        
        # Find all game containers
        game_containers = soup.find_all('section', class_='Scoreboard')
        
        if not game_containers:
            # Try alternate structure
            game_containers = soup.find_all('div', class_='ScoreCell')
        
        print(f"   Found {len(game_containers)} potential game containers")
        
        for container in game_containers:
            try:
                # Get teams
                teams = container.find_all('div', class_='ScoreCell__TeamName')
                if len(teams) < 2:
                    continue
                
                away_team = teams[0].text.strip()
                home_team = teams[1].text.strip()
                
                # Get scores
                scores = container.find_all('div', class_='ScoreCell__Score')
                if len(scores) < 2:
                    continue
                
                away_score_text = scores[0].text.strip()
                home_score_text = scores[1].text.strip()
                
                # Skip if game hasn't started or is in progress
                if not away_score_text.isdigit() or not home_score_text.isdigit():
                    continue
                
                away_score = int(away_score_text)
                home_score = int(home_score_text)
                
                # Check if game is final
                status = container.find('span', class_='ScoreboardScoreCell__Overview')
                if status and 'final' not in status.text.lower():
                    continue
                
                # Get team IDs
                away_id = get_team_id(away_team)
                home_id = get_team_id(home_team)
                
                if away_id and home_id:
                    completed_games.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_id': away_id,
                        'home_id': home_id,
                        'away_score': away_score,
                        'home_score': home_score
                    })
                    print(f"   ✅ {away_team} @ {home_team}: {away_score}-{home_score}")
            
            except Exception as e:
                continue
        
        return completed_games
    
    except Exception as e:
        print(f"   ❌ Error scraping ESPN: {e}")
        return []


def update_dataset_with_scores(dataset_path, completed_games):
    """
    Update dataset with completed game scores
    
    Args:
        dataset_path: Path to the CSV dataset
        completed_games: List of completed game dicts from scraping
    """
    print(f"\n📊 Loading dataset: {dataset_path.name}")
    df = pd.read_csv(dataset_path, low_memory=False, dtype={'GAME_DATE_EST': str})
    print(f"   Loaded {len(df):,} games")
    
    # Parse dates
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
    
    # Track updates
    updates_made = 0
    
    for game in completed_games:
        game_date = pd.to_datetime(game['date']).date()
        
        # Debug: Check what's in the dataset for this date
        date_games = df[df['GAME_DATE_EST'].dt.date == game_date]
        unplayed_games = date_games[date_games['PTS_home'] == 0]
        
        if updates_made == 0 and len(unplayed_games) > 0:  # Only print once
            print(f"\n   [DEBUG] {game['date']} has {len(unplayed_games)} unplayed games in dataset:")
            for _, g in unplayed_games.iterrows():
                print(f"      HOME_TEAM_ID={g['HOME_TEAM_ID']}, VISITOR_TEAM_ID={g['VISITOR_TEAM_ID']}")
        
        # Find matching game in dataset
        mask = (
            (df['GAME_DATE_EST'].dt.date == game_date) &
            (df['HOME_TEAM_ID'] == game['home_id']) &
            (df['VISITOR_TEAM_ID'] == game['away_id']) &
            (df['PTS_home'] == 0)  # Only update unscored games
        )
        
        matching_games = df[mask]
        
        if len(matching_games) > 0:
            # Update scores
            df.loc[mask, 'PTS_home'] = game['home_score']
            df.loc[mask, 'PTS_away'] = game['away_score']
            
            # Update HOME_TEAM_WINS
            home_won = int(game['home_score'] > game['away_score'])
            df.loc[mask, 'HOME_TEAM_WINS'] = home_won
            df.loc[mask, 'TARGET'] = home_won
            
            updates_made += len(matching_games)
            print(f"   ✅ Updated: {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
        else:
            print(f"   ⚠️  No match: {game['away_team']} @ {game['home_team']} (may not be in dataset)")
    
    if updates_made > 0:
        print(f"\n💾 Saving updated dataset with {updates_made} score updates...")
        df.to_csv(dataset_path, index=False)
        print(f"   ✅ Saved")
    else:
        print(f"\n✅ No updates needed (all games already scored)")
    
    return updates_made


def main():
    """Main function"""
    print("="*80)
    print("🏀 NBA COMPLETED GAME SCORE UPDATER")
    print("="*80)
    
    # Define dataset path
    DATAPATH = ROOT / 'data'
    dataset_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
    
    if not dataset_file.exists():
        print(f"❌ Dataset not found: {dataset_file}")
        sys.exit(1)
    
    # Scrape completed games from the last 7 days
    all_completed_games = []
    
    for days_ago in range(7, -1, -1):  # Last 7 days + today
        date = datetime.now().date() - timedelta(days=days_ago)
        completed_games = scrape_completed_games(date)
        all_completed_games.extend(completed_games)
        time.sleep(2)  # Be nice to ESPN servers
    
    print(f"\n📊 Total completed games found: {len(all_completed_games)}")
    
    if all_completed_games:
        # Update dataset
        updates_made = update_dataset_with_scores(dataset_file, all_completed_games)
        
        print("\n" + "="*80)
        print(f"✅ COMPLETED: {updates_made} game scores updated")
        print("="*80)
    else:
        print("\n⚠️  No completed games found in the last 7 days")


if __name__ == '__main__':
    main()

