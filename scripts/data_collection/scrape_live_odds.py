"""
Scrape live NBA betting odds from The Odds API
Requires API key from https://the-odds-api.com/

Free tier: 500 requests/month
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
DATAPATH = Path(__file__).resolve().parents[2] / 'data'
BETTING_PATH = DATAPATH / 'betting'
BETTING_PATH.mkdir(parents=True, exist_ok=True)

def fetch_live_odds(api_key=None, sport='basketball_nba'):
    """
    Fetch live NBA betting odds from The Odds API
    
    Args:
        api_key: API key from the-odds-api.com (or set ODDS_API_KEY env var)
        sport: Sport to fetch (default: basketball_nba)
    
    Returns:
        DataFrame with current betting lines
    """
    if api_key is None:
        api_key = os.environ.get('ODDS_API_KEY')
    
    if not api_key:
        print("❌ No API key provided. Set ODDS_API_KEY environment variable or pass api_key parameter")
        print("💡 Sign up for free at: https://the-odds-api.com/")
        return None
    
    # API endpoint
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/'
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',  # Moneyline, spreads, over/under
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    print(f"🎲 Fetching live odds from The Odds API...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        print(f"✅ API calls remaining: {remaining}")
        
        if not data:
            print("⚠️ No upcoming games with odds available")
            return None
        
        # Parse odds data
        games = []
        for game in data:
            game_info = {
                'game_id': game['id'],
                'sport': game['sport_key'],
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
            }
            
            # Extract bookmaker data (use consensus or first available)
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]  # Use first bookmaker
                game_info['bookmaker'] = bookmaker['key']
                
                # Get markets
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':  # Moneyline
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                game_info['home_ml'] = outcome['price']
                            else:
                                game_info['away_ml'] = outcome['price']
                    
                    elif market['key'] == 'spreads':  # Point spreads
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                game_info['home_spread'] = outcome['point']
                                game_info['home_spread_odds'] = outcome['price']
                            else:
                                game_info['away_spread'] = outcome['point']
                                game_info['away_spread_odds'] = outcome['price']
                    
                    elif market['key'] == 'totals':  # Over/Under
                        game_info['total'] = market['outcomes'][0]['point']
                        game_info['over_odds'] = market['outcomes'][0]['price']
                        game_info['under_odds'] = market['outcomes'][1]['price']
            
            games.append(game_info)
        
        df = pd.DataFrame(games)
        print(f"✅ Fetched odds for {len(df)} games")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return None


def save_live_odds(df, filename=None):
    """Save fetched odds to CSV"""
    if df is None or df.empty:
        print("❌ No odds data to save")
        return
    
    if filename is None:
        filename = f"live_odds_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    filepath = BETTING_PATH / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Saved live odds to: {filepath}")
    
    # Also save as latest.csv for easy access
    latest_path = BETTING_PATH / 'live_odds_latest.csv'
    df.to_csv(latest_path, index=False)
    print(f"✅ Saved as latest: {latest_path}")
    
    return filepath


def convert_to_vegas_format(df):
    """
    Convert The Odds API format to match our Vegas betting lines format
    
    Returns:
        DataFrame compatible with process_real_vegas_lines.py
    """
    if df is None or df.empty:
        return None
    
    print("🔄 Converting to Vegas format...")
    
    # Map team names to standard abbreviations
    # The Odds API uses full names, we need abbreviations
    team_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    converted = []
    for _, row in df.iterrows():
        # Parse date
        date = pd.to_datetime(row['commence_time']).strftime('%Y-%m-%d')
        
        # Get team abbreviations
        home_abbr = team_map.get(row['home_team'], row['home_team'][:3].upper())
        away_abbr = team_map.get(row['away_team'], row['away_team'][:3].upper())
        
        # Create row in Vegas format
        vegas_row = {
            'Date': date,
            'VH': 'H',  # Home
            'Team': home_abbr,
            'Open': row.get('home_spread', 0),
            'Close': row.get('home_spread', 0),
            'ML': row.get('home_ml', 0),
            'OU': row.get('total', 0),
        }
        converted.append(vegas_row)
    
    result = pd.DataFrame(converted)
    print(f"✅ Converted {len(result)} games to Vegas format")
    
    return result


def main():
    """Main execution"""
    print("=" * 60)
    print("NBA LIVE ODDS SCRAPER - The Odds API")
    print("=" * 60)
    print()
    
    # Fetch odds
    df = fetch_live_odds()
    
    if df is not None:
        # Save raw data
        raw_path = save_live_odds(df)
        
        # Convert to Vegas format
        vegas_df = convert_to_vegas_format(df)
        if vegas_df is not None:
            vegas_path = BETTING_PATH / 'live_odds_vegas_format.csv'
            vegas_df.to_csv(vegas_path, index=False)
            print(f"✅ Saved Vegas format: {vegas_path}")
        
        # Display summary
        print()
        print("📊 SUMMARY")
        print("-" * 60)
        print(f"Games fetched: {len(df)}")
        print(f"Upcoming games:")
        for _, row in df.iterrows():
            commence = pd.to_datetime(row['commence_time'])
            print(f"  • {row['away_team']} @ {row['home_team']}")
            print(f"    Time: {commence.strftime('%Y-%m-%d %I:%M %p')}")
            if 'home_spread' in row and pd.notna(row['home_spread']):
                print(f"    Spread: {row['home_spread']:+.1f}")
            if 'total' in row and pd.notna(row['total']):
                print(f"    O/U: {row['total']}")
            if 'home_ml' in row and pd.notna(row['home_ml']):
                print(f"    ML: {row['home_ml']:+.0f} / {row.get('away_ml', 0):+.0f}")
        
        print()
        print("✅ SUCCESS!")
        print("-" * 60)
        print(f"📁 Raw data saved: {raw_path}")
        if vegas_df is not None:
            print(f"📁 Vegas format: {vegas_path}")
        print()
        print("💡 Next steps:")
        print("  1. Data ready in data/betting/")
        print("  2. Use live odds in your Streamlit app")
        print("  3. Or merge with historical data for training")
    else:
        print("❌ Failed to fetch odds")
        print()
        print("💡 Troubleshooting:")
        print("  1. Check your API key: https://the-odds-api.com/account")
        print("  2. Verify API credits remaining")
        print("  3. Set environment variable: export ODDS_API_KEY=your_key")
        print("  4. Or add to GitHub Secrets for automation")


if __name__ == "__main__":
    main()

