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
    
    return filepath


def main():
    """Main execution"""
    print("=" * 60)
    print("NBA LIVE ODDS SCRAPER")
    print("=" * 60)
    print()
    
    # Fetch odds
    df = fetch_live_odds()
    
    if df is not None:
        # Save to CSV
        save_live_odds(df)
        
        # Display summary
        print()
        print("📊 SUMMARY")
        print("-" * 60)
        print(f"Games fetched: {len(df)}")
        print(f"Upcoming games:")
        for _, row in df.iterrows():
            print(f"  • {row['away_team']} @ {row['home_team']}")
            if 'home_spread' in row:
                print(f"    Spread: {row['home_spread']:+.1f}")
            if 'total' in row:
                print(f"    O/U: {row['total']}")
        
        print()
        print("💡 Next steps:")
        print("  1. Review the fetched odds in data/betting/")
        print("  2. Run process_real_vegas_lines.py to merge with games")
        print("  3. Retrain models with updated data")
    else:
        print("❌ Failed to fetch odds")
        print()
        print("💡 Options:")
        print("  1. Get free API key: https://the-odds-api.com/")
        print("  2. Set environment variable: export ODDS_API_KEY=your_key")
        print("  3. Or use Kaggle dataset: python download_real_vegas_data.py")


if __name__ == "__main__":
    main()

