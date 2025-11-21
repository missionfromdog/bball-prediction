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

def fetch_live_odds(api_key=None, sport='basketball_nba', return_raw=False):
    """
    Fetch live NBA betting odds from The Odds API
    
    Args:
        api_key: API key from the-odds-api.com (or set ODDS_API_KEY env var)
        sport: Sport to fetch (default: basketball_nba)
        return_raw: If True, return both processed DataFrame and raw JSON data
    
    Returns:
        DataFrame with current betting lines (or tuple of (DataFrame, raw_data) if return_raw=True)
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
        
        # Parse odds data - extract ALL bookmakers for comparison
        games = []
        for game in data:
            game_info = {
                'game_id': game['id'],
                'sport': game['sport_key'],
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
            }
            
            # Extract bookmaker data (use consensus or first available for backward compatibility)
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]  # Use first bookmaker for main data
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
        
        if return_raw:
            return df, data  # Return both processed df and raw data for bookmaker comparison
        else:
            return df  # Backward compatible: return just DataFrame
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return None


def calculate_vig(home_ml, away_ml):
    """
    Calculate vig (bookmaker margin) from moneyline odds
    
    Vig = (sum of implied probabilities) - 1.0
    
    Returns:
        Vig as percentage (e.g., 4.2 for 4.2%)
    """
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None
    
    # Convert American odds to implied probabilities
    if home_ml > 0:
        home_prob = 100 / (home_ml + 100)
    else:
        home_prob = abs(home_ml) / (abs(home_ml) + 100)
    
    if away_ml > 0:
        away_prob = 100 / (away_ml + 100)
    else:
        away_prob = abs(away_ml) / (abs(away_ml) + 100)
    
    # Vig = sum of probabilities - 1.0
    vig = (home_prob + away_prob - 1.0) * 100
    return vig


def extract_all_bookmakers(raw_data):
    """
    Extract all bookmakers for each game with vig calculations
    
    Returns:
        DataFrame with one row per game-bookmaker combination
    """
    all_bookmakers = []
    
    # Major US sportsbooks (licensed and regulated)
    # Includes: All major US books + Bet365, Hard Rock Bet, ESPN BET
    # Excludes: Offshore books (BetOnline, Bovada, MyBookie, SportsBetting)
    us_bookmakers = [
        # Top Major US Sportsbooks (include all variations)
        'draftkings', 'draftkings_sportsbook',
        'fanduel', 'fanduel_sportsbook',
        'betmgm', 'betmgm_sportsbook',
        'caesars', 'caesars_sportsbook',
        'betrivers', 'betrivers_sportsbook',
        'pointsbet', 'pointsbet_us',
        'wynnbet', 'wynnbet_sportsbook',
        'barstool', 'barstool_sportsbook',
        'foxbet', 'foxbet_sportsbook',
        'unibet', 'unibet_us',
        # Additional Major Books (include variations)
        'bet365',
        'hardrockbet', 'hard_rock_bet',
        'espnbet', 'espn_bet',
        # Regional (optional - include if available, with variations)
        'circasports', 'circa_sports',
        'superbook', 'superbook_sports',
        'betway'
    ]
    
    for game in raw_data:
        game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        if not game.get('bookmakers'):
            continue
        
        for bookmaker in game['bookmakers']:
            bookmaker_key = bookmaker.get('key', '').lower()
            
            # Only include major US sportsbooks (exclude offshore)
            # Check if bookmaker key matches any in our approved list
            # Use flexible matching to catch variations (e.g., "draftkings" matches "draftkings_sportsbook")
            is_approved = False
            for us_name in us_bookmakers:
                # Check if the bookmaker key contains our approved name, or vice versa
                # This handles variations like "draftkings" vs "draftkings_sportsbook"
                if (us_name in bookmaker_key or 
                    bookmaker_key in us_name or 
                    bookmaker_key == us_name):
                    is_approved = True
                    break
            
            if not is_approved:
                # Skip offshore and non-approved bookmakers
                continue
            
            bookmaker_data = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'bookmaker': bookmaker['title'],  # Display name
                'bookmaker_key': bookmaker_key,
            }
            
            # Extract markets
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':  # Moneyline
                    home_ml = None
                    away_ml = None
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            home_ml = outcome['price']
                        else:
                            away_ml = outcome['price']
                    
                    bookmaker_data['home_ml'] = home_ml
                    bookmaker_data['away_ml'] = away_ml
                    bookmaker_data['ml_vig'] = calculate_vig(home_ml, away_ml) if home_ml and away_ml else None
                
                elif market['key'] == 'spreads':  # Point spreads
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            bookmaker_data['home_spread'] = outcome['point']
                            bookmaker_data['home_spread_odds'] = outcome['price']
                        else:
                            bookmaker_data['away_spread'] = outcome['point']
                            bookmaker_data['away_spread_odds'] = outcome['price']
                    
                    # Calculate spread vig if we have both odds
                    if 'home_spread_odds' in bookmaker_data and 'away_spread_odds' in bookmaker_data:
                        bookmaker_data['spread_vig'] = calculate_vig(
                            bookmaker_data['home_spread_odds'],
                            bookmaker_data['away_spread_odds']
                        )
                
                elif market['key'] == 'totals':  # Over/Under
                    bookmaker_data['total'] = market['outcomes'][0]['point']
                    over_odds = market['outcomes'][0]['price']
                    under_odds = market['outcomes'][1]['price']
                    bookmaker_data['over_odds'] = over_odds
                    bookmaker_data['under_odds'] = under_odds
                    bookmaker_data['total_vig'] = calculate_vig(over_odds, under_odds) if over_odds and under_odds else None
            
            all_bookmakers.append(bookmaker_data)
    
    df = pd.DataFrame(all_bookmakers)
    return df


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
    
    # Fetch odds (with raw data for bookmaker comparison)
    result = fetch_live_odds(return_raw=True)
    
    if result is not None:
        df, raw_data = result
        
        # Save raw data (backward compatible)
        raw_path = save_live_odds(df)
        
        # Extract all bookmakers for comparison
        print("\n📊 Extracting all US bookmakers for comparison...")
        try:
            bookmakers_df = extract_all_bookmakers(raw_data)
            
            if not bookmakers_df.empty:
                # Save bookmaker comparison data
                comparison_path = BETTING_PATH / 'live_odds_bookmakers_comparison.csv'
                bookmakers_df.to_csv(comparison_path, index=False)
                print(f"✅ Saved bookmaker comparison: {comparison_path}")
                print(f"   Found {len(bookmakers_df)} bookmaker-game combinations")
                print(f"   Unique bookmakers: {bookmakers_df['bookmaker'].nunique()}")
                print(f"   Unique games: {bookmakers_df['game_id'].nunique()}")
                print(f"   Sample bookmakers: {bookmakers_df['bookmaker'].unique()[:5].tolist()}")
            else:
                print("⚠️ No US bookmakers found in data")
                print("   This might mean:")
                print("   - No upcoming games with US bookmaker odds")
                print("   - API returned different bookmaker keys")
                print("   - All games are from non-US bookmakers")
                
                # Debug: Show what bookmakers are available
                all_bookmaker_keys = set()
                all_bookmaker_titles = set()
                for game in raw_data:
                    for bookmaker in game.get('bookmakers', []):
                        all_bookmaker_keys.add(bookmaker.get('key', '').lower())
                        all_bookmaker_titles.add(bookmaker.get('title', ''))
                
                if all_bookmaker_keys:
                    print(f"   Available bookmaker keys: {sorted(list(all_bookmaker_keys))}")
                    print(f"   Available bookmaker titles: {sorted(list(all_bookmaker_titles))[:10]}")
                    
                    # Show which ones match our approved list
                    approved_keys = [
                        'draftkings', 'fanduel', 'betmgm', 'caesars', 'betrivers',
                        'pointsbet', 'wynnbet', 'barstool', 'foxbet', 'unibet',
                        'bet365', 'hardrockbet', 'espnbet', 'circasports', 'superbook', 'betway'
                    ]
                    matched_keys = [k for k in all_bookmaker_keys if any(approved in k for approved in approved_keys)]
                    unmatched_keys = [k for k in all_bookmaker_keys if k not in matched_keys]
                    print(f"   ✅ Matched approved bookmakers: {sorted(matched_keys)}")
                    if unmatched_keys:
                        print(f"   ⚠️ Unmatched bookmaker keys (not in approved list): {sorted(unmatched_keys)}")
                
                # Create empty file so Streamlit doesn't error
                comparison_path = BETTING_PATH / 'live_odds_bookmakers_comparison.csv'
                empty_df = pd.DataFrame(columns=['game_id', 'home_team', 'away_team', 'bookmaker', 'ml_vig', 'spread_vig', 'total_vig'])
                empty_df.to_csv(comparison_path, index=False)
                print(f"   Created empty comparison file: {comparison_path}")
        except Exception as e:
            print(f"❌ Error extracting bookmakers: {e}")
            import traceback
            traceback.print_exc()
        
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

