"""
Live Odds Display Helper for Streamlit App
"""

import pandas as pd
from pathlib import Path

DATAPATH = Path(__file__).resolve().parent.parent / 'data'


def load_live_odds():
    """Load live betting odds if available"""
    try:
        live_odds_path = DATAPATH / 'betting' / 'live_odds_latest.csv'
        
        # Debug print
        print(f"Looking for live odds at: {live_odds_path}")
        print(f"File exists: {live_odds_path.exists()}")
        
        if live_odds_path.exists():
            df = pd.read_csv(live_odds_path)
            print(f"Loaded {len(df)} games with live odds")
            print(f"Columns: {df.columns.tolist()}")
            return df
        else:
            print("Live odds file not found")
        return None
    except Exception as e:
        print(f"Error loading live odds: {e}")
        return None


def match_game_to_odds(matchup, live_odds_df):
    """Match a game matchup to live odds data"""
    if live_odds_df is None or live_odds_df.empty:
        return None
    
    # Try to match the game
    for _, odds in live_odds_df.iterrows():
        home = str(odds.get('home_team', ''))
        away = str(odds.get('away_team', ''))
        
        # Check if teams match (various formats)
        if (home in matchup or away in matchup or 
            matchup.replace(' @ ', ' ') in f"{away} {home}"):
            return odds
    
    return None


def format_odds_display(odds):
    """Format odds data for display"""
    if odds is None:
        return None
    
    display_data = {}
    
    if 'home_spread' in odds and pd.notna(odds['home_spread']):
        display_data['spread'] = f"{odds['home_spread']:+.1f}"
    
    if 'total' in odds and pd.notna(odds['total']):
        display_data['total'] = f"{odds['total']:.1f}"
    
    if 'home_ml' in odds and pd.notna(odds['home_ml']):
        display_data['ml_home'] = f"{odds['home_ml']:+.0f}"
    
    if 'away_ml' in odds and pd.notna(odds['away_ml']):
        display_data['ml_away'] = f"{odds['away_ml']:+.0f}"
    
    return display_data if display_data else None

