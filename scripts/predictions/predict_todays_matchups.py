"""
Predict today's NBA games using historical data to calculate features on-the-fly

This script:
1. Gets today's matchups
2. Calculates features from each team's past games
3. Adds today's odds and injuries
4. Makes predictions

This is the CORRECT approach - don't try to pre-engineer features for future games!
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Paths
PROJECTPATH = Path(__file__).resolve().parents[2]
DATAPATH = PROJECTPATH / 'data'
MODELPATH = PROJECTPATH / 'models'
PREDICTIONS_PATH = DATAPATH / 'predictions'
PREDICTIONS_PATH.mkdir(exist_ok=True)

def load_historical_data():
    """Load historical games with full features"""
    print("📊 Loading historical data...")
    data_path = DATAPATH / 'games_with_real_vegas_workflow.csv'
    df = pd.read_csv(data_path)
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    print(f"   Loaded {len(df):,} historical games")
    return df


def get_todays_matchups():
    """
    Get today's matchups from the schedule fetch
    Returns list of dicts: [{'home': 'CLE', 'away': 'TOR', 'date': '2025-11-13'}, ...]
    """
    print("\n📅 Getting today's matchups...")
    
    # Try to load from schedule fetch output
    schedule_path = DATAPATH / 'games_with_real_vegas_workflow.csv'
    df = pd.read_csv(schedule_path)
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST']).dt.date
    
    today = datetime.now().date()
    
    # Get games for today that haven't been played yet
    todays_games = df[
        (df['GAME_DATE_EST'] == today) & 
        (df['PTS_home'] == 0)
    ]
    
    if len(todays_games) == 0:
        print(f"   ⚠️  No games found for {today}")
        print(f"   Trying next 7 days...")
        max_date = today + timedelta(days=7)
        todays_games = df[
            (df['GAME_DATE_EST'] >= today) & 
            (df['GAME_DATE_EST'] <= max_date) &
            (df['PTS_home'] == 0)
        ]
    
    matchups = []
    for _, game in todays_games.iterrows():
        matchups.append({
            'home': game['HOME_TEAM_ABBREVIATION'],
            'away': game['VISITOR_TEAM_ABBREVIATION'],
            'date': game['GAME_DATE_EST']
        })
    
    print(f"   Found {len(matchups)} matchups:")
    for m in matchups:
        print(f"      {m['away']} @ {m['home']} on {m['date']}")
    
    return matchups


def calculate_team_features(team_abbr, historical_df, as_of_date, is_home=True):
    """
    Calculate rolling average features for a team based on their past games
    
    Args:
        team_abbr: Team abbreviation (e.g., 'CLE')
        historical_df: Full historical data
        as_of_date: Calculate features as of this date (use past games only)
        is_home: True if calculating for home team, False for away team
    
    Returns:
        dict of features
    """
    # Get team's past games (before as_of_date)
    team_games = historical_df[
        (
            (historical_df['HOME_TEAM_ABBREVIATION'] == team_abbr) |
            (historical_df['VISITOR_TEAM_ABBREVIATION'] == team_abbr)
        ) &
        (historical_df['GAME_DATE_EST'] < pd.to_datetime(as_of_date))
    ].sort_values('GAME_DATE_EST', ascending=False)
    
    # Get last N games
    last_10 = team_games.head(10)
    last_7 = team_games.head(7)
    last_3 = team_games.head(3)
    
    features = {}
    prefix = 'HOME_' if is_home else 'VISITOR_'
    
    # Win percentage (last 3, 7, 10 games)
    for n, games in [(3, last_3), (7, last_7), (10, last_10)]:
        # Count wins
        wins = 0
        for _, game in games.iterrows():
            if game['HOME_TEAM_ABBREVIATION'] == team_abbr:
                wins += game['HOME_TEAM_WINS']
            else:
                wins += (1 - game['HOME_TEAM_WINS'])
        
        win_pct = wins / len(games) if len(games) > 0 else 0.5
        features[f'{prefix}TEAM_WINS_AVG_LAST_{n}'] = win_pct
    
    # Win streak
    streak = 0
    last_result = None
    for _, game in team_games.head(10).iterrows():
        is_win = (game['HOME_TEAM_ABBREVIATION'] == team_abbr and game['HOME_TEAM_WINS'] == 1) or \
                 (game['VISITOR_TEAM_ABBREVIATION'] == team_abbr and game['HOME_TEAM_WINS'] == 0)
        
        if last_result is None:
            last_result = is_win
            streak = 1 if is_win else -1
        elif is_win == last_result:
            streak += 1 if is_win else -1
        else:
            break
    
    features[f'{prefix}TEAM_WIN_STREAK'] = streak
    
    # Home/Away specific features
    if is_home:
        home_games = team_games[team_games['HOME_TEAM_ABBREVIATION'] == team_abbr].head(10)
        for n in [3, 7, 10]:
            games = home_games.head(n)
            if len(games) > 0:
                features[f'{prefix}TEAM_WINS_AVG_LAST_{n}_HOME'] = games['HOME_TEAM_WINS'].mean()
            else:
                features[f'{prefix}TEAM_WINS_AVG_LAST_{n}_HOME'] = 0.5
    else:
        away_games = team_games[team_games['VISITOR_TEAM_ABBREVIATION'] == team_abbr].head(10)
        for n in [3, 7, 10]:
            games = away_games.head(n)
            if len(games) > 0:
                features[f'{prefix}TEAM_WINS_AVG_LAST_{n}_VISITOR'] = (1 - games['HOME_TEAM_WINS']).mean()
            else:
                features[f'{prefix}TEAM_WINS_AVG_LAST_{n}_VISITOR'] = 0.5
    
    return features


def create_prediction_row(matchup, historical_df):
    """
    Create a prediction row for a single matchup
    
    Args:
        matchup: dict with 'home', 'away', 'date'
        historical_df: Full historical data
    
    Returns:
        dict representing one row for prediction
    """
    row = {}
    
    # Basic info
    row['GAME_DATE_EST'] = matchup['date']
    row['HOME_TEAM_ABBREVIATION'] = matchup['home']
    row['VISITOR_TEAM_ABBREVIATION'] = matchup['away']
    row['MATCHUP'] = f"{matchup['away']} @ {matchup['home']}"
    row['MONTH'] = pd.to_datetime(matchup['date']).month
    
    # Calculate features for home team
    home_features = calculate_team_features(
        matchup['home'], 
        historical_df, 
        matchup['date'], 
        is_home=True
    )
    row.update(home_features)
    
    # Calculate features for away team
    away_features = calculate_team_features(
        matchup['away'], 
        historical_df, 
        matchup['date'], 
        is_home=False
    )
    row.update(away_features)
    
    # TODO: Add Vegas odds from live_odds_latest.csv if available
    # TODO: Add injury data if available
    
    return row


def load_model():
    """Load the best trained model"""
    model_options = [
        'histgradient_vegas_calibrated.pkl',
        'ensemble_weighted_vegas.pkl',
        'xgboost_vegas.pkl',
    ]
    
    for model_file in model_options:
        model_path = MODELPATH / model_file
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print(f"✅ Loaded model: {model_file}")
                return model
            except:
                continue
    
    raise FileNotFoundError("No valid model found!")


def main():
    print("="*80)
    print("🎯 NBA DAILY PREDICTIONS - MATCHUP-BASED")
    print("="*80)
    print()
    
    # Load historical data
    historical_df = load_historical_data()
    
    # Get today's matchups
    matchups = get_todays_matchups()
    
    if len(matchups) == 0:
        print("\n❌ No games to predict")
        return
    
    # Create prediction rows
    print("\n🔧 Creating prediction features...")
    prediction_rows = []
    for matchup in matchups:
        row = create_prediction_row(matchup, historical_df)
        prediction_rows.append(row)
        print(f"   ✅ {matchup['away']} @ {matchup['home']}")
    
    df_predict = pd.DataFrame(prediction_rows)
    print(f"\n✅ Created {len(df_predict)} prediction rows")
    print(f"   Features: {len(df_predict.columns)}")
    
    # Load model
    print("\n🤖 Loading model...")
    model = load_model()
    
    # Make predictions
    print("\n🎯 Making predictions...")
    
    # Drop non-feature columns
    metadata_cols = ['GAME_DATE_EST', 'MATCHUP', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION']
    X = df_predict.drop(columns=[col for col in metadata_cols if col in df_predict.columns])
    
    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results dataframe
    results = []
    for idx, (matchup, pred, prob) in enumerate(zip(matchups, predictions, probabilities)):
        results.append({
            'Date': matchup['date'],
            'Matchup': f"{matchup['away']} @ {matchup['home']}",
            'Home_Win_Probability': prob,
            'Predicted_Winner': 'Home' if pred == 1 else 'Away',
            'Confidence': 'High' if abs(prob - 0.5) > 0.15 else 'Medium' if abs(prob - 0.5) > 0.05 else 'Low',
        })
    
    df_results = pd.DataFrame(results)
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"predictions_{timestamp}.csv"
    df_results.to_csv(PREDICTIONS_PATH / filename, index=False)
    df_results.to_csv(PREDICTIONS_PATH / 'predictions_latest.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ PREDICTIONS COMPLETE")
    print("="*80)
    print(f"\n📊 Predicted {len(df_results)} games:")
    print(df_results.to_string(index=False))
    print(f"\n💾 Saved to: {filename}")


if __name__ == "__main__":
    main()

