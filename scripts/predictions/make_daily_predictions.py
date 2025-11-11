#!/usr/bin/env python3
"""
Make Daily NBA Game Predictions

Loads today's games, makes predictions using the best model,
and exports results to CSV for tracking and analysis.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
DATAPATH = Path(__file__).resolve().parents[2] / 'data'
MODELPATH = Path(__file__).resolve().parents[2] / 'models'
PREDICTIONS_PATH = DATAPATH / 'predictions'
PREDICTIONS_PATH.mkdir(exist_ok=True)

# Best model options (try in order)
MODEL_OPTIONS = [
    'histgradient_vegas_calibrated.pkl',
    'xgboost_vegas_calibrated.pkl',
    'best_model_randomforest_vegas.pkl',
    'ensemble_stacking_vegas.pkl',
    'ensemble_weighted_vegas.pkl',
]


def load_model():
    """Load the best performing model"""
    last_error = None
    
    for model_file in MODEL_OPTIONS:
        model_path = MODELPATH / model_file
        if not model_path.exists():
            continue
        
        try:
            # Try joblib first (scikit-learn standard)
            model = joblib.load(model_path)
            print(f"✅ Loaded model with joblib: {model_file}")
            return model
        except Exception as e1:
            try:
                # Fallback to pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ Loaded model with pickle: {model_file}")
                return model
            except Exception as e2:
                last_error = f"{model_file}: joblib error={e1}, pickle error={e2}"
                continue
    
    # If we get here, no model loaded successfully
    raise FileNotFoundError(
        f"Could not load any model. Tried: {MODEL_OPTIONS}\n"
        f"Last error: {last_error}\n"
        f"Models directory: {MODELPATH}"
    )


def load_todays_games():
    """Load today's scheduled games"""
    try:
        # Load games with all features
        df = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
        
        # Get today's games
        today = datetime.now().date()
        df_today = df[df['GAME_DATE_EST'].dt.date == today].copy()
        
        print(f"📅 Found {len(df_today)} games scheduled for {today}")
        return df_today
    
    except Exception as e:
        print(f"❌ Error loading games: {e}")
        return None


def load_live_odds():
    """Load live betting odds if available"""
    try:
        odds_path = DATAPATH / 'betting' / 'live_odds_latest.csv'
        if odds_path.exists():
            df = pd.read_csv(odds_path)
            print(f"🎲 Loaded live odds for {len(df)} games")
            return df
        else:
            print("⚠️  No live odds available")
            return None
    except Exception as e:
        print(f"⚠️  Could not load live odds: {e}")
        return None


def prepare_features(df):
    """Prepare features for prediction - MUST match training data exactly"""
    # Create MATCHUP if it doesn't exist
    if 'MATCHUP' not in df.columns:
        if 'HOME_TEAM_ABBREVIATION' in df.columns and 'VISITOR_TEAM_ABBREVIATION' in df.columns:
            df['MATCHUP'] = df['VISITOR_TEAM_ABBREVIATION'] + ' @ ' + df['HOME_TEAM_ABBREVIATION']
        else:
            # Fallback if team abbreviations don't exist
            df['MATCHUP'] = 'Game ' + df.index.astype(str)
    
    # Features to drop - these are target, metadata, categorical, or leaky features
    # This MUST match what was used in training (from streamlit_app_enhanced.py)
    drop_cols = [
        # Target
        'TARGET',
        # Metadata
        'GAME_DATE_EST', 'GAME_ID', 'MATCHUP', 'SEASON',
        # Team identifiers (categorical)
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        # Leaky features (reveal outcome)
        'HOME_TEAM_WINS', 'VISITOR_TEAM_WINS', 
        'HOME_WL', 'VISITOR_WL',
        'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away',
        'FT_PCT_home', 'FT_PCT_away', 
        'FG3_PCT_home', 'FG3_PCT_away',
        'AST_home', 'AST_away',
        'REB_home', 'REB_away'
    ]
    
    # Drop columns that exist
    existing_drops = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=existing_drops, errors='ignore')
    
    # Remove any other object/string columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"⚠️  Dropping {len(object_cols)} object columns: {list(object_cols)}")
        X = X.drop(columns=object_cols)
    
    print(f"✅ Prepared features: {X.shape[1]} features")
    print(f"   First few features: {list(X.columns[:10])}")
    
    # Keep metadata for display
    metadata_cols = ['GAME_DATE_EST', 'MATCHUP']
    metadata = df[metadata_cols].copy()
    
    # Add team abbreviations if available
    if 'HOME_TEAM_ABBREVIATION' in df.columns:
        metadata['HOME_TEAM_ABBREVIATION'] = df['HOME_TEAM_ABBREVIATION']
    if 'VISITOR_TEAM_ABBREVIATION' in df.columns:
        metadata['VISITOR_TEAM_ABBREVIATION'] = df['VISITOR_TEAM_ABBREVIATION']
    
    return X, metadata


def make_predictions(model, X):
    """Make predictions using the model"""
    try:
        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of home win
        
        return predictions, probabilities
    
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None, None


def match_live_odds(metadata, live_odds_df):
    """Match games with live betting odds"""
    if live_odds_df is None or live_odds_df.empty:
        return [None] * len(metadata)
    
    matched_odds = []
    for _, game in metadata.iterrows():
        matchup = game['MATCHUP']
        
        # Try to find matching game in live odds
        matched = None
        for _, odds in live_odds_df.iterrows():
            odds_matchup = f"{odds.get('away_team', '')} @ {odds.get('home_team', '')}"
            if matchup in odds_matchup or odds_matchup in matchup:
                matched = odds
                break
        
        matched_odds.append(matched)
    
    return matched_odds


def create_predictions_df(metadata, predictions, probabilities, live_odds_matches):
    """Create a DataFrame with predictions and analysis"""
    results = []
    
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        row_data = {
            'Date': metadata.iloc[idx]['GAME_DATE_EST'],
            'Matchup': metadata.iloc[idx]['MATCHUP'],
            'Home_Win_Probability': prob,
            'Predicted_Winner': 'Home' if pred == 1 else 'Away',
            'Confidence': 'High' if abs(prob - 0.5) > 0.15 else 'Medium' if abs(prob - 0.5) > 0.05 else 'Low',
            'Model': 'HistGradient + Vegas (70.20% AUC)',
        }
        
        # Add live odds if available
        live_odds = live_odds_matches[idx]
        if live_odds is not None:
            row_data['Vegas_Spread'] = live_odds.get('home_spread', None)
            row_data['Vegas_Total'] = live_odds.get('total', None)
            row_data['Vegas_ML_Home'] = live_odds.get('home_ml', None)
            row_data['Vegas_ML_Away'] = live_odds.get('away_ml', None)
            
            # Calculate implied probability from moneyline
            if pd.notna(live_odds.get('home_ml')):
                ml = live_odds['home_ml']
                if ml > 0:
                    vegas_prob = 100 / (ml + 100)
                else:
                    vegas_prob = abs(ml) / (abs(ml) + 100)
                row_data['Vegas_Implied_Home_Win_Prob'] = vegas_prob
                row_data['Edge_vs_Vegas'] = prob - vegas_prob
        
        results.append(row_data)
    
    df = pd.DataFrame(results)
    return df


def save_predictions(df, filename=None):
    """Save predictions to CSV"""
    if filename is None:
        filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    
    filepath = PREDICTIONS_PATH / filename
    df.to_csv(filepath, index=False)
    
    # Also save as 'latest' for easy access
    latest_path = PREDICTIONS_PATH / 'predictions_latest.csv'
    df.to_csv(latest_path, index=False)
    
    print(f"✅ Saved predictions: {filepath}")
    print(f"✅ Saved as latest: {latest_path}")
    
    return filepath


def display_summary(df):
    """Display prediction summary"""
    print()
    print("=" * 80)
    print(f"NBA PREDICTIONS FOR {datetime.now().strftime('%A, %B %d, %Y')}")
    print("=" * 80)
    print()
    
    if len(df) == 0:
        print("❌ No games scheduled for today")
        return
    
    for idx, row in df.iterrows():
        print(f"🏀 {row['Matchup']}")
        print(f"   Predicted Winner: {row['Predicted_Winner']}")
        print(f"   Home Win Probability: {row['Home_Win_Probability']:.1%}")
        print(f"   Confidence: {row['Confidence']}")
        
        if 'Vegas_Spread' in row and pd.notna(row['Vegas_Spread']):
            print(f"   Vegas Spread: {row['Vegas_Spread']:+.1f}")
        if 'Vegas_Total' in row and pd.notna(row['Vegas_Total']):
            print(f"   Vegas O/U: {row['Vegas_Total']:.1f}")
        if 'Edge_vs_Vegas' in row and pd.notna(row['Edge_vs_Vegas']):
            edge = row['Edge_vs_Vegas']
            print(f"   Edge vs Vegas: {edge:+.1%} {'✅ STRONG' if abs(edge) > 0.05 else ''}")
        
        print()
    
    print("-" * 80)
    print(f"Total games: {len(df)}")
    print(f"High confidence: {len(df[df['Confidence'] == 'High'])}")
    print(f"Medium confidence: {len(df[df['Confidence'] == 'Medium'])}")
    print(f"Low confidence: {len(df[df['Confidence'] == 'Low'])}")
    print("=" * 80)


def main():
    """Main execution"""
    print()
    print("=" * 80)
    print("NBA DAILY PREDICTION GENERATOR")
    print("=" * 80)
    print()
    
    # Load model
    model = load_model()
    
    # Load today's games
    df_today = load_todays_games()
    
    if df_today is None or len(df_today) == 0:
        print("❌ No games found for today")
        return
    
    # Prepare features
    X, metadata = prepare_features(df_today)
    print(f"✅ Prepared features: {X.shape[1]} features")
    
    # Make predictions
    predictions, probabilities = make_predictions(model, X)
    
    if predictions is None:
        print("❌ Failed to make predictions")
        return
    
    print(f"✅ Made predictions for {len(predictions)} games")
    
    # Load live odds
    live_odds_df = load_live_odds()
    
    # Match with live odds
    live_odds_matches = match_live_odds(metadata, live_odds_df)
    
    # Create predictions DataFrame
    predictions_df = create_predictions_df(metadata, predictions, probabilities, live_odds_matches)
    
    # Save predictions
    filepath = save_predictions(predictions_df)
    
    # Display summary
    display_summary(predictions_df)
    
    print()
    print("💡 Next steps:")
    print(f"  1. Review predictions: {filepath}")
    print("  2. Check data/predictions/predictions_latest.csv")
    print("  3. Track results as games complete")
    print()


if __name__ == "__main__":
    main()

