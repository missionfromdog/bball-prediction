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
PROJECTPATH = Path(__file__).resolve().parents[2]
DATAPATH = PROJECTPATH / 'data'
MODELPATH = PROJECTPATH / 'models'
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


def is_valid_model_file(filepath):
    """Check if model file is valid (not LFS pointer) and can be loaded"""
    if not filepath.exists():
        return False
    
    size = filepath.stat().st_size
    # LFS pointer files are tiny (< 200 bytes), real models are 2-3 MB
    if size < 1000:
        return False
    
    # File size is OK, but can we actually load it?
    # Try loading to detect numpy version mismatches or corruption
    try:
        import joblib
        model = joblib.load(filepath)
        # Successfully loaded!
        return True
    except Exception as e:
        # Can't load (numpy mismatch, corruption, etc.)
        print(f"⚠️  Model file exists but can't be loaded: {e}")
        return False


def retrain_model():
    """Retrain model if needed - called automatically if model is invalid"""
    print("\n" + "="*80)
    print("🔧 MODEL MISSING OR INVALID - RETRAINING")
    print("="*80)
    print("This will take ~2-3 minutes...")
    
    import sys
    PROJECTPATH = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECTPATH / 'scripts' / 'predictions'))
    
    try:
        from setup_model import retrain_model as do_retrain, MODEL_PATH
        success = do_retrain()
        
        if success and is_valid_model_file(MODEL_PATH):
            print("\n✅ Model retrained successfully!")
            return MODEL_PATH
        else:
            print("\n❌ Model retraining failed")
            return None
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_model():
    """Load the best performing model - retrain automatically if invalid"""
    last_error = None
    any_model_exists = False
    
    for model_file in MODEL_OPTIONS:
        model_path = MODELPATH / model_file
        if not model_path.exists():
            continue
        
        any_model_exists = True
        
        # Check if file is valid (not LFS pointer)
        if not is_valid_model_file(model_path):
            file_size = model_path.stat().st_size
            print(f"⚠️  {model_file} is too small ({file_size} bytes) - probably LFS pointer")
            
            # Only try to retrain the first/best model
            if model_file == MODEL_OPTIONS[0]:
                print("🔄 Attempting to retrain best model...")
                retrained_path = retrain_model()
                if retrained_path and is_valid_model_file(retrained_path):
                    model_path = retrained_path
                else:
                    print("❌ Retraining failed, trying other models...")
                    continue
            else:
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
    
    # If NO models exist at all, train one
    if not any_model_exists:
        print("⚠️  No model files found in models directory")
        print("🔄 Training model from scratch...")
        retrained_path = retrain_model()
        if retrained_path and is_valid_model_file(retrained_path):
            model = joblib.load(retrained_path)
            print(f"✅ Loaded newly trained model")
            return model
        else:
            raise FileNotFoundError("Model training failed")
    
    # If we get here, models existed but none loaded successfully
    raise FileNotFoundError(
        f"Could not load any model. Tried: {MODEL_OPTIONS}\n"
        f"Last error: {last_error}\n"
        f"Models directory: {MODELPATH}"
    )


def load_todays_games():
    """Load today's scheduled games (unplayed games with PTS_home == 0)"""
    try:
        # Load games with all features (prioritize master dataset, fallback for workflows)
        # Try master dataset first (30k games with full history)
        data_file = DATAPATH / 'games_master_engineered.csv'
        if not data_file.exists() or data_file.stat().st_size < 1000:
            # Try workflow dataset (5k games, used in GitHub Actions)
            data_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
            if not data_file.exists() or data_file.stat().st_size < 1000:
                # Final fallback to old dataset
                data_file = DATAPATH / 'games_with_real_vegas.csv'
        
        # Force GAME_DATE_EST to be read as string first (prevents auto-parsing issues)
        df = pd.read_csv(data_file, low_memory=False, dtype={'GAME_DATE_EST': str})
        print(f"   [DEBUG] Loaded {len(df):,} games from CSV")
        print(f"   [DEBUG] Sample raw date strings: {df['GAME_DATE_EST'].tail(10).tolist()}")
        
        # Parse dates - handles both '2025-11-14' and '2025-11-11 00:00:00+00:00'
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
        
        # Show which rows have NaT BEFORE timezone conversion
        nat_count_before = df['GAME_DATE_EST'].isna().sum()
        if nat_count_before > 0:
            print(f"   [DEBUG] ⚠️  WARNING: {nat_count_before} NaT values after parsing!")
            # Re-read those specific rows to see raw strings
            df_raw = pd.read_csv(data_file, low_memory=False, dtype=str)
            nat_indices = df[df['GAME_DATE_EST'].isna()].index.tolist()
            print(f"   [DEBUG] NaT row indices: {nat_indices[:10]}")
            print(f"   [DEBUG] Raw date strings for NaT rows:")
            for idx in nat_indices[:5]:
                raw_date = df_raw.iloc[idx]['GAME_DATE_EST']
                print(f"      Row {idx}: '{raw_date}' (len={len(raw_date)}, repr={repr(raw_date)})")
        
        # Strip timezone if present (makes all dates timezone-naive for consistency)
        if pd.api.types.is_datetime64tz_dtype(df['GAME_DATE_EST']):
            df['GAME_DATE_EST'] = df['GAME_DATE_EST'].dt.tz_localize(None)
        
        print(f"   [DEBUG] After date parsing: {len(df):,} games ({df['GAME_DATE_EST'].isna().sum()} NaT values)")
        
        # Get current season
        current_season = datetime.now().year
        if datetime.now().month < 10:
            current_season -= 1
        
        # Filter for current season
        print(f"   [DEBUG] Filtering for SEASON == {current_season}")
        df = df[df['SEASON'] == current_season]
        print(f"   [DEBUG] After SEASON filter: {len(df):,} games")
        
        # Get unplayed games (PTS_home == 0)
        df_unplayed = df[df['PTS_home'] == 0].copy()
        print(f"   [DEBUG] After PTS_home == 0 filter: {len(df_unplayed)} games")
        
        # Note: New games from schedule fetch will have features = 0
        # The model can still make predictions, they just won't be as accurate
        # without historical rolling averages
        print(f"   [DEBUG] Total unplayed games found: {len(df_unplayed)}")
        
        # Filter for upcoming games only (today and next 7 days)
        # This avoids predicting on old games that should have been played already
        from datetime import timedelta
        today = datetime.now().date()
        max_date = today + timedelta(days=7)
        
        print(f"   [DEBUG] GAME_DATE_EST dtype before .dt.date: {df_unplayed['GAME_DATE_EST'].dtype}")
        print(f"   [DEBUG] Sample dates before conversion: {df_unplayed['GAME_DATE_EST'].head(10).tolist()}")
        print(f"   [DEBUG] Any NaT values? {df_unplayed['GAME_DATE_EST'].isna().sum()}")
        
        df_unplayed['GAME_DATE_EST'] = df_unplayed['GAME_DATE_EST'].dt.date
        
        print(f"   [DEBUG] Sample dates after .dt.date: {df_unplayed['GAME_DATE_EST'].head(10).tolist()}")
        
        # Remove rows with NaT dates (can't predict games with invalid dates)
        df_unplayed = df_unplayed.dropna(subset=['GAME_DATE_EST'])
        print(f"   [DEBUG] After dropping NaT dates: {len(df_unplayed)} games")
        
        # Filter out NaT values before getting unique dates
        valid_dates = df_unplayed['GAME_DATE_EST'].unique()
        print(f"   [DEBUG] Unique dates in unplayed games: {sorted(valid_dates)}")
        print(f"   [DEBUG] Date filter: {today} to {max_date}")
        
        df_unplayed = df_unplayed[
            (df_unplayed['GAME_DATE_EST'] >= today) & 
            (df_unplayed['GAME_DATE_EST'] <= max_date)
        ]
        
        print(f"   [DEBUG] Unplayed games after filter: {len(df_unplayed)}")
        
        # Sort by date to get next scheduled games
        df_unplayed = df_unplayed.sort_values('GAME_DATE_EST')
        
        if len(df_unplayed) > 0:
            next_game_date = df_unplayed['GAME_DATE_EST'].iloc[0]
            print(f"📅 Found {len(df_unplayed)} upcoming games")
            print(f"   Next game date: {next_game_date}")
            print(f"   Date range: {today} to {today + pd.Timedelta(days=7)}")
        else:
            print("📅 No unplayed games found")
        
        return df_unplayed
    
    except Exception as e:
        print(f"❌ Error loading games: {e}")
        import traceback
        traceback.print_exc()
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
    # This MUST match what was used in training (setup_model.py logic)
    target_cols = ['HOME_TEAM_WINS', 'TARGET']
    metadata_cols = ['GAME_DATE_EST', 'GAME_ID', 'MATCHUP', 'GAME_STATUS_TEXT', 'SEASON']
    categorical_cols = ['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
                       'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                       'TEAM_ID_home', 'TEAM_ID_away',
                       'data_source', 'whos_favored', 'is_real_vegas_line']
    
    # Leaky features - EXACT post-game stats only (matches training logic)
    leaky_cols = ['FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
                  'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
                  'REB_home', 'REB_away', 'PTS_home', 'PTS_away']
    
    drop_cols = target_cols + metadata_cols + categorical_cols + leaky_cols
    
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
        
        # Ensure matchup is a string (not NaN or float)
        if not isinstance(matchup, str):
            matched_odds.append(None)
            continue
        
        # Try to find matching game in live odds
        matched = None
        for _, odds in live_odds_df.iterrows():
            # Build odds matchup string
            away = odds.get('away_team', '')
            home = odds.get('home_team', '')
            
            # Skip if either team is missing
            if not away or not home:
                continue
            
            odds_matchup = f"{away} @ {home}"
            
            # Fuzzy match: check if either string contains the other
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
    
    # Pull latest data in case workflow didn't (GitHub Actions cache workaround)
    try:
        import subprocess
        print("🔄 Pulling latest data from git...")
        
        # First, show current commit
        current_commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                       capture_output=True, text=True, cwd=PROJECTPATH)
        print(f"   Current commit: {current_commit.stdout.strip()}")
        
        # Do the pull
        result = subprocess.run(['git', 'pull', 'origin', 'main'], 
                              capture_output=True, text=True, cwd=PROJECTPATH)
        print(f"   Git pull stdout: {result.stdout.strip()}")
        print(f"   Git pull stderr: {result.stderr.strip()}")
        
        # Show new commit
        new_commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                   capture_output=True, text=True, cwd=PROJECTPATH)
        print(f"   After pull commit: {new_commit.stdout.strip()}")
        
        if result.returncode == 0:
            print("   ✅ Git pull completed")
        else:
            print(f"   ⚠️  Git pull failed")
    except Exception as e:
        print(f"   ⚠️  Could not pull latest data: {e}")
        print("   Continuing with existing data...")
    print()
    
    # Load model
    model = load_model()
    
    # Load today's games
    df_today = load_todays_games()
    
    if df_today is None or len(df_today) == 0:
        print("❌ No games found for today")
        print()
        print("[DEBUG] Last 10 rows of data file:")
        data_path = DATAPATH / 'games_with_real_vegas_workflow.csv'
        df_check = pd.read_csv(data_path)
        print(df_check[['GAME_DATE_EST', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 'PTS_home']].tail(10))
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

