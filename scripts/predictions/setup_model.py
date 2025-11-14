"""
Setup model for predictions - handles LFS issues by retraining if needed
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'histgradient_vegas_calibrated.pkl'
# Use master dataset (30k games with full history)
DATA_PATH = ROOT / 'data' / 'games_master_engineered.csv'

# Import feature engineering
sys.path.insert(0, str(ROOT))
from src.feature_engineering import process_features


def is_valid_model(filepath):
    """Check if model file is valid (not LFS pointer)"""
    if not filepath.exists():
        return False
    
    size = filepath.stat().st_size
    
    # LFS pointer files are tiny (< 200 bytes)
    # Real models are 2-3 MB (2,000,000+ bytes)
    if size < 1000:
        print(f"⚠️  File is too small ({size} bytes) - probably LFS pointer")
        return False
    
    print(f"✅ File looks valid ({size:,} bytes)")
    return True


def retrain_model():
    """Retrain the model from scratch"""
    print("\n" + "="*80)
    print("🔧 RETRAINING MODEL")
    print("="*80)
    
    # Load data
    print("\n📊 Loading training data...")
    
    # Try master dataset, fallback to workflow dataset
    data_file = DATA_PATH
    if not data_file.exists():
        print(f"⚠️  Master dataset not found: {data_file}")
        # Try workflow dataset
        data_file = ROOT / 'data' / 'games_with_real_vegas_workflow.csv'
        if not data_file.exists():
            print(f"❌ No training data found (tried master and workflow datasets)")
            return False
        print(f"   Using workflow dataset: {data_file}")
    else:
        print(f"   Using master dataset: {data_file}")
    
    df = pd.read_csv(data_file, low_memory=False, dtype={'GAME_DATE_EST': str})
    print(f"   Loaded {len(df):,} games")
    
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: Engineer features before training
    # ═══════════════════════════════════════════════════════════════════
    # Check if features already exist
    rolling_cols = [col for col in df.columns if 'AVG_LAST' in col or 'WIN_STREAK' in col]
    if not rolling_cols or df[rolling_cols[0]].sum() == 0:
        print("\n🔨 Engineering features (this takes 2-5 minutes)...")
        print("   (Features must be engineered during training AND prediction)")
        
        # Parse dates first
        def standardize_date_string(date_str):
            if not isinstance(date_str, str):
                return date_str
            if '+' in date_str:
                date_str = date_str.split('+')[0].strip()
            if date_str.endswith('Z'):
                date_str = date_str[:-1].strip()
            if len(date_str) == 10 and date_str.count('-') == 2:
                date_str = date_str + ' 00:00:00'
            return date_str
        
        df['GAME_DATE_EST'] = df['GAME_DATE_EST'].apply(standardize_date_string)
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
        if pd.api.types.is_datetime64tz_dtype(df['GAME_DATE_EST']):
            df['GAME_DATE_EST'] = df['GAME_DATE_EST'].dt.tz_localize(None)
        
        # Run feature engineering
        df = process_features(df)
        print(f"   ✅ Features engineered: {len(df):,} rows, {len(df.columns)} columns")
        
        # Save engineered dataset for future use
        print(f"   💾 Saving engineered dataset...")
        df.to_csv(data_file, index=False)
        print(f"   ✅ Saved to {data_file.name}")
    else:
        print(f"   ✅ Features already engineered ({len(rolling_cols)} rolling features found)")
    # ═══════════════════════════════════════════════════════════════════
    
    # Prepare features
    print("\n🔧 Preparing features for training...")
    X = df.copy()
    
    # Drop target and metadata
    target_cols = ['HOME_TEAM_WINS', 'TARGET']  # Added TARGET
    metadata_cols = ['GAME_DATE_EST', 'GAME_ID', 'MATCHUP', 'GAME_STATUS_TEXT']
    categorical_cols = ['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 
                       'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
                       'TEAM_ID_home', 'TEAM_ID_away',
                       'data_source', 'whos_favored', 'is_real_vegas_line']
    
    # Drop leaky features (EXACT post-game stats only, not rolling averages)
    leaky_cols = ['FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
                  'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
                  'REB_home', 'REB_away', 'PTS_home', 'PTS_away']
    
    drop_cols = target_cols + metadata_cols + categorical_cols + leaky_cols
    
    X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')
    
    # Drop any remaining object/string columns
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"   Dropping {len(object_cols)} object columns: {object_cols[:10]}")
        X = X.drop(columns=object_cols)
    
    y = df['HOME_TEAM_WINS']
    
    print(f"   Features: {len(X.columns)}")
    print(f"   Samples: {len(X):,}")
    
    # Handle missing values (only for numeric columns)
    if X.isnull().any().any():
        print("   Filling missing values...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        # Drop rows with any remaining NaN (shouldn't be many)
        initial_count = len(X)
        X = X.dropna()
        y = y[X.index]
        if len(X) < initial_count:
            print(f"   Dropped {initial_count - len(X)} rows with remaining NaN values")
    
    # Train/test split
    print("\n✂️  Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,} games")
    print(f"   Test:  {len(X_test):,} games")
    
    # Train model
    print("\n🎯 Training HistGradientBoosting...")
    print("   (This takes ~2-3 minutes)")
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=200,
        max_depth=7,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_features=0.8,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete")
    
    # Calibrate
    print("\n🎨 Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    print("   ✅ Calibration complete")
    
    # Evaluate
    train_score = calibrated.score(X_train, y_train)
    test_score = calibrated.score(X_test, y_test)
    print(f"\n📊 Performance:")
    print(f"   Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"   Test accuracy:     {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Save
    print("\n💾 Saving model...")
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(calibrated, MODEL_PATH)
    
    file_size = MODEL_PATH.stat().st_size
    print(f"   ✅ Model saved to: {MODEL_PATH}")
    print(f"   ✅ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    
    return True


def main():
    print("="*80)
    print("📦 MODEL SETUP")
    print("="*80)
    
    # Check if model exists and is valid
    print(f"\n🔍 Checking model: {MODEL_PATH}")
    
    if is_valid_model(MODEL_PATH):
        print("\n✅ Model is ready to use!")
        return 0
    
    # Model is missing or invalid - retrain
    print("\n⚠️  Model is missing or invalid (LFS pointer)")
    print("🔄 Will retrain from scratch...")
    
    try:
        success = retrain_model()
        
        if success and is_valid_model(MODEL_PATH):
            print("\n" + "="*80)
            print("✅ SUCCESS: Model is ready!")
            print("="*80)
            return 0
        else:
            print("\n" + "="*80)
            print("❌ FAILED: Could not create valid model")
            print("="*80)
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

