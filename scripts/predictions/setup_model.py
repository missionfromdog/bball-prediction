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
DATA_PATH = ROOT / 'data' / 'games_with_real_vegas.csv'


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
    if not DATA_PATH.exists():
        print(f"❌ Training data not found: {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} games")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X = df.copy()
    
    # Drop target and metadata
    target_cols = ['HOME_TEAM_WINS']
    metadata_cols = ['GAME_DATE_EST', 'GAME_ID', 'MATCHUP']
    categorical_cols = ['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 
                       'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']
    
    # Drop leaky features (post-game stats)
    leaky_patterns = ['FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
                     'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
                     'REB_home', 'REB_away', 'PTS_home', 'PTS_away']
    
    drop_cols = target_cols + metadata_cols + categorical_cols
    for pattern in leaky_patterns:
        drop_cols.extend([col for col in X.columns if pattern in col and col not in drop_cols])
    
    X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')
    y = df['HOME_TEAM_WINS']
    
    print(f"   Features: {len(X.columns)}")
    print(f"   Samples: {len(X):,}")
    
    # Handle missing values
    if X.isnull().any().any():
        print("   Filling missing values...")
        X = X.fillna(X.mean())
    
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

