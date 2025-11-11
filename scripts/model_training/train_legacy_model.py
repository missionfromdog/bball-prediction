#!/usr/bin/env python3
"""
Train Legacy XGBoost Model WITH Historical Injury Features

This script:
1. Loads games data with injury features
2. Trains the legacy XGBoost model (with calibration)
3. Compares performance with/without injuries
4. Saves the improved model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

DATAPATH = Path('data')
MODELSPATH = Path('models')
MODELSPATH.mkdir(exist_ok=True)


def train_legacy_with_injuries():
    """
    Train the legacy XGBoost model with injury features
    """
    print("="*70)
    print("🏀 LEGACY MODEL TRAINING WITH INJURY DATA")
    print("="*70)
    
    # Step 1: Load games with injuries
    print("\n📊 Step 1: Loading data with injury features...")
    
    games_with_injuries = pd.read_csv(DATAPATH / 'games_with_injuries.csv')
    
    print(f"   Games loaded: {len(games_with_injuries):,}")
    print(f"   Date range: {games_with_injuries['GAME_DATE_EST'].min()} to {games_with_injuries['GAME_DATE_EST'].max()}")
    
    # Step 2: Prepare features
    print("\n🔧 Step 2: Preparing features...")
    
    # Drop target and metadata
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID']
    
    # Drop categorical features
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON'
    ]
    drop_cols.extend([col for col in categorical_cols if col in games_with_injuries.columns])
    
    # Drop leaky features
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
    ]
    drop_cols.extend([col for col in leaky_features if col in games_with_injuries.columns])
    drop_cols = list(set(drop_cols))
    
    # Check if TARGET exists, if not create from HOME_TEAM_WINS
    if 'TARGET' not in games_with_injuries.columns:
        if 'HOME_TEAM_WINS' in games_with_injuries.columns:
            games_with_injuries['TARGET'] = games_with_injuries['HOME_TEAM_WINS'].astype(int)
            print("   ✅ Created TARGET from HOME_TEAM_WINS")
    
    X = games_with_injuries.drop(columns=drop_cols, errors='ignore').fillna(0)
    y = games_with_injuries['TARGET']
    
    # Identify injury features
    injury_features = [col for col in X.columns if 'injur' in col.lower()]
    
    print(f"   Total features: {X.shape[1]}")
    print(f"   Injury features: {len(injury_features)}")
    print(f"   Samples: {len(X):,}")
    print(f"\n   Top injury features:")
    for feat in injury_features[:10]:
        print(f"      - {feat}")
    
    # Step 3: Train/test split
    print("\n✂️  Step 3: Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Step 4: Train WITHOUT injuries (baseline)
    print("\n" + "="*70)
    print("🔹 Step 4: Training WITHOUT injury features (baseline)...")
    print("="*70)
    
    # Remove injury features
    X_train_no_inj = X_train.drop(columns=injury_features)
    X_test_no_inj = X_test.drop(columns=injury_features)
    
    # XGBoost model (legacy parameters)
    xgb_baseline = xgb.XGBClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        tree_method='hist'
    )
    
    print("   Training...")
    xgb_baseline.fit(X_train_no_inj, y_train)
    
    # Calibrate
    print("   Calibrating...")
    calibrated_baseline = CalibratedClassifierCV(xgb_baseline, method='sigmoid', cv=3)
    calibrated_baseline.fit(X_train_no_inj, y_train)
    
    # Evaluate
    pred_proba_baseline = calibrated_baseline.predict_proba(X_test_no_inj)[:, 1]
    pred_baseline = (pred_proba_baseline > 0.5).astype(int)
    
    auc_baseline = roc_auc_score(y_test, pred_proba_baseline)
    acc_baseline = accuracy_score(y_test, pred_baseline)
    brier_baseline = brier_score_loss(y_test, pred_proba_baseline)
    
    print(f"\n   📊 Results (WITHOUT injuries):")
    print(f"      AUC:         {auc_baseline:.4f}")
    print(f"      Accuracy:    {acc_baseline:.4f}")
    print(f"      Brier Score: {brier_baseline:.4f}")
    
    # Step 5: Train WITH injuries
    print("\n" + "="*70)
    print("🔥 Step 5: Training WITH injury features...")
    print("="*70)
    
    # XGBoost model (same parameters)
    xgb_with_inj = xgb.XGBClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        tree_method='hist'
    )
    
    print("   Training...")
    xgb_with_inj.fit(X_train, y_train)
    
    # Calibrate
    print("   Calibrating...")
    calibrated_with_inj = CalibratedClassifierCV(xgb_with_inj, method='sigmoid', cv=3)
    calibrated_with_inj.fit(X_train, y_train)
    
    # Evaluate
    pred_proba_with_inj = calibrated_with_inj.predict_proba(X_test)[:, 1]
    pred_with_inj = (pred_proba_with_inj > 0.5).astype(int)
    
    auc_with_inj = roc_auc_score(y_test, pred_proba_with_inj)
    acc_with_inj = accuracy_score(y_test, pred_with_inj)
    brier_with_inj = brier_score_loss(y_test, pred_proba_with_inj)
    
    print(f"\n   📊 Results (WITH injuries):")
    print(f"      AUC:         {auc_with_inj:.4f}")
    print(f"      Accuracy:    {acc_with_inj:.4f}")
    print(f"      Brier Score: {brier_with_inj:.4f}")
    
    # Step 6: Compare
    print("\n" + "="*70)
    print("📈 Step 6: COMPARISON")
    print("="*70)
    
    auc_diff = auc_with_inj - auc_baseline
    acc_diff = acc_with_inj - acc_baseline
    brier_diff = brier_baseline - brier_with_inj  # Lower is better
    
    print(f"\n   Metric         | Without  | With     | Improvement")
    print(f"   --------------|----------|----------|-------------")
    print(f"   AUC           | {auc_baseline:.4f}   | {auc_with_inj:.4f}   | {auc_diff:+.4f} ({auc_diff/auc_baseline*100:+.2f}%)")
    print(f"   Accuracy      | {acc_baseline:.4f}   | {acc_with_inj:.4f}   | {acc_diff:+.4f} ({acc_diff/acc_baseline*100:+.2f}%)")
    print(f"   Brier Score   | {brier_baseline:.4f}   | {brier_with_inj:.4f}   | {brier_diff:+.4f} (lower is better)")
    
    # Step 7: Feature importance
    print("\n📊 Step 7: Top Injury Feature Importance...")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_with_inj.feature_importances_
    }).sort_values('importance', ascending=False)
    
    injury_importance = feature_importance[feature_importance['feature'].isin(injury_features)]
    
    print(f"\n   Top 10 injury features by importance:")
    for idx, row in injury_importance.head(10).iterrows():
        print(f"      {row['feature']:45s} {row['importance']:.4f}")
    
    total_importance = feature_importance['importance'].sum()
    injury_importance_sum = injury_importance['importance'].sum()
    injury_pct = (injury_importance_sum / total_importance) * 100
    
    print(f"\n   💡 Injury features account for {injury_pct:.2f}% of total importance")
    
    # Step 8: Verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    if auc_diff > 0.01:
        verdict = "✅ SIGNIFICANT IMPROVEMENT - Injury features help substantially!"
        recommendation = "Use the model WITH injury features"
    elif auc_diff > 0.005:
        verdict = "⚠️  MODEST IMPROVEMENT - Injury features provide some value"
        recommendation = "Consider using injury features, especially for live predictions"
    elif auc_diff > 0:
        verdict = "→  MINIMAL IMPROVEMENT - Injury features have slight positive effect"
        recommendation = "Injury features optional, focus on other improvements"
    else:
        verdict = "❌ NO IMPROVEMENT - Injury features don't help (or hurt)"
        recommendation = "Don't use injury features with this data"
    
    print(f"\n{verdict}")
    print(f"\nRecommendation: {recommendation}")
    
    # Step 9: Save model
    print("\n" + "="*70)
    print("💾 Step 9: Saving Models")
    print("="*70)
    
    # Save the model WITH injuries
    model_file = MODELSPATH / 'legacy_xgboost_with_injuries.pkl'
    joblib.dump(calibrated_with_inj, model_file)
    print(f"\n   ✅ Model WITH injuries saved to: {model_file}")
    
    # Save metadata
    metadata = {
        'model_name': 'XGBoost with Injuries',
        'calibration_method': 'Sigmoid (CV=3)',
        'test_auc': auc_with_inj,
        'test_accuracy': acc_with_inj,
        'brier_score': brier_with_inj,
        'improvement_vs_baseline': {
            'auc': auc_diff,
            'accuracy': acc_diff
        },
        'injury_features': injury_features,
        'injury_feature_importance_pct': injury_pct,
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    metadata_file = MODELSPATH / 'legacy_with_injuries_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"   ✅ Metadata saved to: {metadata_file}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nLegacy Model Performance:")
    print(f"  WITHOUT injuries: {auc_baseline:.2%} AUC, {acc_baseline:.2%} Accuracy")
    print(f"  WITH injuries:    {auc_with_inj:.2%} AUC, {acc_with_inj:.2%} Accuracy")
    print(f"  Improvement:      {auc_diff:+.2%} AUC, {acc_diff:+.2%} Accuracy")
    
    return {
        'baseline': {'auc': auc_baseline, 'accuracy': acc_baseline},
        'with_injuries': {'auc': auc_with_inj, 'accuracy': acc_with_inj},
        'improvement': {'auc': auc_diff, 'accuracy': acc_diff}
    }


if __name__ == "__main__":
    results = train_legacy_with_injuries()

