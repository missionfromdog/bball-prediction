#!/usr/bin/env python3
"""
Train Legacy XGBoost Model WITH REAL Vegas Betting Lines

This is the big test - does REAL Vegas data improve our model significantly?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, precision_score, recall_score

DATAPATH = Path('data')
MODELSPATH = Path('models')


def train_with_real_vegas():
    """Train model with REAL Vegas betting lines"""
    
    print("="*70)
    print("🎰 TRAINING WITH REAL VEGAS BETTING LINES")
    print("="*70)
    
    # Load data with REAL Vegas lines
    print("\n📊 Step 1: Loading data with REAL Vegas lines...")
    
    games_df = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
    
    print(f"   Total games: {len(games_df):,}")
    
    # Check how many have real vs synthetic
    if 'is_real_vegas_line' in games_df.columns:
        real_count = games_df['is_real_vegas_line'].sum()
        print(f"   🎰 Real Vegas lines: {real_count:,} ({real_count/len(games_df)*100:.1f}%)")
        print(f"   📊 Synthetic fallback: {len(games_df) - real_count:,}")
    
    # Prepare features
    print("\n🔧 Step 2: Preparing features...")
    
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', 'merge_key']
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'whos_favored', 'data_source', 'is_real_vegas_line'
    ]
    drop_cols.extend([col for col in categorical_cols if col in games_df.columns])
    
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
        'score_home', 'score_away', 'q1_home', 'q2_home', 'q3_home', 'q4_home'
    ]
    drop_cols.extend([col for col in leaky_features if col in games_df.columns])
    drop_cols = list(set(drop_cols))
    
    # Target
    if 'TARGET' not in games_df.columns and 'HOME_TEAM_WINS' in games_df.columns:
        games_df['TARGET'] = games_df['HOME_TEAM_WINS'].astype(int)
    
    X = games_df.drop(columns=drop_cols, errors='ignore').fillna(0)
    y = games_df['TARGET']
    
    # Remove any duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    
    betting_features = [col for col in X.columns if 
                       'betting' in col.lower() or 
                       col in ['spread', 'total', 'home_ml', 'visitor_ml', 
                              'home_win_prob_implied', 'moneyline_home', 'moneyline_away',
                              'expected_home_pts', 'expected_visitor_pts']]
    
    print(f"   Total features: {X.shape[1]}")
    print(f"   Betting features: {len(betting_features)}")
    print(f"   Samples: {len(X):,}")
    
    # Train/test split
    print("\n✂️  Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")
    
    # Train model
    print("\n" + "="*70)
    print("🚀 Step 4: Training with REAL Vegas data...")
    print("="*70)
    
    xgb_model = xgb.XGBClassifier(
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
    xgb_model.fit(X_train, y_train)
    
    print("   Calibrating...")
    calibrated_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate
    pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, pred)
    brier = brier_score_loss(y_test, pred_proba)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    
    print(f"\n   📊 Results with REAL Vegas:")
    print(f"      AUC:         {auc:.4f}")
    print(f"      Accuracy:    {acc:.4f}")
    print(f"      Precision:   {precision:.4f}")
    print(f"      Recall:      {recall:.4f}")
    print(f"      Brier Score: {brier:.4f}")
    
    # Compare to previous models
    print("\n" + "="*70)
    print("📈 COMPARISON TO PREVIOUS MODELS")
    print("="*70)
    
    print(f"\n   Model Version                  | AUC      | Accuracy | Improvement")
    print(f"   -------------------------------|----------|----------|-------------")
    print(f"   Baseline (no betting)          | 67.44%   | 64.43%   | -")
    print(f"   + Synthetic betting            | 67.68%   | 64.64%   | +0.24%")
    print(f"   + REAL Vegas (current)         | {auc*100:.2f}%   | {acc*100:.2f}%   | {(auc-0.6768)*100:+.2f}%")
    
    improvement = (auc - 0.6768) * 100
    
    # Feature importance
    print("\n📊 Top Betting Feature Importance...")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    betting_importance = feature_importance[feature_importance['feature'].isin(betting_features)]
    
    print(f"\n   Top 10 betting features:")
    for idx, row in betting_importance.head(10).iterrows():
        print(f"      {row['feature']:45s} {row['importance']:.4f}")
    
    total_importance = feature_importance['importance'].sum()
    betting_pct = (betting_importance['importance'].sum() / total_importance) * 100
    
    print(f"\n   💡 Betting features: {betting_pct:.2f}% of total importance")
    print(f"      (vs 3.31% with synthetic)")
    
    # Overall top features
    print(f"\n   🏆 Top 15 features overall:")
    for idx, row in feature_importance.head(15).iterrows():
        feat_type = "🎰" if row['feature'] in betting_features else "📊"
        print(f"      {feat_type} {row['feature']:40s} {row['importance']:.4f}")
    
    # Verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    if improvement > 1.0:
        verdict = "✅ MAJOR BREAKTHROUGH! Real Vegas lines are game-changing!"
        emoji = "🚀"
    elif improvement > 0.5:
        verdict = "✅ SIGNIFICANT IMPROVEMENT! Real Vegas data definitely helps"
        emoji = "🎉"
    elif improvement > 0.2:
        verdict = "✅ SOLID IMPROVEMENT! Real Vegas provides clear value"
        emoji = "✅"
    elif improvement > 0:
        verdict = "⚠️  MODEST IMPROVEMENT - Some benefit from real data"
        emoji = "➕"
    else:
        verdict = "→ MINIMAL CHANGE - Synthetic was surprisingly good"
        emoji = "→"
    
    print(f"\n{emoji} {verdict}")
    print(f"\nImprovement: {improvement:+.2f}% AUC")
    print(f"New AUC: {auc:.2%}")
    
    # Save model
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    
    model_file = MODELSPATH / 'legacy_xgboost_with_real_vegas.pkl'
    joblib.dump(calibrated_model, model_file)
    print(f"\n   ✅ Model saved: {model_file}")
    
    # Save metadata
    metadata = {
        'model_name': 'XGBoost with REAL Vegas Lines',
        'real_vegas_coverage': f"{games_df.get('is_real_vegas_line', pd.Series([False])).sum():,} games",
        'test_auc': float(auc),
        'test_accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'brier_score': float(brier),
        'improvement_vs_synthetic': float(improvement),
        'betting_feature_importance_pct': float(betting_pct),
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = MODELSPATH / 'real_vegas_model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved: {metadata_file}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    if improvement > 0.5:
        print(f"\n🎉 SUCCESS! Real Vegas data improved AUC by {improvement:.2f}%")
        print(f"   This is your new BEST model!")
    elif improvement > 0:
        print(f"\n✅ Real Vegas data provided {improvement:.2f}% improvement")
    else:
        print(f"\n💡 Synthetic betting lines were already quite good")
    
    print(f"\nFinal Model: {auc:.2%} AUC, {acc:.2%} Accuracy")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = train_with_real_vegas()

