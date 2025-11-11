#!/usr/bin/env python3
"""
Train Legacy XGBoost Model WITH Vegas Betting Lines

Compare performance with and without betting features to measure impact.
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
MODELSPATH.mkdir(exist_ok=True)


def train_legacy_with_betting():
    """
    Train the legacy XGBoost model with betting line features
    """
    print("="*70)
    print("🎰 LEGACY MODEL TRAINING WITH VEGAS BETTING LINES")
    print("="*70)
    
    # Step 1: Load games with betting features
    print("\n📊 Step 1: Loading data with betting features...")
    
    games_with_betting = pd.read_csv(DATAPATH / 'games_with_betting.csv')
    
    print(f"   Games loaded: {len(games_with_betting):,}")
    print(f"   Date range: {games_with_betting['GAME_DATE_EST'].min()} to {games_with_betting['GAME_DATE_EST'].max()}")
    
    # Step 2: Prepare features
    print("\n🔧 Step 2: Preparing features...")
    
    # Drop target and metadata
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID']
    
    # Drop categorical features
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'HOME_TEAM', 'VISITOR_TEAM', 'GAME_DATE_EST_betting',
        'spread_category', 'total_category'
    ]
    drop_cols.extend([col for col in categorical_cols if col in games_with_betting.columns])
    
    # Drop leaky features
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
    ]
    drop_cols.extend([col for col in leaky_features if col in games_with_betting.columns])
    drop_cols = list(set(drop_cols))
    
    # Check if TARGET exists
    if 'TARGET' not in games_with_betting.columns:
        if 'HOME_TEAM_WINS' in games_with_betting.columns:
            games_with_betting['TARGET'] = games_with_betting['HOME_TEAM_WINS'].astype(int)
            print("   ✅ Created TARGET from HOME_TEAM_WINS")
    
    X = games_with_betting.drop(columns=drop_cols, errors='ignore').fillna(0)
    y = games_with_betting['TARGET']
    
    # Identify betting features
    betting_features = [col for col in X.columns if 
                       'betting' in col.lower() or 
                       col in ['spread', 'total', 'home_ml', 'visitor_ml', 
                              'home_win_prob_implied', 'betting_edge_exists',
                              'expected_home_pts', 'expected_visitor_pts',
                              'spread_category_encoded', 'total_category_encoded']]
    
    # Identify injury features
    injury_features = [col for col in X.columns if 'injur' in col.lower()]
    
    print(f"   Total features: {X.shape[1]}")
    print(f"   Betting features: {len(betting_features)}")
    print(f"   Injury features: {len(injury_features)}")
    print(f"   Samples: {len(X):,}")
    
    print(f"\n   Key betting features:")
    for feat in betting_features[:10]:
        print(f"      - {feat}")
    
    # Step 3: Train/test split
    print("\n✂️  Step 3: Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Step 4: Train WITHOUT betting features (baseline)
    print("\n" + "="*70)
    print("🔹 Step 4: Training WITHOUT betting features (baseline)...")
    print("="*70)
    
    # Remove betting features
    X_train_no_bet = X_train.drop(columns=betting_features, errors='ignore')
    X_test_no_bet = X_test.drop(columns=betting_features, errors='ignore')
    
    print(f"   Features without betting: {X_train_no_bet.shape[1]}")
    
    # XGBoost model
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
    xgb_baseline.fit(X_train_no_bet, y_train)
    
    # Calibrate
    print("   Calibrating...")
    calibrated_baseline = CalibratedClassifierCV(xgb_baseline, method='sigmoid', cv=3)
    calibrated_baseline.fit(X_train_no_bet, y_train)
    
    # Evaluate
    pred_proba_baseline = calibrated_baseline.predict_proba(X_test_no_bet)[:, 1]
    pred_baseline = (pred_proba_baseline > 0.5).astype(int)
    
    auc_baseline = roc_auc_score(y_test, pred_proba_baseline)
    acc_baseline = accuracy_score(y_test, pred_baseline)
    brier_baseline = brier_score_loss(y_test, pred_proba_baseline)
    precision_baseline = precision_score(y_test, pred_baseline)
    recall_baseline = recall_score(y_test, pred_baseline)
    
    print(f"\n   📊 Results (WITHOUT betting):")
    print(f"      AUC:         {auc_baseline:.4f}")
    print(f"      Accuracy:    {acc_baseline:.4f}")
    print(f"      Precision:   {precision_baseline:.4f}")
    print(f"      Recall:      {recall_baseline:.4f}")
    print(f"      Brier Score: {brier_baseline:.4f}")
    
    # Step 5: Train WITH betting features
    print("\n" + "="*70)
    print("🎰 Step 5: Training WITH betting features...")
    print("="*70)
    
    # XGBoost model
    xgb_with_bet = xgb.XGBClassifier(
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
    xgb_with_bet.fit(X_train, y_train)
    
    # Calibrate
    print("   Calibrating...")
    calibrated_with_bet = CalibratedClassifierCV(xgb_with_bet, method='sigmoid', cv=3)
    calibrated_with_bet.fit(X_train, y_train)
    
    # Evaluate
    pred_proba_with_bet = calibrated_with_bet.predict_proba(X_test)[:, 1]
    pred_with_bet = (pred_proba_with_bet > 0.5).astype(int)
    
    auc_with_bet = roc_auc_score(y_test, pred_proba_with_bet)
    acc_with_bet = accuracy_score(y_test, pred_with_bet)
    brier_with_bet = brier_score_loss(y_test, pred_proba_with_bet)
    precision_with_bet = precision_score(y_test, pred_with_bet)
    recall_with_bet = recall_score(y_test, pred_with_bet)
    
    print(f"\n   📊 Results (WITH betting):")
    print(f"      AUC:         {auc_with_bet:.4f}")
    print(f"      Accuracy:    {acc_with_bet:.4f}")
    print(f"      Precision:   {precision_with_bet:.4f}")
    print(f"      Recall:      {recall_with_bet:.4f}")
    print(f"      Brier Score: {brier_with_bet:.4f}")
    
    # Step 6: Compare
    print("\n" + "="*70)
    print("📈 Step 6: COMPARISON")
    print("="*70)
    
    auc_diff = auc_with_bet - auc_baseline
    acc_diff = acc_with_bet - acc_baseline
    brier_diff = brier_baseline - brier_with_bet  # Lower is better
    precision_diff = precision_with_bet - precision_baseline
    recall_diff = recall_with_bet - recall_baseline
    
    print(f"\n   Metric         | Without  | With     | Improvement")
    print(f"   --------------|----------|----------|-------------")
    print(f"   AUC           | {auc_baseline:.4f}   | {auc_with_bet:.4f}   | {auc_diff:+.4f} ({auc_diff/auc_baseline*100:+.2f}%)")
    print(f"   Accuracy      | {acc_baseline:.4f}   | {acc_with_bet:.4f}   | {acc_diff:+.4f} ({acc_diff/acc_baseline*100:+.2f}%)")
    print(f"   Precision     | {precision_baseline:.4f}   | {precision_with_bet:.4f}   | {precision_diff:+.4f} ({precision_diff/precision_baseline*100:+.2f}%)")
    print(f"   Recall        | {recall_baseline:.4f}   | {recall_with_bet:.4f}   | {recall_diff:+.4f} ({recall_diff/recall_baseline*100:+.2f}%)")
    print(f"   Brier Score   | {brier_baseline:.4f}   | {brier_with_bet:.4f}   | {brier_diff:+.4f} (better)")
    
    # Step 7: Feature importance
    print("\n📊 Step 7: Top Betting Feature Importance...")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_with_bet.feature_importances_
    }).sort_values('importance', ascending=False)
    
    betting_importance = feature_importance[feature_importance['feature'].isin(betting_features)]
    
    print(f"\n   Top 10 betting features by importance:")
    for idx, row in betting_importance.head(10).iterrows():
        print(f"      {row['feature']:45s} {row['importance']:.4f}")
    
    total_importance = feature_importance['importance'].sum()
    betting_importance_sum = betting_importance['importance'].sum()
    betting_pct = (betting_importance_sum / total_importance) * 100
    
    print(f"\n   💡 Betting features account for {betting_pct:.2f}% of total importance")
    
    # Show overall top features
    print(f"\n   🏆 Top 15 features overall:")
    for idx, row in feature_importance.head(15).iterrows():
        feat_type = "🎰 BETTING" if row['feature'] in betting_features else "📊 Other"
        print(f"      {feat_type} {row['feature']:40s} {row['importance']:.4f}")
    
    # Step 8: Confidence analysis
    print("\n📊 Step 8: Confidence Calibration Analysis...")
    
    # Bin predictions by confidence level
    confidence_bins = pd.cut(pred_proba_with_bet, bins=[0, 0.4, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
    
    print("\n   Prediction confidence distribution:")
    for conf_level in ['Low', 'Medium', 'High']:
        mask = confidence_bins == conf_level
        if mask.sum() > 0:
            actual_acc = y_test[mask].mean() if conf_level in ['Medium', 'High'] else (1 - y_test[mask]).mean()
            count = mask.sum()
            print(f"      {conf_level:8s}: {count:5d} games ({count/len(y_test)*100:5.1f}%) - Actual accuracy: {actual_acc:.1%}")
    
    # Step 9: Verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    if auc_diff > 0.03:
        verdict = "✅ SIGNIFICANT IMPROVEMENT - Betting features provide substantial value!"
        emoji = "🚀"
    elif auc_diff > 0.01:
        verdict = "✅ SOLID IMPROVEMENT - Betting features definitely help"
        emoji = "✅"
    elif auc_diff > 0.005:
        verdict = "⚠️  MODEST IMPROVEMENT - Betting features provide some value"
        emoji = "➕"
    elif auc_diff > 0:
        verdict = "→  MINIMAL IMPROVEMENT - Betting features have slight positive effect"
        emoji = "→"
    else:
        verdict = "❌ NO IMPROVEMENT - Betting features don't help"
        emoji = "❌"
    
    print(f"\n{emoji} {verdict}")
    
    # Recommendation
    print(f"\n📝 Recommendation:")
    if auc_diff > 0.01:
        print("   🎯 USE the model WITH betting features")
        print("   💰 Betting lines provide valuable predictive signal")
        print("   📈 This is your best performing model so far")
    else:
        print("   Focus on other feature improvements")
    
    # Step 10: Save model
    print("\n" + "="*70)
    print("💾 Step 10: Saving Models")
    print("="*70)
    
    # Save the model WITH betting
    model_file = MODELSPATH / 'legacy_xgboost_with_betting.pkl'
    joblib.dump(calibrated_with_bet, model_file)
    print(f"\n   ✅ Model WITH betting saved to: {model_file}")
    
    # Save feature names
    feature_file = MODELSPATH / 'betting_model_features.txt'
    with open(feature_file, 'w') as f:
        for feat in X_train.columns:
            f.write(f"{feat}\n")
    print(f"   ✅ Feature names saved to: {feature_file}")
    
    # Save metadata
    metadata = {
        'model_name': 'XGBoost with Betting Lines',
        'calibration_method': 'Sigmoid (CV=3)',
        'test_auc': float(auc_with_bet),
        'test_accuracy': float(acc_with_bet),
        'precision': float(precision_with_bet),
        'recall': float(recall_with_bet),
        'brier_score': float(brier_with_bet),
        'improvement_vs_baseline': {
            'auc': float(auc_diff),
            'accuracy': float(acc_diff),
            'precision': float(precision_diff),
            'recall': float(recall_diff)
        },
        'betting_features': betting_features,
        'betting_feature_importance_pct': float(betting_pct),
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = MODELSPATH / 'betting_model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved to: {metadata_file}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nLegacy Model Performance:")
    print(f"  WITHOUT betting: {auc_baseline:.2%} AUC, {acc_baseline:.2%} Accuracy")
    print(f"  WITH betting:    {auc_with_bet:.2%} AUC, {acc_with_bet:.2%} Accuracy")
    print(f"  Improvement:     {auc_diff:+.2%} AUC, {acc_diff:+.2%} Accuracy")
    
    if auc_diff > 0.01:
        print(f"\n🎉 Betting features provided a {auc_diff:.2%} boost!")
        print("   This is your new best model.")
    
    return {
        'baseline': {'auc': auc_baseline, 'accuracy': acc_baseline},
        'with_betting': {'auc': auc_with_bet, 'accuracy': acc_with_bet},
        'improvement': {'auc': auc_diff, 'accuracy': acc_diff}
    }


if __name__ == "__main__":
    results = train_legacy_with_betting()

