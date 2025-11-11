#!/usr/bin/env python3
"""
Retrain ALL models with Real Vegas + Injury data
Then build new ensemble

This should push us towards 70% AUC!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import xgboost as xgb
from catboost import CatBoostClassifier

DATAPATH = Path('data')
MODELSPATH = Path('models')
RESULTSPATH = Path('results')
RESULTSPATH.mkdir(exist_ok=True)


def prepare_data():
    """Load and prepare data with real Vegas features"""
    print("="*70)
    print("📊 PREPARING DATA")
    print("="*70)
    
    df = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
    
    print(f"\n✅ Loaded {len(df):,} games")
    if 'is_real_vegas_line' in df.columns:
        real_count = df['is_real_vegas_line'].sum()
        print(f"   🎰 Real Vegas: {real_count:,} ({real_count/len(df)*100:.1f}%)")
    
    # Prepare features
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', 'merge_key']
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'whos_favored', 'data_source', 'is_real_vegas_line'
    ]
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
        'score_home', 'score_away', 'q1_home', 'q2_home'
    ]
    
    all_drop = list(set(drop_cols + categorical_cols + leaky_features))
    
    if 'TARGET' not in df.columns and 'HOME_TEAM_WINS' in df.columns:
        df['TARGET'] = df['HOME_TEAM_WINS'].astype(int)
    
    X = df.drop(columns=all_drop, errors='ignore').fillna(0)
    y = df['TARGET']
    X = X.loc[:, ~X.columns.duplicated()]
    
    print(f"\n   Features: {X.shape[1]}")
    print(f"   Samples: {len(X):,}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train RandomForest with real Vegas data"""
    print("\n" + "="*70)
    print("🌲 TRAINING RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("   Training...")
    rf.fit(X_train, y_train)
    
    print("   Calibrating...")
    rf_calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
    rf_calibrated.fit(X_train, y_train)
    
    # Evaluate
    pred_proba = rf_calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, (pred_proba > 0.5).astype(int))
    
    print(f"\n   ✅ RandomForest Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Acc: {acc:.4f}")
    
    # Save
    joblib.dump(rf_calibrated, MODELSPATH / 'best_model_randomforest_vegas.pkl')
    
    return rf_calibrated, {'auc': auc, 'accuracy': acc}


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with real Vegas data"""
    print("\n" + "="*70)
    print("⚡ TRAINING XGBOOST")
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
    xgb_calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
    xgb_calibrated.fit(X_train, y_train)
    
    # Evaluate
    pred_proba = xgb_calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, (pred_proba > 0.5).astype(int))
    
    print(f"\n   ✅ XGBoost Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Acc: {acc:.4f}")
    
    # Save
    joblib.dump(xgb_calibrated, MODELSPATH / 'xgboost_vegas_calibrated.pkl')
    
    return xgb_calibrated, {'auc': auc, 'accuracy': acc}


def train_histgradient(X_train, y_train, X_test, y_test):
    """Train HistGradientBoosting with real Vegas data"""
    print("\n" + "="*70)
    print("📊 TRAINING HISTGRADIENTBOOSTING")
    print("="*70)
    
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    print("   Training...")
    hgb.fit(X_train, y_train)
    
    print("   Calibrating...")
    hgb_calibrated = CalibratedClassifierCV(hgb, method='sigmoid', cv=3)
    hgb_calibrated.fit(X_train, y_train)
    
    # Evaluate
    pred_proba = hgb_calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, (pred_proba > 0.5).astype(int))
    
    print(f"\n   ✅ HistGradientBoosting Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Acc: {acc:.4f}")
    
    # Save
    joblib.dump(hgb_calibrated, MODELSPATH / 'histgradient_vegas_calibrated.pkl')
    
    return hgb_calibrated, {'auc': auc, 'accuracy': acc}


def build_ensemble(models_dict, X_train, y_train, X_test, y_test):
    """Build Stacking and Weighted ensembles"""
    print("\n" + "="*70)
    print("🏆 BUILDING ENSEMBLES")
    print("="*70)
    
    results = {}
    
    # Weighted Voting Ensemble
    print("\n⚖️  Weighted Voting...")
    weights = [models_dict[m]['auc'] for m in ['rf', 'xgb', 'hgb']]
    
    weighted = VotingClassifier(
        estimators=[
            ('rf', models_dict['rf']['model']),
            ('xgb', models_dict['xgb']['model']),
            ('hgb', models_dict['hgb']['model'])
        ],
        voting='soft',
        weights=weights
    )
    
    weighted.fit(X_train, y_train)
    pred_proba = weighted.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, (pred_proba > 0.5).astype(int))
    
    print(f"   AUC: {auc:.4f} | Acc: {acc:.4f}")
    joblib.dump(weighted, MODELSPATH / 'ensemble_weighted_vegas.pkl')
    
    results['weighted'] = {'auc': auc, 'accuracy': acc}
    
    # Stacking Ensemble
    print("\n🏆 Stacking...")
    
    # Use base models (before calibration)
    rf_base = models_dict['rf']['model'].calibrated_classifiers_[0].estimator
    xgb_base = models_dict['xgb']['model'].calibrated_classifiers_[0].estimator
    hgb_base = models_dict['hgb']['model'].calibrated_classifiers_[0].estimator
    
    stacking = StackingClassifier(
        estimators=[
            ('rf', rf_base),
            ('xgb', xgb_base),
            ('hgb', hgb_base)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    
    # Calibrate the stacking ensemble
    stacking_calibrated = CalibratedClassifierCV(stacking, method='sigmoid', cv=3)
    stacking_calibrated.fit(X_train, y_train)
    
    pred_proba = stacking_calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, (pred_proba > 0.5).astype(int))
    
    print(f"   AUC: {auc:.4f} | Acc: {acc:.4f}")
    joblib.dump(stacking_calibrated, MODELSPATH / 'ensemble_stacking_vegas.pkl')
    
    results['stacking'] = {'auc': auc, 'accuracy': acc}
    
    return results


def main():
    print("="*70)
    print("🚀 RETRAIN ALL MODELS WITH REAL VEGAS DATA")
    print("="*70)
    print()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    # Train individual models
    models = {}
    
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models['rf'] = {'model': rf_model, 'auc': rf_metrics['auc']}
    
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models['xgb'] = {'model': xgb_model, 'auc': xgb_metrics['auc']}
    
    hgb_model, hgb_metrics = train_histgradient(X_train, y_train, X_test, y_test)
    models['hgb'] = {'model': hgb_model, 'auc': hgb_metrics['auc']}
    
    # Build ensembles
    ensemble_results = build_ensemble(models, X_train, y_train, X_test, y_test)
    
    # Final results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS WITH REAL VEGAS DATA")
    print("="*70)
    
    all_results = {
        'RandomForest': rf_metrics['auc'],
        'XGBoost': xgb_metrics['auc'],
        'HistGradientBoosting': hgb_metrics['auc'],
        'Weighted Ensemble': ensemble_results['weighted']['auc'],
        'Stacking Ensemble': ensemble_results['stacking']['auc']
    }
    
    print("\n   Model                    | AUC      | vs Old")
    print("   -------------------------|----------|--------")
    
    old_aucs = {
        'RandomForest': 0.6294,
        'XGBoost': 0.6204,
        'HistGradientBoosting': 0.6250,
        'Weighted Ensemble': 0.6314,
        'Stacking Ensemble': 0.6319
    }
    
    for model_name, auc in all_results.items():
        old_auc = old_aucs.get(model_name, 0.58)
        improvement = auc - old_auc
        print(f"   {model_name:24s} | {auc:.4f}   | {improvement:+.4f}")
    
    # Find best
    best_model = max(all_results.items(), key=lambda x: x[1])
    print(f"\n   🏆 Best: {best_model[0]} ({best_model[1]:.4f})")
    
    # Save results
    results_data = {
        'individual_models': {
            'RandomForest': rf_metrics,
            'XGBoost': xgb_metrics,
            'HistGradientBoosting': hgb_metrics
        },
        'ensembles': ensemble_results,
        'best_model': {
            'name': best_model[0],
            'auc': best_model[1]
        },
        'trained_on': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(RESULTSPATH / 'ensemble_results_vegas.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n   💾 Results saved to: {RESULTSPATH / 'ensemble_results_vegas.json'}")
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE!")
    print("="*70)
    print("\n🎯 Next: Update Streamlit app to use new models")
    
    return all_results


if __name__ == "__main__":
    results = main()

