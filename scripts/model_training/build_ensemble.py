#!/usr/bin/env python3
"""
Ensemble Model Builder for NBA Prediction

Combines top models (RandomForest + XGBoost + HistGradientBoosting) into an ensemble.
Uses both voting and stacking approaches to maximize AUC.

Run: python build_ensemble.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


def main():
    print("=" * 70)
    print("NBA Prediction Ensemble Builder")
    print("=" * 70)
    print()
    
    # Paths
    DATAPATH = Path('data')
    MODELS_PATH = Path('models')
    RESULTS_PATH = Path('results')
    
    # Load data
    print("📁 Loading data...")
    train_df = pd.read_csv(DATAPATH / 'train_selected.csv')
    test_df = pd.read_csv(DATAPATH / 'test_selected.csv')
    
    # Prepare features (remove data leakage)
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID']
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
    ]
    drop_cols.extend([col for col in leaky_features if col in train_df.columns])
    drop_cols = list(set(drop_cols))
    
    X_train = train_df.drop(columns=drop_cols).fillna(0)
    y_train = train_df['TARGET']
    X_test = test_df.drop(columns=drop_cols).fillna(0)
    y_test = test_df['TARGET']
    
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Individual model performance
    print("\n" + "=" * 70)
    print("📊 INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 70)
    
    individual_results = {}
    
    # Try to load existing models
    print("\n1. RandomForest (from comparison)")
    try:
        rf_model = joblib.load(MODELS_PATH / 'best_model_randomforest.pkl')
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = rf_model.predict(X_test)
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        rf_acc = accuracy_score(y_test, rf_pred)
        individual_results['RandomForest'] = {'auc': rf_auc, 'acc': rf_acc}
        print(f"   ✅ Loaded: AUC={rf_auc:.4f}, Acc={rf_acc:.4f}")
    except:
        print("   ⚠️  Not found, will retrain")
        rf_model = None
    
    print("\n2. XGBoost (tuned)")
    try:
        xgb_model = joblib.load(MODELS_PATH / 'xgboost_tuned_calibrated.pkl')
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict(X_test)
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        individual_results['XGBoost'] = {'auc': xgb_auc, 'acc': xgb_acc}
        print(f"   ✅ Loaded: AUC={xgb_auc:.4f}, Acc={xgb_acc:.4f}")
    except:
        print("   ⚠️  Not found, will retrain")
        xgb_model = None
    
    # Train base models if not available
    print("\n" + "=" * 70)
    print("🔨 PREPARING BASE MODELS")
    print("=" * 70)
    
    if rf_model is None:
        print("\nTraining RandomForest...")
        rf_base = RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=10,
            min_samples_split=10, n_jobs=-1, verbose=0,
        )
        rf_model = CalibratedClassifierCV(rf_base, method='isotonic', cv=3)
        rf_model.fit(X_train, y_train)
        print("   ✅ Trained")
    
    if xgb_model is None:
        print("\nTraining XGBoost...")
        xgb_base = xgb.XGBClassifier(
            random_state=42, n_estimators=500, learning_rate=0.05,
            max_depth=5, subsample=0.8, tree_method='hist', verbosity=0,
        )
        xgb_model = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
        xgb_model.fit(X_train, y_train)
        print("   ✅ Trained")
    
    # Add HistGradientBoosting
    print("\nTraining HistGradientBoosting...")
    hist_model = HistGradientBoostingClassifier(
        random_state=42, max_iter=500, learning_rate=0.05,
        max_depth=5, verbose=0,
    )
    hist_calibrated = CalibratedClassifierCV(hist_model, method='isotonic', cv=3)
    hist_calibrated.fit(X_train, y_train)
    hist_pred_proba = hist_calibrated.predict_proba(X_test)[:, 1]
    hist_pred = hist_calibrated.predict(X_test)
    hist_auc = roc_auc_score(y_test, hist_pred_proba)
    hist_acc = accuracy_score(y_test, hist_pred)
    individual_results['HistGradientBoosting'] = {'auc': hist_auc, 'acc': hist_acc}
    print(f"   ✅ Trained: AUC={hist_auc:.4f}, Acc={hist_acc:.4f}")
    
    # Build Ensemble 1: Weighted Voting
    print("\n" + "=" * 70)
    print("🔥 BUILDING ENSEMBLE 1: Weighted Voting")
    print("=" * 70)
    
    # Calculate weights based on individual AUC
    weights = []
    for model_name in ['RandomForest', 'XGBoost', 'HistGradientBoosting']:
        if model_name in individual_results:
            weights.append(individual_results[model_name]['auc'])
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    print(f"\nWeights (based on AUC):")
    print(f"  RandomForest: {weights[0]:.3f}")
    print(f"  XGBoost: {weights[1]:.3f}")
    print(f"  HistGradientBoosting: {weights[2]:.3f}")
    
    # Weighted average of predictions
    print("\nCombining predictions...")
    ensemble_pred_proba = (
        weights[0] * rf_model.predict_proba(X_test)[:, 1] +
        weights[1] * xgb_model.predict_proba(X_test)[:, 1] +
        weights[2] * hist_pred_proba
    )
    ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
    
    ensemble1_auc = roc_auc_score(y_test, ensemble_pred_proba)
    ensemble1_acc = accuracy_score(y_test, ensemble_pred)
    ensemble1_brier = brier_score_loss(y_test, ensemble_pred_proba)
    
    print(f"\n✅ Weighted Voting Ensemble:")
    print(f"   Test AUC: {ensemble1_auc:.4f}")
    print(f"   Test Accuracy: {ensemble1_acc:.4f}")
    print(f"   Brier Score: {ensemble1_brier:.4f}")
    
    # Build Ensemble 2: Stacking
    print("\n" + "=" * 70)
    print("🔥 BUILDING ENSEMBLE 2: Stacking (Meta-Learner)")
    print("=" * 70)
    
    # Get base models for stacking (need to retrain without calibration)
    print("\nTraining base models for stacking...")
    
    rf_base = RandomForestClassifier(
        random_state=42, n_estimators=300, max_depth=10,
        min_samples_split=10, n_jobs=-1, verbose=0,
    )
    
    xgb_base = xgb.XGBClassifier(
        random_state=42, n_estimators=500, learning_rate=0.05,
        max_depth=5, subsample=0.8, tree_method='hist', verbosity=0,
    )
    
    hist_base = HistGradientBoostingClassifier(
        random_state=42, max_iter=500, learning_rate=0.05,
        max_depth=5, verbose=0,
    )
    
    # Create stacking ensemble
    estimators = [
        ('rf', rf_base),
        ('xgb', xgb_base),
        ('hist', hist_base),
    ]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    
    print("Training stacking ensemble (this may take a minute)...")
    stacking_model.fit(X_train, y_train)
    
    # Evaluate stacking
    stack_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
    stack_pred = stacking_model.predict(X_test)
    
    stack_auc = roc_auc_score(y_test, stack_pred_proba)
    stack_acc = accuracy_score(y_test, stack_pred)
    stack_brier = brier_score_loss(y_test, stack_pred_proba)
    
    print(f"\n✅ Stacking Ensemble:")
    print(f"   Test AUC: {stack_auc:.4f}")
    print(f"   Test Accuracy: {stack_acc:.4f}")
    print(f"   Brier Score: {stack_brier:.4f}")
    
    # Cross-validate stacking
    print("\nCross-validating stacking ensemble...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        stacking_model, X_train, y_train, 
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    print(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Final Comparison
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON")
    print("=" * 70)
    print()
    
    results = [
        ('Random Baseline', 0.5000, 0.5000),
        ('Home Wins Baseline', 0.5000, 0.5800),
    ]
    
    for name, metrics in individual_results.items():
        results.append((name, metrics['auc'], metrics['acc']))
    
    results.append(('Weighted Voting Ensemble', ensemble1_auc, ensemble1_acc))
    results.append(('Stacking Ensemble', stack_auc, stack_acc))
    
    # Sort by AUC
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<30} {'Test AUC':<12} {'Test Acc':<12}")
    print("=" * 70)
    for name, auc, acc in results:
        marker = "🏆" if auc == max(r[1] for r in results) else "  "
        print(f"{marker} {name:<28} {auc:.4f}       {acc:.4f}")
    
    # Determine best model
    best_ensemble = 'weighted' if ensemble1_auc > stack_auc else 'stacking'
    best_auc = max(ensemble1_auc, stack_auc)
    best_model = stacking_model if best_ensemble == 'stacking' else None
    
    print("\n" + "=" * 70)
    print("🏆 BEST MODEL")
    print("=" * 70)
    
    if best_ensemble == 'stacking':
        print(f"\n✅ Stacking Ensemble wins!")
        print(f"   Test AUC: {best_auc:.4f}")
        print(f"   Improvement over best individual: +{(best_auc - max(individual_results.values(), key=lambda x: x['auc'])['auc']) * 100:.2f}%")
    else:
        print(f"\n✅ Weighted Voting Ensemble wins!")
        print(f"   Test AUC: {best_auc:.4f}")
        print(f"   Improvement over best individual: +{(best_auc - max(individual_results.values(), key=lambda x: x['auc'])['auc']) * 100:.2f}%")
    
    # Save best ensemble
    print("\n" + "=" * 70)
    print("💾 SAVING MODELS")
    print("=" * 70)
    
    if best_ensemble == 'stacking':
        model_path = MODELS_PATH / 'ensemble_stacking.pkl'
        joblib.dump(stacking_model, model_path)
        print(f"✅ Stacking ensemble saved: {model_path}")
    
    # Save weighted ensemble components and weights
    weighted_path = MODELS_PATH / 'ensemble_weighted.pkl'
    weighted_data = {
        'models': [rf_model, xgb_model, hist_calibrated],
        'weights': weights,
        'model_names': ['RandomForest', 'XGBoost', 'HistGradientBoosting'],
    }
    joblib.dump(weighted_data, weighted_path)
    print(f"✅ Weighted ensemble saved: {weighted_path}")
    
    # Save results summary
    results_summary = {
        'individual_models': individual_results,
        'weighted_voting': {
            'auc': float(ensemble1_auc),
            'accuracy': float(ensemble1_acc),
            'brier_score': float(ensemble1_brier),
            'weights': weights.tolist(),
        },
        'stacking': {
            'auc': float(stack_auc),
            'accuracy': float(stack_acc),
            'brier_score': float(stack_brier),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
        },
        'best_model': best_ensemble,
        'best_auc': float(best_auc),
    }
    
    import json
    results_path = RESULTS_PATH / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✅ Results summary saved: {results_path}")
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if best_auc > 0.64:
        print("✅ EXCELLENT! Ensemble beats your previous best (0.64 AUC)")
        print("   → Deploy this ensemble to production")
        print("   → Update Streamlit app to use ensemble model")
    elif best_auc > 0.628:
        print("✅ GOOD! Ensemble beats RandomForest")
        print("   → Consider using this ensemble")
        print("   → Or wait for 100-trial XGBoost tuning results")
    else:
        print("⚠️  Ensemble slightly underperforms")
        print("   → Wait for 100-trial XGBoost tuning")
        print("   → May need more feature engineering")
    
    print("\n📋 Next Steps:")
    print("   1. Run 100-trial XGBoost tuning: python tune_xgboost.py")
    print("   2. If that's better, re-run ensemble with tuned XGBoost")
    print("   3. Deploy best model to Streamlit app")
    print(f"\n🏆 Current Best: {best_ensemble.title()} Ensemble with {best_auc:.4f} AUC")
    print()


if __name__ == "__main__":
    main()

