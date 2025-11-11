#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning with Optuna

Optimizes XGBoost for AUC using the corrected dataset (no data leakage).
Uses your existing Optuna objective function.

Run: python tune_xgboost.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Import your existing objective function
from src.optuna_objectives import XGB_objective


def main():
    print("=" * 70)
    print("XGBoost Hyperparameter Tuning with Optuna")
    print("=" * 70)
    print()
    
    # Configuration
    DATAPATH = Path('data')
    RESULTS_PATH = Path('results')
    MODELS_PATH = Path('models')
    RESULTS_PATH.mkdir(exist_ok=True)
    MODELS_PATH.mkdir(exist_ok=True)
    
    N_TRIALS = 100  # Number of Optuna trials (100 = thorough search, ~25 min)
    N_FOLDS = 5
    SEED = 42
    
    print(f"Configuration:")
    print(f"  Trials: {N_TRIALS}")
    print(f"  CV Folds: {N_FOLDS}")
    print(f"  Metric: AUC-ROC")
    print()
    
    # Load data
    print("📁 Loading data...")
    train_df = pd.read_csv(DATAPATH / 'train_selected.csv')
    test_df = pd.read_csv(DATAPATH / 'test_selected.csv')
    
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    
    # Remove data leakage features
    print("\n🔧 Preparing features...")
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
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Class balance: {y_train.mean():.3f}")
    
    # Static XGBoost parameters
    STATIC_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': SEED,
        'verbosity': 0,
    }
    
    # Create Optuna study
    print("\n🎯 Starting Optuna hyperparameter search...")
    print(f"   This will take approximately {N_TRIALS * 2 / 60:.0f}-{N_TRIALS * 5 / 60:.0f} minutes")
    print()
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_nba_tuning',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    
    # Run optimization
    study.optimize(
        lambda trial: XGB_objective(
            trial=trial,
            train=X_train,
            target=y_train,
            STATIC_PARAMS=STATIC_PARAMS,
            ENABLE_CATEGORICAL=False,
            NUM_BOOST_ROUND=1000,
            OPTUNA_CV='StratifiedKFold',
            OPTUNA_FOLDS=N_FOLDS,
            SEED=SEED,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    
    # Results
    print("\n" + "=" * 70)
    print("🏆 TUNING RESULTS")
    print("=" * 70)
    print()
    
    best_trial = study.best_trial
    print(f"Best Trial: #{best_trial.number}")
    print(f"Best CV AUC: {best_trial.value:.4f}")
    print()
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    print("\n" + "=" * 70)
    print("🔨 Training Final Model with Best Parameters")
    print("=" * 70)
    
    best_params = best_trial.params.copy()
    best_params.update(STATIC_PARAMS)
    
    # Extract num_round separately
    num_round = best_params.pop('num_round', 500)
    
    # Train on full training set
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("\nTraining on full training set...")
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_round,
    )
    
    # Evaluate on test set
    print("\n📊 Test Set Evaluation:")
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Calibrate probabilities
    print("\n🔧 Calibrating probabilities...")
    
    # Convert to sklearn-compatible model for calibration
    xgb_clf = xgb.XGBClassifier(**best_params, n_estimators=num_round)
    xgb_clf.fit(X_train, y_train)
    
    calibrated_model = CalibratedClassifierCV(
        xgb_clf,
        method='isotonic',
        cv=3,
    )
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate calibrated model
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_cal = calibrated_model.predict(X_test)
    
    test_auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    test_acc_cal = accuracy_score(y_test, y_pred_cal)
    
    print(f"  Calibrated Test AUC: {test_auc_cal:.4f}")
    print(f"  Calibrated Test Accuracy: {test_acc_cal:.4f}")
    
    # Compare to baselines
    print("\n" + "=" * 70)
    print("📈 COMPARISON TO BASELINES")
    print("=" * 70)
    print()
    print(f"Random Baseline:        0.5000 AUC, 50.0% accuracy")
    print(f"Home Wins Baseline:     0.5000 AUC, 58.0% accuracy")
    print(f"Default XGBoost:        0.6070 AUC, 58.9% accuracy")
    print(f"Current Best (RF):      0.6282 AUC, 62.5% accuracy")
    print(f"Your Previous XGBoost:  0.6400 AUC, 61.5% accuracy")
    print(f"─" * 70)
    print(f"Tuned XGBoost:          {test_auc:.4f} AUC, {test_acc * 100:.1f}% accuracy")
    print(f"Tuned + Calibrated:     {test_auc_cal:.4f} AUC, {test_acc_cal * 100:.1f}% accuracy")
    
    if test_auc_cal > 0.6282:
        print(f"\n✅ SUCCESS! Beat RandomForest by {(test_auc_cal - 0.6282) * 100:.2f}%")
    else:
        print(f"\n⚠️  Slightly below RandomForest by {(0.6282 - test_auc_cal) * 100:.2f}%")
    
    if test_auc_cal > 0.64:
        print(f"✅ SUCCESS! Beat your previous XGBoost!")
    
    # Save results
    print("\n" + "=" * 70)
    print("💾 SAVING RESULTS")
    print("=" * 70)
    
    # Save best parameters
    params_path = RESULTS_PATH / 'xgboost_best_params.json'
    with open(params_path, 'w') as f:
        save_params = best_trial.params.copy()
        save_params['test_auc'] = float(test_auc)
        save_params['test_accuracy'] = float(test_acc)
        save_params['test_auc_calibrated'] = float(test_auc_cal)
        save_params['test_accuracy_calibrated'] = float(test_acc_cal)
        json.dump(save_params, f, indent=2)
    print(f"✅ Parameters saved: {params_path}")
    
    # Save calibrated model
    model_path = MODELS_PATH / 'xgboost_tuned_calibrated.pkl'
    joblib.dump(calibrated_model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save Optuna study for later analysis
    study_path = RESULTS_PATH / 'optuna_study.pkl'
    joblib.dump(study, study_path)
    print(f"✅ Optuna study saved: {study_path}")
    
    # Save trial history
    trials_df = study.trials_dataframe()
    trials_path = RESULTS_PATH / 'optuna_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"✅ Trial history saved: {trials_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TUNING COMPLETE!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Review the best hyperparameters above")
    print("  2. Check results/optuna_trials.csv for all trial results")
    print("  3. Use models/xgboost_tuned_calibrated.pkl in production")
    print("  4. Consider ensemble with RandomForest for even better results")
    print()
    print(f"Best Model Performance: {test_auc_cal:.4f} AUC")


if __name__ == "__main__":
    main()

