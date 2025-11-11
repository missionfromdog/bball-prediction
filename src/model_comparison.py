"""
Model Comparison Module for NBA Prediction Project

This module provides a comprehensive framework for comparing multiple classification
models with consistent evaluation metrics, focusing on AUC optimization.

Key Features:
- 11 different model types (gradient boosting incl. CatBoost, tree ensembles, linear models)
- Stratified K-Fold cross-validation
- Probability calibration
- Comprehensive metrics (AUC, accuracy, Brier score, log loss)
- Performance timing
- CSV results export
- Visualization generation
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Evaluation
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.calibration import CalibratedClassifierCV

# Utilities
import joblib


class ModelComparator:
    """
    Compare multiple classification models with consistent evaluation.
    Optimized for AUC metric.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_folds: int = 5,
        use_gpu: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize the model comparator.
        
        Args:
            random_state: Random seed for reproducibility
            n_folds: Number of folds for cross-validation
            use_gpu: Whether to use GPU for models that support it
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.results = []
        self.fitted_models = {}
        
    def get_model_configs(self) -> Dict[str, Any]:
        """
        Get configurations for all models to test.
        
        Returns:
            Dictionary mapping model names to model instances
        """
        configs = {
            # Gradient Boosting Models
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                tree_method='hist',  # GPU not available in pip install
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='auc',
                verbosity=0,
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                random_state=self.random_state,
                device='cpu',  # GPU not available in pip install
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            ),
            
            'CatBoost': CatBoostClassifier(
                random_state=self.random_state,
                iterations=500,
                learning_rate=0.05,
                depth=5,
                subsample=0.8,
                verbose=0,
                allow_writing_files=False,  # Don't create training logs
            ),
            
            'HistGradientBoosting': HistGradientBoostingClassifier(
                random_state=self.random_state,
                max_iter=500,
                learning_rate=0.05,
                max_depth=5,
                verbose=0,
            ),
            
            # Tree-Based Ensembles
            'RandomForest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                verbose=0,
            ),
            
            'ExtraTrees': ExtraTreesClassifier(
                random_state=self.random_state,
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                verbose=0,
            ),
            
            # Linear Models
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                n_jobs=-1,
                verbose=0,
            ),
            
            'Ridge': RidgeClassifier(
                random_state=self.random_state,
                alpha=1.0,
            ),
        }
        
        return configs
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model with cross-validation and test set.
        
        Args:
            model: The model instance to evaluate
            model_name: Name of the model for reporting
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*70}")
        
        results = {'model_name': model_name}
        
        # Cross-validation on training set
        if self.verbose >= 1:
            print("Running cross-validation...")
        
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'auc': 'roc_auc',
            'accuracy': 'accuracy',
            'f1': 'f1',
        }
        
        # Time CV training
        cv_start = time.time()
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1 if model_name not in ['XGBoost', 'LightGBM'] else 1,
            return_train_score=False,
        )
        cv_time = time.time() - cv_start
        
        # CV Results
        results['cv_auc_mean'] = cv_scores['test_auc'].mean()
        results['cv_auc_std'] = cv_scores['test_auc'].std()
        results['cv_accuracy_mean'] = cv_scores['test_accuracy'].mean()
        results['cv_accuracy_std'] = cv_scores['test_accuracy'].std()
        results['cv_f1_mean'] = cv_scores['test_f1'].mean()
        results['cv_f1_std'] = cv_scores['test_f1'].std()
        results['cv_time_seconds'] = cv_time
        
        if self.verbose >= 1:
            print(f"  CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
            print(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        
        # Train on full training set
        if self.verbose >= 1:
            print("Training on full training set...")
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        results['train_time_seconds'] = train_time
        
        # Calibrate if requested
        if calibrate and hasattr(model, 'predict_proba'):
            if self.verbose >= 1:
                print("Calibrating probabilities...")
            
            calibrated_model = CalibratedClassifierCV(
                model, method='isotonic', cv=3
            )
            calibrated_model.fit(X_train, y_train)
            model_for_prediction = calibrated_model
            results['calibrated'] = True
        else:
            model_for_prediction = model
            results['calibrated'] = False
        
        # Test set evaluation
        if self.verbose >= 1:
            print("Evaluating on test set...")
        
        # Time prediction
        pred_start = time.time()
        y_pred = model_for_prediction.predict(X_test)
        
        if hasattr(model_for_prediction, 'predict_proba'):
            y_pred_proba = model_for_prediction.predict_proba(X_test)[:, 1]
        else:
            # For Ridge classifier which doesn't have predict_proba
            y_pred_proba = None
        
        pred_time = time.time() - pred_start
        results['prediction_time_seconds'] = pred_time
        results['prediction_time_per_sample_ms'] = (pred_time / len(X_test)) * 1000
        
        # Test metrics
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_precision'] = precision_score(y_test, y_pred)
        results['test_recall'] = recall_score(y_test, y_pred)
        results['test_f1'] = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            results['test_auc'] = roc_auc_score(y_test, y_pred_proba)
            results['test_brier_score'] = brier_score_loss(y_test, y_pred_proba)
            results['test_log_loss'] = log_loss(y_test, y_pred_proba)
        else:
            results['test_auc'] = np.nan
            results['test_brier_score'] = np.nan
            results['test_log_loss'] = np.nan
        
        if self.verbose >= 1:
            print(f"\n  Test Results:")
            print(f"  AUC: {results['test_auc']:.4f}")
            print(f"  Accuracy: {results['test_accuracy']:.4f}")
            print(f"  Precision: {results['test_precision']:.4f}")
            print(f"  Recall: {results['test_recall']:.4f}")
            print(f"  F1: {results['test_f1']:.4f}")
            if y_pred_proba is not None:
                print(f"  Brier Score: {results['test_brier_score']:.4f}")
                print(f"  Log Loss: {results['test_log_loss']:.4f}")
            print(f"\n  Timing:")
            print(f"  Train: {train_time:.2f}s")
            print(f"  Predict: {pred_time:.4f}s ({results['prediction_time_per_sample_ms']:.2f}ms/sample)")
        
        # Store the fitted model
        self.fitted_models[model_name] = model_for_prediction
        
        return results
    
    def compare_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        calibrate: bool = True,
        models_to_test: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            calibrate: Whether to calibrate probabilities
            models_to_test: List of model names to test (None = all)
            
        Returns:
            DataFrame with comparison results
        """
        model_configs = self.get_model_configs()
        
        if models_to_test is not None:
            model_configs = {k: v for k, v in model_configs.items() if k in models_to_test}
        
        print(f"\n{'#'*70}")
        print(f"# Model Comparison - Testing {len(model_configs)} Models")
        print(f"# Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")
        print(f"# Features: {X_train.shape[1]}")
        print(f"# CV Folds: {self.n_folds} | GPU: {self.use_gpu}")
        print(f"{'#'*70}\n")
        
        self.results = []
        
        for model_name, model in model_configs.items():
            try:
                result = self.evaluate_model(
                    model=model,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    calibrate=calibrate,
                )
                self.results.append(result)
                
            except Exception as e:
                print(f"\n❌ Error with {model_name}: {str(e)}")
                if self.verbose >= 2:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Convert to DataFrame and sort by test AUC
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('test_auc', ascending=False)
        
        return results_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        output_dir: Path,
        prefix: str = 'model_comparison',
    ):
        """
        Save comparison results to CSV.
        
        Args:
            results_df: DataFrame with comparison results
            output_dir: Directory to save results
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save full results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_path = output_dir / f'{prefix}_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        
        # Save summary
        summary_cols = [
            'model_name',
            'test_auc', 'cv_auc_mean', 'cv_auc_std',
            'test_accuracy', 'cv_accuracy_mean', 'cv_accuracy_std',
            'test_f1', 'test_precision', 'test_recall',
            'test_brier_score', 'test_log_loss',
            'train_time_seconds', 'prediction_time_per_sample_ms',
            'calibrated',
        ]
        summary_df = results_df[summary_cols]
        summary_path = output_dir / f'{prefix}_summary_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary saved to: {summary_path}")
        
        return results_path, summary_path
    
    def save_best_model(
        self,
        model_name: str,
        output_path: Path,
    ):
        """
        Save the best model to disk.
        
        Args:
            model_name: Name of the model to save
            output_path: Path to save the model
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not found in fitted models")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        joblib.dump(self.fitted_models[model_name], output_path)
        print(f"\n✅ Best model '{model_name}' saved to: {output_path}")
    
    def print_summary(self, results_df: pd.DataFrame, top_n: int = 5):
        """
        Print a formatted summary of results.
        
        Args:
            results_df: DataFrame with comparison results
            top_n: Number of top models to display
        """
        print(f"\n{'='*70}")
        print(f"TOP {top_n} MODELS BY TEST AUC")
        print(f"{'='*70}\n")
        
        for idx, row in results_df.head(top_n).iterrows():
            print(f"#{idx+1}. {row['model_name']}")
            print(f"   Test AUC:      {row['test_auc']:.4f}")
            print(f"   Test Accuracy: {row['test_accuracy']:.4f}")
            print(f"   CV AUC:        {row['cv_auc_mean']:.4f} ± {row['cv_auc_std']:.4f}")
            print(f"   Train Time:    {row['train_time_seconds']:.2f}s")
            print(f"   Predict Time:  {row['prediction_time_per_sample_ms']:.2f}ms/sample")
            print()

