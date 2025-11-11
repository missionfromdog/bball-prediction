#!/usr/bin/env python3
"""
Evaluate Impact of Injury Features on Model Performance

Compares model performance with and without injury features to determine
how valuable injury data is for NBA game prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.model_comparison import ModelComparator

# Paths
DATAPATH = Path('data')
RESULTS_PATH = Path('results')


def prepare_data_with_injuries():
    """
    Load and prepare data with injury features
    """
    print("📊 Loading data with injury features...")
    
    # Load the data with injuries
    games_df = pd.read_csv(DATAPATH / 'games_with_injuries.csv')
    
    # Load the train/test split indices
    train_df = pd.read_csv(DATAPATH / 'train_selected.csv')
    test_df = pd.read_csv(DATAPATH / 'test_selected.csv')
    
    # Merge injury features with train/test splits
    # Use GAME_ID to match
    train_with_injuries = train_df.merge(
        games_df[['GAME_ID'] + [c for c in games_df.columns if 'injur' in c.lower()]],
        on='GAME_ID',
        how='left'
    )
    
    test_with_injuries = test_df.merge(
        games_df[['GAME_ID'] + [c for c in games_df.columns if 'injur' in c.lower()]],
        on='GAME_ID',
        how='left'
    )
    
    # Fill NaN injury features with 0
    injury_cols = [c for c in train_with_injuries.columns if 'injur' in c.lower()]
    train_with_injuries[injury_cols] = train_with_injuries[injury_cols].fillna(0)
    test_with_injuries[injury_cols] = test_with_injuries[injury_cols].fillna(0)
    
    print(f"   Train: {train_with_injuries.shape}")
    print(f"   Test: {test_with_injuries.shape}")
    print(f"   Injury features added: {len(injury_cols)}")
    
    return train_with_injuries, test_with_injuries, injury_cols


def compare_performance():
    """
    Compare model performance with and without injury features
    """
    print("\n" + "="*70)
    print("🏀 INJURY FEATURE IMPACT EVALUATION")
    print("="*70)
    
    # Load both versions of data
    print("\n1️⃣  Loading data WITHOUT injury features...")
    train_no_injury = pd.read_csv(DATAPATH / 'train_selected.csv')
    test_no_injury = pd.read_csv(DATAPATH / 'test_selected.csv')
    
    print("\n2️⃣  Loading data WITH injury features...")
    train_with_injury, test_with_injury, injury_cols = prepare_data_with_injuries()
    
    # Prepare features
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID']
    
    # Leaky features to remove
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
    ]
    drop_cols.extend([col for col in leaky_features if col in train_no_injury.columns])
    drop_cols = list(set(drop_cols))
    
    # Prepare WITHOUT injury
    X_train_no = train_no_injury.drop(columns=drop_cols, errors='ignore').fillna(0)
    y_train = train_no_injury['TARGET']
    X_test_no = test_no_injury.drop(columns=drop_cols, errors='ignore').fillna(0)
    y_test = test_no_injury['TARGET']
    
    # Prepare WITH injury
    X_train_with = train_with_injury.drop(columns=drop_cols, errors='ignore').fillna(0)
    X_test_with = test_with_injury.drop(columns=drop_cols, errors='ignore').fillna(0)
    
    print(f"\n📐 Feature count:")
    print(f"   WITHOUT injuries: {X_train_no.shape[1]} features")
    print(f"   WITH injuries:    {X_train_with.shape[1]} features")
    print(f"   Difference:       +{X_train_with.shape[1] - X_train_no.shape[1]} injury features")
    
    # Test with 3 fast models
    models_to_test = ['RandomForest', 'HistGradientBoosting', 'LogisticRegression']
    
    results = []
    
    for model_name in models_to_test:
        print("\n" + "="*70)
        print(f"Testing: {model_name}")
        print("="*70)
        
        # Test WITHOUT injuries
        print(f"\n   🔹 WITHOUT injury features...")
        comparator_no = ModelComparator(random_state=42, n_folds=3, verbose=0)
        
        # Get model config
        model_configs = comparator_no.get_model_configs()
        model = model_configs[model_name]
        
        results_no = comparator_no.evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train_no,
            y_train=y_train,
            X_test=X_test_no,
            y_test=y_test,
            calibrate=True
        )
        
        # Test WITH injuries
        print(f"   🔹 WITH injury features...")
        comparator_with = ModelComparator(random_state=42, n_folds=3, verbose=0)
        
        model = model_configs[model_name]
        
        results_with = comparator_with.evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train_with,
            y_train=y_train,
            X_test=X_test_with,
            y_test=y_test,
            calibrate=True
        )
        
        # Compare
        auc_diff = results_with['test_auc'] - results_no['test_auc']
        acc_diff = results_with['test_accuracy'] - results_no['test_accuracy']
        
        results.append({
            'model': model_name,
            'auc_no_injury': results_no['test_auc'],
            'auc_with_injury': results_with['test_auc'],
            'auc_improvement': auc_diff,
            'acc_no_injury': results_no['test_accuracy'],
            'acc_with_injury': results_with['test_accuracy'],
            'acc_improvement': acc_diff,
        })
        
        print(f"\n   📊 Results:")
        print(f"      AUC without:  {results_no['test_auc']:.4f}")
        print(f"      AUC with:     {results_with['test_auc']:.4f}")
        print(f"      Improvement:  {auc_diff:+.4f} ({auc_diff/results_no['test_auc']*100:+.2f}%)")
        print(f"\n      Acc without:  {results_no['test_accuracy']:.4f}")
        print(f"      Acc with:     {results_with['test_accuracy']:.4f}")
        print(f"      Improvement:  {acc_diff:+.4f} ({acc_diff/results_no['test_accuracy']*100:+.2f}%)")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("📊 SUMMARY: INJURY FEATURE IMPACT")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Average improvement
    avg_auc_improvement = results_df['auc_improvement'].mean()
    avg_acc_improvement = results_df['acc_improvement'].mean()
    
    print(f"\n📈 Average Improvements:")
    print(f"   AUC: {avg_auc_improvement:+.4f} ({avg_auc_improvement/results_df['auc_no_injury'].mean()*100:+.2f}%)")
    print(f"   Accuracy: {avg_acc_improvement:+.4f} ({avg_acc_improvement/results_df['acc_no_injury'].mean()*100:+.2f}%)")
    
    # Verdict
    print(f"\n" + "="*70)
    print("🎯 VERDICT")
    print("="*70)
    
    if avg_auc_improvement > 0.01:
        verdict = "✅ SIGNIFICANT IMPROVEMENT - Injury features are valuable!"
        recommendation = "Include injury features in production model"
    elif avg_auc_improvement > 0.005:
        verdict = "⚠️  MODEST IMPROVEMENT - Some value but not game-changing"
        recommendation = "Consider including if injury data is reliable"
    else:
        verdict = "❌ MINIMAL IMPROVEMENT - Injury features don't help much"
        recommendation = "Current synthetic injury data may not be realistic enough"
        
    print(f"\n{verdict}")
    print(f"\nRecommendation: {recommendation}")
    
    if avg_auc_improvement <= 0.005:
        print("\n💡 Why might injury features show minimal impact?")
        print("   1. Using SYNTHETIC injury data (not real injuries)")
        print("   2. Real injury data would show player-specific impacts")
        print("   3. Need star player identification + injury severity")
        print("   4. Would benefit from:")
        print("      - Actual injury reports from Basketball-Reference")
        print("      - Player importance scores (All-Stars, usage rate)")
        print("      - Cumulative injury load per team")
        print("      - Recent injury trends")
    
    # Save results
    results_file = RESULTS_PATH / 'injury_impact_comparison.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    return results_df


def main():
    RESULTS_PATH.mkdir(exist_ok=True)
    results = compare_performance()
    return results


if __name__ == "__main__":
    main()

