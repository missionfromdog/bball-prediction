#!/usr/bin/env python3
"""
NBA Model Comparison Runner

Quick script to compare 7+ models and generate results.
Run from project root: python run_model_comparison.py

Results will be saved to results/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our comparison module
from src.model_comparison import ModelComparator

def main():
    print("🏀 NBA Model Comparison")
    print("="*70 + "\n")
    
    # Paths
    DATAPATH = Path('data')
    RESULTS_PATH = Path('results')
    RESULTS_PATH.mkdir(exist_ok=True)
    
    # Load data
    print("📁 Loading data...")
    train_df = pd.read_csv(DATAPATH / 'train_selected.csv')
    test_df = pd.read_csv(DATAPATH / 'test_selected.csv')
    
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    
    # Drop target and metadata columns
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID']
    
    # Drop leaky features (stats from completed game, not available pre-game)
    leaky_features = [
        'HOME_TEAM_WINS',  # This is the target!
        'PTS_home', 'PTS_away',  # Points scored in this game
        'FG_PCT_home', 'FG_PCT_away',  # Shooting % from this game
        'FT_PCT_home', 'FT_PCT_away',  # Free throw % from this game
        'FG3_PCT_home', 'FG3_PCT_away',  # 3-point % from this game
        'AST_home', 'AST_away',  # Assists from this game
        'REB_home', 'REB_away',  # Rebounds from this game
        'PLAYOFF',  # Playoff indicator (if exists)
    ]
    
    drop_cols.extend([col for col in leaky_features if col in train_df.columns])
    drop_cols = list(set(drop_cols))  # Remove duplicates
    
    print(f"   Dropping {len(drop_cols)} columns (target + leaky features)")
    print(f"   Leaky features removed: {[col for col in leaky_features if col in train_df.columns]}")
    
    X_train = train_df.drop(columns=drop_cols).fillna(0)
    y_train = train_df['TARGET']
    
    X_test = test_df.drop(columns=drop_cols).fillna(0)
    y_test = test_df['TARGET']
    
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Initialize comparator
    print("\n🚀 Initializing model comparator...")
    comparator = ModelComparator(
        random_state=42,
        n_folds=5,
        use_gpu=True,  # M3 MacBook Pro has GPU
        verbose=1,
    )
    
    # Run comparison
    print("\n" + "="*70)
    print("Starting model comparison - this will take 5-15 minutes")
    print("="*70 + "\n")
    
    results_df = comparator.compare_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        calibrate=True,  # Calibrate for better probabilities
    )
    
    # Print summary
    comparator.print_summary(results_df, top_n=5)
    
    # Save results
    print("\n💾 Saving results...")
    results_path, summary_path = comparator.save_results(
        results_df=results_df,
        output_dir=RESULTS_PATH,
        prefix='nba_model_comparison',
    )
    
    # Save best model
    best_model_name = results_df.iloc[0]['model_name']
    best_auc = results_df.iloc[0]['test_auc']
    
    comparator.save_best_model(
        model_name=best_model_name,
        output_path=Path('models') / f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    )
    
    # Final summary
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test AUC: {best_auc:.4f}")
    print(f"Test Accuracy: {results_df.iloc[0]['test_accuracy']:.4f}")
    print(f"\nImprovement over baseline (0.50 AUC, 0.58 accuracy):")
    print(f"  AUC: +{best_auc - 0.50:.4f}")
    print(f"  Accuracy: +{results_df.iloc[0]['test_accuracy'] - 0.58:.4f}")
    
    print(f"\n📊 Full results saved to:")
    print(f"   {results_path}")
    print(f"   {summary_path}")
    
    print("\n✅ Done! Check the results/ directory for detailed output.")


if __name__ == "__main__":
    main()

