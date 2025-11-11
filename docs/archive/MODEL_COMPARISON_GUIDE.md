# Model Comparison Guide 🏀

## Quick Start

Run the model comparison in 3 simple steps:

```bash
# 1. Activate your virtual environment
source venv/bin/activate

# 2. Run the comparison script
python run_model_comparison.py

# 3. Check results
ls -lh results/
```

That's it! The script will automatically:
- Load your train/test data (train_selected.csv with 245 features)
- Test 7 different models with 5-fold cross-validation
- Evaluate on AUC (optimized for betting odds comparison)
- Save results to CSV files
- Save the best model

## What Models Are Tested?

### 1. **Gradient Boosting Models** (Usually Best)
- **XGBoost** - Your current model, highly tuned
- **LightGBM** - Fast alternative, often comparable
- **HistGradientBoosting** - Sklearn's fast implementation

### 2. **Tree Ensembles** (Good Baselines)
- **Random Forest** - Robust, interpretable
- **Extra Trees** - Faster, more randomization

### 3. **Linear Models** (Fast, Simple)
- **Logistic Regression** - Classic baseline
- **Ridge Classifier** - Regularized linear model

## What Metrics Are Tracked?

### Primary (for Betting)
- **AUC-ROC**: Quality of probability predictions (most important for odds)
- **Brier Score**: How accurate are the probabilities?
- **Log Loss**: Penalty for confident wrong predictions

### Secondary (for Understanding)
- **Accuracy**: Simple win/loss prediction correctness
- **F1 Score**: Balance of precision and recall
- **Precision/Recall**: Class-specific performance

### Performance
- **Training Time**: How long to train?
- **Prediction Time**: How fast for daily predictions?

## Expected Runtime

On your M3 MacBook Pro:
- **Quick models** (Linear): < 1 minute
- **Tree models** (RF, ET): 2-5 minutes
- **Gradient boosting** (XGB, LGB): 5-10 minutes
- **Total**: ~10-15 minutes for all 7 models

## Output Files

All results saved to `results/` directory:

```
results/
├── nba_model_comparison_YYYYMMDD_HHMMSS.csv       # Full results
├── nba_model_comparison_summary_YYYYMMDD_HHMMSS.csv  # Summary table
└── best_model_*.pkl                                # Best performing model
```

### Results CSV Columns

```python
# Performance Metrics
'test_auc'               # Test set AUC (PRIMARY)
'test_accuracy'          # Test set accuracy
'cv_auc_mean'            # Cross-validation AUC mean
'cv_auc_std'             # Cross-validation AUC std dev
'test_f1'                # F1 score
'test_precision'         # Precision
'test_recall'            # Recall
'test_brier_score'       # Brier score (lower is better)
'test_log_loss'          # Log loss (lower is better)

# Timing
'train_time_seconds'     # Full training time
'prediction_time_per_sample_ms'  # Prediction speed

# Other
'calibrated'             # Was probability calibration applied?
'cv_time_seconds'        # Cross-validation time
```

## Example Output

```
==================================================
TOP 5 MODELS BY TEST AUC
==================================================

#1. XGBoost
   Test AUC:      0.6892
   Test Accuracy: 0.6254
   CV AUC:        0.6831 ± 0.0123
   Train Time:    127.45s
   Predict Time:  0.15ms/sample

#2. LightGBM
   Test AUC:      0.6874
   Test Accuracy: 0.6198
   CV AUC:        0.6809 ± 0.0145
   Train Time:    89.23s
   Predict Time:  0.12ms/sample

...
```

## Interpreting Results

### For Betting Strategy (Optimize for AUC)
- **Best choice**: Highest test_auc
- **Why**: AUC measures how well probabilities rank games (crucial for comparing against odds)
- **Look for**: AUC > 0.65 is good, > 0.70 is excellent

### For Simple Predictions (Optimize for Accuracy)
- **Best choice**: Highest test_accuracy  
- **Why**: Just picking winners, not probability quality
- **Look for**: Accuracy > 0.60 is good (baseline is 0.58)

### For Production (Balance All)
- **Consider**: AUC + Accuracy + Speed
- **Why**: Need good predictions that run fast daily
- **Look for**: Top 3 AUC with prediction_time < 1ms/sample

## What If I Want to Add More Models?

Easy! Edit `src/model_comparison.py` and add to `get_model_configs()`:

```python
# Example: Add CatBoost
'CatBoost': CatBoostClassifier(
    random_state=self.random_state,
    iterations=500,
    learning_rate=0.05,
    depth=5,
    verbose=False,
),
```

Then run: `pip install catboost` and rerun the script.

## Common Issues

### GPU Not Working?
Edit `run_model_comparison.py`, line 47:
```python
use_gpu=False,  # Changed from True
```

### Out of Memory?
Reduce `n_folds` in `run_model_comparison.py`, line 46:
```python
n_folds=3,  # Changed from 5
```

### Taking Too Long?
Test fewer models. Edit `run_model_comparison.py`, add after line 53:
```python
models_to_test=['XGBoost', 'LightGBM', 'RandomForest'],  # Only test these
```

## Advanced: Jupyter Notebook

For interactive exploration, use the notebook instead:

```bash
jupyter notebook notebooks/08_model_comparison.ipynb
```

The notebook includes:
- Step-by-step execution
- Data exploration
- Visualizations (AUC comparison, timing charts)
- Feature importance comparison

## Next Steps

After getting results:

1. **Review** the results CSV to see all models
2. **Compare** to your current baseline (~0.64 AUC)
3. **Choose** best model for your use case (betting = AUC, simple = accuracy)
4. **Tune** the best model with Optuna (notebook 07)
5. **Deploy** by updating `streamlit_app.py` to use new model

## Questions?

Check the model outputs:
- Low AUC (<0.60): Feature engineering needed
- High variance in CV: Model overfitting or data issues
- Slow prediction: Consider simpler model for production
- All models similar: Features are good, ensemble might help

---

**Ready to find the best model?**

```bash
source venv/bin/activate
python run_model_comparison.py
```

Good luck! 🍀🏀

