# XGBoost Hyperparameter Tuning Guide 🎯

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run tuning (100 trials, ~20-30 minutes)
python tune_xgboost.py
```

## What This Does

The script will:
1. ✅ Load corrected data (no data leakage!)
2. ✅ Run 100 Optuna trials to find best hyperparameters
3. ✅ Evaluate each trial with 5-fold cross-validation
4. ✅ Train final model with best parameters
5. ✅ Calibrate probabilities for better AUC
6. ✅ Compare against all baselines
7. ✅ Save best model and parameters

## Expected Runtime

- **Quick test (20 trials)**: ~5-10 minutes
- **Standard (100 trials)**: ~20-30 minutes  
- **Thorough (200 trials)**: ~40-60 minutes

## Customizing Number of Trials

Edit `tune_xgboost.py`, line 35:

```python
N_TRIALS = 100  # Change this number
```

**Recommendations:**
- **20 trials**: Quick test to see if tuning helps
- **50 trials**: Good balance of speed vs quality
- **100 trials**: Standard thorough search
- **200+ trials**: Diminishing returns

## What Gets Tuned

The script optimizes these XGBoost hyperparameters:

### Tree Structure
- `max_depth`: 1-8 (controls overfitting)
- `max_bin`: 2-1000 (histogram granularity)
- `min_child_weight`: 0-12 (minimum samples per leaf)

### Regularization
- `alpha`: 0-12 (L1 regularization)
- `gamma`: 0-12 (complexity penalty)
- `reg_lambda`: 0-12 (L2 regularization)

### Sampling
- `subsample`: 0-1 (row sampling)
- `colsample_bytree`: 0-1 (column sampling)

### Training
- `learning_rate`: 0.001-1.0 (step size)
- `num_round`: 2-1000 (number of trees)
- `scale_pos_weight`: 1-15 (class imbalance)

## Understanding the Output

### During Tuning
```
Trial 50/100: AUC=0.6842 [✔ Best]
Trial 51/100: AUC=0.6721 
...
```

Each trial tests different hyperparameters and reports CV AUC.

### Final Results
```
Best CV AUC: 0.6892
Test AUC: 0.6450
Test Accuracy: 63.2%
```

**Good signs:**
- CV AUC > 0.68
- Test AUC > 0.63
- Small gap between CV and Test (<0.05)

**Red flags:**
- Large gap CV vs Test (>0.08) = overfitting
- Test AUC < 0.60 = something wrong
- All trials similar = hyperparameters don't matter much

## Output Files

All results saved to `results/` and `models/`:

```
results/
├── xgboost_best_params.json      # Best hyperparameters
├── optuna_trials.csv             # All 100 trial results
└── optuna_study.pkl              # Full Optuna study

models/
└── xgboost_tuned_calibrated.pkl  # Best model (ready for production)
```

## Interpreting Results

### Scenario 1: Beats Your Baseline (AUC > 0.64) ✅
```
Tuned + Calibrated: 0.6520 AUC, 63.5% accuracy
✅ SUCCESS! Beat your previous XGBoost!
```

**Action**: Use this model! Update Streamlit app.

### Scenario 2: Matches RandomForest (AUC ~0.628) 📊
```
Tuned + Calibrated: 0.6295 AUC, 62.8% accuracy
```

**Action**: Close to RF. Try ensemble of both.

### Scenario 3: Lower Than Expected (AUC < 0.62) ⚠️
```
Tuned + Calibrated: 0.6150 AUC, 60.5% accuracy
```

**Possible causes:**
- Not enough trials (try 200)
- Data issue (check for remaining leaks)
- Overfitting (try more regularization)

## Advanced: Analyzing Trials

Load the trial history in Python:

```python
import pandas as pd

# Load all trials
trials = pd.read_csv('results/optuna_trials.csv')

# Sort by AUC
best_trials = trials.sort_values('value', ascending=False).head(10)
print(best_trials[['number', 'value', 'params_learning_rate', 'params_max_depth']])

# Check parameter importance
import joblib
study = joblib.load('results/optuna_study.pkl')

from optuna.visualization import plot_param_importances
fig = plot_param_importances(study)
fig.show()
```

## Common Issues

### Issue: "All trials failing"
**Solution**: Check data loading. Run a few lines manually.

### Issue: "Taking too long"
**Solution**: Reduce N_TRIALS to 20 for quick test.

### Issue: "Results same as default"
**Solution**: Normal! Means default params were already good.

### Issue: "Much worse than default"
**Solution**: Try more trials, or Optuna found bad local minimum.

## What If Results Are Good?

If tuned XGBoost beats RandomForest (0.6282):

### Update Streamlit App

Edit `src/streamlit_app.py`, line 112:

```python
# Old:
model_dir = Path.cwd() / "models"
with open(model_dir / "model.pkl", 'rb') as f:
    loaded_model = joblib.load(f)

# New:
model_dir = Path.cwd() / "models"
with open(model_dir / "xgboost_tuned_calibrated.pkl", 'rb') as f:
    loaded_model = joblib.load(f)
```

Then restart: `streamlit run src/streamlit_app.py`

### Create Ensemble

If you want even better results, combine tuned XGBoost + RandomForest:

```python
# I can help you build this!
# Expected improvement: +1-2% AUC
```

## Next Steps After Tuning

Based on results, you can:

**If AUC > 0.64:**
1. ✅ Deploy this model
2. ✅ Update baseline in documentation
3. ✅ Test on more recent data

**If AUC ~0.628:**
1. ✅ Build ensemble with RandomForest
2. ✅ Try different features
3. ✅ Test LightGBM tuning too

**If AUC < 0.62:**
1. ⚠️ Check for data issues
2. ⚠️ Try more trials (200+)
3. ⚠️ Review feature engineering

---

## Ready to Tune?

```bash
source venv/bin/activate
python tune_xgboost.py
```

This will take ~20-30 minutes. Go grab lunch! 🍕

Let me know the results and we'll decide next steps together! 🚀

