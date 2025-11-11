# Enhanced Streamlit App Guide 🏀

## New Features

### ✨ What's New in Enhanced Edition

1. **🎯 Model Selection Dropdown**
   - Choose any trained model from sidebar
   - Switch between ensembles and individual models
   - See model performance metrics instantly

2. **🔬 Model Comparison Mode**
   - Compare predictions from multiple models side-by-side
   - See agreement/disagreement between models
   - Identify high-confidence vs uncertain predictions

3. **📊 Performance Dashboard**
   - Visual comparison of all models
   - AUC and accuracy charts
   - Detailed metrics table

4. **📈 Enhanced Predictions**
   - Confidence levels (High/Medium/Low)
   - Recent performance tracking (last 25 games)
   - Model accuracy statistics

5. **💡 Smart Model Loading**
   - Automatically detects available models
   - Shows which models are loaded
   - Gracefully handles missing models

---

## Quick Start

### Run the Enhanced App

```bash
source venv/bin/activate
streamlit run src/streamlit_app_enhanced.py
```

The app will open at `http://localhost:8501`

---

## Features Walkthrough

### 1. Model Selection

**Sidebar → Select Model**

Available models automatically detected:
- 🏆 **Stacking Ensemble** (best overall)
- ⚖️ **Weighted Ensemble** (fast, good)
- 🌲 **Random Forest** (stable baseline)
- ⚡ **XGBoost (Tuned)** (after running tuning)
- 📦 **Legacy Model** (original)

Each model shows:
- Description
- Test AUC score
- Test Accuracy
- Model type (ensemble vs individual)

### 2. Comparison Mode

**Sidebar → Enable Model Comparison**

When enabled:
- Select 1+ models to compare
- Each game shows predictions from all selected models
- Expandable comparison table per game
- See which models agree/disagree

**Use cases:**
- Verify ensemble consensus
- Find games with high model agreement
- Identify uncertain predictions

### 3. Predictions Tab

**Main Area → Predictions Tab**

Shows:
- **Today's Games** (if any scheduled)
  - Matchup details
  - Home win probability
  - Predicted winner
  - Confidence level (High/Medium/Low)
  - Model comparison (if enabled)

- **Recent Performance** (last 25 games)
  - Actual results vs predictions
  - Running accuracy
  - Average confidence
  - Detailed game-by-game results

**Confidence Levels:**
- 🔥 **High** (>70% probability): Strong prediction
- ⚖️ **Medium** (60-70%): Moderate confidence
- ⚠️ **Low** (<60%): Uncertain, close game

### 4. Performance Tab

**Main Area → Performance Tab**

Visual dashboard showing:
- Bar chart: AUC scores by model
- Bar chart: Accuracy by model
- Full metrics table
- Best model highlighted

**Metrics shown:**
- Test AUC (primary metric for betting)
- Test Accuracy (picking winners)
- Model type (ensemble vs individual)

### 5. About Tab

**Main Area → About Tab**

Documentation including:
- How the system works
- Model types explained
- Current performance
- Season information

---

## Example Workflows

### Workflow 1: Daily Predictions

**Goal**: Check today's games with best model

1. Open app
2. Sidebar: Select "🏆 Stacking Ensemble"
3. Predictions Tab: View today's games
4. Note high-confidence predictions
5. Check recent accuracy

**Time**: 30 seconds

### Workflow 2: Model Comparison

**Goal**: See if models agree on predictions

1. Sidebar: Select primary model (e.g., Stacking Ensemble)
2. Sidebar: Enable "Model Comparison"
3. Sidebar: Select 2-3 models to compare
4. Predictions Tab: Expand comparison for each game
5. Look for:
   - ✅ All models agree → high confidence
   - ⚠️ Models disagree → uncertain game

**Time**: 1-2 minutes

### Workflow 3: Model Evaluation

**Goal**: Find the best model for your use case

1. Performance Tab: View all model metrics
2. Compare AUC scores (for betting odds)
3. Compare accuracy (for simple picks)
4. Predictions Tab: Check recent performance
5. Switch models in sidebar to test

**Time**: 2-3 minutes

### Workflow 4: Confidence Analysis

**Goal**: Find high-confidence bets

1. Select best ensemble model
2. Predictions Tab: Today's games
3. Filter mentally for "🔥 High" confidence
4. Check model comparison for consensus
5. Verify recent accuracy on similar predictions

**Time**: 2-3 minutes

---

## Model Selection Guide

### When to Use Each Model

#### 🏆 Stacking Ensemble
**Best for**: Maximum accuracy, betting analysis
- **Pros**: Highest AUC (~0.632), most robust
- **Cons**: Slower inference, complex
- **Use when**: You want the best predictions

#### ⚖️ Weighted Ensemble
**Best for**: Balanced performance, speed
- **Pros**: Fast, good AUC (~0.630), simple
- **Cons**: Slightly lower than stacking
- **Use when**: Speed matters, good enough accuracy

#### 🌲 Random Forest
**Best for**: Stable baseline, interpretable
- **Pros**: Fast, reliable, good accuracy
- **Cons**: Lower AUC than ensembles
- **Use when**: You want simple, explainable predictions

#### ⚡ XGBoost (Tuned)
**Best for**: High performance single model
- **Pros**: Fast, well-tuned, good AUC
- **Cons**: May overfit to test set
- **Use when**: You want a fast, powerful single model

---

## Comparison vs Original App

| Feature | Original | Enhanced |
|---------|----------|----------|
| Model selection | ❌ Fixed | ✅ Dropdown |
| Multiple models | ❌ No | ✅ Yes |
| Model comparison | ❌ No | ✅ Side-by-side |
| Performance metrics | ❌ Basic | ✅ Dashboard |
| Confidence levels | ❌ No | ✅ High/Med/Low |
| Recent performance | ✅ Yes | ✅ Enhanced |
| Visual charts | ❌ No | ✅ Yes |

---

## Tips & Tricks

### 1. Finding High-Confidence Bets

```
1. Select Stacking Ensemble
2. Enable Model Comparison
3. Add Random Forest + XGBoost
4. Look for games where:
   - All models agree on winner
   - Confidence is "🔥 High"
   - Recent accuracy > 65%
```

### 2. Handling Uncertain Games

When models disagree or confidence is low:
- ⚠️ Avoid betting on these games
- Consider external factors (injuries, etc.)
- Check recent head-to-head history
- Wait for game-day information

### 3. Tracking Model Drift

Monitor recent performance (last 25 games):
- If accuracy drops below 55% → models may need retraining
- If confidence doesn't match results → recalibration needed
- Compare to historical performance

### 4. Optimizing for Your Goal

**For Betting (AUC focus):**
- Use Stacking Ensemble
- Focus on probability, not just winner
- Look for value bets (model prob > odds prob)

**For Simple Picks (Accuracy focus):**
- Use Weighted Ensemble or Random Forest
- Focus on high-confidence predictions only
- Check model consensus

---

## Troubleshooting

### Issue: "No models found!"

**Solution**: Train models first
```bash
python run_model_comparison.py  # Creates individual models
python build_ensemble.py         # Creates ensembles
```

### Issue: Model not appearing in dropdown

**Check**: Is the .pkl file in `models/` directory?
```bash
ls -lh models/
```

**Solution**: Model file must exist and be readable

### Issue: Performance metrics missing

**Solution**: Run ensemble builder
```bash
python build_ensemble.py
```

This creates `results/ensemble_results.json` with metrics.

### Issue: Today's games showing "None"

**Cause**: Off-season or data not updated

**Solutions**:
1. Update data: Run `notebooks/00_update_local_data.ipynb`
2. Check if NBA season is active (October - June)
3. Verify `data/games_engineered.csv` is current

---

## Advanced: Adding Custom Models

To add your own models to the app:

1. **Train your model**
2. **Save as .pkl file** in `models/` directory
3. **Edit app** (line 53): Add to `model_definitions`

```python
'your_model_name': {
    'name': '🎯 Your Model',
    'description': 'Your model description',
    'file': 'your_model.pkl',
    'type': 'individual' or 'ensemble'
}
```

4. **Restart app**: Model appears in dropdown!

---

## Performance Expectations

### Current Standings (with enhanced app):

```
Model                    AUC     Accuracy
==========================================
🏆 Stacking Ensemble    0.6319    61.4%
⚖️ Weighted Ensemble    0.6297    63.9%
🌲 Random Forest        0.6282    62.5%
⚡ XGBoost (20 trials)  0.6245    61.4%
```

### After 100-trial tuning (expected):

```
⚡ XGBoost (100 trials) 0.63-0.65  62-64%
🏆 New Ensemble         0.64-0.66  63-65%
```

---

## Next Steps

1. ✅ **Use enhanced app** for daily predictions
2. ✅ **Run 100-trial tuning**: `python tune_xgboost.py`
3. ✅ **Rebuild ensemble** with better XGBoost
4. ✅ **Deploy best model** in production

---

## Running Both Versions

**Original (simple)**:
```bash
streamlit run src/streamlit_app.py
```

**Enhanced (full features)**:
```bash
streamlit run src/streamlit_app_enhanced.py
```

Keep both! Original for simplicity, enhanced for analysis.

---

**Ready to try it?**

```bash
source venv/bin/activate
streamlit run src/streamlit_app_enhanced.py
```

🚀 Your enhanced NBA prediction dashboard awaits!

