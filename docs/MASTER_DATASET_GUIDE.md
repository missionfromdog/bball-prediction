# 📊 Master Dataset & Production Pipeline Guide

**Last Updated:** November 14, 2025  
**Version:** 2.0 - Master Dataset Edition

---

## 🎯 Overview

This guide documents the complete NBA prediction system built with **30,120 historical games** (2003-2025) and **102 predictive features** including injury data, Vegas betting lines, and rolling performance metrics.

---

## 📈 System Performance

### **Current Model Stats**
- **Test Accuracy:** 62.82%
- **Training Accuracy:** 67.62%
- **Features:** 102 (injury + Vegas + rolling averages)
- **Training Data:** 30,120 games (2003-2025)
- **Model Size:** 1.0 MB (no LFS needed)

### **Feature Breakdown**
- 🏥 **14 Injury Features** - Player availability and impact
- 🎲 **4 Vegas Features** - Spread, total, moneylines
- 📊 **83 Rolling Features** - Performance trends (3/7/10 games)
- 📅 **1 Date Feature** - Seasonality (MONTH)

---

## 🗄️ Data Architecture

### **Master Dataset: `games_master_engineered.csv`**

**Stats:**
- **Rows:** 30,120 games
- **Columns:** 122 total (102 used for training)
- **Size:** 45.5 MB
- **Date Range:** October 5, 2003 → November 13, 2025
- **Location:** `data/games_master_engineered.csv`

**Data Split:**
- Historical (2003-2020): 23,513 games
- Recent (2021-2025): 6,607 games

**Why This Matters:**
- 6x more data than previous 5k-game dataset
- Includes full 22-year NBA history
- All injury and Vegas features preserved
- Consistent feature engineering across all time periods

---

## 🔧 Feature Engineering Pipeline

### **Complete Feature List (102 Features)**

#### **1. Injury Features (14)**
```
HOME_injuries_active          # Number of players currently injured
HOME_star_injuries            # Number of star players injured
HOME_injury_impact            # Overall team impact score
HOME_injuries_severity        # Weighted severity of injuries
HOME_injuries_major           # Count of major injuries
HOME_injuries_recent_7d       # Injuries in last 7 days
HOME_days_since_injury        # Days since last injury report

VISITOR_injuries_active       # Same for away team
VISITOR_star_injuries
VISITOR_injury_impact
VISITOR_injuries_severity
VISITOR_injuries_major
VISITOR_injuries_recent_7d
VISITOR_days_since_injury
```

#### **2. Vegas Betting Features (4)**
```
spread                        # Point spread (+ = home favored)
total                         # Over/under total points
moneyline_home               # Home team moneyline odds
moneyline_away               # Away team moneyline odds
```

#### **3. Rolling Performance Features (83)**

**Win Rates & Streaks:**
```
HOME_TEAM_WIN_STREAK                    # Current win/loss streak
HOME_TEAM_WINS_AVG_LAST_3_HOME         # Win rate, last 3 home games
HOME_TEAM_WINS_AVG_LAST_7_HOME         # Win rate, last 7 home games
HOME_TEAM_WINS_AVG_LAST_10_HOME        # Win rate, last 10 home games
VISITOR_TEAM_WIN_STREAK                 # Away team streak
VISITOR_TEAM_WINS_AVG_LAST_3_VISITOR
VISITOR_TEAM_WINS_AVG_LAST_7_VISITOR
VISITOR_TEAM_WINS_AVG_LAST_10_VISITOR
VISITOR_TEAM_WINS_AVG_LAST_*_VISITOR_MINUS_LEAGUE_AVG  # League-adjusted
```

**Points, Rebounds, Assists (3 time windows × 6 metrics = 18 base features):**
```
HOME_PTS_home_AVG_LAST_3_HOME          # Average points (last 3 home)
HOME_PTS_home_AVG_LAST_7_HOME
HOME_PTS_home_AVG_LAST_10_HOME
HOME_REB_home_AVG_LAST_*_HOME          # Rebounds averages
HOME_AST_home_AVG_LAST_*_HOME          # Assists averages
```

**Shooting Percentages (3 time windows × 6 metrics = 18 features):**
```
HOME_FG_PCT_home_AVG_LAST_*_HOME       # Field goal %
HOME_FT_PCT_home_AVG_LAST_*_HOME       # Free throw %
HOME_FG3_PCT_home_AVG_LAST_*_HOME      # Three-point %
```

**League-Adjusted Metrics (47 features):**
```
*_MINUS_LEAGUE_AVG                     # All rolling averages minus league avg
```

#### **4. Date Features (1)**
```
MONTH                                   # 1-12 for seasonality
```

---

## 🏗️ Building the Master Dataset

### **One-Time Build Process**

The master dataset was built using `scripts/data_collection/build_master_dataset.py`:

```bash
python scripts/data_collection/build_master_dataset.py
```

**What It Does:**
1. Loads raw historical games (2003-2022) from `data/original/games.csv`
2. Loads already-engineered recent games (2021-2025)
3. Integrates Vegas betting data (real data where available)
4. Integrates injury data (real for recent, placeholders for historical)
5. Runs feature engineering on historical data
6. Combines historical + recent into single master dataset
7. Saves to `data/games_master_engineered.csv`

**Build Time:** ~2 seconds (for 30k games!)

**Key Fixes Applied:**
- PLAYOFF column handling for historical data
- Deduplication before merges (prevents Cartesian product)
- Date parsing with `format='mixed'`
- Exact-match leaky feature filtering

---

## 🎯 Model Training

### **Training Script: `scripts/predictions/setup_model.py`**

```bash
python scripts/predictions/setup_model.py
```

**What It Does:**
1. Loads master dataset (30k games)
2. Drops 20 columns:
   - 2 target columns (HOME_TEAM_WINS, TARGET)
   - 5 metadata columns (GAME_DATE_EST, GAME_ID, MATCHUP, etc.)
   - 9 categorical columns (team IDs, data source flags)
   - 12 leaky features (exact post-game stats)
3. Keeps 102 predictive features
4. Trains HistGradientBoostingClassifier
5. Calibrates probabilities
6. Saves to `models/histgradient_vegas_calibrated.pkl`

**Training Configuration:**
```python
HistGradientBoostingClassifier(
    random_state=42,
    max_iter=200,
    max_depth=7,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    max_features=0.8
)
```

**Results:**
- Training: 67.62% accuracy
- Test: 62.82% accuracy
- No data leakage detected

---

## 🔮 Making Predictions

### **Prediction Script: `scripts/predictions/make_daily_predictions.py`**

```bash
python scripts/predictions/make_daily_predictions.py
```

**What It Does:**
1. Loads trained model
2. Loads master dataset
3. Filters for unplayed games (PTS_home == 0)
4. Prepares features (drops same 20 columns as training)
5. Generates predictions & probabilities
6. Matches with live Vegas odds
7. Saves to `data/predictions/predictions_YYYYMMDD.csv`

**Output Columns:**
- Date
- Matchup
- Home_Win_Probability
- Predicted_Winner
- Confidence (High/Medium/Low)
- Vegas_Spread, Vegas_Total, Vegas_ML_Home, Vegas_ML_Away
- Edge_vs_Vegas

---

## 🖥️ Streamlit App

### **Running Locally:**

```bash
streamlit run src/streamlit_app_enhanced.py
```

**Features:**
- Today's game predictions with confidence levels
- Live Vegas odds integration
- Model comparison (all models auto-loaded)
- Performance tracking
- CSV export for Google Sheets
- Data freshness indicators

**Data Loading:**
- Primary: `data/games_master_engineered.csv`
- Fallback: `data/games_with_real_vegas.csv`

---

## 🔄 Daily Update Workflow

### **Incremental Updates (Recommended)**

For daily operations, you only need to engineer features for NEW games (~3-12 games):

1. **Fetch Today's Schedule** (ESPN or NBA.com)
   ```bash
   python scripts/data_collection/fetch_todays_schedule.py
   ```

2. **Engineer Features for New Games Only**
   ```python
   # Add new games to master dataset with PTS_home = 0
   # Run feature engineering (rolling averages calculate automatically)
   ```

3. **Make Predictions**
   ```bash
   python scripts/predictions/make_daily_predictions.py
   ```

### **Monthly Retraining (Optional)**

Retrain the model monthly or when accuracy drops:

```bash
python scripts/predictions/setup_model.py
```

---

## 📂 File Structure

```
data/
├── games_master_engineered.csv       # 30k games, all features (MAIN DATASET)
├── games_with_real_vegas.csv         # Fallback dataset
├── original/                          # Raw source data
│   └── games.csv                      # 2003-2022 raw games
├── predictions/                       # Daily prediction outputs
│   ├── predictions_YYYYMMDD.csv
│   └── predictions_latest.csv
└── betting/                           # Live odds data
    └── live_odds_latest.csv

models/
└── histgradient_vegas_calibrated.pkl  # Production model (1.0 MB)

scripts/
├── data_collection/
│   ├── build_master_dataset.py        # ONE-TIME: Build master dataset
│   ├── fetch_todays_schedule.py       # DAILY: Get today's games
│   └── engineer_features.py           # DAILY: Feature engineering
└── predictions/
    ├── setup_model.py                 # MONTHLY: Retrain model
    └── make_daily_predictions.py      # DAILY: Generate predictions
```

---

## ⚙️ GitHub Actions Workflows

### **Automated Pipelines**

1. **`daily-predictions-v3.yml`** - Daily predictions (9 AM UTC)
2. **`email-daily-predictions-v2.yml`** - Email predictions
3. **`fetch-todays-schedule-v2.yml`** - Fetch today's games (8 AM UTC)
4. **`scrape-live-odds.yml`** - Update Vegas odds (every 6 hours)
5. **`track-performance.yml`** - Track prediction accuracy (11 PM UTC)

**Note:** Some workflows may need updates for the new 102-feature model.

---

## 🐛 Common Issues & Solutions

### **Issue: Model expects wrong number of features**

**Solution:** Ensure feature drop logic is IDENTICAL across all scripts:
- Training: `scripts/predictions/setup_model.py`
- Prediction: `scripts/predictions/make_daily_predictions.py`
- Streamlit: `src/streamlit_app_enhanced.py`

All should drop the SAME 20 columns (exact matches, not pattern matching).

### **Issue: "No games found for today"**

**Cause:** No unplayed games in master dataset for current date.

**Solution:** 
```bash
# Fetch today's schedule first
python scripts/data_collection/fetch_todays_schedule.py

# Then make predictions
python scripts/predictions/make_daily_predictions.py
```

### **Issue: TypeError in odds matching**

**Fixed in v2.0:** Added string type checking before comparison.

---

## 📊 Performance Expectations

### **Accuracy Benchmarks**
- **Baseline (coin flip):** 50%
- **Our model:** 62.82%
- **Vegas implied probability:** ~64-68% (industry standard)

### **Confidence Levels**
- **High:** |prob - 0.5| > 0.15 (e.g., 65%+ or 35%- home win)
- **Medium:** |prob - 0.5| > 0.05 (e.g., 55-65% or 35-45%)
- **Low:** |prob - 0.5| ≤ 0.05 (close to 50/50)

### **Expected ROI**
- Not guaranteed (Vegas typically has the edge)
- Best used for: Identifying value bets, understanding matchups, entertainment

---

## 🚀 Next Steps

### **Production Checklist**

- [x] Master dataset built (30k games)
- [x] Model trained (62.82% accuracy)
- [x] Prediction pipeline working
- [x] Streamlit app updated
- [ ] GitHub Actions workflows tested
- [ ] Documentation complete
- [ ] Performance tracking active

### **Future Enhancements**

1. **Real-time injury updates** - Currently using static/synthetic data for historical
2. **Advanced features** - Pace, defensive rating, recent roster changes
3. **Ensemble models** - Combine multiple algorithms
4. **Deployment** - Host Streamlit app on Streamlit Cloud or Heroku

---

## 📞 Support

For questions or issues:
- **GitHub:** https://github.com/missionfromdog/bball-prediction
- **Documentation:** `docs/` directory
- **Scripts:** Check comments in individual files

---

## 🙏 Acknowledgments

- **Base Project:** Chris Munch's NBA prediction framework
- **Enhancements:** Casey Hess with Cursor AI assistance
- **Data Sources:** NBA.com, ESPN, Kaggle, Basketball-Reference

---

**Built with ❤️ for NBA fans and data enthusiasts!**

