# NBA Prediction Workflow Architecture

## Overview

This system uses **Change Data Capture (CDC)** principles to incrementally update data and make predictions daily. All workflows are automated and run on GitHub Actions.

## Daily Workflow Chain

```
8:00 AM UTC  →  9:00 AM UTC  →  9:30 AM UTC
    │               │                │
    ▼               ▼                ▼
Schedule        Predictions       Email
Fetch          Generation      Delivery
```

## 1. Schedule Fetch (8:00 AM UTC)

**Workflow:** `fetch-todays-schedule.yml`  
**Script:** `scripts/data_collection/fetch_todays_schedule.py`

### What it does:
- Scrapes NBA.com for today's scheduled games
- Loads historical data for feature engineering context
- Calculates rolling statistics for new games
- **Incrementally appends** to existing data (not full reload)
- Maintains workflow dataset at 5,000 most recent games

### Key Features:
- ✅ **Change Data Capture**: Only adds new/updated records
- ✅ **Incremental Updates**: Appends to existing file, doesn't rebuild
- ✅ **Size Management**: Keeps workflow dataset small (17.7 MB)
- ✅ **Feature Engineering**: Full historical context for rolling stats

### Output:
- `data/games_with_real_vegas.csv` - Full history
- `data/games_with_real_vegas_workflow.csv` - Last 5K games (for workflows)

---

## 2. Predictions (9:00 AM UTC)

**Workflow:** `daily-predictions-v2.yml`  
**Script:** `scripts/predictions/make_daily_predictions.py`

### What it does:
- Triggers automatically after schedule fetch completes
- Loads best trained model (HistGradient + Vegas, 70.20% AUC)
- Predicts on all unplayed games (`PTS_home == 0`)
- Saves predictions with confidence levels
- Auto-retrains model if missing/invalid

### Key Features:
- ✅ **Auto-trigger**: Runs when schedule fetch completes
- ✅ **Smart model loading**: Falls back to workflow dataset if needed
- ✅ **Auto-retraining**: Rebuilds model if corrupted (LFS, numpy mismatch, etc.)
- ✅ **Confidence scoring**: High/Medium/Low based on prediction probability

### Output:
- `data/predictions/predictions_YYYYMMDD.csv` - Dated predictions
- `data/predictions/predictions_latest.csv` - Latest predictions

---

## 3. Email Delivery (9:30 AM UTC)

**Workflow:** `email-daily-predictions.yml`  
**Script:** `scripts/predictions/send_email_predictions.py`

### What it does:
- Triggers automatically after predictions complete
- Loads latest predictions
- Formats professional HTML email
- Includes live Vegas odds when available
- Sends to configured email address

### Key Features:
- ✅ **Auto-trigger**: Runs when predictions complete
- ✅ **Professional layout**: Table format with team logos
- ✅ **Vegas integration**: Shows spread, O/U, moneyline
- ✅ **Confidence display**: Visual confidence indicators

---

## Supporting Workflows

### Live Odds Fetch (Every 6 hours)
**Workflow:** `scrape-live-odds.yml`  
**Frequency:** 0:00, 6:00, 12:00, 18:00 UTC

Fetches current betting lines from The Odds API.

### Injury Data Update (Daily 10:00 AM UTC)
**Workflow:** `daily-data-update.yml`  
**Script:** `scripts/data_collection/scrape_real_injuries.py`

Scrapes current injury reports from Basketball-Reference.

### Performance Tracking (Daily 11:00 PM UTC)
**Workflow:** `track-performance.yml`  
**Script:** `scripts/predictions/track_performance.py`

Compares predictions to actual results, calculates ROI.

---

## Data Management Strategy

### Change Data Capture (CDC)

Instead of downloading all historical data daily, we:

1. **Initial Load**: Full history loaded once (28K+ games)
2. **Incremental Updates**: Only add today's new games
3. **Deduplication**: Remove duplicates by date before appending
4. **Size Control**: Workflow dataset limited to 5K games

### File Structure

```
data/
├── games_with_real_vegas.csv           # Full history (104 MB, Git LFS)
├── games_with_real_vegas_workflow.csv  # Last 5K games (17.7 MB, tracked)
├── betting/
│   ├── live_odds_latest.csv            # Current odds
│   └── kaggle/nba_2008-2025.csv       # Historical Vegas data
├── injuries/
│   └── nba_injuries_real_scraped.csv   # Current injuries
└── predictions/
    ├── predictions_latest.csv          # Today's predictions
    └── predictions_YYYYMMDD.csv        # Historical predictions
```

### Why Two Data Files?

1. **`games_with_real_vegas.csv`** (Git LFS)
   - Full 28K+ game history
   - Used locally for model training
   - Too large for workflows to pull (LFS bandwidth limits)

2. **`games_with_real_vegas_workflow.csv`** (Git tracked)
   - Last 5,000 games only
   - Small enough to commit directly (17.7 MB)
   - Used by GitHub Actions workflows
   - Updated daily with new games

---

## Workflow Triggers

### Automatic (Scheduled)
- **8:00 AM**: Fetch schedule
- **9:00 AM**: Make predictions (also auto-triggers after schedule)
- **9:30 AM**: Email predictions (also auto-triggers after predictions)
- **10:00 AM**: Update injuries and Kaggle odds
- **11:00 PM**: Track performance
- **Every 6h**: Fetch live odds

### Manual
All workflows support `workflow_dispatch` for manual triggering via GitHub Actions UI.

---

## Model Training Strategy

### Initial Training
Model is trained on 5,000 most recent games (2021-2025) with:
- Rolling statistics (3/7/10/15 game windows)
- Real Vegas betting lines (spread, total, moneyline)
- Injury data (active injuries, severity, days out)
- Matchup history
- League average comparisons

### Retraining
Model auto-retrains if:
- No valid model file exists
- Model file is corrupted (LFS pointer, numpy mismatch)
- Manual deletion via workflow

Training takes ~2-3 minutes in GitHub Actions.

### Model Performance
- **Algorithm**: HistGradientBoostingClassifier (calibrated)
- **AUC**: 70.20%
- **Accuracy**: ~64%
- **Best features**: Vegas lines, rolling averages, matchup history

---

## Troubleshooting

### Common Issues

**1. Workflow finds no games**
- Check if schedule fetch ran successfully
- Verify NBA season is active (Oct-Jun)
- Manually trigger schedule fetch

**2. Model training fails**
- Check Python version compatibility (3.11)
- Verify numpy/sklearn versions
- May need to clear cached dependencies

**3. LFS bandwidth exceeded**
- All workflows now have `lfs: false`
- Use workflow dataset, not full history
- LFS only needed for local development

**4. Predictions on old games**
- Ensure schedule fetch ran before predictions
- Check workflow trigger chain
- Manually trigger schedule fetch

### Manual Workflow Execution

1. Go to GitHub Actions tab
2. Select workflow (e.g., "Fetch Today's Schedule")
3. Click "Run workflow" → "Run workflow"
4. Monitor progress in Actions tab

---

## Future Enhancements

### Short Term
- [ ] Add player prop predictions
- [ ] Integrate more sportsbooks
- [ ] SMS notifications option
- [ ] Slack/Discord webhooks

### Long Term
- [ ] Real-time in-game predictions
- [ ] Playoff-specific models
- [ ] Player injury impact quantification
- [ ] Arbitrage opportunity detection

---

## Architecture Benefits

✅ **Automated**: No manual intervention needed  
✅ **Incremental**: Only fetch/process new data  
✅ **Efficient**: Small file sizes, fast workflows  
✅ **Reliable**: Auto-retraining, fallback logic  
✅ **Scalable**: Easy to add new data sources  
✅ **Maintainable**: Clear workflow chain, good logging

---

## Contact

For issues or questions:
- GitHub Issues: [github.com/missionfromdog/bball-prediction/issues](https://github.com/missionfromdog/bball-prediction/issues)
- Email: Listed in README.md

---

*Last Updated: 2025-11-12*

