# NBA Prediction & Tracking System - Complete Guide

This guide covers the three new automated systems: Live Odds Display, Daily Predictions, and Performance Tracking.

---

## 🎯 System A: Live Odds Display in Streamlit

### **What It Does:**
Shows live betting lines from The Odds API directly in your Streamlit predictions dashboard.

### **Files:**
- `src/live_odds_display.py` - Helper module for loading and matching odds
- `ADD_TO_STREAMLIT_APP.md` - Instructions for integration

### **Features:**
- ✅ Displays spread, over/under, and moneylines
- ✅ Automatically matches games to odds
- ✅ Shows when odds data is available
- ✅ Gracefully handles missing data

### **Setup:**
The helper module is ready. To add to your Streamlit app:

1. Import the module at the top of `src/streamlit_app_enhanced.py`:
```python
from live_odds_display import load_live_odds, match_game_to_odds, format_odds_display
```

2. Load odds at the start of your `main()` function:
```python
live_odds_df = load_live_odds()
if live_odds_df is not None:
    st.info(f"📊 Live odds loaded for {len(live_odds_df)} games")
```

3. Display odds in your game loop (after showing the matchup):
```python
live_odds = match_game_to_odds(row['MATCHUP'], live_odds_df)
if live_odds is not None:
    odds_display = format_odds_display(live_odds)
    if odds_display:
        st.markdown("**🎲 Live Vegas Odds:**")
        odds_cols = st.columns(3)
        if 'spread' in odds_display:
            with odds_cols[0]:
                st.caption(f"Spread: {odds_display['spread']}")
        if 'total' in odds_display:
            with odds_cols[1]:
                st.caption(f"O/U: {odds_display['total']}")
        if 'ml_home' in odds_display:
            with odds_cols[2]:
                st.caption(f"ML: {odds_display['ml_home']}")
```

---

## 🤖 System B: Automated Daily Predictions

### **What It Does:**
Automatically makes predictions for today's NBA games every morning and exports them to CSV.

### **Files:**
- `scripts/predictions/make_daily_predictions.py` - Prediction engine
- `.github/workflows/daily-predictions.yml` - Automation workflow

### **Features:**
- ✅ Runs automatically every day at 9 AM UTC (1 AM PST / 4 AM EST)
- ✅ Uses best model (HistGradient + Vegas, 70.20% AUC)
- ✅ Includes confidence levels (High/Medium/Low)
- ✅ Matches with live Vegas odds
- ✅ Calculates edge vs Vegas implied probability
- ✅ Exports to CSV for tracking

### **Output Files:**
```
data/predictions/
├── predictions_YYYYMMDD.csv        ← Daily timestamped
└── predictions_latest.csv          ← Always current
```

### **CSV Columns:**
- `Date` - Game date
- `Matchup` - Teams playing
- `Home_Win_Probability` - Model's predicted probability
- `Predicted_Winner` - Home or Away
- `Confidence` - High/Medium/Low
- `Model` - Model used
- `Vegas_Spread` - Current spread
- `Vegas_Total` - Over/Under
- `Vegas_ML_Home` - Home moneyline
- `Vegas_ML_Away` - Away moneyline
- `Vegas_Implied_Home_Win_Prob` - Implied probability from moneyline
- `Edge_vs_Vegas` - Your model's edge over Vegas

### **Manual Run:**
```bash
# Test locally first
cd /path/to/project
source venv/bin/activate
python scripts/predictions/make_daily_predictions.py
```

### **GitHub Actions:**
1. Go to: https://github.com/missionfromdog/bball-prediction/actions
2. Click "Daily NBA Predictions"
3. Click "Run workflow" to trigger manually

### **Schedule:**
- Runs daily at 9 AM UTC
- Perfect timing before most games start
- Automatically commits predictions to repo

---

## 📊 System C: Performance Tracking

### **What It Does:**
Compares your predictions against actual game results and calculates detailed performance metrics.

### **Files:**
- `scripts/predictions/track_performance.py` - Performance tracker
- `.github/workflows/track-performance.yml` - Automation workflow

### **Features:**
- ✅ Runs automatically every night at 11 PM UTC
- ✅ Matches predictions to actual results
- ✅ Calculates accuracy overall and by confidence level
- ✅ Computes betting ROI (assuming $100 bets)
- ✅ Tracks performance vs Vegas lines
- ✅ Shows recent trends (last 10 games)
- ✅ Generates comprehensive reports

### **Output Files:**
```
data/performance/
├── performance_metrics_YYYYMMDD.csv      ← Daily metrics
├── performance_metrics_latest.csv        ← Current metrics
├── detailed_tracking_YYYYMMDD.csv        ← All predictions with results
├── detailed_tracking_latest.csv          ← Current tracking
└── performance_report_YYYYMMDD.txt       ← Human-readable report
```

### **Metrics Tracked:**
- **Overall Accuracy** - Percentage of correct predictions
- **Accuracy by Confidence** - How well High/Medium/Low predictions perform
- **Total Wagered** - Total hypothetical money bet
- **Total Profit/Loss** - Net profit or loss
- **ROI** - Return on investment percentage
- **Strong Edge Accuracy** - Performance when model disagrees with Vegas by >5%
- **Recent Performance** - Last 10 games accuracy

### **Sample Report:**
```
================================================================================
NBA PREDICTION PERFORMANCE REPORT
Generated: 2025-11-11 23:00:00
================================================================================

📊 OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Total Predictions: 45
Correct Predictions: 32
Overall Accuracy: 71.1%

🎯 ACCURACY BY CONFIDENCE LEVEL
--------------------------------------------------------------------------------
High     Confidence:  20 predictions, 80.0% accuracy
Medium   Confidence:  18 predictions, 66.7% accuracy
Low      Confidence:   7 predictions, 57.1% accuracy

💰 BETTING PERFORMANCE
--------------------------------------------------------------------------------
Total Wagered: $4,500.00
Total Profit/Loss: $425.00
ROI: +9.44%

📈 EDGE VS VEGAS
--------------------------------------------------------------------------------
Games with Strong Edge (>5%): 12
Strong Edge Accuracy: 83.3%
```

### **Manual Run:**
```bash
python scripts/predictions/track_performance.py
```

### **GitHub Actions:**
1. Go to: https://github.com/missionfromdog/bball-prediction/actions
2. Click "Track Betting Performance"
3. Click "Run workflow"

---

## 🔄 Complete Automation Flow

```
Every 6 Hours (00, 06, 12, 18 UTC)
    ↓
🎲 Fetch Live Odds
    ↓
data/betting/live_odds_latest.csv updated

Every Day at 9 AM UTC
    ↓
🎯 Make Daily Predictions
    ↓
data/predictions/predictions_latest.csv created

Every Day at 10 AM UTC
    ↓
🏥 Scrape Injuries + Download Vegas Data
    ↓
data/ updated with latest info

Every Day at 11 PM UTC
    ↓
📊 Track Performance
    ↓
data/performance/ updated with metrics
```

---

## 📧 Email/Export Options (System D - Optional)

Want to receive predictions via email? Here are options:

### **Option 1: CSV Download from GitHub**
- Predictions automatically committed to repo
- Download `data/predictions/predictions_latest.csv`
- Import to Google Sheets, Excel, etc.

### **Option 2: GitHub Actions Email**
Add this to `.github/workflows/daily-predictions.yml`:

```yaml
- name: Send Email
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: "NBA Predictions for ${{ steps.date.outputs.date }}"
    to: your@email.com
    from: nba-predictions@yourdomain.com
    body: file://data/predictions/predictions_latest.csv
    attachments: data/predictions/predictions_latest.csv
```

### **Option 3: Slack/Discord Webhook**
Add notification to workflow:

```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        text: "🏀 NBA Predictions Ready!",
        attachments: [{
          color: 'good',
          text: 'Check data/predictions/predictions_latest.csv'
        }]
      }
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### **Option 4: Streamlit Share**
Deploy your Streamlit app to https://share.streamlit.io/ for a live dashboard accessible anywhere.

---

## 🎯 Quick Start Guide

### **Day 1: Setup**
1. ✅ Add `ODDS_API_KEY` to GitHub Secrets (Done!)
2. ✅ Commit all new files to GitHub
3. ✅ Enable workflows in Actions tab

### **Day 2: First Predictions**
1. Run "Daily NBA Predictions" workflow manually
2. Check `data/predictions/predictions_latest.csv`
3. Watch games and see results

### **Day 3: Track Performance**
1. Run "Track Betting Performance" workflow
2. Check `data/performance/performance_report_YYYYMMDD.txt`
3. Review accuracy and ROI

### **Ongoing:**
- Everything runs automatically
- Check Actions tab for workflow status
- Review performance weekly
- Adjust betting strategy based on metrics

---

## 🐛 Troubleshooting

### **No predictions generated:**
- Check if games are scheduled (NBA season runs Oct-June)
- Verify data files exist in `data/`
- Run prediction script locally to see errors

### **Performance tracking shows no matches:**
- Predictions need 1+ day to have actual results
- Check that game dates match format
- Verify `games_with_real_vegas.csv` has recent data

### **Live odds not showing:**
- Check if `data/betting/live_odds_latest.csv` exists
- Verify "Fetch Live Odds" workflow is running
- Confirm API key is valid and has credits

### **Workflows failing:**
- Check Actions tab for error logs
- Verify all permissions are set correctly
- Ensure dependencies are installed

---

## 📚 Further Customization

### **Change Prediction Time:**
Edit `.github/workflows/daily-predictions.yml`:
```yaml
schedule:
  - cron: '0 15 * * *'  # 3 PM UTC instead of 9 AM
```

### **Bet Sizing:**
Modify `track_performance.py` line ~130:
```python
# Change from $100 to your preferred bet size
bet_size = 50  # $50 per bet
```

### **Confidence Thresholds:**
Adjust in `make_daily_predictions.py`:
```python
# Current: High >15%, Medium >5%
# Change to: High >20%, Medium >10%
'Confidence': 'High' if abs(prob - 0.5) > 0.20 else ...
```

---

## 🎉 Success Metrics

After 1 week, you should see:
- ✅ 7-14 predictions made (depending on NBA schedule)
- ✅ Accuracy metrics calculated
- ✅ ROI tracked
- ✅ Performance trends visible

After 1 month:
- ✅ Statistically significant sample size (30+ games)
- ✅ Clear confidence level performance
- ✅ Betting strategy insights
- ✅ Edge vs Vegas validation

---

*System built with Cursor AI • November 2025*

