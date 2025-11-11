# 📥 Kaggle Setup Guide - Download Real NBA Betting Lines

## 🎯 Goal
Download real historical NBA betting odds from Kaggle to improve our model from 67.68% to ~70% AUC.

---

## Step 1: Create Kaggle Account & Get API Token

### 1.1 Sign up for Kaggle (if you don't have an account)
- Go to: https://www.kaggle.com/
- Click "Register" and create a free account

### 1.2 Get your API credentials
1. Go to: https://www.kaggle.com/settings
2. Scroll down to "API" section
3. Click **"Create New Token"**
4. This will download `kaggle.json` to your Downloads folder

### 1.3 Install the credentials
```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file (adjust path if needed)
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set proper permissions (required for security)
chmod 600 ~/.kaggle/kaggle.json
```

---

## Step 2: Find the Best NBA Betting Dataset

### Recommended Datasets on Kaggle

**Option 1: NBA Historical Odds (Best for our use case)**
- Dataset: Search "NBA odds" or "NBA betting lines historical"
- Look for datasets with:
  - Date range: 2003-2024+ 
  - Columns: game_date, home_team, visitor_team, spread, total, moneyline
  - Good ratings/upvotes

**Option 2: Sports Betting Data**
- Dataset: `ehallmar/nba-historical-stats-and-betting-data`
- Usually includes spreads, totals, and outcomes

**Option 3: The Odds API Historical**
- Sometimes available as Kaggle datasets
- Professional-grade odds data

### How to Find It

1. Go to: https://www.kaggle.com/datasets
2. Search: "NBA betting odds" or "NBA spreads historical"
3. Filter by:
   - **File Type:** CSV
   - **License:** Open (so we can use it)
   - **Tags:** sports, basketball, nba
4. Look for recent updates (2023-2024 data)

---

## Step 3: Download the Dataset

Once you find the right dataset, note its path. For example:
```
ehallmar/nba-historical-stats-and-betting-data
```

### Using Kaggle CLI (Automated)

```bash
# Activate your virtual environment
cd /Users/caseyhess/datascience/bball/nba-prediction-main
source venv/bin/activate

# Download the dataset (replace with actual dataset name)
kaggle datasets download -d ehallmar/nba-historical-stats-and-betting-data

# Unzip it
unzip nba-historical-stats-and-betting-data.zip -d data/betting/kaggle/

# Clean up zip file
rm nba-historical-stats-and-betting-data.zip
```

### Using Kaggle Website (Manual)

1. Go to the dataset page
2. Click the **Download** button (top right)
3. Save to: `~/Downloads/`
4. Move and extract:
   ```bash
   mv ~/Downloads/nba-*.zip data/betting/kaggle/
   cd data/betting/kaggle/
   unzip nba-*.zip
   ```

---

## Step 4: Inspect the Downloaded Data

```bash
# See what files we got
ls -lh data/betting/kaggle/

# Check the first few rows
head -20 data/betting/kaggle/*.csv

# Get column names
head -1 data/betting/kaggle/*.csv
```

Common column names to look for:
- `date`, `game_date`, `GAME_DATE`
- `home_team`, `away_team`, `visitor_team`
- `home_spread`, `spread`, `line`
- `total`, `over_under`, `ou`
- `home_ml`, `away_ml`, `moneyline`

---

## Step 5: Process the Real Vegas Data

Once downloaded, run our processing script:

```bash
cd /Users/caseyhess/datascience/bball/nba-prediction-main
./venv/bin/python process_real_vegas_lines.py
```

This will:
1. Load the Kaggle dataset
2. Match it to our game data
3. Replace synthetic betting lines with real ones
4. Create new features from real market data

---

## Expected File Structure

After setup, you should have:

```
data/betting/
├── kaggle/                          # Raw Kaggle data
│   ├── nba_odds.csv                # Real Vegas lines
│   └── README.md                    # Dataset documentation
├── nba_betting_lines_historical.csv # Our synthetic lines (old)
└── nba_betting_lines_real.csv      # Processed real lines (new!)
```

---

## 🔍 What to Look for in Real Vegas Data

### Quality Indicators ✅

1. **Coverage:** At least 10,000+ games (ideally 2003-2024)
2. **Completeness:** Spreads, totals, moneylines for most games
3. **Multiple Books:** Data from different sportsbooks (Pinnacle, BetMGM, etc.)
4. **Opening & Closing Lines:** Line movement data
5. **Timestamps:** When lines were posted

### Red Flags ❌

1. Too many missing values (>20%)
2. Only recent data (2020+)
3. Single sportsbook only
4. No documentation
5. Unrealistic spreads (like 50+ points)

---

## Alternative: The Odds API

If Kaggle doesn't have good historical data, try The Odds API:

```bash
# Sign up for free tier
# Go to: https://the-odds-api.com/

# Get API key
export ODDS_API_KEY="your_key_here"

# Download historical data (if available)
curl "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds" \
  -H "apiKey: $ODDS_API_KEY" \
  > data/betting/odds_api_historical.json
```

**Note:** The Odds API free tier may not include extensive historical data.

---

## Troubleshooting

### "Permission denied" when running kaggle command

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### "Dataset not found"

Double-check the dataset path:
```bash
kaggle datasets list -s "nba betting"
```

### "401 Unauthorized"

Your API token may be invalid. Regenerate it:
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" again
3. Replace `~/.kaggle/kaggle.json`

### CSV encoding issues

```bash
# Check file encoding
file data/betting/kaggle/*.csv

# If needed, convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

---

## 📊 Expected Impact

### Current Model (Synthetic Betting Lines)
- AUC: 67.68%
- Betting features: 3.31% importance
- Issue: Redundant with existing features

### With Real Vegas Lines (Expected)
- **AUC: 69-71%** (+1.5-3.5%)
- **Betting features: 15-20% importance**
- Benefits:
  - Market efficiency signal
  - Insider information (injuries, lineups)
  - Sharp money indicators
  - Line movement patterns

---

## 🎯 Next Steps After Download

1. ✅ Download Kaggle dataset
2. Run: `python process_real_vegas_lines.py`
3. Run: `python train_with_betting.py` (retrain)
4. Compare: Synthetic vs Real performance
5. Deploy: Updated model with real lines

---

## 📚 Resources

- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api
- **The Odds API:** https://the-odds-api.com/
- **OddsPortal:** https://www.oddsportal.com/ (manual download)
- **Sports Reference:** https://www.sports-reference.com/

---

## ⏰ Time Estimate

- Kaggle setup: 5-10 minutes
- Find dataset: 10-15 minutes  
- Download: 2-5 minutes
- Process & integrate: 30 minutes (automated)
- **Total: ~1 hour to real Vegas lines!**

**ROI: 1 hour → +2-4% AUC improvement = Excellent!**

---

*Last Updated: November 11, 2025*  
*Status: Ready to set up Kaggle API*

