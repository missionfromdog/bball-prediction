# Expected Value (EV) & Kelly Criterion Betting Features - Proposal

## 📋 Understanding & Current State

### Current App Capabilities
- ✅ Model predictions with win probabilities (`HOME_WIN_PROBABILITY`)
- ✅ Confidence levels (High/Medium/Low) based on distance from 0.5
- ✅ Live Vegas odds integration (moneylines, spreads, totals)
- ✅ Basic edge calculation exists in `make_daily_predictions.py` (line 587)
- ✅ Model performance tracking

### What's Missing (Per Gemini Feedback)
1. **Proper Edge Calculation**: Current edge doesn't account for vig (house edge)
2. **Expected Value (EV) Calculation**: No EV metric shown
3. **Kelly Criterion**: No bet sizing recommendations
4. **Model Calibration**: Need to verify/improve probability calibration
5. **Value Bet Filtering**: No way to filter/highlight high confidence + high value bets

---

## 🎯 Proposed Features

### Feature 1: Enhanced Edge & Value Calculation

**What it does:**
- Converts model probability to calibrated probability (if needed)
- Calculates book's implied probability from moneyline odds
- Removes vig to get "fair" book probability
- Calculates Edge = P_Model - P_Fair_Book
- Calculates Expected Value (EV) = (P_Model × Potential Profit) - (P_Loss × Potential Loss)

**Display:**
- New columns in predictions table:
  - `Model_Probability` (calibrated)
  - `Book_Implied_Prob` (with vig)
  - `Fair_Book_Prob` (vig removed)
  - `Edge_%` (P_Model - P_Fair_Book)
  - `Expected_Value` (in units, e.g., $EV per $100 bet)

**Implementation:**
```python
def calculate_implied_probability(american_odds):
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def remove_vig(home_prob, away_prob):
    """Remove vig to get fair probabilities"""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total

def calculate_edge(model_prob, fair_book_prob):
    """Calculate edge percentage"""
    return model_prob - fair_book_prob

def calculate_ev(model_prob, american_odds):
    """Calculate expected value"""
    if american_odds > 0:
        profit = american_odds / 100  # e.g., +150 = 1.5x profit
    else:
        profit = 100 / abs(american_odds)  # e.g., -150 = 0.667x profit
    
    loss = 1.0  # Always lose 1 unit if bet loses
    ev = (model_prob * profit) - ((1 - model_prob) * loss)
    return ev
```

---

### Feature 2: Kelly Criterion Bet Sizing

**What it does:**
- Calculates optimal bet size as fraction of bankroll
- Uses formula: `f* = (bp - q) / b`
  - `b` = net odds (decimal odds - 1)
  - `p` = model probability of winning
  - `q` = 1 - p (probability of losing)
- Provides conservative (fractional Kelly) and aggressive (full Kelly) options
- Shows recommended bet amount for given bankroll

**Display:**
- New columns:
  - `Kelly_Fraction` (optimal % of bankroll)
  - `Recommended_Bet_$1000` (bet size for $1000 bankroll)
  - `Kelly_Warning` (if Kelly > 0.25, suggest fractional Kelly)

**Implementation:**
```python
def calculate_kelly_fraction(model_prob, american_odds):
    """Calculate Kelly Criterion optimal bet fraction"""
    # Convert American odds to decimal odds
    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1
    
    # Net odds (b in Kelly formula)
    b = decimal_odds - 1
    
    # Probabilities
    p = model_prob
    q = 1 - p
    
    # Kelly fraction
    kelly = (b * p - q) / b
    
    # Cap at 0 (no negative bets) and 0.25 (conservative max)
    kelly = max(0, min(kelly, 0.25))
    
    return kelly

def calculate_bet_size(kelly_fraction, bankroll):
    """Calculate recommended bet size"""
    return kelly_fraction * bankroll
```

---

### Feature 3: Value Bet Filtering & Highlighting

**What it does:**
- Filters games by criteria:
  - High Confidence: Model probability > 70% (or user-configurable)
  - Positive Edge: Edge > 5% (or user-configurable)
  - Positive EV: Expected Value > 0
- Highlights value bets in the UI
- Adds a "Value Bets" tab/section

**Display:**
- New section: "🎯 Value Bets" with filtered games
- Color coding:
  - 🟢 Green: High confidence + High edge (>10%)
  - 🟡 Yellow: Medium confidence + Medium edge (5-10%)
  - ⚪ Gray: Low edge or negative EV

---

### Feature 4: Model Calibration Check

**What it does:**
- Analyzes historical predictions vs. actual outcomes
- Creates calibration curve
- Adjusts probabilities if model is miscalibrated
- Shows calibration metrics in Performance tab

**Implementation:**
- Use `sklearn.calibration.CalibrationDisplay`
- Calculate Brier Score
- Apply Platt scaling or isotonic regression if needed

---

## 🏗️ Implementation Approach

### Phase 1: Core EV Calculations (Week 1)
1. Create `src/betting_analysis.py` module with:
   - `calculate_implied_probability()`
   - `remove_vig()`
   - `calculate_edge()`
   - `calculate_ev()`
2. Integrate into `streamlit_app_enhanced.py`
3. Add new columns to predictions display
4. Test with current data

### Phase 2: Kelly Criterion (Week 1-2)
1. Add `calculate_kelly_fraction()` to `betting_analysis.py`
2. Add bet sizing columns to predictions
3. Add bankroll input in sidebar
4. Display recommended bet sizes

### Phase 3: Value Bet Filtering (Week 2)
1. Add filtering logic
2. Create "Value Bets" section in UI
3. Add color coding
4. Add user-configurable thresholds

### Phase 4: Calibration (Week 2-3)
1. Add calibration analysis to Performance tab
2. Implement calibration adjustment if needed
3. Re-train model with calibrated probabilities if necessary

---

## ✅ Feasibility Assessment

### ✅ Highly Feasible
- **Edge & EV Calculations**: Straightforward math, already have odds data
- **Kelly Criterion**: Simple formula, easy to implement
- **Value Bet Filtering**: Basic filtering logic
- **UI Integration**: Streamlit makes this easy

### ⚠️ Moderate Complexity
- **Vig Removal**: Need to handle cases where both moneylines aren't available
- **Model Calibration**: Requires historical prediction data analysis
- **Edge Cases**: Handle missing odds, invalid probabilities, etc.

### 🔴 Potential Challenges
- **Model Calibration**: Current model may not be well-calibrated (needs verification)
- **Odds Availability**: Not all games may have moneylines available
- **Kelly Criterion Limitations**: Assumes infinite bankroll, may need fractional Kelly

---

## 📁 Version Control Strategy

### Option A: Git Branch (Recommended)
```bash
# Save current working version
git checkout -b ev-betting-features
git push origin ev-betting-features

# Continue development on new branch
# Main branch remains stable
```

### Option B: Tag Current Version
```bash
# Tag current stable version
git tag -a v1.0-stable -m "Stable version before EV features"
git push origin v1.0-stable

# Continue on main branch
```

### Option C: Backup Directory
```bash
# Copy entire app to backup
cp -r src/ src_backup_v1.0/
cp -r data/ data_backup_v1.0/
```

**Recommendation**: Use **Option A (Git Branch)** for clean version control.

---

## 🎨 UI/UX Design

### New Sections in Streamlit App:

1. **Sidebar Additions:**
   - Bankroll input ($)
   - Confidence threshold slider (50-90%)
   - Edge threshold slider (0-20%)
   - Kelly fraction multiplier (0.25x, 0.5x, 1.0x)

2. **Main Predictions Table:**
   - Add columns: Edge%, EV, Kelly%, Bet Size
   - Color-code rows by value
   - Sort by EV (descending) option

3. **New "Value Bets" Tab:**
   - Filtered list of high-value bets
   - Summary statistics
   - Export to CSV

4. **Performance Tab Enhancements:**
   - EV tracking over time
   - Kelly bet sizing performance
   - Calibration curve

---

## 📊 Example Output

```
Matchup: LAL @ BOS
Model Probability: 75.0%
Book ML: -150 (Implied: 60.0%)
Fair Book Prob: 58.8% (vig removed)
Edge: +16.2% ✅
Expected Value: +$0.25 per $1 bet
Kelly Fraction: 12.5%
Recommended Bet ($1000 bankroll): $125
```

---

## 🚀 Next Steps

1. **Confirm Understanding**: Review this proposal
2. **Save Current Version**: Create git branch or tag
3. **Start Implementation**: Begin with Phase 1 (Core EV Calculations)
4. **Iterate**: Test and refine based on results

---

## ❓ Questions for Clarification

1. **Bankroll**: ✅ Default $100 total (~$10/game), scalable for max games per day, user can override
2. **Kelly Fraction**: ✅ Full Kelly (not fractional)
3. **Calibration**: Should we implement automatic calibration, or just show metrics?
4. **Edge Cases**: How to handle games without moneylines?
5. **Historical Tracking**: Should we track EV performance over time?

