# Odds Comparison Table Metrics Explained

This document explains each metric displayed in the Odds Comparison table in the Streamlit app.

## Table Columns

### Sportsbook
**What it is:** The name of the sportsbook/bookmaker (e.g., DraftKings, FanDuel, BetMGM)

**Why it matters:** Different sportsbooks offer different odds, so comparing across multiple books helps you find the best value.

---

### ML Vig (Moneyline Vig)
**What it is:** The bookmaker's margin/overround for the moneyline bet, expressed as a percentage.

**How it's calculated:** 
```
Vig = (Home Implied Probability + Away Implied Probability - 1.0) × 100
```

**Example:** If home team has -150 odds (implied prob: 60%) and away team has +130 odds (implied prob: 43.48%), the vig is:
- Vig = (0.60 + 0.4348 - 1.0) × 100 = **3.48%**

**What it means:**
- **Lower is better** - Lower vig means the sportsbook is taking a smaller margin
- Typical vig ranges from 2-5% for major US sportsbooks
- A vig of 4.5% means the bookmaker is taking a 4.5% margin on that market

**Why it matters:** Lower vig means better odds for bettors. Finding the sportsbook with the lowest vig for a given market gives you the best value.

---

### Home ML (Home Moneyline)
**What it is:** The American odds for the home team to win the game outright.

**Format:** 
- **Negative numbers** (e.g., -150) = Favorite - You must bet $150 to win $100
- **Positive numbers** (e.g., +130) = Underdog - You bet $100 to win $130

**Example:** 
- `-150` = Home team is favored. Bet $150 to win $100 (total return: $250)
- `+130` = Home team is underdog. Bet $100 to win $130 (total return: $230)

**What it means:**
- More negative = Stronger favorite (e.g., -500 is a heavy favorite)
- More positive = Bigger underdog (e.g., +500 is a heavy underdog)
- Closer to 0 = More even matchup (e.g., -110 vs +100)

---

### Away ML (Away Moneyline)
**What it is:** The American odds for the away team to win the game outright.

**Format:** Same as Home ML (negative = favorite, positive = underdog)

**Example:**
- If Home ML is `-150`, Away ML is typically around `+130` (the opposite side of the same bet)

**Note:** Home ML and Away ML are always opposite sides of the same bet. If you bet on one, you're betting against the other.

---

### Spread Vig (Point Spread Vig)
**What it is:** The bookmaker's margin for the point spread bet, expressed as a percentage.

**How it's calculated:** Same formula as ML Vig, but using the spread odds (typically -110 on both sides).

**Example:** If both sides of the spread are -110:
- Home spread implied prob: 52.38%
- Away spread implied prob: 52.38%
- Vig = (0.5238 + 0.5238 - 1.0) × 100 = **4.76%**

**What it means:**
- **Lower is better** - Lower vig means better value on spread bets
- Spread vig is typically higher than ML vig (often 4-5%)
- Some books offer -105 or even -108 on spreads, which reduces the vig

**Why it matters:** Finding the lowest spread vig can save you money, especially if you bet spreads frequently.

---

### Home Spread
**What it is:** The point spread for the home team, expressed as a number with a sign.

**Format:**
- **Negative number** (e.g., -5.5) = Home team must win by MORE than this many points
- **Positive number** (e.g., +5.5) = Home team can lose by UP TO this many points and still cover

**Example:**
- `-5.5` = Home team is favored by 5.5 points. They must win by 6+ points for the bet to win.
- `+5.5` = Home team is getting 5.5 points. They can lose by 5 or less (or win) for the bet to win.

**What it means:**
- The spread is designed to make both sides of the bet equally attractive
- The number represents the margin the home team is expected to win/lose by
- The `.5` (half-point) prevents ties (pushes)

**Note:** The spread odds are typically -110 on both sides, meaning you bet $110 to win $100.

---

### Total Vig (Over/Under Vig)
**What it is:** The bookmaker's margin for the total (over/under) bet, expressed as a percentage.

**How it's calculated:** Same formula as ML Vig, but using the over/under odds.

**Example:** If over is -110 and under is -110:
- Over implied prob: 52.38%
- Under implied prob: 52.38%
- Vig = (0.5238 + 0.5238 - 1.0) × 100 = **4.76%**

**What it means:**
- **Lower is better** - Lower vig means better value on totals
- Total vig is typically similar to spread vig (4-5%)

---

### Total O/U (Total Over/Under)
**What it is:** The total combined points expected in the game (over/under line).

**Format:** A single number representing the total points (e.g., 225.5)

**Example:**
- `225.5` = The total combined points scored by both teams
- **Over 225.5:** Bet wins if total points > 225.5 (i.e., 226 or more)
- **Under 225.5:** Bet wins if total points < 225.5 (i.e., 225 or less)

**What it means:**
- The `.5` (half-point) prevents ties (pushes)
- This is the bookmaker's prediction of the total scoring in the game
- Both over and under typically have -110 odds

**Note:** The over/under odds are usually -110 on both sides, meaning you bet $110 to win $100.

---

## Summary: What to Look For

### Best Value Indicators:
1. **Lowest Vig** - Across all three markets (ML, Spread, Total), the book with the lowest vig offers the best odds
2. **Consistent Low Vig** - A book that consistently has low vig across multiple markets is generally better value
3. **Best Individual Odds** - Even if vig is higher, sometimes one side of a bet has better odds at a specific book

### Example Comparison:
```
Sportsbook    ML Vig  Spread Vig  Total Vig
DraftKings    4.2%    4.8%        4.9%
FanDuel       3.8%    4.5%        4.6%    ← Best overall (lowest vigs)
BetMGM        4.5%    5.0%        5.1%
```

In this example, **FanDuel** offers the best value across all markets.

---

## Quick Reference

| Metric | What It Is | Lower = Better? | Typical Range |
|--------|------------|-----------------|---------------|
| **ML Vig** | Bookmaker margin on moneyline | ✅ Yes | 2-5% |
| **Home ML** | Odds for home team to win | Depends on bet | -500 to +500 |
| **Away ML** | Odds for away team to win | Depends on bet | -500 to +500 |
| **Spread Vig** | Bookmaker margin on spread | ✅ Yes | 4-5% |
| **Home Spread** | Point spread for home team | N/A | -15.5 to +15.5 |
| **Total Vig** | Bookmaker margin on over/under | ✅ Yes | 4-5% |
| **Total O/U** | Total points line | N/A | 200-250 |

---

## Tips for Using the Table

1. **Compare Vig First** - Start by finding the book with the lowest vig for the market you want to bet
2. **Check Multiple Markets** - A book might have low ML vig but high spread vig (or vice versa)
3. **Look for Consistency** - Books that consistently offer low vig are generally better long-term value
4. **Consider Your Bet Type** - If you only bet moneylines, focus on ML Vig. If you bet spreads, focus on Spread Vig.
5. **Account for Limits** - Some books may have lower vig but also lower betting limits

