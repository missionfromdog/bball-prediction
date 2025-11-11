# Instructions to Add Live Odds to Streamlit App

Add this code to your Streamlit app at line ~405 (after the matchup display):

```python
# Import at top of file
from live_odds_display import load_live_odds, match_game_to_odds, format_odds_display

# At start of main() function (after title)
live_odds_df = load_live_odds()
if live_odds_df is not None:
    st.info(f"📊 Live odds loaded for {len(live_odds_df)} games")

# In the game display loop (after st.caption for date):
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

The helper module `src/live_odds_display.py` has been created for you.
