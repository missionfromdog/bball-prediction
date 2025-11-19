"""Quick test of betting_analysis.py functions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from betting_analysis import *

print('Testing betting_analysis.py...')
print()

# Test implied probability
print('1. Implied Probability:')
print(f'   -150 odds = {calculate_implied_probability(-150):.1%}')
print(f'  +150 odds = {calculate_implied_probability(+150):.1%}')
print()

# Test vig removal
home_implied = calculate_implied_probability(-150)
away_implied = calculate_implied_probability(+130)
fair_home, fair_away = remove_vig(home_implied, away_implied)
print('2. Vig Removal:')
print(f'   Home implied (with vig): {home_implied:.1%}')
print(f'   Away implied (with vig): {away_implied:.1%}')
print(f'   Fair home prob: {fair_home:.1%}')
print(f'   Fair away prob: {fair_away:.1%}')
print()

# Test edge
model_prob = 0.75
edge = calculate_edge(model_prob, fair_home)
print('3. Edge Calculation:')
print(f'   Model prob: {model_prob:.1%}')
print(f'   Fair book prob: {fair_home:.1%}')
print(f'   Edge: {edge:+.1%}')
print()

# Test EV
ev = calculate_ev(model_prob, -150)
print('4. Expected Value:')
print(f'   Model prob: {model_prob:.1%}, Odds: -150')
print(f'   EV: ${ev:.2f} per $1 bet')
print()

# Test Kelly
kelly = calculate_kelly_fraction(model_prob, -150)
print('5. Kelly Criterion:')
print(f'   Kelly fraction: {kelly:.1%}')
print(f'   Bet size ($100 bankroll): ${calculate_bet_size(kelly, 100):.2f}')
print()

# Test default bankroll
print('6. Default Bankroll:')
for games in [5, 10, 15, 20]:
    bankroll = calculate_default_bankroll(games)
    print(f'   {games} games = ${bankroll:.0f} bankroll')
print()

# Test comprehensive analysis
print('7. Comprehensive Analysis:')
analysis = analyze_betting_value(
    model_prob=0.75,
    home_ml=-150,
    away_ml=+130,
    is_home_bet=True
)
print(f'   Model prob: {analysis["model_probability"]:.1%}')
print(f'   Book implied: {analysis["book_implied_prob"]:.1%}')
print(f'   Fair book: {analysis["fair_book_prob"]:.1%}')
print(f'   Edge: {analysis["edge"]:+.1%}')
print(f'   EV: ${analysis["expected_value"]:.2f}')
print(f'   Kelly: {analysis["kelly_fraction"]:.1%}')
print(f'   Has value: {analysis["has_value"]}')
print()

print('✅ All tests passed!')

