"""
Betting Analysis Module for Expected Value (EV) and Kelly Criterion Calculations

This module provides functions to:
- Calculate implied probabilities from betting odds
- Remove vig (house edge) from book odds
- Calculate edge (model probability vs. fair book probability)
- Calculate expected value (EV)
- Calculate Kelly Criterion optimal bet sizing
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_implied_probability(american_odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        american_odds: American odds (e.g., -150 or +150)
    
    Returns:
        Implied probability as a decimal (0.0 to 1.0)
    
    Examples:
        >>> calculate_implied_probability(-150)
        0.6
        >>> calculate_implied_probability(+150)
        0.4
    """
    if pd.isna(american_odds):
        return np.nan
    
    if american_odds > 0:
        # Underdog: +150 means bet $100 to win $150
        return 100 / (american_odds + 100)
    else:
        # Favorite: -150 means bet $150 to win $100
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(home_prob: float, away_prob: float) -> Tuple[float, float]:
    """
    Remove vig (house edge) to get fair probabilities.
    
    The book's implied probabilities sum to > 1.0 due to vig.
    This function normalizes them to sum to 1.0.
    
    Args:
        home_prob: Book's implied probability for home team (with vig)
        away_prob: Book's implied probability for away team (with vig)
    
    Returns:
        Tuple of (fair_home_prob, fair_away_prob)
    
    Example:
        >>> remove_vig(0.6, 0.45)  # Sum = 1.05 (5% vig)
        (0.5714..., 0.4285...)  # Sum = 1.0
    """
    if pd.isna(home_prob) or pd.isna(away_prob):
        return np.nan, np.nan
    
    total = home_prob + away_prob
    if total == 0:
        return np.nan, np.nan
    
    fair_home = home_prob / total
    fair_away = away_prob / total
    
    return fair_home, fair_away


def calculate_edge(model_prob: float, fair_book_prob: float) -> float:
    """
    Calculate edge percentage (model probability - fair book probability).
    
    Positive edge means the model thinks the outcome is more likely than
    the book's fair probability suggests.
    
    Args:
        model_prob: Model's predicted probability (0.0 to 1.0)
        fair_book_prob: Book's fair probability without vig (0.0 to 1.0)
    
    Returns:
        Edge as a decimal (can be negative)
    
    Example:
        >>> calculate_edge(0.75, 0.60)
        0.15  # 15% edge
    """
    if pd.isna(model_prob) or pd.isna(fair_book_prob):
        return np.nan
    
    return model_prob - fair_book_prob


def calculate_ev(model_prob: float, american_odds: float) -> float:
    """
    Calculate Expected Value (EV) per unit bet.
    
    EV = (Win Probability × Potential Profit) - (Loss Probability × Potential Loss)
    
    Positive EV means the bet is profitable in the long run.
    
    Args:
        model_prob: Model's predicted probability of winning (0.0 to 1.0)
        american_odds: American odds for the bet (e.g., -150 or +150)
    
    Returns:
        Expected value per $1 bet (can be negative)
    
    Example:
        >>> calculate_ev(0.75, -150)  # 75% win prob, -150 odds
        0.25  # +$0.25 per $1 bet
        >>> calculate_ev(0.40, +150)  # 40% win prob, +150 odds
        -0.10  # -$0.10 per $1 bet (negative EV)
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
    
    # Calculate profit per $1 bet
    if american_odds > 0:
        # Underdog: +150 means bet $1 to win $1.50
        profit = american_odds / 100
    else:
        # Favorite: -150 means bet $1.50 to win $1, so profit per $1 bet = 100/150
        profit = 100 / abs(american_odds)
    
    # Loss is always $1 (the bet amount)
    loss = 1.0
    
    # EV calculation
    ev = (model_prob * profit) - ((1 - model_prob) * loss)
    
    return ev


def calculate_kelly_fraction(model_prob: float, american_odds: float) -> float:
    """
    Calculate Kelly Criterion optimal bet fraction.
    
    Formula: f* = (bp - q) / b
    Where:
        b = net odds (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Args:
        model_prob: Model's predicted probability of winning (0.0 to 1.0)
        american_odds: American odds for the bet (e.g., -150 or +150)
    
    Returns:
        Optimal fraction of bankroll to bet (0.0 to 1.0, capped at 0.25 for safety)
    
    Example:
        >>> calculate_kelly_fraction(0.75, -150)
        0.125  # Bet 12.5% of bankroll
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
    
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
    if b == 0:
        return 0.0
    
    kelly = (b * p - q) / b
    
    # Cap at 0 (no negative bets) and 0.25 (conservative max for full Kelly)
    kelly = max(0.0, min(kelly, 0.25))
    
    return kelly


def calculate_bet_size(kelly_fraction: float, bankroll: float) -> float:
    """
    Calculate recommended bet size based on Kelly fraction and bankroll.
    
    Args:
        kelly_fraction: Kelly fraction (0.0 to 1.0)
        bankroll: Total bankroll amount
    
    Returns:
        Recommended bet size in dollars
    """
    if pd.isna(kelly_fraction) or pd.isna(bankroll):
        return np.nan
    
    return kelly_fraction * bankroll


def calculate_default_bankroll(num_games: int, base_bankroll: float = 100.0) -> float:
    """
    Calculate default bankroll based on number of games.
    
    Default: $100 total, ~$10/game, but scales up for max games per day.
    Formula: base_bankroll * max(1, num_games / 10)
    
    Args:
        num_games: Number of games to bet on
        base_bankroll: Base bankroll amount (default $100)
    
    Returns:
        Default bankroll amount
    """
    if num_games <= 0:
        return base_bankroll
    
    # Scale up if more than 10 games, but maintain ~$10/game minimum
    multiplier = max(1.0, num_games / 10.0)
    return base_bankroll * multiplier


def analyze_betting_value(
    model_prob: float,
    home_ml: Optional[float] = None,
    away_ml: Optional[float] = None,
    is_home_bet: bool = True
) -> dict:
    """
    Comprehensive betting value analysis for a single game.
    
    Args:
        model_prob: Model's predicted probability (0.0 to 1.0)
        home_ml: Home team moneyline odds (optional)
        away_ml: Away team moneyline odds (optional)
        is_home_bet: True if betting on home team, False for away team
    
    Returns:
        Dictionary with all betting metrics:
        - model_probability
        - book_implied_prob (with vig)
        - fair_book_prob (vig removed)
        - edge
        - expected_value
        - kelly_fraction
        - has_value (boolean)
    """
    result = {
        'model_probability': model_prob,
        'book_implied_prob': np.nan,
        'fair_book_prob': np.nan,
        'edge': np.nan,
        'expected_value': np.nan,
        'kelly_fraction': np.nan,
        'has_value': False
    }
    
    # Determine which odds to use
    if is_home_bet:
        ml_odds = home_ml
        model_prob_used = model_prob
    else:
        ml_odds = away_ml
        model_prob_used = 1 - model_prob  # Flip probability for away team
    
    if pd.isna(ml_odds):
        return result
    
    # Calculate implied probability (with vig)
    book_implied = calculate_implied_probability(ml_odds)
    result['book_implied_prob'] = book_implied
    
    # Calculate fair probability (vig removed) if we have both moneylines
    if not pd.isna(home_ml) and not pd.isna(away_ml):
        home_implied = calculate_implied_probability(home_ml)
        away_implied = calculate_implied_probability(away_ml)
        fair_home, fair_away = remove_vig(home_implied, away_implied)
        
        if is_home_bet:
            result['fair_book_prob'] = fair_home
        else:
            result['fair_book_prob'] = fair_away
    else:
        # If we only have one moneyline, use it as fair (less accurate)
        result['fair_book_prob'] = book_implied
    
    # Calculate edge
    result['edge'] = calculate_edge(model_prob_used, result['fair_book_prob'])
    
    # Calculate EV
    result['expected_value'] = calculate_ev(model_prob_used, ml_odds)
    
    # Calculate Kelly fraction
    result['kelly_fraction'] = calculate_kelly_fraction(model_prob_used, ml_odds)
    
    # Determine if bet has value (positive EV and positive edge)
    result['has_value'] = (
        result['expected_value'] > 0 and
        result['edge'] > 0 and
        not pd.isna(result['kelly_fraction']) and
        result['kelly_fraction'] > 0
    )
    
    return result

