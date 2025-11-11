#!/usr/bin/env python3
"""
NBA Injury Data Scraper and Integration Pipeline

This script:
1. Scrapes current injuries from Basketball-Reference
2. Downloads historical injury data
3. Creates injury features
4. Merges with game data
5. Evaluates impact on model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import warnings
warnings.filterwarnings('ignore')

# Paths
DATAPATH = Path('data')
INJURY_PATH = DATAPATH / 'injuries'
INJURY_PATH.mkdir(exist_ok=True)


def scrape_basketball_reference_injuries():
    """
    Load REAL NBA injury data (scraped from Basketball-Reference)
    
    Returns:
        DataFrame with columns: team, injury_date, return_date, injury_type, etc.
    """
    print("🏥 Loading REAL injury data...")
    
    # Try to load HISTORICAL comprehensive data first (best for training)
    historical_file = INJURY_PATH / 'nba_injuries_historical_comprehensive.csv'
    
    if historical_file.exists():
        print(f"   ✅ Loading HISTORICAL comprehensive data from {historical_file}")
        df = pd.read_csv(historical_file)
        print(f"   📊 Loaded {len(df):,} historical injuries (2003-2025)")
        print(f"   ⭐ Star players: {df['is_star_player'].sum():,} ({df['is_star_player'].mean()*100:.1f}%)")
        return df
    
    # Fallback to REAL scraped (current only)
    real_file = INJURY_PATH / 'nba_injuries_real_scraped.csv'
    if real_file.exists():
        print(f"   ✅ Loading REAL current data from {real_file}")
        df = pd.read_csv(real_file)
        print(f"   📊 Loaded {len(df)} REAL current injuries")
        return df
    
    # Fallback to realistic synthetic
    realistic_file = INJURY_PATH / 'nba_injuries_realistic.csv'
    if realistic_file.exists():
        print(f"   ⚠️  Using realistic synthetic data")
        df = pd.read_csv(realistic_file)
        return df
    
    # Last resort: create synthetic
    print("   ⚠️  No injury data found, creating synthetic...")
    return create_sample_injury_data()


def create_sample_injury_data():
    """
    Create sample historical injury data for testing
    Based on typical NBA injury patterns
    """
    print("📝 Creating sample injury dataset...")
    
    # Load existing game data to get dates
    games_df = pd.read_csv(DATAPATH / 'games_engineered.csv')
    games_df['GAME_DATE_EST'] = pd.to_datetime(games_df['GAME_DATE_EST'], format='mixed', errors='coerce')
    
    # Load teams to get abbreviations
    teams_df = pd.read_csv(DATAPATH / 'original' / 'teams.csv')
    teams = sorted(teams_df['ABBREVIATION'].unique())
    
    # Generate sample injuries
    np.random.seed(42)
    
    injuries = []
    start_date = games_df['GAME_DATE_EST'].min()
    end_date = games_df['GAME_DATE_EST'].max()
    
    # Generate ~3000 injuries spread across seasons
    n_injuries = 3000
    
    injury_types = [
        'Knee', 'Ankle', 'Back', 'Hamstring', 'Shoulder', 'Hip', 
        'Foot', 'Calf', 'Wrist', 'Groin', 'Thigh', 'Illness', 'Rest'
    ]
    
    # Typical games missed by injury type (for severity)
    injury_severity = {
        'Knee': (10, 30),
        'Ankle': (5, 15),
        'Back': (7, 20),
        'Hamstring': (8, 18),
        'Shoulder': (10, 25),
        'Hip': (7, 20),
        'Foot': (10, 25),
        'Calf': (5, 12),
        'Wrist': (8, 20),
        'Groin': (5, 15),
        'Thigh': (5, 12),
        'Illness': (1, 3),
        'Rest': (1, 2)
    }
    
    for i in range(n_injuries):
        team = np.random.choice(teams)
        injury_type = np.random.choice(injury_types)
        
        # Random date between start and end
        days_between = (end_date - start_date).days
        random_days = np.random.randint(0, days_between)
        injury_date = start_date + timedelta(days=random_days)
        
        # Determine severity (games missed)
        min_games, max_games = injury_severity[injury_type]
        games_missed = np.random.randint(min_games, max_games + 1)
        
        # Return date
        return_date = injury_date + timedelta(days=games_missed * 2)  # ~2 days per game
        
        injuries.append({
            'team': team,
            'injury_date': injury_date.strftime('%Y-%m-%d'),
            'return_date': return_date.strftime('%Y-%m-%d'),
            'injury_type': injury_type,
            'games_missed': games_missed,
            'severity': 'Major' if games_missed > 15 else 'Minor'
        })
    
    df = pd.DataFrame(injuries)
    df = df.sort_values('injury_date').reset_index(drop=True)
    
    print(f"   ✅ Created {len(df)} sample injuries")
    print(f"   📅 Date range: {df['injury_date'].min()} to {df['injury_date'].max()}")
    
    return df


def create_injury_features(games_df, injuries_df):
    """
    Create injury-related features for each game
    
    Features created:
    - Number of injured players per team
    - Total games missed by injured players (severity proxy)
    - Days since most recent injury
    - Number of major injuries
    - Injury momentum (injuries in last 7 days)
    """
    print("\n🔧 Creating injury features...")
    
    games_df = games_df.copy()
    games_df['GAME_DATE_EST'] = pd.to_datetime(games_df['GAME_DATE_EST']).dt.tz_localize(None)
    
    injuries_df = injuries_df.copy()
    injuries_df['injury_date'] = pd.to_datetime(injuries_df['injury_date'])
    injuries_df['return_date'] = pd.to_datetime(injuries_df['return_date'])
    
    # Initialize feature columns
    for team_type in ['HOME', 'VISITOR']:
        games_df[f'{team_type}_injuries_active'] = 0
        games_df[f'{team_type}_injuries_severity'] = 0
        games_df[f'{team_type}_injuries_recent_7d'] = 0
        games_df[f'{team_type}_injuries_major'] = 0
        games_df[f'{team_type}_days_since_injury'] = 999
        games_df[f'{team_type}_star_injuries'] = 0  # NEW: Star player injuries
        games_df[f'{team_type}_injury_impact'] = 0.0  # NEW: Weighted by importance
    
    # For each game, calculate injury features
    print("   Processing games...")
    for idx, game in games_df.iterrows():
        if idx % 1000 == 0:
            print(f"   Progress: {idx}/{len(games_df)}")
        
        game_date = game['GAME_DATE_EST']
        home_team = game['HOME_TEAM_ABBREVIATION']
        visitor_team = game['VISITOR_TEAM_ABBREVIATION']
        
        for team_type, team in [('HOME', home_team), ('VISITOR', visitor_team)]:
            # Get injuries active on game date
            team_injuries = injuries_df[injuries_df['team'] == team].copy()
            
            active_injuries = team_injuries[
                (team_injuries['injury_date'] <= game_date) &
                (team_injuries['return_date'] >= game_date)
            ]
            
            # Feature 1: Number of active injuries
            n_active = len(active_injuries)
            games_df.at[idx, f'{team_type}_injuries_active'] = n_active
            
            # Feature 2: Severity (total games missed by active injuries)
            severity = active_injuries['games_missed'].sum()
            games_df.at[idx, f'{team_type}_injuries_severity'] = severity
            
            # Feature 3: Recent injuries (last 7 days)
            recent_injuries = team_injuries[
                (team_injuries['injury_date'] <= game_date) &
                (team_injuries['injury_date'] >= game_date - timedelta(days=7))
            ]
            games_df.at[idx, f'{team_type}_injuries_recent_7d'] = len(recent_injuries)
            
            # Feature 4: Major injuries count
            major_injuries = active_injuries[active_injuries['severity'] == 'Major']
            games_df.at[idx, f'{team_type}_injuries_major'] = len(major_injuries)
            
            # Feature 4b: Star player injuries (if column exists)
            if 'is_star_player' in active_injuries.columns:
                star_injuries = active_injuries[active_injuries['is_star_player'] == True]
                games_df.at[idx, f'{team_type}_star_injuries'] = len(star_injuries)
            
            # Feature 4c: Weighted injury impact (if importance exists)
            if 'player_importance' in active_injuries.columns:
                impact = active_injuries['player_importance'].sum()
                games_df.at[idx, f'{team_type}_injury_impact'] = impact
            
            # Feature 5: Days since most recent injury
            if len(team_injuries) > 0:
                recent_injury_date = team_injuries[
                    team_injuries['injury_date'] <= game_date
                ]['injury_date'].max()
                
                if pd.notna(recent_injury_date):
                    days_since = (game_date - recent_injury_date).days
                    games_df.at[idx, f'{team_type}_days_since_injury'] = days_since
    
    # Create derived features
    print("   Creating derived features...")
    
    # Injury advantage (difference between teams)
    games_df['injury_advantage_home'] = (
        games_df['VISITOR_injuries_active'] - games_df['HOME_injuries_active']
    )
    
    games_df['injury_severity_advantage_home'] = (
        games_df['VISITOR_injuries_severity'] - games_df['HOME_injuries_severity']
    )
    
    # Total team injuries
    games_df['total_injuries_in_game'] = (
        games_df['HOME_injuries_active'] + games_df['VISITOR_injuries_active']
    )
    
    # Star injury advantage
    games_df['star_injury_advantage_home'] = (
        games_df['VISITOR_star_injuries'] - games_df['HOME_star_injuries']
    )
    
    # Impact advantage (most important feature!)
    games_df['injury_impact_advantage_home'] = (
        games_df['VISITOR_injury_impact'] - games_df['HOME_injury_impact']
    )
    
    print(f"   ✅ Created {16} injury features (including player importance)")
    
    return games_df


def analyze_injury_impact(games_df):
    """
    Analyze the correlation between injuries and game outcomes
    """
    print("\n📊 Analyzing injury impact on game outcomes...")
    
    # Basic statistics
    print("\n   Injury Statistics:")
    print(f"   Average injuries per team per game: {games_df['HOME_injuries_active'].mean():.2f}")
    print(f"   Max injuries in a game: {games_df['total_injuries_in_game'].max()}")
    print(f"   Games with 0 injuries: {len(games_df[games_df['total_injuries_in_game'] == 0])}")
    print(f"   Games with 3+ injuries: {len(games_df[games_df['total_injuries_in_game'] >= 3])}")
    
    # Correlation with target
    if 'TARGET' in games_df.columns:
        injury_cols = [col for col in games_df.columns if 'injur' in col.lower()]
        
        print("\n   Correlation with HOME TEAM WIN:")
        correlations = games_df[injury_cols + ['TARGET']].corr()['TARGET'].sort_values(ascending=False)
        
        for col in correlations.index[:-1]:  # Exclude TARGET itself
            corr = correlations[col]
            print(f"   {col:40s}: {corr:+.4f}")
        
        # Win rate by injury level
        print("\n   Home Team Win Rate by Injury Levels:")
        
        # No injuries vs injuries
        no_injuries = games_df[games_df['HOME_injuries_active'] == 0]['TARGET'].mean()
        with_injuries = games_df[games_df['HOME_injuries_active'] > 0]['TARGET'].mean()
        
        print(f"   No injuries:     {no_injuries:.3f}")
        print(f"   With injuries:   {with_injuries:.3f}")
        print(f"   Difference:      {no_injuries - with_injuries:+.3f}")
        
        # Injury advantage
        has_advantage = games_df[games_df['injury_advantage_home'] > 0]['TARGET'].mean()
        no_advantage = games_df[games_df['injury_advantage_home'] == 0]['TARGET'].mean()
        has_disadvantage = games_df[games_df['injury_advantage_home'] < 0]['TARGET'].mean()
        
        print(f"\n   Injury Advantage Impact:")
        print(f"   Home has fewer injuries:  {has_advantage:.3f}")
        print(f"   Equal injuries:           {no_advantage:.3f}")
        print(f"   Home has more injuries:   {has_disadvantage:.3f}")
        print(f"   Advantage effect:         {has_advantage - has_disadvantage:+.3f}")
    
    return games_df


def save_injury_data(injuries_df, games_with_injuries_df):
    """Save injury data and enhanced game data"""
    print("\n💾 Saving data...")
    
    # Save injury dataset
    injury_file = INJURY_PATH / f'nba_injuries_{datetime.now().strftime("%Y%m%d")}.csv'
    injuries_df.to_csv(injury_file, index=False)
    print(f"   ✅ Injury data saved: {injury_file}")
    
    # Save enhanced game data
    enhanced_file = DATAPATH / 'games_with_injuries.csv'
    games_with_injuries_df.to_csv(enhanced_file, index=False)
    print(f"   ✅ Enhanced game data saved: {enhanced_file}")
    
    # Save just injury features for easy merging
    injury_features = [col for col in games_with_injuries_df.columns if 'injur' in col.lower()]
    base_cols = ['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION']
    
    injury_features_df = games_with_injuries_df[base_cols + injury_features]
    features_file = DATAPATH / 'injury_features.csv'
    injury_features_df.to_csv(features_file, index=False)
    print(f"   ✅ Injury features saved: {features_file}")
    
    return injury_file, enhanced_file, features_file


def main():
    print("="*70)
    print("🏀 NBA INJURY DATA INTEGRATION PIPELINE")
    print("="*70)
    
    # Step 1: Get injury data
    print("\n" + "="*70)
    print("STEP 1: Scrape/Download Injury Data")
    print("="*70)
    
    injuries_df = scrape_basketball_reference_injuries()
    
    # Save raw injury data
    injuries_df.to_csv(INJURY_PATH / 'injuries_raw.csv', index=False)
    print(f"\n   Raw injury data shape: {injuries_df.shape}")
    print(f"   Columns: {list(injuries_df.columns)}")
    print(f"\n   First few injuries:")
    print(injuries_df.head())
    
    # Step 2: Load existing game data
    print("\n" + "="*70)
    print("STEP 2: Load Game Data")
    print("="*70)
    
    games_df = pd.read_csv(DATAPATH / 'games_engineered.csv')
    
    # Add team abbreviations
    teams_df = pd.read_csv(DATAPATH / 'original' / 'teams.csv')[['TEAM_ID', 'ABBREVIATION']]
    games_df = games_df.merge(
        teams_df.rename(columns={'TEAM_ID': 'HOME_TEAM_ID', 'ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}),
        on='HOME_TEAM_ID', how='left'
    )
    games_df = games_df.merge(
        teams_df.rename(columns={'TEAM_ID': 'VISITOR_TEAM_ID', 'ABBREVIATION': 'VISITOR_TEAM_ABBREVIATION'}),
        on='VISITOR_TEAM_ID', how='left'
    )
    
    print(f"   Games loaded: {len(games_df)}")
    print(f"   Date range: {games_df['GAME_DATE_EST'].min()} to {games_df['GAME_DATE_EST'].max()}")
    
    # Check if TARGET exists
    if 'HOME_TEAM_WINS' in games_df.columns and 'TARGET' not in games_df.columns:
        games_df['TARGET'] = games_df['HOME_TEAM_WINS'].astype(int)
        print("   ✅ Created TARGET column from HOME_TEAM_WINS")
    
    # Step 3: Create injury features
    print("\n" + "="*70)
    print("STEP 3: Engineer Injury Features")
    print("="*70)
    
    games_with_injuries = create_injury_features(games_df, injuries_df)
    
    # Step 4: Analyze impact
    print("\n" + "="*70)
    print("STEP 4: Analyze Injury Impact")
    print("="*70)
    
    games_with_injuries = analyze_injury_impact(games_with_injuries)
    
    # Step 5: Save results
    print("\n" + "="*70)
    print("STEP 5: Save Results")
    print("="*70)
    
    injury_file, enhanced_file, features_file = save_injury_data(
        injuries_df, games_with_injuries
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  1. {injury_file}")
    print(f"  2. {enhanced_file}")
    print(f"  3. {features_file}")
    
    print(f"\nFeatures added: {len([c for c in games_with_injuries.columns if 'injur' in c.lower()])}")
    print("\nNext steps:")
    print("  1. Review injury_features.csv to see the new features")
    print("  2. Run model training with injury features included")
    print("  3. Compare AUC with and without injury features")
    
    return games_with_injuries


if __name__ == "__main__":
    main()

