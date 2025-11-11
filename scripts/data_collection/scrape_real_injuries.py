#!/usr/bin/env python3
"""
Real NBA Injury Scraper using Selenium

Scrapes current injuries from Basketball-Reference and processes them
into features for the prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

DATAPATH = Path('data')
INJURY_PATH = DATAPATH / 'injuries'
INJURY_PATH.mkdir(exist_ok=True)


def scrape_basketball_reference_selenium():
    """
    Scrape Basketball-Reference injury report using Selenium
    
    Returns:
        DataFrame with real injury data including player names and descriptions
    """
    print("🏥 Scraping Basketball-Reference with Selenium...")
    print("   (This may take 10-15 seconds)")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run without opening browser window
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    
    try:
        # Initialize Chrome driver
        print("   🌐 Starting Chrome driver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Load the injury page
        url = "https://www.basketball-reference.com/friv/injuries.fcgi"
        print(f"   📡 Loading {url}...")
        driver.get(url)
        
        # Wait for the table to load
        time.sleep(3)
        
        # Get page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find the injuries table by ID
        table = soup.find('table', id='injuries')
        
        if not table:
            print("   ⚠️  Could not find injuries table with id='injuries'")
            return pd.DataFrame()
        
        injuries = []
        
        # Get all rows except header
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            # Find player cell (th with data-stat="player")
            player_th = row.find('th', {'data-stat': 'player'})
            if not player_th:
                continue
            
            # Get player info
            player_link = player_th.find('a')
            if not player_link:
                continue
                
            player_name = player_link.text.strip()
            player_id = player_th.get('data-append-csv', '')
            
            # Get team
            team_td = row.find('td', {'data-stat': 'team_name'})
            team_link = team_td.find('a') if team_td else None
            team = team_link.text.strip() if team_link else ''
            
            # Get update date
            date_td = row.find('td', {'data-stat': 'date_update'})
            update_date = date_td.text.strip() if date_td else ''
            
            # Get description
            note_td = row.find('td', {'data-stat': 'note'})
            description = note_td.text.strip() if note_td else ''
            
            injuries.append({
                'player_name': player_name,
                'player_id': player_id,
                'team': team,
                'update_date': update_date,
                'description': description,
                'scrape_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if len(injuries) > 0:
            df = pd.DataFrame(injuries)
            print(f"   ✅ Successfully scraped {len(df)} real injuries!")
            print(f"\n   Sample injuries:")
            if len(df) > 0:
                for idx, inj in df.head(5).iterrows():
                    print(f"      - {inj['player_name']} ({inj['team']}): {inj['description'][:50]}...")
            return df
        else:
            print("   ⚠️  No injuries found in table")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"   ❌ Error during scraping: {e}")
        print(f"   Error type: {type(e).__name__}")
        return pd.DataFrame()
        
    finally:
        if driver:
            driver.quit()
            print("   🔒 Chrome driver closed")


def extract_injury_info(description):
    """
    Parse injury description to extract structured information
    
    Example: "Out (Knee) - Will be re-evaluated in 4 weeks"
    """
    info = {
        'status': 'Unknown',
        'injury_type': 'Unknown',
        'severity': 'Minor',
        'estimated_games_out': 0
    }
    
    # Extract status
    if 'Out' in description:
        info['status'] = 'Out'
    elif 'Day To Day' in description or 'Questionable' in description:
        info['status'] = 'Day To Day'
    elif 'Probable' in description:
        info['status'] = 'Probable'
    elif 'Out For Season' in description:
        info['status'] = 'Out For Season'
        info['severity'] = 'Major'
    
    # Extract injury type (in parentheses)
    if '(' in description and ')' in description:
        start = description.find('(')
        end = description.find(')')
        info['injury_type'] = description[start+1:end].strip()
    
    # Estimate games out based on description
    desc_lower = description.lower()
    
    if 'out for season' in desc_lower or 'season-ending' in desc_lower:
        info['estimated_games_out'] = 82
        info['severity'] = 'Major'
    elif 'weeks' in desc_lower:
        # Extract number of weeks
        import re
        weeks = re.findall(r'(\d+)[\s-]week', desc_lower)
        if weeks:
            num_weeks = int(weeks[0])
            info['estimated_games_out'] = num_weeks * 3  # ~3 games per week
            info['severity'] = 'Major' if num_weeks >= 4 else 'Minor'
    elif 'month' in desc_lower:
        import re
        months = re.findall(r'(\d+)[\s-]month', desc_lower)
        if months:
            num_months = int(months[0])
            info['estimated_games_out'] = num_months * 12
            info['severity'] = 'Major'
    elif info['status'] == 'Day To Day':
        info['estimated_games_out'] = 2
        info['severity'] = 'Minor'
    elif info['status'] == 'Out':
        info['estimated_games_out'] = 7  # Default for "Out"
        info['severity'] = 'Minor'
    
    return info


def enhance_with_player_data(injuries_df):
    """
    Add player importance metrics from existing player data
    """
    print("\n🎯 Enhancing with player statistics...")
    
    try:
        # Load player data
        players_df = pd.read_csv(DATAPATH / 'original' / 'players.csv')
        
        # For now, mark random players as "stars" (would need actual stats)
        # In a real implementation, you'd join with season stats for PPG, All-Star status, etc.
        
        np.random.seed(42)
        injuries_df['is_star_player'] = np.random.random(len(injuries_df)) < 0.30
        injuries_df['player_importance'] = injuries_df['is_star_player'].apply(
            lambda x: np.random.uniform(0.7, 1.0) if x else np.random.uniform(0.1, 0.5)
        )
        
        print(f"   ✅ Added player importance scores")
        print(f"   ⭐ Star players: {injuries_df['is_star_player'].sum()} ({injuries_df['is_star_player'].mean()*100:.1f}%)")
        
    except Exception as e:
        print(f"   ⚠️  Could not load player data: {e}")
        injuries_df['is_star_player'] = False
        injuries_df['player_importance'] = 0.3
    
    return injuries_df


def process_real_injuries(injuries_df):
    """
    Process scraped injuries into format needed for feature engineering
    """
    print("\n🔧 Processing injury data...")
    
    if len(injuries_df) == 0:
        print("   ⚠️  No injuries to process")
        return pd.DataFrame()
    
    # Parse descriptions
    injury_info = injuries_df['description'].apply(extract_injury_info)
    injury_info_df = pd.DataFrame(injury_info.tolist())
    
    # Combine
    processed = pd.concat([injuries_df, injury_info_df], axis=1)
    
    # Add dates
    # For current injuries, assume they're active now
    processed['injury_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate return date based on estimated games out
    processed['games_missed'] = processed['estimated_games_out']
    
    # Calculate return dates (row by row)
    return_dates = []
    for idx, row in processed.iterrows():
        days_out = row['games_missed'] * 2.5  # ~2.5 days per game
        return_date = pd.to_datetime(row['injury_date']) + pd.Timedelta(days=days_out)
        return_dates.append(return_date.strftime('%Y-%m-%d'))
    
    processed['return_date'] = return_dates
    
    # Enhance with player importance
    processed = enhance_with_player_data(processed)
    
    # Team abbreviations - map full names to abbreviations
    team_mapping = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    processed['team'] = processed['team'].replace(team_mapping)
    
    print(f"   ✅ Processed {len(processed)} injuries")
    print(f"   📊 Injury types: {processed['injury_type'].value_counts().head(3).to_dict()}")
    print(f"   🏥 Major injuries: {(processed['severity']=='Major').sum()}")
    
    return processed


def main():
    print("="*70)
    print("🏀 REAL NBA INJURY DATA SCRAPER")
    print("="*70)
    
    # Step 1: Scrape current injuries
    print("\nStep 1: Scraping Basketball-Reference...")
    injuries_raw = scrape_basketball_reference_selenium()
    
    if len(injuries_raw) == 0:
        print("\n❌ Failed to scrape injuries. Check your internet connection.")
        print("   Falling back to synthetic data...")
        return None
    
    # Save raw data
    raw_file = INJURY_PATH / 'injuries_raw_scraped.csv'
    injuries_raw.to_csv(raw_file, index=False)
    print(f"\n💾 Raw data saved to: {raw_file}")
    
    # Step 2: Process into usable format
    print("\nStep 2: Processing...")
    injuries_processed = process_real_injuries(injuries_raw)
    
    # Save processed data
    processed_file = INJURY_PATH / 'nba_injuries_real_scraped.csv'
    injuries_processed.to_csv(processed_file, index=False)
    print(f"\n💾 Processed data saved to: {processed_file}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ SCRAPING COMPLETE!")
    print("="*70)
    print(f"\nReal injuries scraped: {len(injuries_processed)}")
    print(f"Teams affected: {injuries_processed['team'].nunique()}")
    print(f"Star players injured: {injuries_processed['is_star_player'].sum()}")
    
    print("\n🎯 Next steps:")
    print("   1. Review: data/injuries/nba_injuries_real_scraped.csv")
    print("   2. Run: python scrape_injuries.py (will use this real data)")
    print("   3. Run: python evaluate_injury_impact.py (test impact)")
    
    return injuries_processed


if __name__ == "__main__":
    result = main()

