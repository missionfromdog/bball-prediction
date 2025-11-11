#!/usr/bin/env python3
"""
Track Betting Performance

Compares predictions against actual results and calculates
accuracy, ROI, and other performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
DATAPATH = Path(__file__).resolve().parents[2] / 'data'
PREDICTIONS_PATH = DATAPATH / 'predictions'
TRACKING_PATH = DATAPATH / 'performance'
TRACKING_PATH.mkdir(exist_ok=True)


def load_predictions():
    """Load all prediction files"""
    pred_files = list(PREDICTIONS_PATH.glob('predictions_*.csv'))
    
    if not pred_files:
        print("❌ No prediction files found")
        return None
    
    # Load and combine all predictions
    all_preds = []
    for file in pred_files:
        if 'latest' not in file.name:  # Skip the 'latest' file
            df = pd.read_csv(file)
            df['Prediction_File'] = file.name
            all_preds.append(df)
    
    if not all_preds:
        return None
    
    df = pd.concat(all_preds, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Loaded {len(df)} predictions from {len(pred_files)} files")
    return df


def load_actual_results():
    """Load actual game results"""
    try:
        df = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
        
        # Keep only completed games with results
        df_complete = df[pd.notna(df['TARGET'])].copy()
        
        print(f"✅ Loaded {len(df_complete)} completed games")
        return df_complete
    
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None


def match_predictions_to_results(predictions_df, results_df):
    """Match predictions with actual game results"""
    matched = []
    
    for _, pred in predictions_df.iterrows():
        pred_date = pred['Date'].date() if isinstance(pred['Date'], pd.Timestamp) else pd.to_datetime(pred['Date']).date()
        matchup = pred['Matchup']
        
        # Find matching game in results
        matching_games = results_df[
            (results_df['GAME_DATE_EST'].dt.date == pred_date) &
            (results_df['MATCHUP'] == matchup)
        ]
        
        if len(matching_games) > 0:
            result = matching_games.iloc[0]
            
            match_data = pred.to_dict()
            match_data['Actual_Winner'] = 'Home' if result['TARGET'] == 1 else 'Away'
            match_data['Correct'] = match_data['Predicted_Winner'] == match_data['Actual_Winner']
            match_data['Home_Score'] = result.get('PTS_home', None)
            match_data['Away_Score'] = result.get('PTS_away', None)
            
            matched.append(match_data)
    
    if not matched:
        print("⚠️  No matches found between predictions and results")
        return None
    
    df = pd.DataFrame(matched)
    print(f"✅ Matched {len(df)} predictions to actual results")
    
    return df


def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Overall accuracy
    metrics['Total_Predictions'] = len(df)
    metrics['Correct_Predictions'] = df['Correct'].sum()
    metrics['Overall_Accuracy'] = df['Correct'].mean()
    
    # Accuracy by confidence
    for conf in ['High', 'Medium', 'Low']:
        conf_df = df[df['Confidence'] == conf]
        if len(conf_df) > 0:
            metrics[f'{conf}_Confidence_Predictions'] = len(conf_df)
            metrics[f'{conf}_Confidence_Accuracy'] = conf_df['Correct'].mean()
    
    # Calculate betting performance (assuming $100 bets)
    df['Bet_Result'] = 0.0
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('Vegas_ML_Home')) and pd.notna(row.get('Vegas_ML_Away')):
            # Determine which side we bet
            if row['Predicted_Winner'] == 'Home':
                ml = row['Vegas_ML_Home']
            else:
                ml = row['Vegas_ML_Away']
            
            # Calculate payout
            if row['Correct']:
                if ml > 0:
                    payout = 100 * (ml / 100)
                else:
                    payout = 100 * (100 / abs(ml))
                df.at[idx, 'Bet_Result'] = payout
            else:
                df.at[idx, 'Bet_Result'] = -100
    
    # ROI calculations
    total_wagered = len(df[df['Bet_Result'] != 0]) * 100
    total_profit = df['Bet_Result'].sum()
    
    metrics['Total_Wagered'] = total_wagered
    metrics['Total_Profit'] = total_profit
    metrics['ROI'] = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    # Edge vs Vegas accuracy
    if 'Edge_vs_Vegas' in df.columns:
        strong_edge = df[df['Edge_vs_Vegas'].abs() > 0.05]
        if len(strong_edge) > 0:
            metrics['Strong_Edge_Games'] = len(strong_edge)
            metrics['Strong_Edge_Accuracy'] = strong_edge['Correct'].mean()
    
    return metrics, df


def create_performance_report(metrics, detailed_df):
    """Create a comprehensive performance report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NBA PREDICTION PERFORMANCE REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall Performance
    report_lines.append("📊 OVERALL PERFORMANCE")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Predictions: {metrics['Total_Predictions']}")
    report_lines.append(f"Correct Predictions: {metrics['Correct_Predictions']}")
    report_lines.append(f"Overall Accuracy: {metrics['Overall_Accuracy']:.1%}")
    report_lines.append("")
    
    # Performance by Confidence
    report_lines.append("🎯 ACCURACY BY CONFIDENCE LEVEL")
    report_lines.append("-" * 80)
    for conf in ['High', 'Medium', 'Low']:
        if f'{conf}_Confidence_Predictions' in metrics:
            count = metrics[f'{conf}_Confidence_Predictions']
            acc = metrics[f'{conf}_Confidence_Accuracy']
            report_lines.append(f"{conf:8} Confidence: {count:3} predictions, {acc:.1%} accuracy")
    report_lines.append("")
    
    # Betting Performance
    report_lines.append("💰 BETTING PERFORMANCE")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Wagered: ${metrics['Total_Wagered']:,.2f}")
    report_lines.append(f"Total Profit/Loss: ${metrics['Total_Profit']:,.2f}")
    report_lines.append(f"ROI: {metrics['ROI']:+.2f}%")
    report_lines.append("")
    
    # Edge vs Vegas
    if 'Strong_Edge_Games' in metrics:
        report_lines.append("📈 EDGE VS VEGAS")
        report_lines.append("-" * 80)
        report_lines.append(f"Games with Strong Edge (>5%): {metrics['Strong_Edge_Games']}")
        report_lines.append(f"Strong Edge Accuracy: {metrics['Strong_Edge_Accuracy']:.1%}")
        report_lines.append("")
    
    # Recent Performance (last 10 games)
    report_lines.append("🕒 RECENT PERFORMANCE (Last 10 Games)")
    report_lines.append("-" * 80)
    recent = detailed_df.sort_values('Date', ascending=False).head(10)
    recent_acc = recent['Correct'].mean()
    report_lines.append(f"Last 10 Games Accuracy: {recent_acc:.1%}")
    report_lines.append("")
    
    for _, game in recent.iterrows():
        result_icon = "✅" if game['Correct'] else "❌"
        report_lines.append(f"{result_icon} {game['Date'].strftime('%Y-%m-%d')} | {game['Matchup']}")
        report_lines.append(f"   Predicted: {game['Predicted_Winner']} | Actual: {game['Actual_Winner']} | Conf: {game['Confidence']}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def save_performance_data(metrics, detailed_df):
    """Save performance data to files"""
    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_path = TRACKING_PATH / f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics: {metrics_path}")
    
    # Save detailed tracking
    detailed_path = TRACKING_PATH / f"detailed_tracking_{datetime.now().strftime('%Y%m%d')}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✅ Saved detailed tracking: {detailed_path}")
    
    # Also save as 'latest'
    latest_metrics = TRACKING_PATH / 'performance_metrics_latest.csv'
    latest_detailed = TRACKING_PATH / 'detailed_tracking_latest.csv'
    metrics_df.to_csv(latest_metrics, index=False)
    detailed_df.to_csv(latest_detailed, index=False)
    print(f"✅ Updated latest tracking files")


def main():
    """Main execution"""
    print()
    print("=" * 80)
    print("NBA PREDICTION PERFORMANCE TRACKER")
    print("=" * 80)
    print()
    
    # Load predictions
    predictions_df = load_predictions()
    if predictions_df is None:
        return
    
    # Load actual results
    results_df = load_actual_results()
    if results_df is None:
        return
    
    # Match predictions to results
    matched_df = match_predictions_to_results(predictions_df, results_df)
    if matched_df is None:
        return
    
    # Calculate performance metrics
    metrics, detailed_df = calculate_performance_metrics(matched_df)
    
    # Create report
    report = create_performance_report(metrics, detailed_df)
    
    # Display report
    print(report)
    
    # Save performance data
    save_performance_data(metrics, detailed_df)
    
    # Save report
    report_path = TRACKING_PATH / f"performance_report_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Saved report: {report_path}")
    
    print()
    print("💡 Performance tracking complete!")
    print(f"   View details in: {TRACKING_PATH}")
    print()


if __name__ == "__main__":
    main()

