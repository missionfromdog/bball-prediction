"""
Send NBA Predictions via Email
Formats daily predictions into a nice HTML email
VERSION: 2.0 (Column Normalization Fix)
"""

import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import os
import sys

# Print version immediately
print("\n" + "="*80, flush=True)
print("📧 EMAIL SCRIPT VERSION 2.0 - COLUMN NORMALIZATION", flush=True)
print("="*80 + "\n", flush=True)
sys.stdout.flush()


def load_latest_predictions():
    """Load the most recent predictions file"""
    predictions_dir = Path(__file__).resolve().parents[2] / 'data' / 'predictions'
    
    # Ensure directory exists
    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory not found: {predictions_dir}\n"
            "Run 'Daily NBA Predictions' workflow first to generate predictions."
        )
    
    # Find today's predictions file
    today_str = datetime.now().strftime('%Y%m%d')
    
    # Try both naming patterns (daily_predictions_* and predictions_*)
    predictions_file = predictions_dir / f'daily_predictions_{today_str}.csv'
    if not predictions_file.exists():
        predictions_file = predictions_dir / f'predictions_{today_str}.csv'
    
    if not predictions_file.exists():
        # Try to find the most recent file with either pattern
        prediction_files = sorted(
            list(predictions_dir.glob('daily_predictions_*.csv')) + 
            list(predictions_dir.glob('predictions_*.csv')),
            reverse=True
        )
        if prediction_files:
            predictions_file = prediction_files[0]
            print(f"⚠️  Today's predictions not found, using most recent: {predictions_file.name}")
        else:
            raise FileNotFoundError(
                f"No prediction files found in {predictions_dir}\n"
                "Available files: " + str(list(predictions_dir.glob('*.csv'))) + "\n"
                "Run 'Daily NBA Predictions' workflow first to generate predictions."
            )
    
    df = pd.read_csv(predictions_file)
    return df, predictions_file.stem


def format_html_email(df, filename):
    """Format predictions as HTML email"""
    
    print(f"   [DEBUG] Original columns: {df.columns.tolist()}", flush=True)
    
    # Normalize column names (handle case variations)
    df.columns = df.columns.str.strip()
    column_map = {col.lower(): col for col in df.columns}
    
    # Map common column name variations to standard names
    rename_map = {}
    
    # Matchup column
    for col_lower, col_actual in column_map.items():
        if 'matchup' in col_lower or 'match_up' in col_lower:
            if col_actual != 'Matchup':
                rename_map[col_actual] = 'Matchup'
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"   [DEBUG] Renamed columns: {rename_map}", flush=True)
    
    # Ensure Matchup column exists
    if 'Matchup' not in df.columns:
        print(f"   [ERROR] Matchup column not found! Available: {df.columns.tolist()}", flush=True)
        # Try to create it from other columns if possible
        if 'Date' in df.columns:
            df['Matchup'] = 'Game ' + df.index.astype(str)
    
    print(f"   [DEBUG] Final columns: {df.columns.tolist()}", flush=True)
    
    # Extract date from filename (handle both naming patterns)
    date_str = filename.replace('daily_predictions_', '').replace('predictions_', '').replace('.csv', '')
    
    # Handle "latest" filename - use today's date
    if date_str == 'latest':
        formatted_date = datetime.now().strftime('%A, %B %d, %Y')
    else:
        formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%A, %B %d, %Y')
    
    # Count games and high confidence predictions
    total_games = len(df)
    
    # Handle missing Confidence column (calculate from probability if needed)
    if 'Confidence' not in df.columns:
        prob_col = None
        for col in ['Home_Win_Probability', 'home_win_probability', 'Probability', 'probability']:
            if col in df.columns:
                prob_col = col
                break
        
        if prob_col:
            df['Confidence'] = df[prob_col].apply(
                lambda p: 'High' if abs(float(p) - 0.5) > 0.15 else 'Medium' if abs(float(p) - 0.5) > 0.05 else 'Low'
            )
        else:
            df['Confidence'] = 'Medium'  # Default
    
    high_conf = len(df[df['Confidence'] == 'High'])
    medium_conf = len(df[df['Confidence'] == 'Medium'])
    low_conf = len(df[df['Confidence'] == 'Low'])
    
    # Start HTML with improved styling
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #1a1a1a;
                max-width: 700px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 20px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 32px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}
            .header p {{
                margin: 12px 0 0 0;
                opacity: 0.95;
                font-size: 16px;
                font-weight: 500;
            }}
            .summary {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .summary h2 {{
                margin: 0 0 20px 0;
                color: #667eea;
                font-size: 20px;
                font-weight: 700;
            }}
            .stats {{
                display: table;
                width: 100%;
                border-collapse: separate;
                border-spacing: 10px 0;
            }}
            .stat {{
                display: table-cell;
                text-align: center;
                padding: 15px 10px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: 700;
                color: #667eea;
                display: block;
                margin-bottom: 5px;
            }}
            .stat-label {{
                font-size: 11px;
                color: #6c757d;
                text-transform: uppercase;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            .game {{
                background: white;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            .game-header {{
                background: #f8f9fa;
                padding: 15px 20px;
                border-bottom: 2px solid #e9ecef;
            }}
            .matchup {{
                font-size: 20px;
                font-weight: 700;
                color: #1a1a1a;
                margin: 0;
            }}
            .prediction-section {{
                padding: 20px;
            }}
            .prediction-row {{
                display: table;
                width: 100%;
                margin-bottom: 15px;
            }}
            .prediction-left {{
                display: table-cell;
                vertical-align: middle;
                width: 70%;
            }}
            .prediction-right {{
                display: table-cell;
                vertical-align: middle;
                width: 30%;
                text-align: right;
            }}
            .predicted-winner {{
                font-size: 18px;
                font-weight: 700;
                color: #667eea;
                margin: 0 0 5px 0;
            }}
            .win-probability {{
                font-size: 14px;
                color: #6c757d;
                margin: 0;
            }}
            .confidence {{
                display: inline-block;
                padding: 8px 20px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .conf-high {{
                background-color: #10b981;
                color: white;
            }}
            .conf-medium {{
                background-color: #f59e0b;
                color: white;
            }}
            .conf-low {{
                background-color: #ef4444;
                color: white;
            }}
            .odds-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                margin-top: 15px;
                border-top: 2px solid #e9ecef;
                padding-top: 15px;
            }}
            .odds-table td {{
                padding: 10px 15px;
                text-align: center;
                border-right: 1px solid #e9ecef;
            }}
            .odds-table td:last-child {{
                border-right: none;
            }}
            .odds-label {{
                font-size: 11px;
                color: #6c757d;
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.5px;
                display: block;
                margin-bottom: 5px;
            }}
            .odds-value {{
                font-size: 18px;
                font-weight: 700;
                color: #1a1a1a;
                display: block;
            }}
            .no-odds {{
                color: #adb5bd;
                font-style: italic;
            }}
            .footer {{
                text-align: center;
                padding: 30px 20px;
                color: #6c757d;
                font-size: 13px;
            }}
            .footer a {{
                color: #667eea;
                text-decoration: none;
                font-weight: 600;
            }}
            .footer-divider {{
                margin: 15px 0;
                color: #adb5bd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏀 NBA Daily Predictions</h1>
            <p>{formatted_date}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Today's Summary</h2>
            <div class="stats">
                <div class="stat">
                    <span class="stat-value">{total_games}</span>
                    <span class="stat-label">Total Games</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{high_conf}</span>
                    <span class="stat-label">High Confidence</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{medium_conf}</span>
                    <span class="stat-label">Medium Confidence</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{low_conf}</span>
                    <span class="stat-label">Low Confidence</span>
                </div>
            </div>
        </div>
    """
    
    # Add each game with improved table layout
    for idx, row in df.iterrows():
        try:
            matchup = row.get('Matchup', row.get('matchup', row.get('MATCHUP', f'Game {idx}')))
            predicted_winner = row.get('Predicted_Winner', row.get('predicted_winner', 'Unknown'))
            home_win_prob = float(row.get('Home_Win_Probability', row.get('home_win_probability', 0.5)))
            confidence = row.get('Confidence', row.get('confidence', 'Medium'))
        except Exception as e:
            print(f"   [ERROR] Row {idx} access failed: {e}", flush=True)
            print(f"   [ERROR] Available columns: {row.index.tolist()}", flush=True)
            raise
        
        # Determine confidence class
        conf_class = f"conf-{confidence.lower()}"
        
        # Format probability
        if home_win_prob >= 0.5:
            prob_text = f"{home_win_prob:.1%} Home Win"
        else:
            prob_text = f"{(1-home_win_prob):.1%} Away Win"
        
        # Add game card with table layout
        html += f"""
        <div class="game">
            <div class="game-header">
                <h3 class="matchup">{matchup}</h3>
            </div>
            <div class="prediction-section">
                <div class="prediction-row">
                    <div class="prediction-left">
                        <p class="predicted-winner">🎯 {predicted_winner} wins</p>
                        <p class="win-probability">{prob_text}</p>
                    </div>
                    <div class="prediction-right">
                        <span class="confidence {conf_class}">{confidence}</span>
                    </div>
                </div>
        """
        
        # Add Vegas odds table if any odds are available
        has_spread = 'Vegas_Spread' in row and pd.notna(row.get('Vegas_Spread'))
        has_total = 'Vegas_Total' in row and pd.notna(row.get('Vegas_Total'))
        has_ml_home = 'Vegas_ML_Home' in row and pd.notna(row.get('Vegas_ML_Home'))
        has_ml_away = 'Vegas_ML_Away' in row and pd.notna(row.get('Vegas_ML_Away'))
        
        if has_spread or has_total or has_ml_home:
            # Format the odds values
            spread_val = f"{row.get('Vegas_Spread', 0):+.1f}" if has_spread else '<span class="no-odds">--</span>'
            total_val = f"{row.get('Vegas_Total', 0):.1f}" if has_total else '<span class="no-odds">--</span>'
            
            # Format moneyline
            if has_ml_home:
                ml_home = row.get('Vegas_ML_Home')
                ml_val = f"{ml_home:+.0f}" if ml_home != 0 else '<span class="no-odds">--</span>'
            else:
                ml_val = '<span class="no-odds">--</span>'
            
            html += f"""
                <table class="odds-table">
                    <tr>
                        <td>
                            <span class="odds-label">Spread</span>
                            <span class="odds-value">{spread_val}</span>
                        </td>
                        <td>
                            <span class="odds-label">Over/Under</span>
                            <span class="odds-value">{total_val}</span>
                        </td>
                        <td>
                            <span class="odds-label">Moneyline</span>
                            <span class="odds-value">{ml_val}</span>
                        </td>
                    </tr>
                </table>
            """
        
        html += """
            </div>
        </div>
        """
    
    # Add footer
    html += """
        <div class="footer">
            <p>🤖 Generated by NBA Prediction System</p>
            <p class="footer-divider">• • •</p>
            <p>Model: HistGradient Boosting (70.20% AUC) with Vegas Lines + Injury Data</p>
            <p style="margin-top: 15px;">
                <a href="https://github.com/missionfromdog/bball-prediction">View on GitHub</a>
            </p>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(html_content, to_email, subject=None):
    """Send email with predictions"""
    
    # Get email configuration from environment variables
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    from_email = os.getenv('EMAIL_FROM')
    email_password = os.getenv('EMAIL_PASSWORD')
    
    if not from_email or not email_password:
        raise ValueError(
            "Email credentials not configured. Set EMAIL_FROM and EMAIL_PASSWORD "
            "environment variables or GitHub Secrets."
        )
    
    # Default subject
    if subject is None:
        subject = f"🏀 NBA Predictions - {datetime.now().strftime('%B %d, %Y')}"
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    
    # Attach HTML content
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, email_password)
            server.send_message(msg)
            print(f"✅ Email sent successfully to {to_email}")
            return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False


def main():
    """Main function"""
    print("📧 NBA Predictions Email Sender")
    print("=" * 80)
    
    # Get recipient email
    to_email = os.getenv('EMAIL_TO')
    if not to_email:
        print("❌ EMAIL_TO environment variable not set")
        print("Set EMAIL_TO to specify recipient email address")
        sys.exit(1)
    
    try:
        # Load predictions
        print("📊 Loading latest predictions...")
        df, filename = load_latest_predictions()
        print(f"✅ Loaded {len(df)} predictions from {filename}")
        
        # Format email
        print("📝 Formatting email...")
        html_content = format_html_email(df, filename)
        print("✅ Email formatted")
        
        # Send email
        print(f"📧 Sending email to {to_email}...")
        success = send_email(html_content, to_email)
        
        if success:
            print("✅ Email sent successfully!")
            print(f"📬 Recipient: {to_email}")
            print(f"🎯 Games: {len(df)}")
        else:
            print("❌ Failed to send email")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

