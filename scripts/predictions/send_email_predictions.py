"""
Send NBA Predictions via Email
Formats daily predictions into a nice HTML email
"""

import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import os
import sys


def load_latest_predictions():
    """Load the most recent predictions file"""
    predictions_dir = Path(__file__).resolve().parents[2] / 'data' / 'predictions'
    
    # Find today's predictions file
    today_str = datetime.now().strftime('%Y%m%d')
    predictions_file = predictions_dir / f'daily_predictions_{today_str}.csv'
    
    if not predictions_file.exists():
        # Try to find the most recent file
        prediction_files = sorted(predictions_dir.glob('daily_predictions_*.csv'), reverse=True)
        if prediction_files:
            predictions_file = prediction_files[0]
        else:
            raise FileNotFoundError("No prediction files found")
    
    df = pd.read_csv(predictions_file)
    return df, predictions_file.stem


def format_html_email(df, filename):
    """Format predictions as HTML email"""
    
    # Extract date from filename
    date_str = filename.replace('daily_predictions_', '')
    formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%A, %B %d, %Y')
    
    # Count games and high confidence predictions
    total_games = len(df)
    high_conf = len(df[df['Confidence'] == 'High'])
    medium_conf = len(df[df['Confidence'] == 'Medium'])
    low_conf = len(df[df['Confidence'] == 'Low'])
    
    # Start HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .summary {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary h2 {{
                margin-top: 0;
                color: #667eea;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin-top: 15px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .game {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .game-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 10px;
            }}
            .matchup {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }}
            .game-time {{
                color: #666;
                font-size: 14px;
            }}
            .prediction {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .predicted-winner {{
                font-size: 20px;
                font-weight: bold;
                color: #667eea;
            }}
            .confidence {{
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
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
            .odds {{
                display: flex;
                justify-content: space-around;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #f0f0f0;
            }}
            .odds-item {{
                text-align: center;
            }}
            .odds-label {{
                font-size: 11px;
                color: #999;
                text-transform: uppercase;
            }}
            .odds-value {{
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 12px;
            }}
            .footer a {{
                color: #667eea;
                text-decoration: none;
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
                    <div class="stat-value">{total_games}</div>
                    <div class="stat-label">Total Games</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{high_conf}</div>
                    <div class="stat-label">High Confidence</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{medium_conf}</div>
                    <div class="stat-label">Medium Confidence</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{low_conf}</div>
                    <div class="stat-label">Low Confidence</div>
                </div>
            </div>
        </div>
    """
    
    # Add each game
    for _, row in df.iterrows():
        matchup = row['Matchup']
        predicted_winner = row['Predicted_Winner']
        home_win_prob = row['Home_Win_Probability']
        confidence = row['Confidence']
        
        # Determine confidence class
        conf_class = f"conf-{confidence.lower()}"
        
        # Format probability
        if home_win_prob >= 0.5:
            prob_text = f"{home_win_prob:.1%} Home Win"
        else:
            prob_text = f"{(1-home_win_prob):.1%} Away Win"
        
        # Add game card
        html += f"""
        <div class="game">
            <div class="game-header">
                <div class="matchup">{matchup}</div>
            </div>
            <div class="prediction">
                <div>
                    <div class="predicted-winner">🎯 {predicted_winner} wins</div>
                    <div style="color: #666; margin-top: 5px;">{prob_text}</div>
                </div>
                <div class="confidence {conf_class}">{confidence}</div>
            </div>
        """
        
        # Add Vegas odds if available
        if 'Vegas_Spread' in df.columns and pd.notna(row.get('Vegas_Spread')):
            html += f"""
            <div class="odds">
                <div class="odds-item">
                    <div class="odds-label">Spread</div>
                    <div class="odds-value">{row.get('Vegas_Spread', 'N/A')}</div>
                </div>
                <div class="odds-item">
                    <div class="odds-label">Total</div>
                    <div class="odds-value">{row.get('Vegas_Total', 'N/A')}</div>
                </div>
                <div class="odds-item">
                    <div class="odds-label">Moneyline</div>
                    <div class="odds-value">{row.get('Vegas_ML_Home', 'N/A')}</div>
                </div>
            </div>
            """
        
        html += """
        </div>
        """
    
    # Add footer
    html += """
        <div class="footer">
            <p>🤖 Generated by NBA Prediction System</p>
            <p>Model: HistGradient Boosting (70.20% AUC)</p>
            <p><a href="https://github.com/missionfromdog/bball-prediction">View on GitHub</a></p>
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

