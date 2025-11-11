# 📧 Email Predictions Setup Guide

Get NBA predictions delivered to your inbox daily with beautifully formatted HTML emails!

## 🎯 Features

- **Beautiful HTML Emails**: Professional design with color-coded confidence levels
- **Daily Summary**: Game count, high/medium/low confidence breakdown
- **Detailed Predictions**: Each game with predicted winner, probability, confidence
- **Vegas Odds**: Spread, total, and moneyline (when available)
- **Automated**: Runs daily after predictions are generated
- **Manual Trigger**: Run anytime from GitHub Actions

---

## 🔧 Setup Instructions

### Step 1: Get Email App Password

#### For Gmail:
1. Go to Google Account settings: https://myaccount.google.com/
2. Security → 2-Step Verification (must be enabled)
3. Scroll to "App passwords"
4. Generate new app password for "Mail"
5. Copy the 16-character password (remove spaces)

#### For Outlook/Hotmail:
1. Go to: https://account.microsoft.com/security
2. Advanced security options → App passwords
3. Create new app password
4. Copy the password

#### For Other Providers:
- Most email providers support "app passwords" or "application-specific passwords"
- Search "[your provider] app password" for instructions

---

### Step 2: Add GitHub Secrets

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add these secrets:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `EMAIL_FROM` | Your email address | `your.email@gmail.com` |
| `EMAIL_PASSWORD` | App password from Step 1 | `abcd efgh ijkl mnop` |
| `EMAIL_TO` | Recipient email (can be same) | `your.email@gmail.com` |

**Optional Secrets** (for non-Gmail):

| Secret Name | Default | Description |
|-------------|---------|-------------|
| `SMTP_SERVER` | `smtp.gmail.com` | Your email provider's SMTP server |
| `SMTP_PORT` | `587` | SMTP port (usually 587 or 465) |

#### SMTP Server Examples:
- Gmail: `smtp.gmail.com:587`
- Outlook: `smtp-mail.outlook.com:587`
- Yahoo: `smtp.mail.yahoo.com:587`
- iCloud: `smtp.mail.me.com:587`

---

### Step 3: Enable the Workflow

1. Go to: GitHub → Actions → "Email Daily Predictions"
2. Click "Enable workflow" (if disabled)
3. Schedule: Runs daily at 9:30 AM UTC (after predictions)
4. Manual run: Click "Run workflow" to test immediately

---

## ✉️ Email Format Preview

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│         🏀 NBA Daily Predictions                    │
│            Monday, November 11, 2025                │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  📊 Today's Summary                                 │
│                                                     │
│    8           6           1          1            │
│  Total      High       Medium      Low             │
│  Games    Confidence  Confidence  Confidence       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Lakers @ Celtics                                   │
│                                                     │
│  🎯 Home wins                          [HIGH]       │
│     67.2% Home Win                                  │
│                                                     │
│  Spread: -5.5  │  Total: 218.5  │  ML: -220       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Warriors @ Nets                                    │
│                                                     │
│  🎯 Away wins                         [MEDIUM]      │
│     58.3% Away Win                                  │
│                                                     │
│  Spread: +3.0  │  Total: 225.0  │  ML: +140       │
└─────────────────────────────────────────────────────┘

... (all games)

┌─────────────────────────────────────────────────────┐
│  🤖 Generated by NBA Prediction System              │
│  Model: HistGradient Boosting (70.20% AUC)         │
│  View on GitHub                                     │
└─────────────────────────────────────────────────────┘
```

**Features:**
- ✅ Color-coded confidence (green/yellow/red)
- ✅ Clean, professional design
- ✅ Mobile-responsive
- ✅ Vegas odds included
- ✅ Summary statistics
- ✅ Direct links to GitHub

---

## 🧪 Testing

### Test Locally:

```bash
# Set environment variables
export EMAIL_FROM="your.email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_TO="recipient@email.com"

# Run script
python scripts/predictions/send_email_predictions.py
```

### Test on GitHub:

1. Go to Actions → "Email Daily Predictions"
2. Click "Run workflow"
3. Select branch: `main`
4. Click "Run workflow"
5. Check your inbox!

---

## 🔒 Security Notes

### ✅ Safe:
- App passwords are **not** your main email password
- They can be revoked anytime without affecting your account
- GitHub Secrets are encrypted and never exposed in logs
- Each app password is for one application only

### ⚠️ Important:
- **Never commit** email credentials to the repository
- **Use GitHub Secrets** for all sensitive data
- App passwords have **limited permissions** (safer than main password)
- Revoke old app passwords you're not using

---

## 📅 Schedule

### Default Schedule:
- **Daily**: 9:30 AM UTC (4:30 AM EST / 1:30 AM PST)
- **After**: Daily predictions workflow (9:00 AM UTC)
- **Days**: Every day (7 days a week)

### Customize Schedule:

Edit `.github/workflows/email-daily-predictions.yml`:

```yaml
schedule:
  # Run at 10 AM UTC
  - cron: '0 10 * * *'
  
  # Run at 9 AM and 5 PM UTC
  - cron: '0 9,17 * * *'
  
  # Run only on game days (Mon, Tue, Wed, Fri, Sat)
  - cron: '30 9 * * 1,2,3,5,6'
```

**Cron Format**: `minute hour day month weekday`

---

## 🎨 Customization

### Change Email Subject:

Edit `send_email_predictions.py`, line ~250:

```python
subject = f"🏀 Your Custom Subject - {datetime.now().strftime('%B %d, %Y')}"
```

### Change Colors:

Edit the `<style>` section in `format_html_email()`:

```python
.header {{
    background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
}}
```

### Add More Details:

Add to the game card in `format_html_email()`:

```python
html += f"""
<div>Edge vs Vegas: {row.get('Betting_Edge', 'N/A')}</div>
<div>Injury Impact: {row.get('Injury_Advantage', 'N/A')}</div>
"""
```

---

## 🐛 Troubleshooting

### Email Not Received:

1. **Check spam folder** - First-time automated emails often go to spam
2. **Verify secrets** - Go to Settings → Secrets → Check all 3 are set
3. **Check workflow logs** - Actions → "Email Daily Predictions" → Latest run
4. **Test manually** - Run workflow manually to see detailed logs

### Common Errors:

**"Email credentials not configured"**
- Missing `EMAIL_FROM` or `EMAIL_PASSWORD` secret
- Solution: Add secrets in GitHub Settings → Secrets

**"Authentication failed"**
- Wrong app password
- Solution: Generate new app password and update secret

**"Connection refused"**
- Wrong SMTP server or port
- Solution: Check provider's SMTP settings

**"No prediction files found"**
- Predictions not generated yet
- Solution: Run "Daily NBA Predictions" workflow first

---

## 📊 Workflow Integration

The complete daily workflow:

```
9:00 AM UTC  → Daily NBA Predictions (make predictions)
             ↓
9:30 AM UTC  → Email Daily Predictions (send email)
             ↓
11:00 PM UTC → Track Performance (after games complete)
```

All three workflows work together automatically!

---

## 🎯 Multiple Recipients

### Send to Multiple Emails:

Option 1: Comma-separated in secret:
```
EMAIL_TO = "email1@example.com,email2@example.com,email3@example.com"
```

Option 2: Update script to support list:
```python
# In send_email_predictions.py
to_emails = os.getenv('EMAIL_TO').split(',')
for email in to_emails:
    send_email(html_content, email.strip())
```

---

## 💡 Pro Tips

1. **Mark as Important**: Add email to contacts so it doesn't go to spam
2. **Create Filter**: Auto-label emails with "NBA Predictions"
3. **Morning Routine**: Set email to arrive with your morning coffee time
4. **Mobile Friendly**: Email format works great on phones
5. **Archive System**: Keep predictions to track your own picks

---

## 📞 Support

If you encounter issues:

1. Check the workflow logs in GitHub Actions
2. Verify your email provider's SMTP settings
3. Try the test script locally first
4. Check that predictions exist in `data/predictions/`

---

## 🎉 You're All Set!

Once configured, you'll get beautiful NBA prediction emails every day automatically. Enjoy! 🏀📧

