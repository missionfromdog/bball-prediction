# NBA Prediction - Automation Guide

This guide explains how to set up automated daily scraping of NBA injury data and betting odds.

## 🤖 Overview

We've set up **GitHub Actions** workflows to automatically:
1. **Scrape NBA injuries** every day from Basketball-Reference
2. **Download updated Vegas odds** from Kaggle (when available)
3. **Process and merge data** into your prediction dataset
4. **Commit changes** back to your repository

---

## 📋 Setup Instructions

### Step 1: Add GitHub Secrets

Your workflows need API credentials to function. Add these to your GitHub repository:

1. Go to your repo: `https://github.com/missionfromdog/bball-prediction`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each:

**Required Secrets:**

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `KAGGLE_USERNAME` | `caseyhess` | Your Kaggle username |
| `KAGGLE_KEY` | `e2911ebf...` | Your Kaggle API key |

**Optional Secrets (for live odds):**

| Secret Name | Value | Purpose |
|-------------|-------|---------|
| `ODDS_API_KEY` | Your key | The Odds API key (get at [the-odds-api.com](https://the-odds-api.com/)) |

---

## 🔄 Automated Workflows

### 1. Daily Data Update (Main Workflow)

**File:** `.github/workflows/daily-data-update.yml`

**Schedule:** Every day at 10:00 AM UTC (2:00 AM PST / 5:00 AM EST)

**What it does:**
1. ✅ Scrapes current NBA injuries from Basketball-Reference
2. ✅ Downloads latest Vegas odds from Kaggle (if updated)
3. ✅ Processes and merges data into `games_with_real_vegas.csv`
4. ✅ Commits changes back to your repo

**Manual trigger:**
- Go to **Actions** tab in your repo
- Select **"Daily Data Update"**
- Click **"Run workflow"**

### 2. Live Odds Scraper (Optional)

**File:** `.github/workflows/scrape-live-odds.yml`

**Schedule:** Manual trigger only

**What it does:**
- Fetches **live betting odds** from The Odds API
- Useful for getting odds closer to game time

**How to use:**
1. Sign up at [the-odds-api.com](https://the-odds-api.com/) (500 free requests/month)
2. Add `ODDS_API_KEY` to GitHub Secrets
3. Manually trigger workflow when needed

---

## 📊 Data Update Flow

```
┌─────────────────────┐
│  GitHub Actions     │
│  (Daily @ 10 AM)    │
└──────────┬──────────┘
           │
           ├─────────────────────────────────┐
           │                                 │
           ▼                                 ▼
┌─────────────────────┐          ┌──────────────────────┐
│  Scrape Injuries    │          │  Download Vegas Odds │
│  (Basketball-Ref)   │          │  (Kaggle)            │
└──────────┬──────────┘          └──────────┬───────────┘
           │                                 │
           │     ┌─────────────────┐        │
           └────►│  Process & Merge│◄───────┘
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Commit to Repo │
                 └─────────────────┘
```

---

## 🛠️ Alternative Automation Methods

### Option 1: Cron Job (Local Server)

If you have a server or always-on computer:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 10 AM)
0 10 * * * cd /path/to/bball-prediction && source venv/bin/activate && python scripts/data_collection/scrape_real_injuries.py && python scripts/data_collection/download_real_vegas_data.py && python scripts/data_collection/process_real_vegas_lines.py
```

### Option 2: Cloud Functions

**AWS Lambda / Google Cloud Functions:**
- Deploy scraping scripts as serverless functions
- Schedule with CloudWatch Events / Cloud Scheduler
- More reliable than GitHub Actions for heavy scraping

**Example structure:**
```python
# lambda_handler.py
import boto3
from scrape_real_injuries import main as scrape_injuries
from process_real_vegas_lines import main as process_vegas

def lambda_handler(event, context):
    scrape_injuries()
    process_vegas()
    return {'statusCode': 200, 'body': 'Data updated successfully'}
```

### Option 3: Airflow / Prefect

For complex pipelines with dependencies:

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('nba_prediction_etl', schedule_interval='@daily')

scrape_injuries = PythonOperator(
    task_id='scrape_injuries',
    python_callable=scrape_injuries_func,
    dag=dag
)

download_odds = PythonOperator(
    task_id='download_odds',
    python_callable=download_odds_func,
    dag=dag
)

process_data = PythonOperator(
    task_id='process_data',
    python_callable=process_data_func,
    dag=dag
)

[scrape_injuries, download_odds] >> process_data
```

---

## 🔍 Monitoring

### GitHub Actions Dashboard

1. Go to **Actions** tab in your repo
2. See all workflow runs and their status
3. Click any run to see detailed logs

### Workflow Notifications

**Email notifications:**
- GitHub sends emails on workflow failures (enabled by default)

**Slack/Discord integration:**
Add to your workflow:
```yaml
- name: Notify on Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Workflow fails with "403 Forbidden"**
- Basketball-Reference may be blocking automated requests
- Solution: Add delays, rotate user agents, or use residential proxies

**2. Kaggle download fails**
- Check that `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are correct
- Verify Kaggle API is enabled on your account

**3. No data changes detected**
- This is normal if no games or injuries have changed
- Workflow will skip commit if no changes

**4. Selenium/Chrome issues in GitHub Actions**
- Chrome installation sometimes fails
- Workflow includes fallback to synthetic data

### Debug Mode

Enable debug logging in GitHub Actions:

1. Go to **Settings** → **Secrets** → **Actions**
2. Add secret: `ACTIONS_RUNNER_DEBUG` = `true`
3. Add secret: `ACTIONS_STEP_DEBUG` = `true`

---

## 📈 Data Freshness

| Data Source | Update Frequency | Automation Status |
|-------------|------------------|-------------------|
| **Injuries** | Daily | ✅ Automated (Basketball-Reference) |
| **Vegas Odds (Historical)** | Weekly | ✅ Automated (Kaggle) |
| **Vegas Odds (Live)** | Real-time | ⚠️ Manual (requires API key) |
| **Game Results** | Daily | ⚠️ Not yet automated |

---

## 🚀 Future Enhancements

**Possible additions:**
- [ ] Auto-retrain models when data changes significantly
- [ ] Scrape today's NBA game schedule and make predictions
- [ ] Send predictions via email/Slack before games
- [ ] Track model performance over time
- [ ] Alert when model drift is detected
- [ ] Integration with sports betting APIs for automated betting

---

## 💡 Best Practices

1. **Run manually first** - Test workflow before relying on scheduled runs
2. **Monitor quota** - GitHub Actions: 2,000 minutes/month (free tier)
3. **Rate limiting** - Add delays between requests to avoid bans
4. **Error handling** - Workflows continue even if one step fails
5. **Data validation** - Check data quality after each scrape
6. **Backup strategy** - Keep historical copies of scraped data

---

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [The Odds API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [Basketball-Reference Terms of Use](https://www.sports-reference.com/data_use.html)

---

## 📞 Support

If you encounter issues:
1. Check the **Actions** tab for error logs
2. Review `docs/` for experiment results
3. Test scripts locally first: `python scripts/data_collection/scrape_real_injuries.py`
4. Open an issue on GitHub

---

*Last Updated: November 11, 2025*
*Automation Status: ✅ Ready to deploy*

