<div align="center">
    <h1>NBA Game Predictor - Enhanced Edition</h1>
    <h3>70.20% AUC with Vegas Betting Lines + Injury Data</h3>
</div>
<br />
<div align="center">
    <sub>Let's connect 🤗</sub>
    <br />
    <a href="https://github.com/missionfromdog">GitHub</a> •
    <a href="https://www.linkedin.com/in/caseyhesschicago">LinkedIn</a> •
    <a href="https://twitter.com/caseyhess">Twitter</a> •
    <a href="https://www.missionfromdog.com">Website</a>
<br />
</div>

**Project Repository:** [https://github.com/missionfromdog/bball-prediction](https://github.com/missionfromdog/bball-prediction)

**Built from:** Chris Munch's [NBA Prediction Base Project](https://github.com/cmunch1/nba-prediction)

---

## 🚀 What's New in This Enhanced Version

This project significantly improves upon the original NBA prediction model by **Casey Hess** using **Cursor AI** to iterate and enhance the application:

### **Major Improvements:**
- ✅ **+2.52% AUC improvement** (67.68% → 70.20%)
- ✅ **Real Vegas betting lines integration** - 23,118 historical games (2007-2024) from Kaggle + live odds via The Odds API
- ✅ **NBA injury data scraping** - Real-time injury tracking from Basketball-Reference with Selenium
- ✅ **11 ML models tested** - Including XGBoost, LightGBM, HistGradientBoosting, CatBoost, RandomForest
- ✅ **Ensemble models** - Stacking and Weighted voting classifiers
- ✅ **Enhanced Streamlit app** - Model comparison, CSV export, confidence indicators, performance tracking
- ✅ **Fully automated workflows** - Daily schedule fetch, predictions, score updates, and email notifications via GitHub Actions
- ✅ **Organized codebase** - Professional structure with `scripts/` directory and comprehensive documentation

### **Performance:**
| Model | AUC | Accuracy | Status |
|-------|-----|----------|--------|
| **HistGradientBoosting + Vegas** | **70.20%** | **~63%** | 🏆 **BEST** (In Production) |
| Stacking Ensemble | 69.91% | ~62% | 🥈 |
| Weighted Ensemble | 69.81% | ~62% | 🥉 |
| RandomForest + Vegas | 69.37% | ~61% | ⭐ |
| XGBoost + Vegas | 68.85% | ~60% | ⭐ |
| Legacy XGBoost (baseline) | 67.68% | ~59% | 📊 |

### **Automated Daily Workflows:**
| Workflow | Schedule | Purpose |
|----------|----------|---------|
| **Fetch Today's Schedule** | 8:00 AM UTC (3 AM EST) | Scrapes ESPN for today's NBA games |
| **Daily Predictions** | 9:00 AM UTC (4 AM EST) | Generates predictions with feature engineering & model training |
| **Update Scores** | 12:00 PM UTC (7 AM EST) | Updates completed game scores from ESPN |
| **Email Predictions** | Auto-triggered | Sends formatted email with predictions after workflow completes |

**See [ENHANCEMENTS.md](ENHANCEMENTS.md) for full details on improvements.**

---

#### Table of contents
- [About This Project](#about-this-project)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture) ⭐ **NEW**
- [Introduction (Original)](#introduction)
- [Problem](#problem-increase-the-profitability-of-betting-on-nba-games)
- [Initial step](#initial-step-predict-the-probability-that-the-home-team-will-win-each-game)
- [Plan](#plan)
- [Overview](#overview)
- [Enhancements](#enhancements)
- [Future Possibilities](#future-possibilities)
- [Structure](#structure)
- [Data](#data)
- [EDA and data processing](#eda-and-data-processing)
- [Train/validation/test split](#train--testvalidation-split)
- [Baseline models](#baseline-models)
- [Feature engineering](#feature-engineering)
- [Model training/testing](#model-training-pipeline)
- [Streamlit app](#streamlit-app)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## About This Project

**By Casey Hess** | Data Science Team Lead | Chicago, IL

This is an enhanced version of Chris Munch's NBA Game Prediction project. I discovered his excellent baseline Streamlit app and decided to push it further using **Cursor AI** as my development partner.

### 🎯 **My Goals:**
1. **Improve model accuracy** by integrating additional data sources
2. **Test multiple ML models** to find the best performer
3. **Build production-ready features** including CSV export and model comparison
4. **Learn by doing** - hands-on iteration with real-world sports prediction

### 💡 **Development Process:**
Using Cursor AI, I iteratively:
- Added real Vegas betting lines (+1.17% AUC) from Kaggle
- Integrated live odds via The Odds API for real-time betting data
- Scraped live NBA injury data with Selenium (+0.14% AUC)
- Tested 11 different ML models (HistGradientBoosting won)
- Built ensemble models for robustness
- Enhanced the Streamlit app with comparison tools and export features
- Built fully automated GitHub Actions workflows for daily predictions and email notifications
- Organized the codebase for production and collaboration

### 📊 **Results:**
**+2.52% AUC improvement** over the baseline, achieving **70.20% AUC** with the best model.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/missionfromdog/bball-prediction.git
cd bball-prediction

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run src/streamlit_app_enhanced.py
```

**✨ Features:**
- 🤖 **Automated daily predictions** via GitHub Actions (runs at 4 AM EST)
- 📧 **Email notifications** with formatted predictions
- 📊 **Local Streamlit app** for interactive exploration
- 🎯 **240+ engineered features** including Vegas odds and injury data
- 🏀 **Varied predictions** with confidence levels (High/Medium/Low)

**Note:** The app works perfectly locally. Streamlit Cloud deployment is not supported due to large dataset size (>100MB).

For workflow details and manual predictions, see [docs/SCRIPTS_README.md](docs/SCRIPTS_README.md)

---

## System Architecture

### 📊 Visual Overview

View the complete system architecture with interactive diagrams:

**📖 [Technical Architecture Documentation](docs/ARCHITECTURE.md)**
- Mermaid diagram with data flow
- Feature engineering pipeline (45 → 240 columns)
- ML model architecture
- Deployment details
- Performance optimizations

**🎨 [Visual System Diagrams](docs/SYSTEM_DIAGRAM.md)**
- ASCII art flowcharts
- Phase-by-phase data flows
- Automation schedule
- Technology decision trees

### 🔄 Automated Workflow Pipeline

```
8:00 AM UTC → Fetch Schedule (ESPN)
    ↓
9:00 AM UTC → Generate Predictions (Feature Engineering + ML)
    ↓
12:00 PM UTC → Update Scores (ESPN)
    ↓
Auto-trigger → Send Email Notification
```

**Daily Runtime:** ~8-9 minutes total
**Predictions:** 8-15 games with varied probabilities (27%-71%)
**Email:** HTML formatted with confidence levels

---

## Introduction (Original)

*The following introduction is from Chris Munch's original project:*

This project is a demonstration of the ability to quickly learn, develop, and deploy end-to-end machine learning technologies.

Why predict NBA games:
 - Multiple games are played every day during the season for daily performance tracking
 - Picking a game winner is easy for a casual audience to understand
 - Lots of data available
 - Can be used to develop betting strategies

The original project established the foundation with excellent feature engineering and a production-ready pipeline.

## Problem: Increase the profitability of betting on NBA games

### Initial Step: Predict the probability that the home team will win each game

Machine learning classification models will be used to predict the probability of the winner of each game based upon historical data. This is a first step in developing a betting strategy that will increase the profitability of betting on NBA games. 

*Disclaimer*

In reality, a betting strategy is a rather complex problem with many elements beyond simply picking the winner of each game. Huge amounts of manpower and money have been invested in developing such strategies, and it is not likely that a learning project will be able to compete very well with such efforts. However, it may provide an extra element of insight that could be used to improve the profitability of an existing betting strategy.

### Plan
 
 - Gradient boosted tree models (Xgboost and LightGBM) will be utilized to determine the probability that the home team will win each game. 
 - The model probability will be calibrated against the true probability distribution using sklearn's CalibratedClassifierCV. 
 - The probability of winning will be important in developing betting strategies because such strategies will not bet on every game, just on games with better expected values.
 - Pipelines will be setup to scrape new data from NBA website every day and retrain the model when desired.
 - The model will be deployed online using a [streamlit app](https://cmunch1-nba-prediction.streamlit.app/p/) to predict and report winning probabilities every day. 

<img src="./images/streamlit_example.jpg" width="800"/>


### Overview

 - Historical game data is retrieved from Kaggle.
 - EDA, Data Processing, and Feature Engineering are used to develop best model in either XGboost or LightGBM.
 - Data and model is added to serverless Feature Store and Model Registry
 - Model is deployed online as a Streamlit app
 - Pipelines are setup to:
   - Scrape new data from NBA website and add to Feature Store every day using Github Actions
   - Retrain model and tune hyperparameters

*Initial Modeling Development Cycle*

<img src="./images/modeling_cycle.png" width="800"/>


*Initial Data Setup*

<img src="./images/initial_setup.png" width="800"/>


*Production Pipeline*

<img src="./images/production_pipeline.png" width="800"/>

 
 Tools Used:

 - VS Code w/ Copilot - IDE
 - Pandas - data manipulation
 - XGboost - modeling
 - LightGBM - modeling
 - Scikit-learn - probability calibration
 - Optuna - hyperparameter tuning
 - Neptune.ai - experiment tracking
 - Selenium - data scraping and processing
 - ScrapingAnt - data scraping
 - BeautifulSoup - data processing of scraped data
 - Hopsworks.ai - Feature Store and Model Registry
 - Github Actions - running notebooks to scrape new data, predict winning probabilities, and retrain models
 - Streamlit - web app deployment

NOTE: As of October 2024, I am temporarily removing Hopsworks feature store and model registry from this project until it becomes more stable.

## Enhancements

### ✅ **Completed Improvements**

**1. Vegas Betting Lines Integration (+1.17% AUC)**
- Integrated 23,118 historical Vegas betting lines (2007-2024) from Kaggle
- Live odds integration via The Odds API for real-time data
- Features: spreads, totals, moneylines, implied probabilities
- Major breakthrough in prediction accuracy

**2. Injury Data Integration (+0.14% AUC)**
- Real-time injury scraping from Basketball-Reference using Selenium
- 19 injury-related features including severity, star player impact, team burden
- Combined with Vegas data for best results

**3. Multiple Model Testing**
- Tested 11 different ML models
- HistGradientBoosting emerged as the winner (70.20% AUC, ~63% accuracy)
- Built Stacking and Weighted ensemble models

**4. Enhanced Streamlit App**
- Model comparison tool (compare up to 3 models side-by-side)
- CSV export for Google Sheets integration
- Confidence indicators (High/Medium/Low)
- Historical performance tracking with recent game results
- Live odds display integration
- Data freshness indicators

**5. Fully Automated Production Workflows**
- Daily schedule fetching from ESPN (8 AM UTC)
- Automated predictions with feature engineering (9 AM UTC)
- Completed score updates from ESPN (12 PM UTC)
- Email notifications with formatted predictions
- All workflows run via GitHub Actions

**6. Production-Ready Codebase**
- Organized `scripts/` directory structure
- Data collection, model training, and analysis scripts separated
- Comprehensive documentation in `docs/`
- Contributing guidelines and enhancement summaries
- 240+ engineered features with automated feature engineering

### 🔮 **Future Possibilities**

High-impact opportunities:
 - Player-level statistics (minutes played, usage rate, PER)
 - Rest days / back-to-back game tracking
 - Travel distance between games
 - Referee assignments and tendencies
 - Live betting odds during season
 - Neural networks / deep learning approaches
 - AutoML with AutoGluon (when Python 3.13 support available)
 - Real-time model updates and drift detection
 - Betting strategy development and backtesting
 - A/B testing against other prediction models


### Structure

Jupyter Notebooks were used for initial development and testing and are labeled 01 through 10 in the main directory. Notebooks 01 thru 06 are primarily just historical records and notes for the development process.

Key functions were moved to .py files in src directory once the functions were stable.

Notebooks 07, 09, and 10 are used in production. I chose to keep the notebooks instead of full conversion to scripts because:

 - I think they look better in terms of documentation
 - I prefer to peruse the notebook output after model testing and retraining sometimes instead of relying just on experiment tracking logs
 - I haven't yet conceptually decided on my preferred way of structuring my model testing pipelines for best reusability and maintainability (e.g. should I use custom wrapper functions to invoke experiment logging so that I can easily change providers, or should I just choose one provider and stick with their API?)



### Data

Data from the 2013 thru 2021 season has been archived on Kaggle. New data is scraped from NBA website. 

Currently available data includes:

 - games_details.csv .. (each-game player stats for everyone on the roster)
 - games.csv .......... (each-game team stats: final scores, points scored, field-goal & free-throw percentages, etc...)
 - players.csv ........ (index of players' names and teams)
 - ranking.csv ........ (incremental daily record of standings, games played, won, lost, win%, home record, road record)
 - teams.csv .......... (index of team info such as city and arena names and also head coach) 
 
 NOTES 
 - games.csv is the primary data source and will be the only data used initially
 - games_details.csv details individual player stats for each game and may be added to the model later
 - ranking.csv data is essentially cumulative averages from the beginning of the season and is not really needed as these and other rolling averages can be calculated from the games.csv data 


**New Data**

New data is scraped from [https://www.nba.com/stats/teams/boxscores](https://www.nba.com/stats/teams/boxscores)

 
**Data Leakage**

The data for each game are stats for the *completed* game. We want to predict the winner *before* the game is played, not after. The model should only use data that would be available before the game is played. Our model features will primarily be rolling stats for the previous games (e.g. average assists for previous 5 games) while excluding the current game.

I mention this because I did see several similar projects online that failed to take this into account. If the goal is simply to predict which stats are important for winning games, then the model can be trained on the entire dataset. However, if the goal is to predict the winner of a game like we are trying to do, then the model must be trained on data that would only be available before the game is played.

### EDA and Data Processing

Exploratory Data Analysis (EDA) and Data Processing are summarized and detailed in the notebooks. Some examples include:

Histograms of various features

<img src="./images/distributions.png" width="500"/> 

Correlations between features

<img src="./images/correlation_bar_chart.png" width="500"/>


### Train / Test/Validation Split
  
  - Latest season is used as Test/Validation data and previous seasons are used as Train data
  
### Baseline Models
  
Simple If-Then Models

 - Home team always wins (Accuracy = 0.59, AUC = 0.50 on Train data, Accuracy = 0.49, AUC = 0.50 on Test data)
 
ML Models

 - LightGBM (Accuracy = 0.58, AUC = 0.64 on Test data)
 - XGBoost (Accuracy = 0.59, AUC = 0.61 on Test data)

### Feature Engineering

 - Convert game date to month only
 - Compile rolling means for various time periods for each team as home team and as visitor team 
 - Compile current win streak for each team as home team and as visitor team
 - Compile head-to-head matchup data for each team pair 
 - Compile rolling means for various time periods for each team regardless of home or visitor status
 - Compile current win streak for each team regardless of home or visitor status
 - Subtract the league average rolling means from each team's rolling means


### Model Training/Testing

**Models**
 - LightGBM 
 - XGBoost 

The native Python API (rather than the Scikit-learn wrapper) is used for initial testing of both models because of ease of built-in Shapley values, which are used for feature importance analysis and for adversarial validation (since Shapley values are local to each dataset, they can be used to determine if the train and test datasets have the same feature importances. If they do not, then it may indicate that the model does not generalize very well.)

The Scikit-learn wrapper is used later in production because it allows for easier probability calibration using sklearn's CalibratedClassifierCV.

<img src="./images/train_vs_test_shapley.jpg" width="500"/>

**Evaluation**
 - AUC is the primary training metric for now. This may change once betting strategy comes into play.
 - Accuracy is the secondary metric - easier for casual users to appreciate and easy to compare to public predictions
 - Shapley values compared: Train set vs Test/Validation set
 - Test/Validation set is split: early half vs later half

<img src="./images/confusion_matrix.png" width="300"/>

 
**Experiment Tracking**
 
Notebook 07 integrates Neptune.ai for experiment tracking and Optuna for hyperparameter tuning.

Experiment tracking logs can be viewed here: [https://app.neptune.ai/cmunch1/nba-prediction/experiments?split=tbl&dash=charts&viewId=979e20ed-e172-4c33-8aae-0b1aa1af3602](https://app.neptune.ai/cmunch1/nba-prediction/experiments?split=tbl&dash=charts&viewId=979e20ed-e172-4c33-8aae-0b1aa1af3602)

<img src="./images/neptune.png" width="500"/>


**Probability Calibration**

SKlearn's CalibratedClassifierCV is used to ensure that the model probabilities are calibrated against the true probability distribution. The Brier loss score is used to by the software to automatically select the best calibration method (sigmoid, isotonic, or none).

<img src="./images/calibration.png" width="500"/>



### Production Features Pipeline

Notebook 09 is run from a Github Actions every morning.

- It scrapes the stats from the previous day's games, updates all the rolling statistics and streaks, and adds them to the Feature Store.
- It scrapes the upcoming game matchups for the current day and adds them to the Feature Store so that the streamlit app can use these to make it's daily predictions.

A variable can be set to either use Selenium or ScrapingAnt for scraping the data. ScrapingAnt is used in production because of its built-in proxy server.

 - The Selenium notebook worked fine when ran locally, but there were issues when running the notebook in Github Actions, likely due to the ip address and anti-bot measures on the NBA website (which would require a proxy server to address)
 - ScrapingAnt is a cloud-based scraper with a Python API that handles the proxy server issues. An account is required, but the free account is sufficient for this project.

### Model Training Pipeline

Notebook 10 retrieves the most current data, executes Notebook 07 to handle hyperparameter tuning, model training, and calibration, and then adds the model to the Model Registry. The time periods used for the train set and test set can be adjusted so that the model can be tested only on the most current games.

### Streamlit App

The streamlit app is deployed at streamlit.io and can be accessed here: [https://cmunch1-nba-prediction.streamlit.app/](https://cmunch1-nba-prediction.streamlit.app/)

It uses the model in the Model Registry to predict the win probability of the home team for the current day's upcoming games.

<img src="./images/streamlit2.png" width="500"/>

### Model Performance 

The current model was tested over the completed 2022-2023 regular season (not playoffs) and had an accuracy of 0.615.

Baseline performance of "home team always wins" is 0.58 for this same time period.

One of the [top public prediction models](https://nflpickwatch.com/profile/nba/157/) had an accuracy of 0.656 for this same time period.

Overall, the performance for the regular season is not bad, but there is room for improvement.

<img src="./images/regular_season_accuracy.png" width="800"/>


## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Additional data sources (player stats, rest days, travel distance)
- Model improvements and tuning
- Betting strategy development
- Documentation and examples
- Bug fixes and performance optimization

## Contact

**Casey Hess**
- 🌐 Website: [www.missionfromdog.com](https://www.missionfromdog.com)
- 💼 LinkedIn: [linkedin.com/in/caseyhesschicago](https://www.linkedin.com/in/caseyhesschicago)
- 🐦 Twitter: [@caseyhess](https://twitter.com/caseyhess)
- 📧 GitHub: [@missionfromdog](https://github.com/missionfromdog)
- 📍 Location: Chicago, IL

Currently overseeing a team of data scientists, developers, and analysts as the product owner for a media management and prediction platform.

---

### Acknowledgements

**Original Project:**
- **Chris Munch** - Created the excellent baseline NBA prediction project that served as the foundation
- Original Repository: [github.com/cmunch1/nba-prediction](https://github.com/cmunch1/nba-prediction)
- Pau Labarto Bajo - Mentored Chris on the original project ([datamachines.xyz](https://datamachines.xyz/))

**Data Sources:**
- **Kaggle** - Vegas betting lines dataset (2007-2024)
- **Basketball-Reference.com** - Live NBA injury data
- **NBA Stats API** - Historical game data

**Tools & Technologies:**
- **Cursor AI** - AI-powered development partner for iterative improvements
- **Streamlit** - Web application framework
- **Scikit-learn, XGBoost, LightGBM, CatBoost** - Machine learning libraries
- **Selenium & BeautifulSoup** - Web scraping tools

---

<div align="center">
<sub>Built with ❤️ and Cursor AI in Chicago</sub>
</div>
