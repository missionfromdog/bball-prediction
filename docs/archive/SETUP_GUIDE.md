# NBA Prediction Project - Setup Guide

## ✅ Virtual Environment Setup Complete!

Your virtual environment has been successfully created and all dependencies have been installed.

## 📦 What Was Installed

### Core ML & Data Science
- **pandas** 2.3.3 - Data manipulation
- **numpy** 1.26.4 - Numerical computing
- **scikit-learn** 1.7.2 - Machine learning utilities
- **xgboost** 2.1.4 - Gradient boosting (main prediction model)
- **lightgbm** 4.6.0 - Alternative gradient boosting
- **optuna** 3.6.2 - Hyperparameter tuning

### Visualization & Analysis
- **matplotlib** 3.10.7 - Plotting
- **seaborn** 0.13.2 - Statistical visualizations
- **shap** 0.49.1 - Model interpretation
- **sweetviz** 2.3.1 - EDA reports

### Experiment Tracking
- **neptune** 1.14.0 - Experiment tracking platform
- **neptune-xgboost**, **neptune-lightgbm**, **neptune-optuna** - Integrations

### Web Scraping
- **selenium** 4.38.0 - Browser automation
- **beautifulsoup4** 4.14.2 - HTML parsing
- **scrapingant-client** 1.0.1 - Web scraping service

### Deployment
- **streamlit** 1.51.0 - Web app framework
- **jupyter** 1.1.1 - Notebook environment

## 🚀 How to Use the Virtual Environment

### On macOS/Linux:
```bash
# Activate the virtual environment
source venv/bin/activate

# Verify installation
python --version
pip list

# Run Jupyter notebooks
jupyter notebook

# Run the Streamlit app
streamlit run src/streamlit_app.py

# Deactivate when done
deactivate
```

### On Windows:
```bash
# Activate the virtual environment
venv\Scripts\activate

# (Same commands as above for verification and running)
```

## 📊 Project Structure

```
nba-prediction-main/
├── venv/                           # Virtual environment (newly created)
├── src/                            # Source code
│   ├── feature_engineering.py      # Rolling stats, streaks
│   ├── model_training.py           # Model training & evaluation
│   ├── streamlit_app.py            # Web application
│   ├── webscraping.py              # Data collection from NBA.com
│   └── constants.py                # Configuration constants
├── notebooks/                       # Jupyter notebooks
│   ├── 00_update_local_data.ipynb  # Data updates
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 05_feature_engineering.ipynb
│   ├── 07_model_testing.ipynb      # Model evaluation
│   ├── 09_production_features_pipeline.ipynb
│   └── 10_model_training_pipeline.ipynb
├── data/                           # Datasets
├── models/                         # Trained models
├── configs/                        # Model configurations
└── requirements_updated.txt        # Updated dependencies

```

## 🏀 Quick Start Guide

### 1. Explore the Data
```bash
source venv/bin/activate
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Run Feature Engineering
```bash
# View feature engineering notebook
jupyter notebook notebooks/05_feature_engineering.ipynb
```

### 3. Train a Model
```bash
# Open model training notebook
jupyter notebook notebooks/10_model_training_pipeline.ipynb
```

### 4. Run the Streamlit App (if data is available)
```bash
streamlit run src/streamlit_app.py
```

## 🔑 Environment Variables

The project uses a `.env` file for API keys. Create one if needed:

```bash
# .env file
NEPTUNE_API_TOKEN=your_neptune_token_here
SCRAPINGANT_API_KEY=your_scrapingant_key_here
# HOPSWORKS_API_KEY=your_hopsworks_key_here  # Currently not used
```

## 📝 Key Features Engineered

The model uses these types of features:
- **Rolling averages** (3, 7, 10, 15 games) for points, assists, rebounds, FG%, etc.
- **Win/loss streaks** (home, away, and overall)
- **Head-to-head matchup statistics** between specific teams
- **Home vs league average comparisons**
- **Date features** (month of season)

## 🎯 Model Performance

- **Current Accuracy**: ~61.5% on 2023-24 season
- **Baseline** (home team always wins): 54.7%
- **Top public models**: ~65-66%
- **Primary metric**: AUC (Area Under ROC Curve)
- **Secondary metric**: Accuracy

## ⚠️ Important Notes

1. **Data Leakage Prevention**: The model only uses stats available BEFORE game time (rolling averages exclude the current game)

2. **Season Timing**: NBA runs October - June. Predictions only available during season.

3. **Hopsworks Disabled**: As of Oct 2024, Hopsworks feature store is temporarily disabled due to stability issues. Data is now read from local CSV files.

4. **Web Scraping**: ScrapingAnt is used for production scraping (handles proxy servers). Selenium can be used locally.

## 🛠️ Troubleshooting

### Import Errors
If you get import errors, ensure the venv is activated:
```bash
which python  # Should show path to venv/bin/python
```

### Missing Data
If data files are missing, run:
```bash
jupyter notebook notebooks/00_update_local_data.ipynb
```

### Module Not Found
Ensure you're in the project root directory and venv is activated.

## 📚 Next Steps

Now that your environment is set up, you mentioned wanting to make updates to the project. What would you like to modify?

Common improvements:
1. Add new features (player stats, injuries, etc.)
2. Try different models or ensemble methods
3. Improve data visualizations
4. Add betting strategy logic
5. Update data pipeline automation
6. Improve model performance with advanced feature engineering

Let me know what you'd like to work on!

