# Scripts Directory Guide

This directory contains all the scripts used to enhance the NBA prediction model.

## 📁 Directory Structure

### `data_collection/`
Scripts for collecting and processing external data sources:

- **`scrape_injuries.py`** - Main injury data pipeline (supports real & synthetic data)
- **`scrape_real_injuries.py`** - Selenium scraper for Basketball-Reference injuries
- **`scrape_betting_lines.py`** - Betting line scraper (with synthetic fallback)
- **`download_historical_injuries.py`** - Generate comprehensive historical injury dataset
- **`download_real_vegas_data.py`** - Kaggle API setup and Vegas data downloader
- **`create_realistic_injuries.py`** - Generate realistic synthetic injury data
- **`integrate_betting_features.py`** - Merge synthetic betting lines with games
- **`process_real_vegas_lines.py`** - Merge real Vegas lines with games (USED IN PRODUCTION)

### `model_training/`
Scripts for training and evaluating models:

- **`run_model_comparison.py`** - Compare 11 different ML models
- **`train_legacy_model.py`** - Train baseline XGBoost model
- **`train_with_betting.py`** - Train with synthetic betting features
- **`train_with_real_vegas.py`** - Train with real Vegas data (+1.17% AUC!)
- **`retrain_all_with_vegas.py`** - Retrain all models with Vegas data (FINAL MODELS)
- **`build_ensemble.py`** - Build voting and stacking ensembles

### `analysis/`
Scripts for testing and evaluation:

- **`evaluate_injury_impact.py`** - Measure impact of injury features
- **`test_installation.py`** - Verify environment setup

## 🚀 Key Scripts for Production

If you want to retrain models from scratch:

1. **Data Collection:**
   ```bash
   python scripts/data_collection/process_real_vegas_lines.py
   ```

2. **Model Training:**
   ```bash
   python scripts/model_training/retrain_all_with_vegas.py
   ```

3. **Run App:**
   ```bash
   streamlit run src/streamlit_app_enhanced.py
   ```

## 📊 Results Summary

Best models (trained with real Vegas + injury data):
- **HistGradientBoosting: 70.20% AUC** ⭐ (BEST)
- Stacking Ensemble: 69.91% AUC
- Weighted Ensemble: 69.81% AUC
- RandomForest: 69.37% AUC
- XGBoost: 68.85% AUC

Baseline (no Vegas data): 67.68% AUC

**Improvement: +2.52% AUC from real Vegas lines!**

## 📝 Notes

- All scripts have been tested and validated
- Synthetic data generators are included as fallbacks
- Real data sources: Kaggle (Vegas), Basketball-Reference (Injuries)
- Models are saved in `models/` directory
- Data is saved in `data/` directory

