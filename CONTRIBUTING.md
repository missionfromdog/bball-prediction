# Contributing to NBA Game Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🏗️ Project Structure

```
nba-prediction-main/
├── src/                    # Main source code
│   ├── streamlit_app_enhanced.py  # Production Streamlit app
│   ├── model_comparison.py        # Model training pipeline
│   └── ...
├── scripts/                # Data collection & training scripts
│   ├── data_collection/    # Scraping & data processing
│   ├── model_training/     # Model training & evaluation
│   └── analysis/           # Testing & analysis
├── data/                   # Data files (Git LFS)
├── models/                 # Trained model files (.pkl)
├── configs/                # Configuration files
├── docs/                   # Documentation & experiment results
├── requirements_updated.txt  # Python dependencies
└── README.md              # Main documentation
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd nba-prediction-main
   ```

2. **Set up environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements_updated.txt
   ```

3. **Download data:**
   - Data files use Git LFS
   - Or retrain models from scratch using scripts in `scripts/`

## 🔬 Development Workflow

### Adding New Features

1. **Feature Engineering:**
   - Add features in `src/feature_engineering.py`
   - Document feature meaning and purpose
   - Avoid data leakage (no post-game stats!)

2. **New Data Sources:**
   - Add collection scripts to `scripts/data_collection/`
   - Process and merge with `games.csv`
   - Test impact on model performance

3. **New Models:**
   - Add to `src/model_comparison.py`
   - Compare against baseline models
   - Document performance improvements

### Testing Changes

```bash
# Test data pipeline
python scripts/data_collection/process_real_vegas_lines.py

# Test model training
python scripts/model_training/retrain_all_with_vegas.py

# Test Streamlit app
streamlit run src/streamlit_app_enhanced.py
```

## 📊 Model Evaluation Standards

- **Primary Metric:** AUC-ROC (Area Under Curve)
- **Secondary Metrics:** Accuracy, Precision, Recall, Log Loss
- **Validation:** Use TimeSeriesSplit (respect temporal ordering)
- **Baseline:** 67.68% AUC (legacy XGBoost without Vegas data)
- **Current Best:** 70.20% AUC (HistGradientBoosting with Vegas + injuries)

### Performance Benchmarks

- **Significant Improvement:** +1% AUC or more
- **Minor Improvement:** +0.1% to +1% AUC
- **Negligible:** < +0.1% AUC

## 🎯 Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and modular

## 📝 Documentation

When adding features:
1. Update relevant docstrings
2. Add usage examples
3. Document in `docs/` if significant
4. Update README if user-facing

## 🐛 Reporting Bugs

Include:
- Python version
- Operating system
- Error message and traceback
- Steps to reproduce

## 💡 Suggesting Enhancements

Good enhancement proposals include:
- Problem description
- Proposed solution
- Expected benefits (e.g., AUC improvement)
- Implementation approach

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## 🙏 Acknowledgments

- Basketball-Reference.com for injury data
- Kaggle for Vegas betting lines
- NBA Stats API for game data

