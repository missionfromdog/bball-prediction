# NBA Prediction Project - Quick Start ⚡

## 1️⃣ Activate Environment
```bash
cd /Users/caseyhess/datascience/bball/nba-prediction-main
source venv/bin/activate
```

Or use the convenience script:
```bash
source activate_env.sh
```

## 2️⃣ Verify Setup
```bash
python test_installation.py
```

## 3️⃣ Explore the Data
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## 4️⃣ Common Commands

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Key notebooks:
# - 01_eda.ipynb - Explore the data
# - 05_feature_engineering.ipynb - See how features are created
# - 07_model_testing.ipynb - Test models
# - 10_model_training_pipeline.ipynb - Full training pipeline
```

### Streamlit App
```bash
# Run the web app
streamlit run src/streamlit_app.py
```

### Python Scripts
```bash
# Import and use the modules
python
>>> from src import feature_engineering, model_training
>>> import pandas as pd
>>> df = pd.read_csv('data/games.csv')
```

### Package Management
```bash
# List installed packages
pip list

# Install new package
pip install package_name

# Freeze current environment
pip freeze > requirements_frozen.txt
```

## 5️⃣ Project Structure
```
├── venv/              # Virtual environment (activated)
├── src/               # Source code
│   ├── feature_engineering.py  # Create rolling stats
│   ├── model_training.py       # Train models
│   ├── streamlit_app.py        # Web interface
│   └── webscraping.py          # Scrape NBA data
├── notebooks/         # Jupyter notebooks
├── data/              # Datasets
├── models/            # Trained models
└── configs/           # Configuration files
```

## 6️⃣ Deactivate When Done
```bash
deactivate
```

## 🆘 Need Help?
- Read: `SETUP_GUIDE.md` for detailed instructions
- Read: `ENVIRONMENT_SETUP_SUMMARY.md` for what was set up
- Read: `README.md` for project overview
- Check: Data files exist in `data/` directory
- Verify: `which python` shows path to venv

## 🎯 What's Next?
Tell me what you want to work on:
- Improve model accuracy?
- Add new features?
- Update the web interface?
- Automate predictions?
- Something else?

