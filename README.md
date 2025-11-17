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
- [Original Project](#original-project)
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

## Original Project

This enhanced version builds upon **Chris Munch's** excellent [NBA Prediction baseline project](https://github.com/cmunch1/nba-prediction).

**For detailed information about the original methodology, data processing, and modeling approach, please visit:**

📖 **[Chris Munch's Original README](https://github.com/cmunch1/nba-prediction/blob/main/README.md)**

The original project established the foundation with:
- ✅ Excellent feature engineering (rolling averages, streaks, matchups)
- ✅ Production-ready pipeline (data scraping, model training, deployment)
- ✅ Comprehensive EDA and documentation
- ✅ XGBoost baseline model achieving 67.68% AUC

This enhanced version adds Vegas betting lines, injury data, automated workflows, and improved model performance (+2.52% AUC improvement).

---

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
