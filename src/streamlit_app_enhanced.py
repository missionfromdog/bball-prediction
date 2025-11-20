"""
Enhanced NBA Prediction Streamlit App with Model Selection & Comparison

Features:
- Multiple model support (individual models + ensembles)
- Live model comparison
- Performance metrics dashboard
- Prediction confidence analysis
- Model switching
"""

import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import subprocess
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try different import paths for compatibility
try:
    from feature_engineering import fix_datatypes, remove_non_rolling, process_features
    from constants import LONG_INTEGER_FIELDS, SHORT_INTEGER_FIELDS, DATE_FIELDS, DROP_COLUMNS, NBA_TEAMS_NAMES
    from live_odds_display import load_live_odds, match_game_to_odds, format_odds_display
    from betting_analysis import (
        analyze_betting_value, calculate_default_bankroll,
        calculate_bet_size, calculate_ev, calculate_edge
    )
except:
    from src.feature_engineering import fix_datatypes, remove_non_rolling, process_features
    from src.constants import LONG_INTEGER_FIELDS, SHORT_INTEGER_FIELDS, DATE_FIELDS, DROP_COLUMNS, NBA_TEAMS_NAMES
    try:
        from src.live_odds_display import load_live_odds, match_game_to_odds, format_odds_display
    except:
        # Fallback if live_odds_display not available
        def load_live_odds():
            return None
        def match_game_to_odds(matchup, odds_df):
            return None
        def format_odds_display(odds):
            return None
    try:
        from src.betting_analysis import (
            analyze_betting_value, calculate_default_bankroll,
            calculate_bet_size, calculate_ev, calculate_edge
        )
    except ImportError:
        # Fallback if betting_analysis not available
        def analyze_betting_value(*args, **kwargs):
            return {}
        def calculate_default_bankroll(*args, **kwargs):
            return 100.0
        def calculate_bet_size(*args, **kwargs):
            return 0.0
        def calculate_ev(*args, **kwargs):
            return 0.0
        def calculate_edge(*args, **kwargs):
            return 0.0


# Page configuration
st.set_page_config(
    page_title="NBA Game Predictor - Enhanced",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATAPATH = Path('data')
MODELSPATH = Path('models')
PREDICTIONS_PATH = DATAPATH / 'predictions'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fancy_header(text, font_size=24, color="#ff5f27"):
    """Render fancy colored header"""
    st.markdown(
        f'<span style="color:{color}; font-size: {font_size}px; font-weight: bold;">{text}</span>',
        unsafe_allow_html=True
    )


def refresh_data_from_github():
    """Pull latest data from GitHub repository"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        
        # Run git pull
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Check if there were updates
            if 'Already up to date' in result.stdout:
                return 'up_to_date', result.stdout
            else:
                return 'updated', result.stdout
        else:
            return 'error', result.stderr
            
    except subprocess.TimeoutExpired:
        return 'error', 'Git pull timed out after 30 seconds'
    except Exception as e:
        return 'error', str(e)


def load_available_models():
    """Scan models directory and load available models"""
    models = {}
    model_info = {}
    
    # Define available models and their info
    # NOTE: Only NEW models trained on 215-feature workflow dataset (30k games)
    # Old models were trained on different feature sets and are INCOMPATIBLE
    model_definitions = {
        'histgradient_vegas': {
            'name': '🏆 HistGradient + Vegas (NEW - 62.46%)',
            'description': '62.46% accuracy on 30k games with 215 features',
            'file': 'histgradient_vegas_calibrated.pkl',
            'type': 'individual',
            'is_best': True
        },
        # OLD MODELS BELOW - INCOMPATIBLE WITH NEW 215-FEATURE DATASET
        # Uncomment after retraining with new workflow dataset
        # 'ensemble_stacking_vegas': {
        #     'name': '🥈 Stacking Ensemble + Vegas (69.91%)',
        #     'description': '69.91% AUC - New stacking with Vegas data',
        #     'file': 'ensemble_stacking_vegas.pkl',
        #     'type': 'ensemble'
        # },
        # 'ensemble_weighted_vegas': {
        #     'name': '🥉 Weighted Ensemble + Vegas (69.81%)',
        #     'description': '69.81% AUC - Weighted with Vegas data',
        #     'file': 'ensemble_weighted_vegas.pkl',
        #     'type': 'ensemble'
        # },
        # 'randomforest_vegas': {
        #     'name': '🌲 RandomForest + Vegas (69.37%)',
        #     'description': '69.37% AUC - Retrained with Vegas data',
        #     'file': 'best_model_randomforest_vegas.pkl',
        #     'type': 'individual'
        # },
        # 'xgboost_vegas': {
        #     'name': '⚡ XGBoost + Vegas (68.85%)',
        #     'description': '68.85% AUC - XGBoost with Vegas data',
        #     'file': 'xgboost_vegas_calibrated.pkl',
        #     'type': 'individual'
        # },
    }
    
    # Load models that exist
    for key, info in model_definitions.items():
        model_path = MODELSPATH / info['file']
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                models[key] = model
                model_info[key] = info
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {info['name']}: {str(e)[:50]}")
    
    return models, model_info


def load_performance_metrics():
    """Load saved performance metrics if available"""
    metrics = {}
    
    # Try to load ensemble results
    ensemble_path = Path('results/ensemble_results.json')
    if ensemble_path.exists():
        with open(ensemble_path, 'r') as f:
            ensemble_data = json.load(f)
            
            # Extract individual model metrics
            for model_name, model_metrics in ensemble_data.get('individual_models', {}).items():
                key = model_name.lower().replace(' ', '_')
                if 'randomforest' in key:
                    key = 'best_model_randomforest'
                elif 'xgboost' in key:
                    key = 'xgboost_tuned_calibrated'
                    
                metrics[key] = {
                    'test_auc': model_metrics.get('auc', 0),
                    'test_accuracy': model_metrics.get('acc', 0)
                }
            
            # Extract ensemble metrics
            if 'weighted_voting' in ensemble_data:
                metrics['ensemble_weighted'] = {
                    'test_auc': ensemble_data['weighted_voting'].get('auc', 0),
                    'test_accuracy': ensemble_data['weighted_voting'].get('accuracy', 0)
                }
            
            if 'stacking' in ensemble_data:
                metrics['ensemble_stacking'] = {
                    'test_auc': ensemble_data['stacking'].get('auc', 0),
                    'test_accuracy': ensemble_data['stacking'].get('accuracy', 0),
                    'cv_auc': ensemble_data['stacking'].get('cv_auc_mean', 0),
                    'cv_std': ensemble_data['stacking'].get('cv_auc_std', 0)
                }
    
    return metrics


def predict_with_model(model, X, model_key):
    """Make predictions handling different model types"""
    try:
        if model_key == 'ensemble_weighted':
            # Weighted ensemble needs special handling
            models_list = model['models']
            weights = model['weights']
            
            pred_proba = np.zeros(len(X))
            for m, w in zip(models_list, weights):
                pred_proba += w * m.predict_proba(X)[:, 1]
            
            predictions = (pred_proba > 0.5).astype(int)
            return predictions, pred_proba
        else:
            # Standard sklearn interface
            predictions = model.predict(X)
            pred_proba = model.predict_proba(X)[:, 1]
            return predictions, pred_proba
    except Exception as e:
        import traceback
        st.error(f"❌ Error making predictions: {str(e)}")
        st.error(f"Model: {model_key}")
        st.error(f"Features in X: {X.shape[1]}")
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())
        return None, None


def prepare_data_for_prediction(df):
    """Prepare data for model prediction - keeps ALL features including injuries + Vegas"""
    # Fix datatypes
    df = fix_datatypes(df, DATE_FIELDS, SHORT_INTEGER_FIELDS, LONG_INTEGER_FIELDS)
    
    # Add matchup column for display
    df['MATCHUP'] = df['VISITOR_TEAM_ID'].map(NBA_TEAMS_NAMES) + " @ " + df['HOME_TEAM_ID'].map(NBA_TEAMS_NAMES)
    
    # Drop columns to match training logic (EXACT matches only)
    # Target and metadata
    target_cols = ['HOME_TEAM_WINS', 'TARGET']
    metadata_cols = ['GAME_DATE_EST', 'GAME_ID', 'MATCHUP', 'GAME_STATUS_TEXT', 'merge_key']
    
    # Categorical features
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'TEAM_ID_home', 'TEAM_ID_away',
        'whos_favored', 'data_source', 'is_real_vegas_line'
    ]
    
    # Leaky features (EXACT post-game stats only, not rolling averages)
    leaky_cols = [
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PTS_home', 'PTS_away',
        'PLAYOFF', 'score_home', 'score_away', 
        'q1_home', 'q2_home', 'q3_home', 'q4_home'
    ]
    
    drop_cols = target_cols + metadata_cols + categorical_cols + leaky_cols
    
    # Create feature matrix
    X = df.drop(columns=drop_cols, errors='ignore').fillna(0)
    
    # Remove duplicate columns if any
    X = X.loc[:, ~X.columns.duplicated()]
    
    return X, df


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title('🏀 NBA Game Predictor - Enhanced Edition')
    st.markdown('---')
    
    # Sidebar - Model Selection
    st.sidebar.header('⚙️ Configuration')
    
    # Load available models
    with st.spinner('Loading models...'):
        models, model_info = load_available_models()
        performance_metrics = load_performance_metrics()
    
    if not models:
        st.error("❌ No models found! Please train models first.")
        st.info("Run: `python run_model_comparison.py` or `python build_ensemble.py`")
        return
    
    st.sidebar.success(f"✅ {len(models)} model(s) loaded")
    
    # Model selector
    st.sidebar.subheader('🎯 Select Model')
    
    model_options = {key: info['name'] for key, info in model_info.items()}
    
    # Default to best model if available
    default_index = 0
    if 'histgradient_vegas' in model_options:
        default_index = list(model_options.keys()).index('histgradient_vegas')
    elif 'ensemble_stacking_vegas' in model_options:
        default_index = list(model_options.keys()).index('ensemble_stacking_vegas')
    elif 'legacy_xgboost_real_vegas' in model_options:
        default_index = list(model_options.keys()).index('legacy_xgboost_real_vegas')
    elif 'ensemble_stacking' in model_options:
        default_index = list(model_options.keys()).index('ensemble_stacking')
    
    selected_model_key = st.sidebar.selectbox(
        'Choose prediction model:',
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=default_index
    )
    
    selected_model = models[selected_model_key]
    selected_info = model_info[selected_model_key]
    
    # Show model info
    st.sidebar.markdown(f"**{selected_info['name']}**")
    st.sidebar.caption(selected_info['description'])
    
    # Show if it's the best model
    if selected_info.get('is_best'):
        st.sidebar.success("⭐ This is your BEST model!")
    
    # Show performance if available
    if selected_model_key in performance_metrics:
        metrics = performance_metrics[selected_model_key]
        st.sidebar.markdown("**Performance:**")
        if 'test_auc' in metrics and metrics['test_auc'] > 0:
            st.sidebar.metric("Test AUC", f"{metrics['test_auc']:.4f}")
        if 'test_accuracy' in metrics and metrics['test_accuracy'] > 0:
            st.sidebar.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
    
    # Data refresh button
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔄 Data Management**")
    
    if st.sidebar.button("🔄 Refresh Data from GitHub", help="Pull latest games and predictions from GitHub"):
        with st.spinner("Pulling latest data from GitHub..."):
            status, message = refresh_data_from_github()
            
            if status == 'updated':
                st.sidebar.success("✅ Data updated! Reloading app...")
                st.rerun()
            elif status == 'up_to_date':
                st.sidebar.info("ℹ️ Already up to date")
            else:
                st.sidebar.error(f"❌ Update failed: {message}")
    
    # Auto-load all models for comparison (no checkbox needed)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔬 Model Comparison**")
    
    # Get all other models for comparison
    compare_models = [k for k in model_options.keys() if k != selected_model_key]
    
    if compare_models:
        st.sidebar.caption(f"Comparing {len(compare_models)} other model(s)")
    else:
        st.sidebar.caption("No other models available for comparison")
        st.sidebar.info("💡 Old models need retraining for new 102-feature dataset")
    
    # Betting Analysis Settings (will be populated after loading games)
    # Store bankroll in session state
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 100.0
    
    # Load live odds
    live_odds_df = load_live_odds()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Predictions', '💰 Odds Comparison', '📈 Performance', '📜 History', 'ℹ️ About'])
    
    # ========================================================================
    # TAB 1: PREDICTIONS
    # ========================================================================
    with tab1:
        fancy_header('Today\'s Game Predictions', font_size=28)
        
        # Show data freshness info
        col1, col2, col3 = st.columns(3)
        with col1:
            # Check live odds freshness
            odds_path = DATAPATH / 'betting' / 'live_odds_latest.csv'
            if odds_path.exists():
                odds_time = datetime.fromtimestamp(odds_path.stat().st_mtime)
                st.info(f"🎲 Live Odds: {odds_time.strftime('%b %d, %I:%M %p')}")
            else:
                st.warning("🎲 No live odds")
        
        with col2:
            # Check injury data freshness
            injury_path = DATAPATH / 'injuries' / 'nba_injuries_real_scraped.csv'
            if injury_path.exists():
                injury_time = datetime.fromtimestamp(injury_path.stat().st_mtime)
                st.info(f"🏥 Injuries: {injury_time.strftime('%b %d, %I:%M %p')}")
            else:
                st.warning("🏥 No injury data")
        
        with col3:
            # Check main data freshness
            data_path = DATAPATH / 'games_with_real_vegas.csv'
            if data_path.exists():
                data_time = datetime.fromtimestamp(data_path.stat().st_mtime)
                st.info(f"📊 Data: {data_time.strftime('%b %d, %I:%M %p')}")
        
        # Show live odds status
        if live_odds_df is not None and len(live_odds_df) > 0:
            # Note: Odds API returns multiple days, we filter to today's games below
            st.success(f"✅ Live Vegas odds available ({len(live_odds_df)} games in feed)")
        
        st.markdown("")
        
        # Load data
        try:
            # Load data with ALL features (injuries + Vegas)
            # Priority: workflow dataset > streamlit sample > old dataset
            workflow_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
            streamlit_sample = DATAPATH / 'games_streamlit_sample.csv'
            
            if workflow_file.exists() and workflow_file.stat().st_size > 1000:
                df_full = pd.read_csv(workflow_file)
                data_source = "workflow"
            elif streamlit_sample.exists():
                # Use smaller sample for Streamlit Cloud (500 recent games)
                df_full = pd.read_csv(streamlit_sample)
                data_source = "sample"
            else:
                # Final fallback to old dataset
                df_full = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
                data_source = "legacy"
            
            # Check if features are already engineered (240+ columns means engineered)
            if len(df_full.columns) < 200:
                with st.spinner(f"Engineering features from {data_source} dataset... (this may take a minute)"):
                    # Features not present - need to engineer them
                    df_full = process_features(df_full)
                    st.success(f"✅ Engineered {len(df_full.columns)} features")
            
            # Get current season
            current_season = datetime.today().year
            if datetime.today().month < 10:
                current_season = current_season - 1
            
            df_current_season = df_full[df_full['SEASON'] == current_season]
            
            # Get today's games (unplayed + today's date only)
            today = pd.to_datetime(datetime.today().date())
            df_current_season['GAME_DATE_EST'] = pd.to_datetime(df_current_season['GAME_DATE_EST'], errors='coerce')
            
            # Filter for today's date AND unplayed
            df_today = df_current_season[
                (df_current_season['PTS_home'] == 0) & 
                (df_current_season['GAME_DATE_EST'].dt.date == today.date())
            ]
            df_past = df_current_season[df_current_season['PTS_home'] != 0]
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Make sure data/games_with_real_vegas.csv exists and is up to date.")
            import traceback
            st.code(traceback.format_exc())
            return
        
        # Today's games section
        if len(df_today) == 0:
            st.warning("🤷‍♂️ No games scheduled for today!")
            st.info("NBA season typically runs October - June. Check back during the season!")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"🏀 {len(df_today)} game(s) scheduled today!")
            with col2:
                st.markdown("")  # Spacing
            
            # Prepare data
            X_today, df_today_display = prepare_data_for_prediction(df_today)
            
            # Main model predictions
            predictions, pred_proba = predict_with_model(selected_model, X_today, selected_model_key)
            
            # Check if predictions succeeded
            if predictions is None or pred_proba is None:
                st.error("❌ Prediction failed. Please check the error above or try a different model.")
                return
            
            df_today_display['HOME_WIN_PROB'] = pred_proba
            df_today_display['PREDICTION'] = ['Home' if p == 1 else 'Away' for p in predictions]
            
            # Add confidence levels
            df_today_display['CONFIDENCE'] = ['High' if abs(p-0.5) > 0.15 else 'Medium' if abs(p-0.5) > 0.05 else 'Low' 
                                              for p in pred_proba]
            
            # Reset index to avoid indexing issues
            df_today_display = df_today_display.reset_index(drop=True)
            
            # Calculate default bankroll and update sidebar
            default_bankroll = calculate_default_bankroll(len(df_today_display))
            if 'bankroll' not in st.session_state:
                st.session_state.bankroll = default_bankroll
            
            # Bankroll input in sidebar (update the placeholder)
            with st.sidebar:
                st.markdown("**💰 Betting Analysis**")
                user_bankroll = st.number_input(
                    "Bankroll ($)",
                    min_value=10.0,
                    max_value=10000.0,
                    value=float(st.session_state.bankroll),
                    step=10.0,
                    help=f"Default: ${default_bankroll:.0f} (~$10/game for {len(df_today_display)} games)"
                )
                st.session_state.bankroll = user_bankroll
            
            # Calculate betting analysis for each game
            betting_analyses = []
            for idx, row in df_today_display.iterrows():
                # Get live odds for this game
                live_odds = match_game_to_odds(row['MATCHUP'], live_odds_df) if live_odds_df is not None else None
                
                # Determine if we're betting on home or away
                is_home_bet = predictions[idx] == 1
                
                # Extract moneylines
                home_ml = None
                away_ml = None
                if live_odds is not None:
                    home_ml = live_odds.get('home_ml', None)
                    away_ml = live_odds.get('away_ml', None)
                    # Convert to float if they're strings
                    try:
                        if home_ml is not None:
                            home_ml = float(home_ml)
                        if away_ml is not None:
                            away_ml = float(away_ml)
                    except (ValueError, TypeError):
                        home_ml = None
                        away_ml = None
                
                # Perform betting analysis
                analysis = analyze_betting_value(
                    model_prob=pred_proba[idx],
                    home_ml=home_ml,
                    away_ml=away_ml,
                    is_home_bet=is_home_bet
                )
                
                # Add bet size calculation
                if not pd.isna(analysis.get('kelly_fraction', np.nan)):
                    analysis['bet_size'] = calculate_bet_size(
                        analysis['kelly_fraction'],
                        st.session_state.bankroll
                    )
                else:
                    analysis['bet_size'] = 0.0
                
                betting_analyses.append(analysis)
            
            # Add betting columns to display dataframe
            df_today_display['EDGE'] = [a.get('edge', np.nan) for a in betting_analyses]
            df_today_display['EV'] = [a.get('expected_value', np.nan) for a in betting_analyses]
            df_today_display['KELLY'] = [a.get('kelly_fraction', np.nan) for a in betting_analyses]
            df_today_display['BET_SIZE'] = [a.get('bet_size', 0.0) for a in betting_analyses]
            df_today_display['HAS_VALUE'] = [a.get('has_value', False) for a in betting_analyses]
            
            # Comparison predictions if enabled
            comparison_results = {}
            if compare_models:
                for comp_key in compare_models:
                    comp_model = models[comp_key]
                    comp_pred, comp_proba = predict_with_model(comp_model, X_today, comp_key)
                    # Only add if predictions succeeded
                    if comp_pred is not None and comp_proba is not None:
                        comparison_results[comp_key] = {
                            'proba': comp_proba,
                            'name': model_info[comp_key]['name']
                        }
            
            # Create export DataFrame (include betting metrics)
            export_df = pd.DataFrame({
                'Date': [datetime.now().strftime('%Y-%m-%d')] * len(df_today_display),
                'Matchup': df_today_display['MATCHUP'],
                'Home_Win_Probability': [f"{p:.1%}" for p in pred_proba],
                'Predicted_Winner': df_today_display['PREDICTION'],
                'Confidence': df_today_display['CONFIDENCE'],
                'Edge_%': [f"{e:+.1%}" if not pd.isna(e) else "N/A" for e in df_today_display['EDGE']],
                'Expected_Value': [f"${ev:.2f}" if not pd.isna(ev) else "N/A" for ev in df_today_display['EV']],
                'Kelly_%': [f"{k:.1%}" if not pd.isna(k) else "N/A" for k in df_today_display['KELLY']],
                'Bet_Size': [f"${bs:.2f}" if bs > 0 else "N/A" for bs in df_today_display['BET_SIZE']],
                'Has_Value': df_today_display['HAS_VALUE'].apply(lambda x: "Yes" if x else "No"),
                'Model': [selected_info['name']] * len(df_today_display)
            })
            
            # Add download button and value bets filter
            col_title, col_value, col_download = st.columns([2, 1, 1])
            with col_title:
                st.subheader("📊 Today's Predictions")
            with col_value:
                # Count value bets
                value_bets_count = df_today_display['HAS_VALUE'].sum()
                if value_bets_count > 0:
                    st.success(f"🎯 {value_bets_count} Value Bet(s)")
                else:
                    st.info("🎯 No value bets")
            with col_download:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"nba_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download predictions as CSV for Google Sheets"
                )
            
            st.markdown("")  # Spacing
            
            # Value Bets Section (if any)
            value_bets_df = df_today_display[df_today_display['HAS_VALUE'] == True]
            if len(value_bets_df) > 0:
                with st.expander(f"🎯 Value Bets ({len(value_bets_df)} games)", expanded=True):
                    st.markdown("**Games with positive EV and edge:**")
                    value_display = value_bets_df[['MATCHUP', 'HOME_WIN_PROB', 'EDGE', 'EV', 'KELLY', 'BET_SIZE']].copy()
                    value_display['HOME_WIN_PROB'] = value_display['HOME_WIN_PROB'].apply(lambda x: f"{x:.1%}")
                    value_display['EDGE'] = value_display['EDGE'].apply(lambda x: f"{x:+.1%}" if not pd.isna(x) else "N/A")
                    value_display['EV'] = value_display['EV'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
                    value_display['KELLY'] = value_display['KELLY'].apply(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
                    value_display['BET_SIZE'] = value_display['BET_SIZE'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                    value_display.columns = ['Matchup', 'Win Prob', 'Edge', 'EV', 'Kelly %', 'Bet Size']
                    st.dataframe(value_display, use_container_width=True, hide_index=True)
                st.markdown("")  # Spacing
            
            # Display predictions
            for idx, row in df_today_display.iterrows():
                # Use different layout if betting metrics available
                has_betting_data = not pd.isna(row.get('EDGE', np.nan))
                
                if has_betting_data:
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                else:
                    col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Highlight value bets
                    if row.get('HAS_VALUE', False):
                        st.markdown(f"### 🎯 {row['MATCHUP']} ⭐ VALUE BET")
                    else:
                        st.markdown(f"### {row['MATCHUP']}")
                    st.caption(f"{row['GAME_DATE_EST']}")
                    
                    # Display live odds if available
                    live_odds = match_game_to_odds(row['MATCHUP'], live_odds_df)
                    if live_odds is not None:
                        odds_display = format_odds_display(live_odds)
                        if odds_display:
                            st.markdown("**🎲 Live Vegas Odds:**")
                            odds_cols = st.columns(3)
                            
                            with odds_cols[0]:
                                if 'spread' in odds_display:
                                    st.caption(f"Spread: {odds_display['spread']}")
                                else:
                                    st.caption("Spread: --")
                            
                            with odds_cols[1]:
                                if 'total' in odds_display:
                                    st.caption(f"O/U: {odds_display['total']}")
                                else:
                                    st.caption("O/U: --")
                            
                            with odds_cols[2]:
                                if 'ml_home' in odds_display:
                                    st.caption(f"ML: {odds_display['ml_home']}")
                                else:
                                    st.caption("ML: --")
                
                with col2:
                    prob = row['HOME_WIN_PROB']
                    st.metric(
                        "Home Win Probability",
                        f"{prob:.1%}",
                        delta=f"{prob - 0.5:.1%}" if prob != 0.5 else None
                    )
                
                with col3:
                    prediction = row['PREDICTION']
                    confidence = abs(prob - 0.5) * 2
                    
                    if confidence > 0.4:
                        conf_text = "🔥 High"
                        conf_color = "green"
                    elif confidence > 0.2:
                        conf_text = "⚖️ Medium"
                        conf_color = "orange"
                    else:
                        conf_text = "⚠️ Low"
                        conf_color = "red"
                    
                    st.markdown(f"**Winner:** {prediction}")
                    st.markdown(f"**Confidence:** :{conf_color}[{conf_text}]")
                
                # Add betting metrics column if available
                if has_betting_data:
                    with col4:
                        st.markdown("**💰 Betting Analysis:**")
                        edge = row.get('EDGE', np.nan)
                        ev = row.get('EV', np.nan)
                        kelly = row.get('KELLY', np.nan)
                        bet_size = row.get('BET_SIZE', 0.0)
                        
                        if not pd.isna(edge):
                            edge_color = "green" if edge > 0.05 else "orange" if edge > 0 else "red"
                            st.markdown(f"**Edge:** :{edge_color}[{edge:+.1%}]")
                        
                        if not pd.isna(ev):
                            ev_color = "green" if ev > 0 else "red"
                            st.markdown(f"**EV:** :{ev_color}[${ev:.2f}]")
                        
                        if not pd.isna(kelly) and kelly > 0:
                            st.markdown(f"**Kelly:** {kelly:.1%}")
                            if bet_size > 0:
                                st.markdown(f"**Bet:** ${bet_size:.2f}")
                        else:
                            st.caption("No bet recommended")
                
                # Always show comparison (expanded by default)
                if comparison_results:
                    with st.expander(f"📊 Compare All Models for this game", expanded=True):
                        comp_df = pd.DataFrame({
                            'Model': [selected_info['name']] + [comparison_results[k]['name'] for k in comparison_results],
                            'Home Win Prob': [prob] + [comparison_results[k]['proba'][idx] for k in comparison_results],
                        })
                        comp_df['Winner'] = comp_df['Home Win Prob'].apply(lambda x: 'Home' if x > 0.5 else 'Away')
                        comp_df['Confidence'] = comp_df['Home Win Prob'].apply(
                            lambda x: 'High' if abs(x - 0.5) > 0.15 else 'Medium' if abs(x - 0.5) > 0.05 else 'Low'
                        )
                        comp_df['Home Win Prob'] = comp_df['Home Win Prob'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
        
        # Past games section
        st.markdown("### 📅 Recent Performance (Last 25 Games)")
        
        if len(df_past) > 0:
            # Sort and limit FIRST (before preparing data)
            df_past_display = df_past.sort_values('GAME_DATE_EST', ascending=False).head(25).copy()
            df_past_display = df_past_display.reset_index(drop=True)
            
            # Prepare past games data for the sorted/limited dataframe
            X_past_limited, df_past_display = prepare_data_for_prediction(df_past_display)
            
            # Get predictions for the sorted/limited dataframe
            past_predictions, past_proba = predict_with_model(selected_model, X_past_limited, selected_model_key)
            
            df_past_display['HOME_WIN_PROB'] = past_proba
            df_past_display['PREDICTED_WINNER'] = past_predictions
            df_past_display['ACTUAL_WINNER'] = df_past_display['HOME_TEAM_WINS']
            df_past_display['CORRECT'] = df_past_display['PREDICTED_WINNER'] == df_past_display['ACTUAL_WINNER']
            
            # Calculate betting analysis for past games
            past_betting_analyses = []
            for idx, row in df_past_display.iterrows():
                # Get live odds for this game (may not be available for past games)
                live_odds = match_game_to_odds(row['MATCHUP'], live_odds_df) if live_odds_df is not None else None
                
                # Determine if we're betting on home or away
                # idx now matches the position in past_predictions since we reset_index
                is_home_bet = past_predictions[idx] == 1
                
                # Extract moneylines
                home_ml = None
                away_ml = None
                if live_odds is not None:
                    home_ml = live_odds.get('home_ml', None)
                    away_ml = live_odds.get('away_ml', None)
                    # Convert to float if they're strings
                    try:
                        if home_ml is not None:
                            home_ml = float(home_ml)
                        if away_ml is not None:
                            away_ml = float(away_ml)
                    except (ValueError, TypeError):
                        home_ml = None
                        away_ml = None
                
                # Also check if odds are in the dataset itself (historical odds)
                if home_ml is None and 'moneyline_home' in row and pd.notna(row.get('moneyline_home')):
                    try:
                        home_ml = float(row['moneyline_home'])
                    except (ValueError, TypeError):
                        pass
                if away_ml is None and 'moneyline_away' in row and pd.notna(row.get('moneyline_away')):
                    try:
                        away_ml = float(row['moneyline_away'])
                    except (ValueError, TypeError):
                        pass
                
                # Perform betting analysis
                # idx now matches the position in past_proba since we reset_index
                analysis = analyze_betting_value(
                    model_prob=past_proba[idx],
                    home_ml=home_ml,
                    away_ml=away_ml,
                    is_home_bet=is_home_bet
                )
                
                past_betting_analyses.append(analysis)
            
            # Add betting columns to display dataframe
            df_past_display['EDGE'] = [a.get('edge', np.nan) for a in past_betting_analyses]
            df_past_display['HAS_VALUE'] = [a.get('has_value', False) for a in past_betting_analyses]
            
            # Calculate accuracy
            accuracy = df_past_display['CORRECT'].mean()
            correct_count = df_past_display['CORRECT'].sum()
            total_count = len(df_past_display)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy (Last 25)", f"{accuracy:.1%}")
            with col2:
                st.metric("Correct Predictions", f"{correct_count}/{total_count}")
            with col3:
                avg_confidence = (df_past_display['HOME_WIN_PROB'] - 0.5).abs().mean() * 2
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Display table with betting metrics
            display_df = df_past_display[[
                'GAME_DATE_EST', 'MATCHUP', 'HOME_WIN_PROB', 
                'ACTUAL_WINNER', 'CORRECT', 'EDGE', 'HAS_VALUE'
            ]].copy()
            
            display_df.columns = ['Date', 'Matchup', 'Home Win Prob', 'Actual Winner', 'Correct', 'Edge', 'Value Bet']
            display_df['Home Win Prob'] = display_df['Home Win Prob'].apply(lambda x: f"{x:.1%}")
            display_df['Actual Winner'] = display_df['Actual Winner'].apply(lambda x: 'Home' if x == 1 else 'Away')
            display_df['Correct'] = display_df['Correct'].apply(lambda x: '✅' if x else '❌')
            
            # Format Edge column
            display_df['Edge'] = display_df['Edge'].apply(
                lambda x: f"{x:+.1%}" if not pd.isna(x) else "N/A"
            )
            
            # Format Value Bet column
            display_df['Value Bet'] = display_df['Value Bet'].apply(
                lambda x: '✅ Yes' if x else '❌ No' if not pd.isna(x) else 'N/A'
            )
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No past games available for this season yet.")
    
    # ========================================================================
    # TAB 2: ODDS COMPARISON
    # ========================================================================
    with tab2:
        fancy_header('US Sportsbook Odds Comparison', font_size=28)
        st.markdown("Compare vig (bookmaker margin) across US sportsbooks to find the best odds.")
        st.markdown("---")
        
        # Load bookmaker comparison data
        comparison_path = DATAPATH / 'betting' / 'live_odds_bookmakers_comparison.csv'
        
        if comparison_path.exists():
            try:
                df_comparison = pd.read_csv(comparison_path)
                
                if df_comparison.empty:
                    st.warning("⚠️ No bookmaker comparison data available.")
                else:
                    # Create matchup column
                    df_comparison['Matchup'] = df_comparison['away_team'] + ' @ ' + df_comparison['home_team']
                    
                    # Get unique games
                    unique_games = df_comparison['Matchup'].unique()
                    
                    if len(unique_games) == 0:
                        st.warning("⚠️ No games found in comparison data.")
                    else:
                        # Select game
                        selected_game = st.selectbox(
                            "Select Game:",
                            options=unique_games,
                            index=0
                        )
                        
                        # Filter for selected game
                        game_data = df_comparison[df_comparison['Matchup'] == selected_game].copy()
                        
                        if game_data.empty:
                            st.warning(f"⚠️ No bookmaker data for {selected_game}")
                        else:
                            # Create comparison table
                            comparison_rows = []
                            
                            for _, row in game_data.iterrows():
                                bookmaker_name = row['bookmaker']
                                
                                # Format vig values
                                ml_vig = f"{row['ml_vig']:.2f}%" if pd.notna(row.get('ml_vig')) else "N/A"
                                spread_vig = f"{row['spread_vig']:.2f}%" if pd.notna(row.get('spread_vig')) else "N/A"
                                total_vig = f"{row['total_vig']:.2f}%" if pd.notna(row.get('total_vig')) else "N/A"
                                
                                # Format odds
                                home_ml = f"{row['home_ml']:+.0f}" if pd.notna(row.get('home_ml')) else "N/A"
                                away_ml = f"{row['away_ml']:+.0f}" if pd.notna(row.get('away_ml')) else "N/A"
                                spread = f"{row.get('home_spread', 0):+.1f}" if pd.notna(row.get('home_spread')) else "N/A"
                                total = f"{row.get('total', 0):.1f}" if pd.notna(row.get('total')) else "N/A"
                                
                                comparison_rows.append({
                                    'Sportsbook': bookmaker_name,
                                    'Moneyline Vig': ml_vig,
                                    'Home ML': home_ml,
                                    'Away ML': away_ml,
                                    'Spread Vig': spread_vig,
                                    'Spread': spread,
                                    'Total Vig': total_vig,
                                    'O/U': total,
                                })
                            
                            comparison_df = pd.DataFrame(comparison_rows)
                            
                            # Helper function to extract numeric value from vig string
                            def extract_vig_value(vig_str):
                                """Extract numeric value from vig string, handling emojis and %"""
                                if vig_str == "N/A" or pd.isna(vig_str):
                                    return None
                                # Remove emojis, stars, and % signs, then convert to float
                                cleaned = str(vig_str).replace('⭐', '').replace('%', '').strip()
                                try:
                                    return float(cleaned)
                                except (ValueError, TypeError):
                                    return None
                            
                            # Find best (lowest) vig for each market from original values
                            ml_vigs = [extract_vig_value(v) for v in comparison_df['Moneyline Vig']]
                            spread_vigs = [extract_vig_value(v) for v in comparison_df['Spread Vig']]
                            total_vigs = [extract_vig_value(v) for v in comparison_df['Total Vig']]
                            
                            # Filter out None values and find minimums
                            best_ml = min([v for v in ml_vigs if v is not None]) if any(v is not None for v in ml_vigs) else None
                            best_spread = min([v for v in spread_vigs if v is not None]) if any(v is not None for v in spread_vigs) else None
                            best_total = min([v for v in total_vigs if v is not None]) if any(v is not None for v in total_vigs) else None
                            
                            # Format rows with star for best vig
                            def format_row(row):
                                """Format row with best vig highlighting"""
                                ml_vig_val = row['Moneyline Vig']
                                spread_vig_val = row['Spread Vig']
                                total_vig_val = row['Total Vig']
                                
                                # Format with star for best (compare numeric values)
                                if ml_vig_val != "N/A" and best_ml is not None:
                                    ml_num = extract_vig_value(ml_vig_val)
                                    if ml_num is not None and abs(ml_num - best_ml) < 0.001:  # Use small epsilon for float comparison
                                        row['Moneyline Vig'] = f"⭐ {ml_vig_val.replace('⭐', '').strip()}"
                                
                                if spread_vig_val != "N/A" and best_spread is not None:
                                    spread_num = extract_vig_value(spread_vig_val)
                                    if spread_num is not None and abs(spread_num - best_spread) < 0.001:
                                        row['Spread Vig'] = f"⭐ {spread_vig_val.replace('⭐', '').strip()}"
                                
                                if total_vig_val != "N/A" and best_total is not None:
                                    total_num = extract_vig_value(total_vig_val)
                                    if total_num is not None and abs(total_num - best_total) < 0.001:
                                        row['Total Vig'] = f"⭐ {total_vig_val.replace('⭐', '').strip()}"
                                
                                return row
                            
                            display_df = comparison_df.apply(format_row, axis=1)
                            
                            # Display table
                            st.subheader(f"📊 {selected_game}")
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Add legend
                            st.caption("⭐ = Best (lowest) vig for that market")
                            
                            # Summary stats
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if best_ml is not None:
                                    st.metric("Best ML Vig", f"{best_ml:.2f}%")
                            
                            with col2:
                                if best_spread is not None:
                                    st.metric("Best Spread Vig", f"{best_spread:.2f}%")
                            
                            with col3:
                                if best_total is not None:
                                    st.metric("Best Total Vig", f"{best_total:.2f}%")
                            
                            # Show all games summary
                            st.markdown("---")
                            st.subheader("📋 All Upcoming Games")
                            
                            # Create summary table for all games
                            summary_rows = []
                            for game in unique_games:
                                game_df = df_comparison[df_comparison['Matchup'] == game]
                                
                                # Find best vigs
                                ml_vigs = game_df['ml_vig'].dropna()
                                spread_vigs = game_df['spread_vig'].dropna()
                                total_vigs = game_df['total_vig'].dropna()
                                
                                best_ml_vig = ml_vigs.min() if not ml_vigs.empty else None
                                best_spread_vig = spread_vigs.min() if not spread_vigs.empty else None
                                best_total_vig = total_vigs.min() if not total_vigs.empty else None
                                
                                # Find which bookmaker has best vig
                                best_ml_book = game_df.loc[game_df['ml_vig'].idxmin(), 'bookmaker'] if not ml_vigs.empty else "N/A"
                                best_spread_book = game_df.loc[game_df['spread_vig'].idxmin(), 'bookmaker'] if not spread_vigs.empty else "N/A"
                                best_total_book = game_df.loc[game_df['total_vig'].idxmin(), 'bookmaker'] if not total_vigs.empty else "N/A"
                                
                                summary_rows.append({
                                    'Game': game,
                                    'Best ML Vig': f"{best_ml_vig:.2f}%" if best_ml_vig else "N/A",
                                    'Best ML Book': best_ml_book,
                                    'Best Spread Vig': f"{best_spread_vig:.2f}%" if best_spread_vig else "N/A",
                                    'Best Spread Book': best_spread_book,
                                    'Best Total Vig': f"{best_total_vig:.2f}%" if best_total_vig else "N/A",
                                    'Best Total Book': best_total_book,
                                })
                            
                            summary_df = pd.DataFrame(summary_rows)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
            except Exception as e:
                st.error(f"❌ Error loading bookmaker comparison data: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Bookmaker comparison data not found.")
            st.info("💡 Run the 'Fetch Live NBA Odds' workflow to generate comparison data.")
    
    # ========================================================================
    # TAB 3: PERFORMANCE
    # ========================================================================
    with tab3:
        fancy_header('Model Performance Analysis', font_size=28)
        st.markdown("")
        
        # ====================================================================
        # SECTION 1: Model Training Performance (Always Available)
        # ====================================================================
        st.subheader("🎯 Model Training Performance")
        st.caption("Performance on historical test data (2003-2025)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", "62.82%", help="Accuracy on held-out test set")
        with col2:
            st.metric("Training Accuracy", "67.62%", help="Accuracy on training set")
        with col3:
            st.metric("Training Games", "30,120", help="Total games used for training")
        with col4:
            st.metric("Features", "102", help="Injury + Vegas + Rolling averages")
        
        st.markdown("")
        
        # Model details
        with st.expander("📋 Model Details", expanded=False):
            st.markdown("""
            **Model Type:** HistGradientBoostingClassifier (Calibrated)
            
            **Feature Breakdown:**
            - 🏥 **14 Injury Features** - Player availability, star injuries, impact scores
            - 🎲 **4 Vegas Features** - Spread, total, moneylines
            - 📊 **83 Rolling Features** - Win rates, points, rebounds, assists (3/7/10 game windows)
            - 📅 **1 Date Feature** - Month (seasonality)
            
            **Training Configuration:**
            - Algorithm: HistGradientBoosting with isotonic calibration
            - Data Split: 80% train / 20% test
            - Date Range: Oct 2003 → Nov 2025
            - Stratified sampling: Yes (preserves win/loss ratio)
            
            **Performance Comparison:**
            - Baseline (coin flip): 50%
            - Our model: 62.82%
            - Vegas (professional): ~64-68%
            - **Gap to Vegas:** ~1-5% (getting closer!)
            """)
        
        st.markdown("---")
        
        # ====================================================================
        # SECTION 2: Live Prediction Tracking (From Workflow)
        # ====================================================================
        st.subheader("📈 Live Prediction Tracking")
        
        # Try to load performance metrics from workflow
        perf_metrics_path = DATAPATH / 'predictions' / 'performance_metrics_latest.csv'
        perf_detailed_path = DATAPATH / 'predictions' / 'detailed_tracking_latest.csv'
        
        if perf_metrics_path.exists() and perf_detailed_path.exists():
            # Load performance data
            try:
                metrics_df = pd.read_csv(perf_metrics_path)
                detailed_df = pd.read_csv(perf_detailed_path)
                
                st.caption("Real-world prediction performance (updated nightly)")
                st.markdown("")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    acc = metrics_df['Overall_Accuracy'].iloc[0]
                    st.metric("Overall Accuracy", f"{acc:.1%}")
                with col2:
                    total = metrics_df['Total_Predictions'].iloc[0]
                    correct = metrics_df['Correct_Predictions'].iloc[0]
                    st.metric("Predictions", f"{correct}/{total}")
                with col3:
                    if 'ROI' in metrics_df.columns:
                        roi = metrics_df['ROI'].iloc[0]
                        st.metric("ROI", f"{roi:+.1f}%")
                    else:
                        st.metric("Games Tracked", f"{total}")
                with col4:
                    if 'Total_Profit' in metrics_df.columns:
                        profit = metrics_df['Total_Profit'].iloc[0]
                        st.metric("Profit/Loss", f"${profit:+,.0f}")
                    else:
                        # Calculate win rate
                        win_rate = correct / total if total > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1%}")
                
                st.markdown("")
                
                # Performance by confidence
                st.markdown("**🎯 Accuracy by Confidence Level**")
                conf_cols = st.columns(3)
                
                for idx, conf in enumerate(['High', 'Medium', 'Low']):
                    count_col = f'{conf}_Confidence_Predictions'
                    acc_col = f'{conf}_Confidence_Accuracy'
                    
                    if count_col in metrics_df.columns and acc_col in metrics_df.columns:
                        with conf_cols[idx]:
                            count = int(metrics_df[count_col].iloc[0])
                            acc = metrics_df[acc_col].iloc[0]
                            st.metric(f"{conf} Confidence", f"{acc:.1%}", f"{count} games")
                
                st.markdown("---")
                
                # Recent games
                st.markdown("**🕒 Recent Predictions (Last 10)**")
                recent = detailed_df.sort_values('Date', ascending=False).head(10)
                
                display_recent = recent[['Date', 'Matchup', 'Predicted_Winner', 'Actual_Winner', 'Correct', 'Confidence']].copy()
                display_recent['Date'] = pd.to_datetime(display_recent['Date']).dt.strftime('%Y-%m-%d')
                display_recent['Result'] = display_recent['Correct'].apply(lambda x: '✅' if x else '❌')
                display_recent = display_recent.drop('Correct', axis=1)
                
                st.dataframe(display_recent, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error loading workflow performance data: {e}")
                st.caption("Showing training performance only (above)")
        else:
            # No workflow data yet - show helpful message
            st.caption("Real-world tracking not available yet")
            st.info("""
            **📊 How to enable live tracking:**
            
            1. **Make Predictions** - Run `Daily NBA Predictions` workflow
            2. **Wait for Results** - Games must complete (next day)
            3. **Track Performance** - Run `Track Betting Performance` workflow
            
            **What you'll see:**
            - Overall accuracy on real predictions
            - Accuracy by confidence level (High/Medium/Low)
            - ROI and profit/loss (if betting $100/game)
            - Recent prediction results with outcomes
            
            💡 For now, see **Model Training Performance** above for test set accuracy.
            """)
        
        st.markdown("---")
        
        # ====================================================================
        # SECTION 3: Dataset Statistics
        # ====================================================================
        st.subheader("📊 Dataset Statistics")
        
        try:
            # Load dataset stats - use same priority as predictions
            workflow_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
            streamlit_sample = DATAPATH / 'games_streamlit_sample.csv'
            legacy_file = DATAPATH / 'games_with_real_vegas.csv'
            
            df_stats = None
            data_source_name = None
            
            if workflow_file.exists() and workflow_file.stat().st_size > 1000:
                df_stats = pd.read_csv(workflow_file, low_memory=False)
                data_source_name = "Workflow Dataset"
            elif streamlit_sample.exists():
                df_stats = pd.read_csv(streamlit_sample, low_memory=False)
                data_source_name = "Sample Dataset"
            elif legacy_file.exists():
                df_stats = pd.read_csv(legacy_file, low_memory=False)
                data_source_name = "Legacy Dataset"
            
            if df_stats is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_games = len(df_stats)
                    st.metric("Total Games", f"{total_games:,}")
                with col2:
                    completed_games = len(df_stats[df_stats['PTS_home'] != 0])
                    st.metric("Completed Games", f"{completed_games:,}")
                with col3:
                    upcoming_games = len(df_stats[df_stats['PTS_home'] == 0])
                    st.metric("Upcoming Games", f"{upcoming_games}")
                with col4:
                    num_features = len(df_stats.columns)
                    st.metric("Features", f"{num_features}")
                
                st.markdown("")
                
                # Date range
                df_stats['GAME_DATE_EST'] = pd.to_datetime(df_stats['GAME_DATE_EST'], errors='coerce')
                min_date = df_stats['GAME_DATE_EST'].min().strftime('%B %d, %Y')
                max_date = df_stats['GAME_DATE_EST'].max().strftime('%B %d, %Y')
                
                st.caption(f"📅 Date Range: {min_date} → {max_date}")
                st.caption(f"📂 Source: {data_source_name}")
                
            else:
                st.warning("No dataset found. Please ensure data files exist in the data/ directory.")
        except Exception as e:
            st.error(f"Error loading dataset stats: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 4: HISTORY
    # ========================================================================
    with tab4:
        fancy_header('Historical Performance', font_size=28)
        st.markdown("Comprehensive performance metrics across all predictions since tracking began.")
        st.markdown("---")
        
        @st.cache_data(ttl=300)  # Cache for 5 minutes (shorter to pick up new files faster)
        def load_all_historical_predictions():
            """Load all historical prediction files and match with actual results"""
            try:
                # Load all prediction files (including latest, but deduplicate)
                pred_files = list(PREDICTIONS_PATH.glob('predictions_*.csv'))
                # Include latest file as it may have the most recent predictions
                # We'll deduplicate by date+matchup later
                
                if not pred_files:
                    return None, None
                
                # Load and combine all predictions
                all_preds = []
                for file in pred_files:
                    try:
                        df = pd.read_csv(file)
                        df['Prediction_File'] = file.name
                        all_preds.append(df)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {file.name}: {e}")
                        continue
                
                if not all_preds:
                    return None, None
                
                predictions_df = pd.concat(all_preds, ignore_index=True)
                predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
                
                # Deduplicate by Date + Matchup (keep most recent if duplicates)
                # This handles cases where predictions_latest.csv overlaps with dated files
                predictions_df = predictions_df.sort_values('Date', ascending=False)
                predictions_df = predictions_df.drop_duplicates(subset=['Date', 'Matchup'], keep='first')
                
                # Load actual game results
                try:
                    # Try workflow dataset first (has latest games)
                    results_file = DATAPATH / 'games_with_real_vegas_workflow.csv'
                    if not results_file.exists():
                        results_file = DATAPATH / 'games_master_engineered.csv'
                    if not results_file.exists():
                        results_file = DATAPATH / 'games_with_real_vegas.csv'
                    
                    results_df = pd.read_csv(results_file)
                    results_df['GAME_DATE_EST'] = pd.to_datetime(results_df['GAME_DATE_EST'])
                    
                    # Keep only completed games
                    results_df = results_df[results_df['PTS_home'] > 0].copy()
                    results_df['Actual_Winner'] = results_df['HOME_TEAM_WINS'].apply(lambda x: 'Home' if x == 1 else 'Away')
                    
                    # Create MATCHUP from team IDs (convert to abbreviations to match predictions)
                    # Map full team names to abbreviations
                    TEAM_NAME_TO_ABBREV = {
                        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
                        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
                        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
                        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
                        "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
                        "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
                        "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
                        "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
                        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
                        "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
                        "Washington Wizards": "WAS"
                    }
                    
                    # Check if we have abbreviation columns, otherwise use ID mapping
                    if 'VISITOR_TEAM_ABBREVIATION' in results_df.columns and 'HOME_TEAM_ABBREVIATION' in results_df.columns:
                        results_df['MATCHUP'] = results_df['VISITOR_TEAM_ABBREVIATION'] + ' @ ' + results_df['HOME_TEAM_ABBREVIATION']
                    elif 'VISITOR_TEAM_ID' in results_df.columns and 'HOME_TEAM_ID' in results_df.columns:
                        # Convert IDs to full names, then to abbreviations
                        visitor_names = results_df['VISITOR_TEAM_ID'].map(NBA_TEAMS_NAMES)
                        home_names = results_df['HOME_TEAM_ID'].map(NBA_TEAMS_NAMES)
                        results_df['VISITOR_TEAM_ABBREVIATION'] = visitor_names.map(TEAM_NAME_TO_ABBREV)
                        results_df['HOME_TEAM_ABBREVIATION'] = home_names.map(TEAM_NAME_TO_ABBREV)
                        results_df['MATCHUP'] = results_df['VISITOR_TEAM_ABBREVIATION'] + ' @ ' + results_df['HOME_TEAM_ABBREVIATION']
                    else:
                        st.warning("⚠️ Could not find team columns in results dataset")
                        return predictions_df, None
                    
                except Exception as e:
                    st.error(f"Error loading game results: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return predictions_df, None
                
                # Match predictions with results
                # Create a matching key from date and matchup
                predictions_df['Match_Key'] = predictions_df['Date'].dt.strftime('%Y-%m-%d') + '|' + predictions_df['Matchup']
                results_df['Match_Key'] = results_df['GAME_DATE_EST'].dt.strftime('%Y-%m-%d') + '|' + results_df['MATCHUP']
                
                # Debug: Show date ranges and sample match keys
                pred_dates = predictions_df['Date'].dt.date.unique()
                result_dates = results_df['GAME_DATE_EST'].dt.date.unique()
                
                # Filter results to only dates that have predictions
                pred_date_set = set(pred_dates)
                results_df_filtered = results_df[results_df['GAME_DATE_EST'].dt.date.isin(pred_date_set)].copy()
                
                # Sample match keys for debugging
                sample_pred_keys = predictions_df['Match_Key'].head(3).tolist()
                sample_result_keys = results_df_filtered['Match_Key'].head(3).tolist() if len(results_df_filtered) > 0 else []
                
                # Merge - preserve all prediction columns and add result columns
                matched_df = predictions_df.merge(
                    results_df[['Match_Key', 'Actual_Winner', 'HOME_TEAM_WINS', 'PTS_home', 'PTS_away']],
                    on='Match_Key',
                    how='left'
                )
                
                # Ensure all prediction columns are preserved (Edge, EV, Kelly, Bet_Size, Value_Bet, etc.)
                # The merge should preserve them, but let's verify they exist
                for col in ['Edge', 'EV', 'Kelly', 'Bet_Size', 'Value_Bet']:
                    if col not in matched_df.columns and col in predictions_df.columns:
                        matched_df[col] = predictions_df[col]
                
                # Calculate correctness (only for rows that have results)
                matched_df['Has_Result'] = matched_df['Actual_Winner'].notna()
                matched_df['Correct'] = None
                matched_df.loc[matched_df['Has_Result'], 'Correct'] = (
                    matched_df.loc[matched_df['Has_Result'], 'Predicted_Winner'] == 
                    matched_df.loc[matched_df['Has_Result'], 'Actual_Winner']
                )
                
                # Store debug info
                matched_df.attrs['debug_info'] = {
                    'total_predictions': len(predictions_df),
                    'total_results': len(results_df),
                    'results_in_pred_date_range': len(results_df_filtered),
                    'matched_count': matched_df['Has_Result'].sum(),
                    'pred_date_range': (min(pred_dates), max(pred_dates)) if len(pred_dates) > 0 else None,
                    'result_date_range': (min(result_dates), max(result_dates)) if len(result_dates) > 0 else None,
                    'sample_pred_keys': sample_pred_keys,
                    'sample_result_keys': sample_result_keys,
                }
                
                return matched_df, predictions_df
                
            except Exception as e:
                st.error(f"Error loading historical predictions: {e}")
                import traceback
                st.code(traceback.format_exc())
                return None, None
        
        # Cache clear button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Data", help="Clear cache and reload predictions"):
                load_all_historical_predictions.clear()
                st.rerun()
        
        # Load historical data
        with st.spinner('Loading historical predictions...'):
            matched_df, all_predictions_df = load_all_historical_predictions()
        
        if matched_df is None or len(matched_df) == 0:
            st.warning("⚠️ No historical predictions found. Predictions will appear here once you start making daily predictions.")
            st.info("💡 Run the 'Daily NBA Predictions' workflow to generate predictions.")
        elif 'Has_Result' not in matched_df.columns:
            st.warning("⚠️ Could not match predictions with game results. Check that game results are available.")
            st.info("💡 This may happen if predictions and game results use different date/matchup formats.")
        else:
            # Show debug info if available
            if hasattr(matched_df, 'attrs') and 'debug_info' in matched_df.attrs:
                debug = matched_df.attrs['debug_info']
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write(f"**Total Predictions:** {debug['total_predictions']}")
                    st.write(f"**Total Completed Games in Dataset:** {debug['total_results']}")
                    st.write(f"**Results in Prediction Date Range:** {debug.get('results_in_pred_date_range', 'N/A')}")
                    st.write(f"**Matched Predictions:** {debug['matched_count']}")
                    if debug['pred_date_range']:
                        st.write(f"**Prediction Date Range:** {debug['pred_date_range'][0]} to {debug['pred_date_range'][1]}")
                    if debug['result_date_range']:
                        st.write(f"**Results Date Range:** {debug['result_date_range'][0]} to {debug['result_date_range'][1]}")
                    if debug.get('sample_pred_keys'):
                        st.write("**Sample Prediction Match Keys:**")
                        for key in debug['sample_pred_keys']:
                            st.code(key)
                    if debug.get('sample_result_keys'):
                        st.write("**Sample Result Match Keys (in pred date range):**")
                        for key in debug['sample_result_keys']:
                            st.code(key)
                    else:
                        st.warning("⚠️ No results found in prediction date range. Games may not be completed yet.")
            
            # Filter to only games with results
            completed_df = matched_df[matched_df['Has_Result']].copy()
            pending_df = matched_df[~matched_df['Has_Result']].copy()
            
            if len(completed_df) == 0:
                st.warning("⚠️ No completed games found in predictions yet. Check back after games finish!")
                
                # Show pending predictions
                if len(pending_df) > 0:
                    st.info(f"📅 You have {len(pending_df)} prediction(s) waiting for results:")
                    pending_display = pending_df[['Date', 'Matchup', 'Predicted_Winner', 'Home_Win_Probability', 'Confidence']].copy()
                    pending_display['Home_Win_Probability'] = pending_display['Home_Win_Probability'].apply(lambda x: f"{x:.1%}")
                    pending_display = pending_display.sort_values('Date', ascending=False)
                    st.dataframe(pending_display, use_container_width=True, hide_index=True)
            else:
                # ========================================================================
                # MAIN KPIs
                # ========================================================================
                st.subheader("📊 Overall Performance Metrics")
                
                total_games = len(completed_df)
                correct_predictions = completed_df['Correct'].sum()
                accuracy = completed_df['Correct'].mean()
                avg_confidence = completed_df['Home_Win_Probability'].apply(lambda x: abs(x - 0.5) * 2).mean()
                
                # Calculate confidence from probability
                completed_df['Confidence_Score'] = completed_df['Home_Win_Probability'].apply(
                    lambda x: abs(x - 0.5) * 2
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Games Predicted", f"{total_games:,}")
                with col2:
                    st.metric("Accuracy", f"{accuracy:.1%}", 
                             delta=f"{correct_predictions:,} / {total_games:,}")
                with col3:
                    st.metric("Correct Predictions", f"{correct_predictions:,}", 
                             delta=f"{total_games - correct_predictions:,} incorrect")
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                st.markdown("---")
                
                # ========================================================================
                # BETTING METRICS
                # ========================================================================
                st.subheader("💰 Betting Performance")
                
                # Filter to games with betting data (check if columns exist first)
                betting_cols = ['Edge', 'EV', 'Value_Bet']
                has_betting_data = all(col in completed_df.columns for col in betting_cols)
                
                if has_betting_data:
                    betting_df = completed_df[
                        (completed_df['Edge'].notna()) & 
                        (completed_df['EV'].notna()) &
                        (completed_df['Value_Bet'].notna())
                    ].copy()
                else:
                    betting_df = pd.DataFrame()  # Empty dataframe if columns don't exist
                
                if len(betting_df) > 0:
                    # Calculate betting metrics
                    total_value_bets = betting_df['Value_Bet'].sum()
                    value_bet_accuracy = betting_df[betting_df['Value_Bet']]['Correct'].mean() if total_value_bets > 0 else 0
                    avg_edge = betting_df['Edge'].mean()
                    avg_ev = betting_df['EV'].mean()
                    total_ev = betting_df['EV'].sum()
                    avg_kelly = betting_df['Kelly'].mean()
                    total_bet_size = betting_df['Bet_Size'].sum()
                    
                    # Calculate ROI (simplified - assumes $10 per bet)
                    total_wagered = len(betting_df) * 10  # $10 per bet
                    roi = (total_ev / total_wagered * 100) if total_wagered > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Value Bets", f"{total_value_bets:,}", 
                                 delta=f"{value_bet_accuracy:.1%} accuracy" if total_value_bets > 0 else None)
                    with col2:
                        st.metric("Avg Edge", f"{avg_edge:.2%}", 
                                 delta="vs Vegas" if avg_edge > 0 else None)
                    with col3:
                        st.metric("Total EV", f"${total_ev:.2f}", 
                                 delta=f"{roi:.1f}% ROI" if roi > 0 else None)
                    with col4:
                        st.metric("Avg Kelly Fraction", f"{avg_kelly:.2%}")
                    
                    # Additional betting metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Expected Value", f"${avg_ev:.2f}")
                    with col2:
                        st.metric("Total Bet Size (Kelly)", f"${total_bet_size:.2f}")
                else:
                    st.info("💡 Betting metrics will appear once predictions include odds data.")
                
                st.markdown("---")
                
                # ========================================================================
                # PERFORMANCE BY CONFIDENCE LEVEL
                # ========================================================================
                st.subheader("📈 Performance by Confidence Level")
                
                # Add explanation of confidence levels
                st.markdown("""
                **Confidence Level Definitions:**
                - **High**: Win probability > 65% or < 35% (more than 15% away from 50/50)
                - **Medium**: Win probability 55-65% or 35-45% (5-15% away from 50/50)
                - **Low**: Win probability 45-55% (within 5% of 50/50)
                """)
                
                confidence_breakdown = completed_df.groupby('Confidence').agg({
                    'Correct': ['count', 'sum', 'mean'],
                    'Confidence_Score': 'mean'
                }).round(3)
                confidence_breakdown.columns = ['Total', 'Correct', 'Accuracy', 'Avg_Confidence_Score']
                confidence_breakdown = confidence_breakdown.sort_values('Accuracy', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(
                        confidence_breakdown.style.format({
                            'Total': '{:.0f}',
                            'Correct': '{:.0f}',
                            'Accuracy': '{:.1%}',
                            'Avg_Confidence_Score': '{:.1%}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    # Accuracy by confidence chart
                    try:
                        import plotly.express as px
                        fig = px.bar(
                            confidence_breakdown.reset_index(),
                            x='Confidence',
                            y='Accuracy',
                            title='Accuracy by Confidence Level',
                            labels={'Accuracy': 'Accuracy (%)', 'Confidence': 'Confidence Level'},
                            color='Accuracy',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(yaxis_tickformat='.1%', showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.info("Install plotly for charts: `pip install plotly`")
                
                st.markdown("---")
                
                # ========================================================================
                # TIME SERIES ANALYSIS
                # ========================================================================
                st.subheader("📅 Performance Over Time")
                
                # Calculate rolling accuracy
                completed_df_sorted = completed_df.sort_values('Date')
                completed_df_sorted['Rolling_Accuracy'] = completed_df_sorted['Correct'].expanding().mean()
                
                # Group by date for daily metrics
                # Build aggregation dict based on available columns
                agg_dict = {
                    'Correct': ['count', 'sum', 'mean'],
                    'Confidence_Score': 'mean'
                }
                
                # Add betting columns if they exist
                if 'Edge' in completed_df_sorted.columns:
                    agg_dict['Edge'] = 'mean'
                if 'EV' in completed_df_sorted.columns:
                    agg_dict['EV'] = 'sum'
                
                daily_metrics = completed_df_sorted.groupby(completed_df_sorted['Date'].dt.date).agg(agg_dict).round(3)
                
                # Flatten column names
                if isinstance(daily_metrics.columns, pd.MultiIndex):
                    daily_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_metrics.columns.values]
                
                # Rename columns to standard names
                column_mapping = {
                    'Correct_count': 'Games',
                    'Correct_sum': 'Correct',
                    'Correct_mean': 'Accuracy',
                    'Confidence_Score_mean': 'Avg_Confidence'
                }
                if 'Edge_mean' in daily_metrics.columns:
                    column_mapping['Edge_mean'] = 'Avg_Edge'
                if 'EV_sum' in daily_metrics.columns:
                    column_mapping['EV_sum'] = 'Total_EV'
                
                daily_metrics = daily_metrics.rename(columns=column_mapping)
                daily_metrics = daily_metrics.reset_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    # Accuracy over time
                    try:
                        import plotly.express as px
                        if 'Accuracy' in daily_metrics.columns:
                            fig1 = px.line(
                                daily_metrics,
                                x='Date',
                                y='Accuracy',
                                title='Daily Accuracy Over Time',
                                labels={'Accuracy': 'Accuracy (%)', 'Date': 'Date'},
                                markers=True
                            )
                            fig1.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                          annotation_text="50% Baseline")
                            fig1.update_layout(yaxis_tickformat='.1%')
                            st.plotly_chart(fig1, use_container_width=True)
                        else:
                            st.info("Accuracy data not available")
                    except ImportError:
                        st.info("Install plotly for charts")
                    except Exception as e:
                        st.warning(f"Could not create accuracy chart: {e}")
                
                with col2:
                    # Rolling accuracy
                    try:
                        import plotly.express as px
                        if 'Rolling_Accuracy' in completed_df_sorted.columns:
                            fig2 = px.line(
                                completed_df_sorted,
                                x='Date',
                                y='Rolling_Accuracy',
                                title='Cumulative Rolling Accuracy',
                                labels={'Rolling_Accuracy': 'Cumulative Accuracy (%)', 'Date': 'Date'},
                            )
                            fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                          annotation_text="50% Baseline")
                            fig2.update_layout(yaxis_tickformat='.1%')
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("Rolling accuracy data not available")
                    except ImportError:
                        st.info("Install plotly for charts")
                    except Exception as e:
                        st.warning(f"Could not create rolling accuracy chart: {e}")
                
                st.markdown("---")
                
                # ========================================================================
                # BEST VIG PLATFORM ANALYSIS
                # ========================================================================
                st.subheader("🎲 Best Vig Platform Analysis")
                
                # Load bookmaker comparison data if available
                comparison_path = DATAPATH / 'betting' / 'live_odds_bookmakers_comparison.csv'
                if comparison_path.exists():
                    try:
                        bookmaker_df = pd.read_csv(comparison_path)
                        
                        # Calculate average vig by bookmaker
                        bookmaker_stats = bookmaker_df.groupby('bookmaker').agg({
                            'ml_vig': 'mean',
                            'spread_vig': 'mean',
                            'total_vig': 'mean'
                        }).round(4)
                        bookmaker_stats.columns = ['Avg ML Vig', 'Avg Spread Vig', 'Avg Total Vig']
                        bookmaker_stats = bookmaker_stats.sort_values('Avg Total Vig')
                        
                        st.markdown("**Average Vig by Sportsbook (Lower is Better)**")
                        st.dataframe(
                            bookmaker_stats.style.format({
                                'Avg ML Vig': '{:.2f}%',
                                'Avg Spread Vig': '{:.2f}%',
                                'Avg Total Vig': '{:.2f}%'
                            }).highlight_min(axis=0, subset=['Avg ML Vig', 'Avg Spread Vig', 'Avg Total Vig']),
                            use_container_width=True
                        )
                        
                        # Best overall platform
                        bookmaker_stats['Overall_Avg_Vig'] = (
                            bookmaker_stats['Avg ML Vig'] + 
                            bookmaker_stats['Avg Spread Vig'] + 
                            bookmaker_stats['Avg Total Vig']
                        ) / 3
                        best_platform = bookmaker_stats['Overall_Avg_Vig'].idxmin()
                        best_vig = bookmaker_stats.loc[best_platform, 'Overall_Avg_Vig']
                        
                        st.success(f"🏆 **Best Overall Platform**: {best_platform} (Avg Vig: {best_vig:.2f}%)")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not load bookmaker comparison data: {e}")
                else:
                    st.info("💡 Bookmaker comparison data will appear once the 'Fetch Live NBA Odds' workflow runs.")
                
                st.markdown("---")
                
                # ========================================================================
                # DETAILED TABLE
                # ========================================================================
                st.subheader("📋 All Historical Predictions")
                
                # Create display dataframe - only include columns that exist
                base_cols = ['Date', 'Matchup', 'Predicted_Winner', 'Actual_Winner', 
                            'Correct', 'Home_Win_Probability', 'Confidence']
                betting_cols = ['Edge', 'EV', 'Value_Bet']
                
                display_cols = base_cols + [col for col in betting_cols if col in completed_df.columns]
                display_df = completed_df[display_cols].copy()
                
                display_df['Home_Win_Probability'] = display_df['Home_Win_Probability'].apply(lambda x: f"{x:.1%}")
                if 'Edge' in display_df.columns:
                    display_df['Edge'] = display_df['Edge'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                if 'EV' in display_df.columns:
                    display_df['EV'] = display_df['EV'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                display_df['Correct'] = display_df['Correct'].apply(lambda x: "✅" if x else "❌")
                if 'Value_Bet' in display_df.columns:
                    display_df['Value_Bet'] = display_df['Value_Bet'].apply(lambda x: "⭐" if x else "")
                
                display_df = display_df.sort_values('Date', ascending=False)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = completed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full History (CSV)",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # TAB 5: ABOUT
    # ========================================================================
    with tab5:
        fancy_header('About This App', font_size=28)
        st.markdown("")
        
        st.markdown("""
        ### 🏀 NBA Game Prediction System
        
        This enhanced app provides:
        
        **🎯 Multiple Model Support**
        - Individual models (Random Forest, XGBoost)
        - Ensemble methods (Stacking, Weighted Voting)
        - Automatic model loading and selection
        
        **📊 Live Predictions**
        - Today's game predictions with confidence levels
        - Historical performance tracking
        - Model comparison mode
        
        **📈 Performance Metrics**
        - AUC and accuracy tracking
        - Model-by-model comparison
        - Visual performance dashboards
        
        ### 🔧 Model Types
        
        **Ensemble Models** (Recommended)
        - Combine multiple algorithms for better accuracy
        - More robust and stable predictions
        - Typically 1-2% better AUC than individual models
        
        **Individual Models**
        - Single algorithm predictions
        - Faster inference
        - Good for understanding specific model behavior
        
        ### 📊 Current Performance
        
        - **🏆 Current Model**: 62.82% Accuracy (HistGradient + Vegas on 30k games)
        - **Training Data**: 30,120 games (2003-2025) - 6x more historical data!
        - **Features**: 102 predictive features (injury + Vegas + rolling averages)
        - **Baseline**: 50% (coin flip)
        - **Professional Target**: 64-68% (Vegas level) - 📈 Getting closer!
        
        ### 🎓 How It Works
        
        1. **Feature Engineering**: Rolling averages, win streaks, matchup history
        2. **Model Training**: XGBoost, Random Forest, HistGradientBoosting
        3. **Ensemble**: Combine predictions for better accuracy
        4. **Calibration**: Adjust probabilities for better confidence estimates
        
        ### 📝 Notes
        
        - NBA season: October - June
        - Predictions update daily
        - Models retrained periodically
        - No betting advice provided!
        
        ---
        
        **Version**: 2.0 Enhanced  
        **Last Updated**: November 2025
        """)


if __name__ == "__main__":
    main()

