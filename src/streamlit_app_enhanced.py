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
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try different import paths for compatibility
try:
    from feature_engineering import fix_datatypes, remove_non_rolling
    from constants import LONG_INTEGER_FIELDS, SHORT_INTEGER_FIELDS, DATE_FIELDS, DROP_COLUMNS, NBA_TEAMS_NAMES
    from live_odds_display import load_live_odds, match_game_to_odds, format_odds_display
except:
    from src.feature_engineering import fix_datatypes, remove_non_rolling
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


# Page configuration
st.set_page_config(
    page_title="NBA Game Predictor - Enhanced",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATAPATH = Path('data')
MODELSPATH = Path('models')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fancy_header(text, font_size=24, color="#ff5f27"):
    """Render fancy colored header"""
    st.markdown(
        f'<span style="color:{color}; font-size: {font_size}px; font-weight: bold;">{text}</span>',
        unsafe_allow_html=True
    )


def load_available_models():
    """Scan models directory and load available models"""
    models = {}
    model_info = {}
    
    # Define available models and their info
    model_definitions = {
        'histgradient_vegas': {
            'name': '🏆 HistGradient + Vegas (BEST - 70.20%)',
            'description': '70.20% AUC - Champion model!',
            'file': 'histgradient_vegas_calibrated.pkl',
            'type': 'individual',
            'is_best': True
        },
        'ensemble_stacking_vegas': {
            'name': '🥈 Stacking Ensemble + Vegas (69.91%)',
            'description': '69.91% AUC - New stacking with Vegas data',
            'file': 'ensemble_stacking_vegas.pkl',
            'type': 'ensemble'
        },
        'ensemble_weighted_vegas': {
            'name': '🥉 Weighted Ensemble + Vegas (69.81%)',
            'description': '69.81% AUC - Weighted with Vegas data',
            'file': 'ensemble_weighted_vegas.pkl',
            'type': 'ensemble'
        },
        'randomforest_vegas': {
            'name': '🌲 RandomForest + Vegas (69.37%)',
            'description': '69.37% AUC - Retrained with Vegas data',
            'file': 'best_model_randomforest_vegas.pkl',
            'type': 'individual'
        },
        'xgboost_vegas': {
            'name': '⚡ XGBoost + Vegas (68.85%)',
            'description': '68.85% AUC - XGBoost with Vegas data',
            'file': 'xgboost_vegas_calibrated.pkl',
            'type': 'individual'
        },
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
    
    # Drop only the columns that should never be in predictions
    # Keep ALL features including injury and Vegas features
    drop_cols = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', 'merge_key', 'MATCHUP']
    
    # Drop categorical features
    categorical_cols = [
        'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION',
        'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
        'whos_favored', 'data_source', 'is_real_vegas_line'
    ]
    drop_cols.extend(categorical_cols)
    
    # Drop leaky features (game results)
    leaky_features = [
        'HOME_TEAM_WINS', 'PTS_home', 'PTS_away',
        'FG_PCT_home', 'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away',
        'FG3_PCT_home', 'FG3_PCT_away', 'AST_home', 'AST_away',
        'REB_home', 'REB_away', 'PLAYOFF',
        'score_home', 'score_away', 'q1_home', 'q2_home', 'q3_home', 'q4_home'
    ]
    drop_cols.extend(leaky_features)
    
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
    
    # Comparison mode
    st.sidebar.markdown("---")
    comparison_mode = st.sidebar.checkbox('🔬 Enable Model Comparison', value=False)
    
    if comparison_mode and len(models) > 1:
        compare_models = st.sidebar.multiselect(
            'Compare with:',
            options=[k for k in model_options.keys() if k != selected_model_key],
            format_func=lambda x: model_options[x],
            default=[]
        )
    else:
        compare_models = []
    
    # Load live odds
    live_odds_df = load_live_odds()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(['📊 Predictions', '📈 Performance', 'ℹ️ About'])
    
    # ========================================================================
    # TAB 1: PREDICTIONS
    # ========================================================================
    with tab1:
        fancy_header('Today\'s Game Predictions', font_size=28)
        
        # Show live odds status
        if live_odds_df is not None and len(live_odds_df) > 0:
            st.success(f"🎲 Live Vegas odds loaded for {len(live_odds_df)} games from The Odds API")
        
        st.markdown("")
        
        # Load data
        try:
            # Load data with ALL features (injuries + Vegas)
            df_current_season = pd.read_csv(DATAPATH / 'games_with_real_vegas.csv')
            
            # Get current season
            current_season = datetime.today().year
            if datetime.today().month < 10:
                current_season = current_season - 1
            
            df_current_season = df_current_season[df_current_season['SEASON'] == current_season]
            
            # Get today's games
            df_today = df_current_season[df_current_season['PTS_home'] == 0]
            df_past = df_current_season[df_current_season['PTS_home'] != 0]
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Make sure data/games_with_real_vegas.csv exists and is up to date.")
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
            
            # Comparison predictions if enabled
            comparison_results = {}
            if compare_models:
                for comp_key in compare_models:
                    comp_model = models[comp_key]
                    comp_pred, comp_proba = predict_with_model(comp_model, X_today, comp_key)
                    comparison_results[comp_key] = {
                        'proba': comp_proba,
                        'name': model_info[comp_key]['name']
                    }
            
            # Create export DataFrame
            export_df = pd.DataFrame({
                'Date': [datetime.now().strftime('%Y-%m-%d')] * len(df_today_display),
                'Matchup': df_today_display['MATCHUP'],
                'Home_Win_Probability': [f"{p:.1%}" for p in pred_proba],
                'Predicted_Winner': df_today_display['PREDICTION'],
                'Confidence': df_today_display['CONFIDENCE'],
                'Model': [selected_info['name']] * len(df_today_display)
            })
            
            # Add download button
            col_title, col_download = st.columns([3, 1])
            with col_title:
                st.subheader("📊 Today's Predictions")
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
            
            # Display predictions
            for idx, row in df_today_display.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
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
                
                # Show comparison if enabled
                if comparison_results:
                    with st.expander(f"📊 Compare Models for this game"):
                        comp_df = pd.DataFrame({
                            'Model': [selected_info['name']] + [comparison_results[k]['name'] for k in comparison_results],
                            'Home Win Prob': [prob] + [comparison_results[k]['proba'][idx] for k in comparison_results],
                        })
                        comp_df['Winner'] = comp_df['Home Win Prob'].apply(lambda x: 'Home' if x > 0.5 else 'Away')
                        comp_df['Home Win Prob'] = comp_df['Home Win Prob'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
        
        # Past games section
        st.markdown("### 📅 Recent Performance (Last 25 Games)")
        
        if len(df_past) > 0:
            # Prepare past games data
            X_past, df_past_display = prepare_data_for_prediction(df_past)
            
            # Get predictions
            past_predictions, past_proba = predict_with_model(selected_model, X_past, selected_model_key)
            
            df_past_display['HOME_WIN_PROB'] = past_proba
            df_past_display['PREDICTED_WINNER'] = past_predictions
            df_past_display['ACTUAL_WINNER'] = df_past_display['TARGET']
            df_past_display['CORRECT'] = df_past_display['PREDICTED_WINNER'] == df_past_display['ACTUAL_WINNER']
            
            # Sort and limit
            df_past_display = df_past_display.sort_values('GAME_DATE_EST', ascending=False).head(25)
            
            # Reset index to avoid display issues
            df_past_display = df_past_display.reset_index(drop=True)
            
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
            
            # Display table
            display_df = df_past_display[[
                'GAME_DATE_EST', 'MATCHUP', 'HOME_WIN_PROB', 
                'ACTUAL_WINNER', 'CORRECT'
            ]].copy()
            
            display_df.columns = ['Date', 'Matchup', 'Home Win Prob', 'Actual Winner', 'Correct']
            display_df['Home Win Prob'] = display_df['Home Win Prob'].apply(lambda x: f"{x:.1%}")
            display_df['Actual Winner'] = display_df['Actual Winner'].apply(lambda x: 'Home' if x == 1 else 'Away')
            display_df['Correct'] = display_df['Correct'].apply(lambda x: '✅' if x else '❌')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No past games available for this season yet.")
    
    # ========================================================================
    # TAB 2: PERFORMANCE
    # ========================================================================
    with tab2:
        fancy_header('Model Performance Comparison', font_size=28)
        st.markdown("")
        
        if performance_metrics:
            # Create comparison dataframe
            perf_data = []
            for key, metrics in performance_metrics.items():
                if key in model_info:
                    perf_data.append({
                        'Model': model_info[key]['name'],
                        'Type': model_info[key]['type'].title(),
                        'Test AUC': metrics.get('test_auc', 0),
                        'Test Accuracy': metrics.get('test_accuracy', 0),
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_df = perf_df.sort_values('Test AUC', ascending=False)
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 AUC Scores")
                    st.bar_chart(perf_df.set_index('Model')['Test AUC'])
                
                with col2:
                    st.subheader("🎯 Accuracy")
                    st.bar_chart(perf_df.set_index('Model')['Test Accuracy'])
                
                # Full table
                st.subheader("📋 Detailed Metrics")
                display_perf = perf_df.copy()
                display_perf['Test AUC'] = display_perf['Test AUC'].apply(lambda x: f"{x:.4f}")
                display_perf['Test Accuracy'] = display_perf['Test Accuracy'].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_perf, use_container_width=True, hide_index=True)
            else:
                st.info("No performance metrics available. Run model comparison to generate metrics.")
        else:
            st.info("No performance metrics found. Run `python build_ensemble.py` to generate comparison metrics.")
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
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
        
        - **🏆 Best Model**: 70.20% AUC (HistGradient + Real Vegas) - WE HIT 70%!
        - **🥈 Stacking**: 69.91% AUC (New ensemble with Vegas data)
        - **🥉 Weighted**: 69.81% AUC (Weighted ensemble with Vegas)
        - **Baseline**: 58% AUC (home team always wins)
        - **Professional Target**: 70-75% AUC - ✅ ACHIEVED!
        
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

