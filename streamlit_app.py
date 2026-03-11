import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import TICKERS, START_DATE, END_DATE, MODEL_FEATURES, N_REGIMES
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from hmm_model import HMMModel
from regime_analysis import RegimeAnalyzer
from backtest_engine import BacktestEngine
from visualization import Visualizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced HMM Regime Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #636EFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1c24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #636EFA !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/64/combo-chart.png", width=60)
    st.title("HMM Quant Lab")
    st.markdown("---")
    
    selected_asset = st.selectbox("Select Asset", list(TICKERS.keys()), index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.strptime(START_DATE, "%Y-%m-%d"))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        
    n_states = st.slider("Number of Regimes", 2, 6, N_REGIMES)
    
    retrain_btn = st.button("🚀 Run Analysis Engine", use_container_width=True)
    
    st.markdown("---")
    st.info("**HMM (Hidden Markov Model)** is used to cluster market periods into hidden states based on Return & Volatility.")

# --- MAIN APP LOGIC ---
st.title("📊 Indian Market Regime Analytics")
st.caption(f"Analyzing {selected_asset} using Gaussian Hidden Markov Models")

if retrain_btn:
    with st.spinner("Executing Pipeline..."):
        # Progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Data Fetching
        status_text.text("Fetching Market Data...")
        progress_bar.progress(20)
        dl = DataLoader()
        # Create a dict with just the necessary tickers for speed
        relevant_tickers = {selected_asset: TICKERS[selected_asset], "VIX": TICKERS["VIX"]}
        dfs = dl.fetch_data(relevant_tickers, str(start_date), str(end_date))
        
        if selected_asset not in dfs or dfs[selected_asset].empty:
            st.warning("Live data fetch failed. Using fallback mock data.")
            dfs = dl.generate_mock_data(relevant_tickers, str(start_date), str(end_date))
        
        # 2. Feature Engineering
        status_text.text("Engineering Features...")
        progress_bar.progress(40)
        fe = FeatureEngineer(main_asset_name=selected_asset)
        df = fe.generate_features(dfs)
        
        # 3. HMM Training
        status_text.text("Training HMM Engine...")
        progress_bar.progress(60)
        X = fe.scale_features(df, MODEL_FEATURES)
        hmm = HMMModel(n_regimes=n_states)
        hmm.train(X)
        
        # 4. Regime Analysis
        status_text.text("Analyzing Regimes...")
        progress_bar.progress(80)
        df['state'] = hmm.predict(X)
        ra = RegimeAnalyzer()
        stats = ra.calculate_regime_stats(df)
        labels = ra.map_regime_labels(stats)
        df['regime_label'] = df['state'].map(labels)
        
        # 5. Backtesting
        status_text.text("Simulating Strategy...")
        progress_bar.progress(95)
        be = BacktestEngine()
        df_results = be.run_backtest(df)
        
        # Save to Session State
        st.session_state.processed_data = df_results
        st.session_state.model = hmm
        st.session_state.stats = stats
        st.session_state.labels = labels
        
        progress_bar.progress(100)
        status_text.text("Pipeline Complete!")
        st.balloons()

# --- DISPLAY RESULTS ---
if st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    hmm = st.session_state.model
    stats = st.session_state.stats
    labels = st.session_state.labels
    
    # KPIs Row
    be = BacktestEngine()
    m_strat = be.calculate_performance_metrics(df, 'strat_returns')
    m_mkt = be.calculate_performance_metrics(df, 'returns')
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Strategy CAGR", f"{m_strat['CAGR']:.2%}", f"{(m_strat['CAGR'] - m_mkt['CAGR']):.2f} vs B&H")
    kpi2.metric("Sharpe Ratio", f"{m_strat['Sharpe']:.2f}", f"{(m_strat['Sharpe'] - m_mkt['Sharpe']):.2f}")
    kpi3.metric("Max Drawdown", f"{m_strat['Max Drawdown']:.2%}", f"{(m_strat['Max Drawdown'] - m_mkt['Max Drawdown']):.2%}")
    kpi4.metric("Win Rate", f"{m_strat['Win Rate']:.2%}")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Regime Visualization", "🔬 Detailed Analysis", "💰 Strategy Backtest"])
    
    viz = Visualizer()
    
    with tab1:
        st.subheader("Price Action Colored by Market Regime")
        fig_regime = viz.plot_regimes(df, ticker=selected_asset)
        st.plotly_chart(fig_regime, use_container_width=True)
        
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Regime Key")
            for i, (k, v) in enumerate(labels.items()):
                st.write(f"State {k}: **{v}**")
        with col_r:
            st.subheader("Regime Transitions")
            fig_trans = viz.plot_transition_heatmap(hmm.get_transition_matrix(), [labels[i] for i in range(n_states)])
            st.plotly_chart(fig_trans, use_container_width=True)

    with tab2:
        st.subheader("Regime Characteristics")
        st.dataframe(stats, use_container_width=True)
        
        st.subheader("Return Distributions per State")
        fig_dist = viz.plot_regime_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.subheader("Performance Comparison (Cumulative Returns)")
        fig_equity = viz.plot_equity_curve(df)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**NIFTY 50 Benchmark Statistics**")
            st.json(m_mkt)
        with col_m2:
            st.write("**HMM Strategy Statistics**")
            st.json(m_strat)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Analysis Engine' to begin.")
    
    # Placeholder Display
    st.image("https://www.tradingview.com/static/images/free-images/charting-library-preview.png")
