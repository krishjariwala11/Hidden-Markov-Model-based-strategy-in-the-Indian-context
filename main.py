import pandas as pd
import numpy as np
from config import TICKERS, START_DATE, END_DATE, MODEL_FEATURES, N_REGIMES
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from hmm_model import HMMModel
from regime_analysis import RegimeAnalyzer
from backtest_engine import BacktestEngine
from visualization import Visualizer

def run_pipeline():
    print("--- Starting HMM Regime Detection Pipeline ---")
    
    # 1. Data Loading
    dl = DataLoader()
    dfs = dl.fetch_data(TICKERS, START_DATE, END_DATE)
    
    if "NIFTY50" not in dfs or dfs["NIFTY50"].empty:
        print("\nWARNING: Live data fetch failed (potentially rate limited). Switching to MOCK DATA for demonstration.")
        dfs = dl.generate_mock_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Feature Engineering
    fe = FeatureEngineer(main_asset_name="NIFTY50")
    df = fe.generate_features(dfs)
    print(f"Features generated. Shape: {df.shape}")
    
    # 3. Model Training
    X = fe.scale_features(df, MODEL_FEATURES)
    hmm = HMMModel(n_regimes=N_REGIMES)
    hmm.train(X)
    
    # 4. Regime Prediction
    df['state'] = hmm.predict(X)
    
    # 5. Regime Analysis & Labeling
    ra = RegimeAnalyzer()
    stats = ra.calculate_regime_stats(df)
    labels = ra.map_regime_labels(stats)
    df['regime_label'] = df['state'].map(labels)
    
    print("\nRegime Statistics:")
    print(stats)
    print("\nRegime Labels Assigned:")
    for k, v in labels.items():
        print(f"State {k} -> {v}")
        
    # 6. Backtesting
    be = BacktestEngine()
    df_results = be.run_backtest(df)
    
    metrics_strat = be.calculate_performance_metrics(df_results, 'strat_returns')
    metrics_mkt = be.calculate_performance_metrics(df_results, 'returns')
    
    print("\n--- Performance Metrics ---")
    perf_df = pd.DataFrame([metrics_mkt, metrics_strat], index=['NIFTY Buy&Hold', 'HMM Strategy'])
    print(perf_df.T)
    
    # 7. Visualization
    viz = Visualizer()
    print("\nGenerating Visualizations...")
    viz.plot_regimes(df_results)
    viz.plot_equity_curve(df_results)
    
    label_list = [labels.get(i, f"S{i}") for i in range(N_REGIMES)]
    viz.plot_transition_heatmap(hmm.get_transition_matrix(), label_list)
    viz.plot_regime_distribution(df_results)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"Results saved in 'output/' directory.")

if __name__ == "__main__":
    run_pipeline()
