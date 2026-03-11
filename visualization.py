import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import os
from config import OUTPUT_DIR

class Visualizer:
    """
    Creates high-quality, interactive Plotly charts for regime detection and strategy performance.
    """
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir

    def plot_regimes(self, df: pd.DataFrame, ticker: str = "NIFTY50"):
        """
        Plots the price chart colored by regime.
        """
        fig = go.Figure()
        
        regimes = df['regime_label'].unique()
        # Use a qualitative color palette
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
        color_map = {r: colors[i % len(colors)] for i, r in enumerate(regimes)}
        
        for regime in regimes:
            mask = df['regime_label'] == regime
            fig.add_trace(go.Scatter(
                x=df.index[mask],
                y=df.loc[mask, 'Close'],
                mode='markers',
                marker=dict(size=4, color=color_map[regime]),
                name=regime
            ))
            
        fig.update_layout(
            title=f"{ticker} Regimes Detected via HMM",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            legend_title="Detected Regime"
        )
        fig.write_html(os.path.join(self.output_dir, "regime_chart.html"))
        return fig

    def plot_equity_curve(self, df: pd.DataFrame):
        """
        Compares Strategy Equity Curve vs Buy & Hold.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['cum_market_returns'], name="Buy & Hold (NIFTY50)", line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['cum_strat_returns'], name="HMM Regime Strategy", line=dict(color='cyan', width=2)))
        
        fig.update_layout(
            title="Equity Curve: HMM Strategy vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.write_html(os.path.join(self.output_dir, "equity_curve.html"))
        return fig

    def plot_transition_heatmap(self, trans_mat: np.ndarray, labels: list):
        """
        Plots the regime transition probability heatmap using Plotly.
        """
        fig = px.imshow(
            trans_mat,
            labels=dict(x="To State", y="From State", color="Probability"),
            x=labels,
            y=labels,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Regime Transition Probabilities"
        )
        fig.update_layout(template="plotly_dark")
        fig.write_html(os.path.join(self.output_dir, "transition_matrix.html"))
        return fig

    def plot_regime_distribution(self, df: pd.DataFrame):
        """
        Distribution of returns per regime using Plotly box plots.
        """
        fig = px.box(
            df, 
            x="regime_label", 
            y="returns", 
            color="regime_label",
            title="Daily Return Distribution per Regime",
            template="plotly_dark"
        )
        fig.write_html(os.path.join(self.output_dir, "return_dist.html"))
        return fig
