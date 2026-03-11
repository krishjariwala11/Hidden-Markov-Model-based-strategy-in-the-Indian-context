import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Simulates a regime-switching trading strategy.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run_backtest(self, df: pd.DataFrame, label_col: str = 'regime_label') -> pd.DataFrame:
        """
        Strategy:
        Bull -> 100% Long
        Bear -> -50% Short (or 0% Cash to be safer, user asked for short)
        Sideways -> 0% (Cash)
        High Volatility -> 25% exposure (Reduced risk)
        """
        df = df.copy()
        
        # Define Exposure
        df['exposure'] = 0.0
        df.loc[df[label_col] == "Bull Market", 'exposure'] = 1.0
        df.loc[df[label_col] == "Bear Market", 'exposure'] = -0.5 # Shorting index
        df.loc[df[label_col] == "Sideways", 'exposure'] = 0.0
        df.loc[df[label_col] == "High Volatility", 'exposure'] = 0.25 # Caution
        
        # Calculate Strategy Returns
        # We use the returns from 'next day' to avoid look-ahead bias
        df['strat_returns'] = df['exposure'] * df['returns']
        
        # Cumulative Returns
        df['cum_market_returns'] = (1 + df['returns']).cumprod()
        df['cum_strat_returns'] = (1 + df['strat_returns']).cumprod()
        
        # Equity Curve
        df['equity_curve'] = self.initial_capital * df['cum_strat_returns']
        
        return df

    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame, returns_col: str) -> dict:
        """
        Computes CAGR, Sharpe, Sortino, Max Drawdown.
        """
        returns = df[returns_col].dropna()
        if len(returns) == 0:
            return {}
            
        # Annualized Return (CAGR)
        total_ret = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_ret) ** (1/n_years) - 1
        
        # Annualized Vol
        vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 0 risk free rate)
        sharpe = cagr / vol if vol != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol != 0 else 0
        
        # Max Drawdown
        cum_ret = (1 + returns).cumprod()
        roll_max = cum_ret.cummax()
        dd = (cum_ret - roll_max) / roll_max
        max_dd = dd.min()
        
        # Win Rate
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate
        }
