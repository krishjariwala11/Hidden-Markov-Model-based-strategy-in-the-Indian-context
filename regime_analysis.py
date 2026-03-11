import pandas as pd
import numpy as np
from typing import Dict, List
from config import REGIME_LABELS

class RegimeAnalyzer:
    """
    Analyzes and labels detected market regimes.
    """
    
    @staticmethod
    def calculate_regime_stats(df: pd.DataFrame, state_col: str = 'state') -> pd.DataFrame:
        """
        Calculates return, volatility, and Sharpe per regime.
        """
        stats = []
        for state in sorted(df[state_col].unique()):
            subset = df[df[state_col] == state]
            
            mean_ret = subset['returns'].mean() * 252
            vol = subset['returns'].std() * np.sqrt(252)
            sharpe = mean_ret / vol if vol != 0 else 0
            
            # Drawdown per regime
            roll_max = subset['Close'].cummax()
            drawdown = (subset['Close'] - roll_max) / roll_max
            avg_dd = drawdown.mean()
            max_dd = drawdown.min()
            
            # Duration
            # Count consecutive occurrences
            duration = (df[state_col] == state).sum() / len(df) * 100 # % of time
            
            stats.append({
                'Regime': state,
                'Ann. Return': mean_ret,
                'Ann. Vol': vol,
                'Sharpe': sharpe,
                'Avg Drawdown': avg_dd,
                'Max Drawdown': max_dd,
                'Occurrence %': duration
            })
            
        return pd.DataFrame(stats)

    @staticmethod
    def map_regime_labels(stats_df: pd.DataFrame) -> Dict[int, str]:
        """
        Automatically labels regimes based on Return/Vol characteristics.
        High Return, Low Vol -> Bull
        Low/Neg Return, High Vol -> Bear
        Low Return, Low Vol -> Sideways
        High Vol, Variable Return -> High Volatility
        """
        # Sort by mean return
        sorted_by_ret = stats_df.sort_values('Ann. Return', ascending=False)
        
        labels = {}
        # Simple heuristic
        # 1. Highest return state is Bull
        bull_state = sorted_by_ret.iloc[0]['Regime']
        labels[int(bull_state)] = "Bull Market"
        
        # 2. Lowest return state is Bear
        bear_state = sorted_by_ret.iloc[-1]['Regime']
        labels[int(bear_state)] = "Bear Market"
        
        # 3. State with highest volatility that isn't Bear (could be Bull but usually Crisis)
        remaining = stats_df[~stats_df['Regime'].isin([bull_state, bear_state])]
        if not remaining.empty:
            vol_state = remaining.sort_values('Ann. Vol', ascending=False).iloc[0]['Regime']
            labels[int(vol_state)] = "High Volatility"
            
            # 4. The last one is Sideways
            sideways_states = remaining[remaining['Regime'] != vol_state]['Regime']
            for s in sideways_states:
                labels[int(s)] = "Sideways"
        
        return labels

if __name__ == "__main__":
    # Test labeling logic
    data = {
        'Regime': [0, 1, 2, 3],
        'Ann. Return': [0.15, -0.20, 0.05, -0.05],
        'Ann. Vol': [0.12, 0.35, 0.10, 0.25]
    }
    df = pd.DataFrame(data)
    ra = RegimeAnalyzer()
    print(ra.map_regime_labels(df))
