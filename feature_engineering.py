import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class FeatureEngineer:
    """
    Generates advanced financial features for HMM modeling.
    """
    
    def __init__(self, main_asset_name: str = "NIFTY50"):
        self.main_asset = main_asset_name
        self.scaler = StandardScaler()

    def generate_features(self, data_frames: dict) -> pd.DataFrame:
        """
        Combines multiple assets and generates technical features.
        """
        if self.main_asset not in data_frames:
            raise ValueError(f"{self.main_asset} not found in data_frames")
            
        df = data_frames[self.main_asset].copy()
        
        # 1. Log Returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Rolling Volatility (annualized)
        df['volatility_10'] = df['returns'].rolling(window=10).std() * np.sqrt(252)
        df['volatility_30'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
        
        # 3. Moving Averages
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        df['ma_200'] = df['Close'].rolling(window=200).mean()
        
        # 4. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 5. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 6. Volume Change
        df['vol_change'] = df['Volume'].pct_change()
        
        # 7. Add External Data: VIX
        if 'VIX' in data_frames and not data_frames['VIX'].empty:
            vix_df = data_frames['VIX']['Close'].copy()
            # Reindex to match main asset
            df['vix_level'] = vix_df
            df['vix_change'] = df['vix_level'].pct_change()
        else:
            print("Warning: VIX data not available. Some features will be missing.")
            df['vix_level'] = 0.0 # Placeholder or handle in feature list
            df['vix_change'] = 0.0
        # 8. Add Optional: BANKNIFTY returns for correlation
        if 'BANKNIFTY' in data_frames:
            bank_returns = np.log(data_frames['BANKNIFTY']['Close'] / data_frames['BANKNIFTY']['Close'].shift(1))
            df['bank_returns'] = bank_returns

        # 9. Drawdown
        roll_max = df['Close'].cummax()
        df['drawdown'] = (df['Close'] - roll_max) / roll_max

        # Cleanup
        df = df.dropna()
        
        return df

    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Scales the specified features using StandardScaler.
        """
        return self.scaler.fit_transform(df[feature_cols])

if __name__ == "__main__":
    # Mock data for testing
    dates = pd.date_range('2023-01-01', periods=300)
    data = {'Close': np.random.randn(300).cumsum() + 18000, 
            'Volume': np.random.randint(100, 1000, 300),
            'High': np.random.randn(300),
            'Low': np.random.randn(300),
            'Open': np.random.randn(300)}
    df_nifty = pd.DataFrame(data, index=dates)
    
    vix_data = {'Close': np.random.randn(300) + 15}
    df_vix = pd.DataFrame(vix_data, index=dates)
    
    fe = FeatureEngineer()
    processed_df = fe.generate_features({'NIFTY50': df_nifty, 'VIX': df_vix})
    print(processed_df.tail())
    print("Features generated:", processed_df.columns.tolist())
