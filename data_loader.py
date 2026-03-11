import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, Optional
from config import TICKERS, START_DATE, END_DATE, DATA_DIR

class DataLoader:
    """
    Handles fetching and saving historical market data for Indian assets with robust error handling.
    """
    
    @staticmethod
    def fetch_data(tickers: Dict[str, str], start: str, end: str, retries: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical data for multiple tickers with retry logic.
        """
        data_frames = {}
        for name, ticker in tickers.items():
            print(f"Fetching data for {name} ({ticker})...")
            success = False
            for attempt in range(retries):
                try:
                    # Use yf.download with threads disabled to avoid potential race conditions
                    # and auto_adjust to avoid multi-index columns where possible
                    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
                    
                    if df.empty:
                        print(f"  Attempt {attempt+1}: Empty dataframe for {ticker}. Retrying...")
                        time.sleep(2)
                        continue
                    
                    # Modern yfinance returns columns like (Close, Ticker) if Ticker is passed as list or index
                    # Let's flatten if multi-index
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    data_frames[name] = df
                    
                    # Save to CSV for caching
                    if not os.path.exists(DATA_DIR):
                        os.makedirs(DATA_DIR)
                    file_path = os.path.join(DATA_DIR, f"{name.lower()}.csv")
                    df.to_csv(file_path)
                    
                    success = True
                    print(f"  Successfully fetched {name}")
                    break
                    
                except Exception as e:
                    print(f"  Attempt {attempt+1}: Error fetching {ticker}: {e}")
                    time.sleep(2)
            
            if not success:
                print(f"Failed to fetch {name} after {retries} attempts.")
        
        return data_frames

    @staticmethod
    def load_cached_data() -> Dict[str, pd.DataFrame]:
        """
        Loads data from local CSV cache.
        """
        data_frames = {}
        if not os.path.exists(DATA_DIR):
            return {}
            
        for file in os.listdir(DATA_DIR):
            if file.endswith(".csv"):
                name = file.replace(".csv", "").upper()
                df = pd.read_csv(os.path.join(DATA_DIR, file), index_col=0, parse_dates=True)
                data_frames[name] = df
        return data_frames

    @staticmethod
    def generate_mock_data(tickers: Dict[str, str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        Generates synthetic market data for demonstration purposes.
        """
        print("Generating mock data for demonstration...")
        dates = pd.date_range(start, end)
        data_frames = {}
        for name in tickers.keys():
            n = len(dates)
            # Simulate a random walk with drift
            returns = np.random.normal(0.0005, 0.015, n)
            prices = 10000 * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'Open': prices * (1 + 0.001 * np.random.randn(n)),
                'High': prices * (1 + 0.005 * np.abs(np.random.randn(n))),
                'Low': prices * (1 - 0.005 * np.abs(np.random.randn(n))),
                'Close': prices,
                'Adj Close': prices,
                'Volume': np.random.randint(100000, 1000000, n)
            }, index=dates)
            data_frames[name] = df
        return data_frames

if __name__ == "__main__":
    # Test mock generator
    dl = DataLoader()
    dfs = dl.generate_mock_data(TICKERS, START_DATE, END_DATE)
    print(f"Generated mock data for {list(dfs.keys())}")
