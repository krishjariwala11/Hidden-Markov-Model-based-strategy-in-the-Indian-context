import os

# Project Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Tickers
TICKERS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "VIX": "^INDIAVIX",  # Switching to index ticker for better reliability
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS"
}

# Date Range for Training/Analysis
START_DATE = "2015-01-01"
END_DATE = "2024-03-10"

# HMM Hyperparameters
N_REGIMES = 4
COVARIANCE_TYPE = "full"
N_ITER = 1000
RANDOM_STATE = 42

# Feature Lists
MODEL_FEATURES = [
    "returns",
    "volatility_30",
    "vix_level",
    "vix_change"
]

# Regime Labels (Placeholders - will be assigned dynamically based on analysis)
REGIME_LABELS = {
    0: "Regime 0",
    1: "Regime 1",
    2: "Regime 2",
    3: "Regime 3"
}

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
