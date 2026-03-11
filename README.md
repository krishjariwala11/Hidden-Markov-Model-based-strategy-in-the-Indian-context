# HMM Market Regime Detection

A production-grade quantitative research framework for detecting market regimes in the Indian stock market using Hidden Markov Models (HMM).

## Overview

This project implements a Gaussian Hidden Markov Model (HMM) to identify various market regimes (e.g., Bull, Bear, Sideways, High Volatility) based on historical stock data. It includes a complete pipeline from data ingestion and feature engineering to model training, backtesting, and interactive visualization.

### Key Features
- **Data Ingestion**: Fetches historical data for NIFTY 50 and other major Indian tickers using `yfinance`.
- **Feature Engineering**: Calculates log returns, rolling volatility, and VIX-based indicators.
- **HMM Modeling**: Uses `hmmlearn` to train a Gaussian HMM for regime detection.
- **Regime-Based Backtesting**: A backtesting engine that evaluates performance based on regime transitions.
- **Interactive Visualizations**: Plotly-based charts for equity curves, regime distributions, and volatility analysis.
- **Streamlit Dashboard**: A user-friendly interface for exploring model results and adjusting parameters.

## Project Structure

```text
├── data/               # Local cache for downloaded data (ignored by git)
├── output/             # Generated charts and reports (ignored by git)
├── backtest_engine.py  # Simulation of regime-based strategies
├── config.py           # Project configuration and hyperparameters
├── data_loader.py      # Utilities for fetching historical data
├── feature_engineering.py # Signal processing and feature extraction
├── hmm_model.py        # HMM model definition and training
├── main.py             # CLI entry point for the full pipeline
├── regime_analysis.py  # Tools for interpreting HMM states
├── streamlit_app.py    # Interactive web dashboard
└── visualization.py    # Plotly visualization utilities
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hmm-regime-detection.git
   cd hmm-regime-detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline
To fetch data, train the model, and generate reports, run:
```bash
python main.py
```

### Launching the Dashboard
To explore results interactively:
```bash
streamlit run streamlit_app.py
```

## Configuration

You can customize the symbols, date ranges, and HMM hyperparameters in `config.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
