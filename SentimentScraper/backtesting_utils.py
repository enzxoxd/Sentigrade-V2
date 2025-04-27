# backtesting_utils.py
import yfinance as yf
import ta
import vectorbt as vbt
import numpy as np
import pandas as pd
import logging
import streamlit as st
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def fetch_stock_prices(ticker, start_date, end_date):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
    end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date

    try:
        stock_price_df = yf.download(ticker, start=start_date_str, end=end_date_str)
        if stock_price_df.empty:
            st.error(f"No stock price data found for ticker {ticker} between {start_date_str} and {end_date_str}.")
            return None
        return stock_price_df
    except Exception as e:
        st.error(f"Failed to fetch stock prices for {ticker}: {e}")
        return None

def align_sentiment_with_prices(stock_price_df, sentiment_df):
    """Align sentiment scores with historical stock prices."""
    if stock_price_df is None or sentiment_df is None or sentiment_df.empty:
        logger.warning("No stock price or sentiment data to align.")
        return pd.DataFrame()

    # Ensure 'publishedAt' is in sentiment_df and contains datetime objects
    if 'publishedAt' not in sentiment_df.columns:
        logger.error("Missing 'publishedAt' column in sentiment_df.")
        return pd.DataFrame()

    # Convert 'publishedAt' to datetime, handling errors
    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'], errors='coerce')
    sentiment_df.dropna(subset=['publishedAt'], inplace=True)  # Drop rows with NaT

    # Verify that stock_price_df index is datetime
    if not isinstance(stock_price_df.index, pd.DatetimeIndex):
        stock_price_df.index = pd.to_datetime(stock_price_df.index)

    # Aggregate sentiment scores by date
    daily_sentiment = sentiment_df.groupby(sentiment_df['publishedAt'].dt.date)['combined_sentiment'].mean().reset_index()
    daily_sentiment['publishedAt'] = pd.to_datetime(daily_sentiment['publishedAt'])  # Ensure datetime

    # Merge price data with sentiment data
    aligned_df = pd.merge(stock_price_df, daily_sentiment, left_index=True, right_on='publishedAt', how='left')
    aligned_df.set_index(stock_price_df.index, inplace=True)
    aligned_df.rename(columns={'combined_sentiment': 'avg_sentiment'}, inplace=True)

    # Fill missing sentiment scores with 0 (neutral)
    aligned_df['avg_sentiment'].fillna(0, inplace=True)

    return aligned_df

def calculate_positions(aligned_df, fast_ma_window, slow_ma_window, sentiment_threshold):
    """Calculate trading positions based on moving averages and sentiment."""
    # Moving Averages
    aligned_df['fast_ma'] = ta.trend.sma_indicator(aligned_df['Close'], window=fast_ma_window)
    aligned_df['slow_ma'] = ta.trend.sma_indicator(aligned_df['Close'], window=slow_ma_window)

    # Generate signals
    aligned_df['long_signal'] = (
        (aligned_df['fast_ma'] > aligned_df['slow_ma']) & (aligned_df['avg_sentiment'] > sentiment_threshold)
    )
    aligned_df['short_signal'] = (
        (aligned_df['fast_ma'] < aligned_df['slow_ma']) & (aligned_df['avg_sentiment'] < -sentiment_threshold)
    )

    # Combine signals to create positions
    aligned_df['positions'] = 0
    aligned_df['positions'] = np.where(aligned_df['long_signal'], 1, aligned_df['positions'])
    aligned_df['positions'] = np.where(aligned_df['short_signal'], -1, aligned_df['positions'])

    return aligned_df

def run_backtest(aligned_df, initial_capital, risk_per_trade):
    """Run a backtest using VectorBT."""
    close = aligned_df['Close']
    positions = aligned_df['positions']
    
    # Handle missing data
    if close.isnull().any() or positions.isnull().any():
        logger.error("Missing data in close prices or positions. Ensure data is complete.")
        st.error("Backtest cannot run with missing data. Please check the data and parameters.")
        return None

    try:
        # Convert positions to integer type expected by VectorBT
        positions = positions.astype(int)
        
        # Create a VectorBT portfolio
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=positions == 1,  # Long positions
            exits=positions == -1,  # Short positions
            init_cash=initial_capital,
            risk_per_trade=risk_per_trade / 100,
            fees=0.001,  # Example fee
            freq='D'  # Daily frequency
        )
        return pf
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        st.error(f"Backtest failed: {e}")
        return None
