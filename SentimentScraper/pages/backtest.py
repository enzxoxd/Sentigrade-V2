# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import date, timedelta

# # Set up the page
# st.set_page_config(page_title="Backtesting Strategy", page_icon="📊")
# st.title("📊 Stock Strategy Backtesting")

# # Access ticker from session state with a safe default
# ticker_symbol = st.session_state.get('ticker', '')

# # If ticker is not in session state, provide a way to input it
# if not ticker_symbol:
#     ticker_symbol = st.text_input("Enter ticker symbol:", placeholder="e.g., AAPL")
#     st.warning("No ticker found from main app. Please enter a ticker symbol above.")

# # Allow users to select a custom date range for backtesting
# if ticker_symbol:
#     today = date.today()
#     start_date = st.date_input("Select start date for backtesting", today - timedelta(days=365))
#     end_date = st.date_input("Select end date for backtesting", today)

#     # Ensure the start date is before the end date
#     if start_date > end_date:
#         st.error("Start date must be before the end date.")
#     else:
#         try:
#             # Fetch data for backtesting
#             data = yf.download(ticker_symbol, start=start_date, end=end_date)

#             if data.empty:
#                 st.error("No data found for the specified ticker symbol.")
#             else:
#                 # Add dropdown for selecting the strategy
#                 strategy = st.selectbox("Select Strategy", ("SMA Crossover", "EMA Crossover", "RSI Strategy", "Bollinger Bands"))

#                 # Setting up different strategies and corresponding sliders
#                 if strategy == "SMA Crossover":
#                     short_window = st.slider("Short Window (days)", min_value=5, max_value=100, value=20)
#                     long_window = st.slider("Long Window (days)", min_value=50, max_value=200, value=50)

#                     # Calculate moving averages
#                     data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
#                     data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

#                     # Generate trading signals: 1 when short MA > long MA
#                     data['Signal'] = 0
#                     data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1
#                     data['Position'] = data['Signal'].diff()

#                 elif strategy == "EMA Crossover":
#                     short_window = st.slider("Short Window (days)", min_value=5, max_value=100, value=20)
#                     long_window = st.slider("Long Window (days)", min_value=50, max_value=200, value=50)

#                     # Calculate exponential moving averages
#                     data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
#                     data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()

#                     # Generate trading signals: 1 when short EMA > long EMA
#                     data['Signal'] = 0
#                     data.loc[data['Short_EMA'] > data['Long_EMA'], 'Signal'] = 1
#                     data['Position'] = data['Signal'].diff()

#                 elif strategy == "RSI Strategy":
#                     rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14)
#                     rsi_lower = st.slider("RSI Lower Threshold", min_value=10, max_value=40, value=30)
#                     rsi_upper = st.slider("RSI Upper Threshold", min_value=60, max_value=90, value=70)

#                     # Calculate RSI
#                     delta = data['Close'].diff()
#                     gain = delta.where(delta > 0, 0)
#                     loss = -delta.where(delta < 0, 0)

#                     avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
#                     avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

#                     rs = avg_gain / avg_loss
#                     data['RSI'] = 100 - (100 / (1 + rs))

#                     # Generate trading signals based on RSI
#                     data['Signal'] = 0
#                     data.loc[data['RSI'] < rsi_lower, 'Signal'] = 1  # Buy signal
#                     data.loc[data['RSI'] > rsi_upper, 'Signal'] = -1  # Sell signal

#                 elif strategy == "Bollinger Bands":
#                     bb_window = st.slider("Bollinger Bands Window", min_value=10, max_value=50, value=20)
#                     bb_std_dev = st.slider("Bollinger Bands Std Dev", min_value=1, max_value=3, value=2)

#                     # Calculate Bollinger Bands
#                     data['Rolling_Mean'] = data['Close'].rolling(window=bb_window).mean()
#                     data['Rolling_Std'] = data['Close'].rolling(window=bb_window).std()

#                     data['Upper_Band'] = data['Rolling_Mean'] + (data['Rolling_Std'] * bb_std_dev)
#                     data['Lower_Band'] = data['Rolling_Mean'] - (data['Rolling_Std'] * bb_std_dev)

#                     # Generate trading signals
#                     data['Signal'] = 0
#                     data.loc[data['Close'] < data['Lower_Band'], 'Signal'] = 1  # Buy signal
#                     data.loc[data['Close'] > data['Upper_Band'], 'Signal'] = -1  # Sell signal

#                 # Performance metrics
#                 data['Daily_Return'] = data['Close'].pct_change()
#                 data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)

#                 # Cumulative returns for market and strategy
#                 data['Cumulative_Market_Returns'] = (1 + data['Daily_Return']).cumprod() - 1
#                 data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Return']).cumprod() - 1

#                 # Display results
#                 st.subheader(f"Backtest Results for {ticker_symbol}")
#                 st.dataframe(data[['Close', 'Signal', 'Cumulative_Market_Returns', 'Cumulative_Strategy_Returns']])

#                 # Calculate and display the final PnL
#                 final_market_pnl = data['Cumulative_Market_Returns'].iloc[-1]
#                 final_strategy_pnl = data['Cumulative_Strategy_Returns'].iloc[-1]
#                 st.write(f"Final Market PnL: {final_market_pnl * 100:.2f}%")
#                 st.write(f"Final Strategy PnL: {final_strategy_pnl * 100:.2f}%")

#                 # Plot the cumulative returns
#                 st.subheader("Cumulative Returns: Market vs Strategy")

#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.plot(data['Cumulative_Market_Returns'], label='Market Returns')
#                 ax.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns')

#                 ax.set_title(f"{ticker_symbol} Cumulative Returns")
#                 ax.legend()

#                 # Pass the figure to st.pyplot()
#                 st.pyplot(fig)

#         except Exception as e:
#             st.error(f"An error occurred: {e}")
