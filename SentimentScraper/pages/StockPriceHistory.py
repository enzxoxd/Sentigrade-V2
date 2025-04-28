import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="Stock Price History", page_icon="📈")
st.title("📈 Stock Price History")

# Access ticker from session state with a safe default
ticker_symbol = st.session_state.get('ticker', '')

# If ticker is not in session state, provide a way to input it
if not ticker_symbol:
    ticker_symbol = st.text_input("Enter ticker symbol:", placeholder="e.g., AAPL")
    st.warning("No ticker found from main app. Please enter a ticker symbol above.")


if ticker_symbol:
    today = date.today()
    past_date = today - timedelta(days=7)

    try:
        data = yf.download(ticker_symbol, start=past_date, end=today)

        if data.empty:
            st.error("No data found for the specified ticker symbol.")
        else:
            st.subheader(f"Closing Prices for {ticker_symbol} (Last 7 Days)")
            df = pd.DataFrame(data['Close'])
            df.index = df.index.strftime('%Y-%m-%d')
            st.dataframe(df)

            st.line_chart(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a ticker symbol to view stock price history.")
