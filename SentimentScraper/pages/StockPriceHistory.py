import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="Stock Price History", page_icon="📈")
st.title("📈 Stock Price History")

# Access ticker from session state
ticker_symbol = st.session_state.get('ticker', 'AAPL')  # Default to AAPL if not found

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
    st.warning("Please enter a ticker symbol in the main app.")
