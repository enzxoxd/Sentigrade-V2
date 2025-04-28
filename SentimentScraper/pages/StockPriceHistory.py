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

# Allow users to select a custom date range
if ticker_symbol:
    today = date.today()
    start_date = st.date_input("Select start date", today - timedelta(days=30))
    end_date = st.date_input("Select end date", today)

    # Ensure the start date is before the end date
    if start_date > end_date:
        st.error("Start date must be before the end date.")
    else:
        try:
            data = yf.download(ticker_symbol, start=start_date, end=end_date)

            if data.empty:
                st.error("No data found for the specified ticker symbol.")
            else:
                st.subheader(f"Stock Data for {ticker_symbol} from {start_date} to {end_date}")
                df = pd.DataFrame(data[['Open', 'High', 'Low', 'Close', 'Volume']])
                df.index = df.index.strftime('%Y-%m-%d')
                st.dataframe(df)

                # Display line chart for Closing prices
                st.line_chart(df['Close'])

                # Optionally, show other charts (like Open, High, Low, or Volume)
                chart_type = st.selectbox("Select chart type", ["Close", "Volume", "Open", "High", "Low"])
                st.line_chart(df[chart_type])

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a ticker symbol to view stock price history.")
