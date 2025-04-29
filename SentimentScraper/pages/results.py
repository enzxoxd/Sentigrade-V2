import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")

# Function to load historical sentiment data
def load_historical_data():
    if os.path.isfile('historical_sentiment.csv'):
        return pd.read_csv('historical_sentiment.csv')
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no historical data is found

# Function to save current session data to the CSV
def save_to_csv(analyzed_df, ticker_symbol):
    # Ensure no NaN values in critical columns like 'headline', 'summary', 'combined_sentiment'
    analyzed_df = analyzed_df.fillna({'headline': 'No headline available', 'summary': 'No summary available', 'combined_sentiment': 0.0})
    
    analyzed_df['publishedAt'] = pd.to_datetime(analyzed_df['publishedAt'], errors='coerce')
    file_name = 'historical_sentiment.csv'
    
    # Check if file exists to decide whether to add headers or not
    file_exists = os.path.isfile(file_name)
    analyzed_df['ticker'] = ticker_symbol  # Add ticker symbol to the data

    # Append the data to the CSV file
    analyzed_df.to_csv(file_name, mode='a', header=not file_exists, index=False)

# --- Sidebar for filtering ---
# Load historical sentiment data from CSV file
historical_df = load_historical_data()

# Get the tickers from the historical data or set to empty if none available
if not historical_df.empty:
    options = historical_df['ticker'].unique().tolist()
else:
    options = []

# Sidebar elements to allow the user to select tickers and date range
ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
date_to = st.sidebar.date_input("To Date", value=datetime.now())

# --- Ensure the 'publishedAt' column is in datetime format for filtering ---
if 'publishedAt' in historical_df.columns:
    historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')

# Filter historical data based on user input for ticker and date range
filtered_data = historical_df[
    (historical_df['ticker'].isin(ticker_filter)) &
    (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
    (historical_df['publishedAt'] <= pd.to_datetime(date_to))
]

# Display filtered results
st.subheader("📰 Headlines & Sentiment Scores")
if not filtered_data.empty:
    for _, row in filtered_data.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row['summary']}*")
        st.caption(f"Source: {row['source']} | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()
else:
    st.warning("No results match the selected filters.")

# Display a download button for the filtered historical data
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name=f"filtered_sentiment_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)

# --- Fetch and Analyze Sentiment in Real-Time ---
if 'analyzed_df' not in st.session_state:
    st.warning("No sentiment data found. Please run sentiment analysis first.")
else:
    analyzed_df = st.session_state['analyzed_df']

    # Ensure no NaN values are present in sentiment data
    analyzed_df = analyzed_df.fillna({'headline': 'No headline available', 'summary': 'No summary available', 'combined_sentiment': 0.0})

    # Access ticker from session state with a safe default
    ticker_symbol = st.session_state.get('ticker', '')

    # If ticker is not in session state, provide a way to input it
    if not ticker_symbol:
        ticker_symbol = st.text_input("Enter ticker symbol:", placeholder="e.g., AAPL")
        if ticker_symbol:
            st.session_state['ticker'] = ticker_symbol  # Store the ticker in the session state
        st.warning("No ticker found from main app. Please enter a ticker symbol above.")

    # Ensure that sentiment data is valid before saving to CSV
    if not analyzed_df.empty:
        # Save the current session data to the historical CSV
        save_to_csv(analyzed_df, ticker_symbol)

        # --- Display the current session's results ---
        st.subheader("📈 Current Session Sentiment Analysis Results")
        if not analyzed_df.empty:
            for _, row in analyzed_df.iterrows():
                st.markdown(f"**[{row['headline']}]({row['url']})**")
                st.markdown(f"*{row['summary']}*")
                st.caption(f"Source: {row['source']} | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
                st.divider()

    else:
        st.warning("No sentiment data available to save or display.")

    # --- Display metrics and sentiment analysis by date ---
    avg_sentiment = analyzed_df['combined_sentiment'].mean()
    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # Categorize sentiment
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
    )

    # Display sentiment by date
    sentiment_by_date = analyzed_df.copy()
    sentiment_by_date['date'] = pd.to_datetime(sentiment_by_date['publishedAt'], errors='coerce').dt.normalize()
    daily_sentiment = sentiment_by_date.groupby('date').agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()
    daily_sentiment.set_index('date', inplace=True)

    st.subheader("📅 Sentiment by Date")
    st.line_chart(daily_sentiment['avg_sentiment'])
