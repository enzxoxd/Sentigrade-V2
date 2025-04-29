import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")

# --- Functions ---
def load_historical_data():
    if os.path.isfile('historical_sentiment.csv'):
        return pd.read_csv('historical_sentiment.csv')
    else:
        return pd.DataFrame()

def save_to_csv(analyzed_df, ticker_symbol):
    analyzed_df = analyzed_df.fillna({
        'headline': 'No headline available',
        'summary': 'No summary available',
        'combined_sentiment': 0.0
    })
    analyzed_df['publishedAt'] = pd.to_datetime(analyzed_df['publishedAt'], errors='coerce')
    analyzed_df['ticker'] = ticker_symbol
    file_name = 'historical_sentiment.csv'
    file_exists = os.path.isfile(file_name)
    analyzed_df.to_csv(file_name, mode='a', header=not file_exists, index=False)

def get_filtered_data(historical_df, ticker_filter, date_from, date_to):
    if 'publishedAt' in historical_df.columns:
        historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')
    return historical_df[
        (historical_df['ticker'].isin(ticker_filter)) &
        (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
        (historical_df['publishedAt'] <= pd.to_datetime(date_to))
    ]

# --- Handle session sentiment data ---
if 'analyzed_df' in st.session_state:
    analyzed_df = st.session_state['analyzed_df']
    ticker_symbol = st.session_state.get('ticker', '')

    if not ticker_symbol:
        ticker_symbol = st.text_input("Enter ticker symbol:", placeholder="e.g., AAPL")
        if ticker_symbol:
            st.session_state['ticker'] = ticker_symbol
        st.warning("No ticker found from main app. Please enter a ticker symbol above.")

    if not analyzed_df.empty and ticker_symbol:
        save_to_csv(analyzed_df, ticker_symbol)

# --- Reload historical data after saving new session data ---
historical_df = load_historical_data()
options = historical_df['ticker'].unique().tolist() if not historical_df.empty else []

# --- Sidebar filters ---
st.sidebar.header("🔍 Filter Historical Data")
ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
date_to = st.sidebar.date_input("To Date", value=datetime.now())

# --- Filter and display historical data ---
filtered_data = get_filtered_data(historical_df, ticker_filter, date_from, date_to)
# --- Filter and display historical data in pivoted table ---
# --- Display filtered results as a simple table ---
st.subheader("📰 Filtered Historical Sentiment Data (Table View)")

if not filtered_data.empty:
    # Format datetime concisely
    filtered_data['publishedAt'] = pd.to_datetime(filtered_data['publishedAt'], errors='coerce')
    filtered_data['publishedAt'] = filtered_data['publishedAt'].dt.strftime('%Y-%m-%d %H:%M')

    # Reset index and add row numbers starting from 1
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data.index += 1  # So first row starts at 1
    filtered_data.rename_axis("N", inplace=True)

    # Select and rename columns for clarity
    display_df = filtered_data[[
        'publishedAt', 'ticker', 'headline', 'combined_sentiment', 'source', 'url'
    ]].rename(columns={
        'publishedAt': 'Published At',
        'ticker': 'Ticker',
        'headline': 'Headline',
        'combined_sentiment': 'Sentiment Score',
        'source': 'Source',
        'url': 'URL'
    })

    st.dataframe(display_df, use_container_width=True)
else:
    st.warning("No results match the selected filters.")



# --- Download filtered data ---
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name=f"filtered_sentiment_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)

# --- Display current session data ---
if 'analyzed_df' in st.session_state and not st.session_state['analyzed_df'].empty:
    analyzed_df = st.session_state['analyzed_df']
    analyzed_df = analyzed_df.fillna({
        'headline': 'No headline available',
        'summary': 'No summary available',
        'combined_sentiment': 0.0
    })

    # Categorize sentiment
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
    )

    # Display current session in table view
    st.subheader("📈 Current Session Sentiment Analysis Results (Table View)")
    display_cols = ['publishedAt', 'headline', 'combined_sentiment', 'sentiment_category', 'source', 'url']
    st.dataframe(analyzed_df[display_cols].sort_values(by='publishedAt', ascending=False), use_container_width=True)

    # Average sentiment metric
    avg_sentiment = analyzed_df['combined_sentiment'].mean()
    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    # Sentiment over time
    sentiment_by_date = analyzed_df.copy()
    sentiment_by_date['date'] = pd.to_datetime(sentiment_by_date['publishedAt'], errors='coerce').dt.normalize()
    daily_sentiment = sentiment_by_date.groupby('date').agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()
    daily_sentiment.set_index('date', inplace=True)

    st.subheader("📅 Sentiment by Date")
    st.line_chart(daily_sentiment['avg_sentiment'])
