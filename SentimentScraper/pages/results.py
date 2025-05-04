import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import altair as alt

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")

# Function to load historical sentiment data
def load_historical_data():
    if os.path.isfile('historical_sentiment.csv'):
        return pd.read_csv('historical_sentiment.csv')
    else:
        return pd.DataFrame()

# Function to save current session data to the CSV
def save_to_csv(analyzed_df, ticker_symbol):
    analyzed_df = analyzed_df.fillna({
        'headline': 'No headline available',
        'summary': 'No summary available',
        'combined_sentiment': 0.0
    })
    analyzed_df['publishedAt'] = pd.to_datetime(analyzed_df['publishedAt'], errors='coerce')
    analyzed_df['ticker'] = ticker_symbol
    file_name = 'historical_sentiment.csv'
    analyzed_df.to_csv(file_name, mode='a', header=not os.path.isfile(file_name), index=False)

# --- Load and Filter Historical Data ---
historical_df = load_historical_data()

if not historical_df.empty:
    historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')
    options = historical_df['ticker'].unique().tolist()
else:
    options = []

ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
date_to = st.sidebar.date_input("To Date", value=datetime.now())

# Filter data
filtered_data = historical_df[
    (historical_df['ticker'].isin(ticker_filter)) &
    (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
    (historical_df['publishedAt'] <= pd.to_datetime(date_to))
]

# Display table with the ticker column
st.subheader("📰 Filtered Historical Sentiment Data (Table View)")
if not filtered_data.empty:
    st.dataframe(
        filtered_data[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'source', 'url']]
        .sort_values(by='publishedAt', ascending=False),
        use_container_width=True
    )
else:
    st.warning("No results match the selected filters.")


# Download button
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name=f"filtered_sentiment_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)

# --- Save Current Session Sentiment Data if Available ---
if 'analyzed_df' in st.session_state and not st.session_state['analyzed_df'].empty:
    analyzed_df = st.session_state['analyzed_df'].copy()
    analyzed_df = analyzed_df.fillna({
        'headline': 'No headline available',
        'summary': 'No summary available',
        'combined_sentiment': 0.0
    })
    analyzed_df['publishedAt'] = pd.to_datetime(analyzed_df['publishedAt'], errors='coerce')
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
    )
    ticker_symbol = st.session_state.get('ticker', '')
    if ticker_symbol:
        save_to_csv(analyzed_df, ticker_symbol)
        st.subheader("📈 Current Session Sentiment Analysis Results (Table View)")
        st.dataframe(
            analyzed_df[['publishedAt', 'headline', 'combined_sentiment', 'sentiment_category', 'source', 'url']]
            .sort_values(by='publishedAt', ascending=False),
            use_container_width=True
        )
        avg_sentiment = analyzed_df['combined_sentiment'].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
    else:
        st.warning("No ticker symbol found for saving current session data.")
else:
    st.info("No new sentiment analysis this session. Only historical data will be used for the chart.")

# --- Sentiment Chart: Based on Historical and/or Current Data ---
# Recombine for plotting (filtered historical + optional session)
combined_data = filtered_data.copy()
if 'analyzed_df' in st.session_state and not st.session_state['analyzed_df'].empty:
    ticker_symbol = st.session_state.get('ticker', '')
    if ticker_symbol:
        session_df = st.session_state['analyzed_df'].copy()
        session_df['ticker'] = ticker_symbol
        session_df['publishedAt'] = pd.to_datetime(session_df['publishedAt'], errors='coerce')
        combined_data = pd.concat([filtered_data, session_df], ignore_index=True)

# Prepare chart data
if not combined_data.empty:
    combined_data['date'] = pd.to_datetime(combined_data['publishedAt'], errors='coerce').dt.date
    combined_data['combined_sentiment'] = pd.to_numeric(combined_data['combined_sentiment'], errors='coerce').fillna(0)

    daily_sentiment = combined_data.groupby(['date', 'ticker']).agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()

    sentiment_pivot = daily_sentiment.pivot(index='date', columns='ticker', values='avg_sentiment')
    sentiment_pivot.sort_index(inplace=True)

    # Altair interactive line chart with points and hover
    st.subheader("📅 Sentiment by Date (All Tickers on the Same Chart)")

    # Create base chart
    base = alt.Chart(daily_sentiment).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('avg_sentiment:Q', title='Average Sentiment Score'),
        color=alt.Color('ticker:N', title='Ticker')
    )

    # Line chart
    lines = base.mark_line()

    # Points on hover
    points = base.mark_circle(size=60).encode(
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('ticker:N', title='Ticker'),
            alt.Tooltip('avg_sentiment:Q', title='Avg Sentiment'),
            alt.Tooltip('article_count:Q', title='Articles')
        ]
    )

    # Layer and interactive
    chart = (lines + points).interactive().properties(
        width='container',
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.warning("No data available for sentiment chart.")
