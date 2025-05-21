import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import altair as alt

# --- Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")

DATA_FILE = os.path.join(os.getcwd(), 'historical_sentiment.csv')

# --- Helper Functions ---

def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    else:
        return "Negative"

def load_historical_data():
    if os.path.isfile(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            df = df.drop_duplicates(subset=['ticker', 'headline'])
            return df
        except Exception as e:
            st.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
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
    analyzed_df = analyzed_df.drop_duplicates(subset=['ticker', 'headline'])

    if os.path.isfile(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE)
        combined_df = pd.concat([existing_df, analyzed_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['ticker', 'headline'], inplace=True)
    else:
        combined_df = analyzed_df

    combined_df.to_csv(DATA_FILE, index=False)

def load_batch_ticker_data():
    popular_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
    batch_data = {}
    for ticker in popular_tickers:
        session_key = f"analyzed_df_{ticker}"
        if session_key in st.session_state and st.session_state[session_key] is not None:
            batch_data[ticker] = st.session_state[session_key].copy()
    return batch_data

# --- Load Historical Data ---
historical_df = load_historical_data()

if not historical_df.empty:
    historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')
    options = historical_df['ticker'].unique().tolist()
else:
    options = []

# --- Sidebar Filters ---
ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)

date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
date_to = st.sidebar.date_input("To Date", value=datetime.now())

if date_from > date_to:
    st.sidebar.error("Invalid date range: 'From Date' must be before 'To Date'.")

# --- Filter Historical Data ---
filtered_data = historical_df[
    (historical_df['ticker'].isin(ticker_filter)) &
    (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
    (historical_df['publishedAt'] <= pd.to_datetime(date_to))
]

# --- Display Filtered Table ---
st.subheader("📰 Filtered Historical Sentiment Data (Table View)")
if not filtered_data.empty:
    st.dataframe(
        filtered_data[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'source', 'url']]
        .sort_values(by='publishedAt', ascending=False),
        use_container_width=True
    )
else:
    st.warning("No results match the selected filters.")

# --- Download CSV Button ---
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name=f"filtered_sentiment_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)

# --- Save Session Data (if available) ---
if 'analyzed_df' in st.session_state and not st.session_state['analyzed_df'].empty:
    analyzed_df = st.session_state['analyzed_df'].copy()
    analyzed_df = analyzed_df.fillna({
        'headline': 'No headline available',
        'summary': 'No summary available',
        'combined_sentiment': 0.0
    })
    analyzed_df['publishedAt'] = pd.to_datetime(analyzed_df['publishedAt'], errors='coerce')
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(classify_sentiment)
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
    batch_data = load_batch_ticker_data()
    if batch_data:
        st.info(f"Found batch analysis data for {len(batch_data)} tickers: {', '.join(batch_data.keys())}")
        for ticker, ticker_df in batch_data.items():
            ticker_df = ticker_df.fillna({
                'headline': 'No headline available',
                'summary': 'No summary available',
                'combined_sentiment': 0.0
            })
            save_to_csv(ticker_df, ticker)
    else:
        st.info("No new sentiment analysis this session. Only historical data will be used for the chart.")

# --- Sentiment Chart ---
combined_data = filtered_data.copy()

if 'analyzed_df' in st.session_state and not st.session_state['analyzed_df'].empty:
    ticker_symbol = st.session_state.get('ticker', '')
    if ticker_symbol:
        session_df = st.session_state['analyzed_df'].copy()
        session_df['ticker'] = ticker_symbol
        session_df['publishedAt'] = pd.to_datetime(session_df['publishedAt'], errors='coerce')
        combined_data = pd.concat([filtered_data, session_df], ignore_index=True)

if not combined_data.empty:
    combined_data['date'] = pd.to_datetime(combined_data['publishedAt'], errors='coerce').dt.date
    combined_data['combined_sentiment'] = pd.to_numeric(combined_data['combined_sentiment'], errors='coerce').fillna(0)

    daily_sentiment = combined_data.groupby(['date', 'ticker']).agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()

    # --- Chart 1: Sentiment by Ticker and Date ---
    st.subheader("📅 Sentiment by Date (All Tickers on the Same Chart)")

    base = alt.Chart(daily_sentiment).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('avg_sentiment:Q', title='Average Sentiment Score'),
        color=alt.Color('ticker:N', title='Ticker')
    )

    lines = base.mark_line()
    points = base.mark_circle(size=60).encode(
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('ticker:N', title='Ticker'),
            alt.Tooltip('avg_sentiment:Q', title='Avg Sentiment'),
            alt.Tooltip('article_count:Q', title='Articles')
        ]
    )

    chart = (lines + points).interactive().properties(
        width='container',
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

    # --- Chart 2: Average Sentiment of 6 Benchmark Tickers ---
    benchmark_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
    benchmark_data = combined_data[combined_data['ticker'].isin(benchmark_tickers)]

    if not benchmark_data.empty:
        avg_benchmark_sentiment = benchmark_data.groupby('date').agg(
            avg_sentiment=('combined_sentiment', 'mean'),
            article_count=('headline', 'count')
        ).reset_index()

        st.subheader("📈 Average Sentiment Across 6 Tickers (SPY, AAPL, MSFT, NVDA, AMZN, META)")

        avg_chart = alt.Chart(avg_benchmark_sentiment).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('avg_sentiment:Q', title='Average Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
            tooltip=[
                alt.Tooltip('date:T', title='Date'),
                alt.Tooltip('avg_sentiment:Q', title='Avg Sentiment'),
                alt.Tooltip('article_count:Q', title='Total Articles')
            ]
        ).interactive().properties(
            width='container',
            height=350
        )

        st.altair_chart(avg_chart, use_container_width=True)
    else:
        st.info("No data available for the benchmark tickers (SPY, AAPL, MSFT, NVDA, AMZN, META).")
else:
    st.warning("No data available for sentiment chart.")
