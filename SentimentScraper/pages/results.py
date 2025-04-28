import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from db_utils import init_db

# --- Streamlit page config ---
st.set_page_config(page_title="Stored Sentiment Results", page_icon="📋", layout="wide")
st.title("📋 Stored Sentiment Analysis Results")
st.markdown("View and filter results from previous sentiment analysis runs.")

# --- Initialize database ---
init_db()

# --- Helper Functions ---
def load_results() -> pd.DataFrame:
    """Load all sentiment analysis results from SQLite database."""
    try:
        conn = sqlite3.connect('results.db')
        query = "SELECT * FROM sentiment_results"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return pd.DataFrame()

# --- Main App Logic ---
results_df = load_results()

if results_df.empty:
    st.warning("No sentiment analysis results found in the database. Run the sentiment analysis to store results.")
else:
    # Sidebar for filtering
    with st.sidebar:
        st.subheader("Filter Results")
        ticker_filter = st.multiselect(
            "Select Ticker(s)",
            options=sorted(results_df['ticker'].unique()),
            default=[]
        )
        date_from = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
        date_to = st.date_input("To Date", value=datetime.now())

    # Apply filters
    filtered_df = results_df.copy()
    if ticker_filter:
        filtered_df = filtered_df[filtered_df['ticker'].isin(ticker_filter)]
    filtered_df = filtered_df[
        (filtered_df['run_timestamp'].dt.date >= date_from) &
        (filtered_df['run_timestamp'].dt.date <= date_to)
    ]

    if filtered_df.empty:
        st.warning("No results match the selected filters.")
    else:
        # Display results table
        st.subheader("Stored Results")
        display_df = filtered_df[[
            'run_id', 'ticker', 'run_timestamp', 'headline', 'summary',
            'sentiment_score', 'source', 'published_at', 'url'
        ]].rename(columns={
            'run_id': 'Run ID',
            'ticker': 'Ticker',
            'run_timestamp': 'Run Timestamp',
            'headline': 'Headline',
            'summary': 'Summary',
            'sentiment_score': 'Sentiment Score',
            'source': 'Source',
            'published_at': 'Published At',
            'url': 'URL'
        })
        st.dataframe(display_df, use_container_width=True)

        # Visualization: Sentiment trend by ticker
        st.subheader("Sentiment Trend by Ticker")
        if len(ticker_filter) == 1:
            trend_df = filtered_df[filtered_df['ticker'] == ticker_filter[0]]
            fig = px.scatter(
                trend_df,
                x='run_timestamp',
                y='sentiment_score',
                title=f"Sentiment Trend for {ticker_filter[0]}",
                labels={
                    "run_timestamp": "Run Timestamp",
                    "sentiment_score": "Sentiment Score"
                },
                color='sentiment_score',
                color_continuous_scale=['red', 'green'],
                hover_data=['headline', 'summary']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a single ticker in the filter to view the sentiment trend.")

        # Summary statistics
        st.subheader("Summary Statistics")
        avg_sentiment = filtered_df.groupby('ticker')['sentiment_score'].mean().reset_index()
        st.write("Average Sentiment by Ticker:")
        st.dataframe(avg_sentiment.rename(columns={'sentiment_score': 'Average Sentiment'}), use_container_width=True)