import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
import yfinance as yf
import altair as alt

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")

# --- Load Historical Sentiment Data ---
def load_historical_data(file_path='historical_sentiment.csv'):
    """Load historical sentiment data from a CSV file."""
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# --- Clean Ticker Symbol ---
def clean_ticker(ticker):
    """Remove any leading special characters like $ from tickers."""
    return str(ticker).strip().lstrip('$').upper()

# --- Fetch Stock Closing Price from Yahoo Finance ---
def fetch_stock_data(ticker, date):
    """Fetch stock closing price for the given ticker and date."""
    try:
        data = yf.download(ticker, start=date, end=date + timedelta(days=1), progress=False, actions=False)
        if not data.empty and 'Close' in data.columns:
            closing_price = data['Close'].iloc[0]
            return pd.DataFrame({'ticker': [ticker], 'published_date': [date], 'closing_price': [closing_price]})
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- Add Closing Prices to Sentiment Data ---
def add_closing_prices(df):
    """Add closing price for each row based on ticker and published date."""
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['published_date'] = df['publishedAt'].dt.date
    df['ticker'] = df['ticker'].apply(clean_ticker)

    closing_prices = []

    for ticker in df['ticker'].dropna().unique():
        sub_df = df[df['ticker'] == ticker]
        for date_item in sub_df['published_date'].unique():
            price_df = fetch_stock_data(ticker, date_item)
            if not price_df.empty:
                closing_prices.append(price_df)

    if closing_prices:
        all_prices = pd.concat(closing_prices, ignore_index=True)
        merged = pd.merge(df, all_prices, how='left', on=['ticker', 'published_date'])

        # Remove rows with missing price
        merged = merged.dropna(subset=['closing_price'])

        # Format closing price and drop helper column
        merged['closing_price'] = merged['closing_price'].astype(float)
        return merged.drop(columns=['published_date'])

    df['closing_price'] = None
    return df

# --- Filter Historical Sentiment Data ---
def filter_sentiment_data(historical_df, ticker_filter, date_from, date_to):
    """Filter the sentiment data based on tickers and date range."""
    return historical_df[
        (historical_df['ticker'].isin(ticker_filter)) &
        (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
        (historical_df['publishedAt'] <= pd.to_datetime(date_to))
    ]

# --- Rebase Prices to 100 as of a Reference Date ---
def rebase_prices(df, base_date=date(2025, 5, 19)):
    base_prices = df[df['publishedAt'] == base_date][['ticker', 'closing_price']].drop_duplicates()
    base_prices = base_prices.rename(columns={'closing_price': 'base_price'})
    rebased_df = pd.merge(df, base_prices, on='ticker', how='left')
    rebased_df = rebased_df.dropna(subset=['base_price'])
    rebased_df['rebased_price'] = (rebased_df['closing_price'] / rebased_df['base_price']) * 100
    return rebased_df

# --- Main Streamlit App ---
def main():
    historical_df = load_historical_data()

    if not historical_df.empty:
        historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')
        historical_df['ticker'] = historical_df['ticker'].apply(clean_ticker)

        # Drop rows with NaN in key fields
        historical_df = historical_df.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment'])

        # Remove duplicates: same headline for same ticker
        historical_df = historical_df.drop_duplicates(subset=['ticker', 'headline'])

        options = historical_df['ticker'].dropna().unique().tolist()
    else:
        options = []

    # Sidebar filters
    ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
    date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
    date_to = st.sidebar.date_input("To Date", value=datetime.now())

    # Filter and enrich sentiment data
    filtered_data = filter_sentiment_data(historical_df, ticker_filter, date_from, date_to)

    if not filtered_data.empty:
        filtered_data = add_closing_prices(filtered_data)

        # Drop rows with remaining NaN values in important fields
        filtered_data = filtered_data.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment', 'closing_price'])

        # Ensure numeric types for plotting
        filtered_data['closing_price'] = filtered_data['closing_price'].astype(float)
        filtered_data['combined_sentiment'] = filtered_data['combined_sentiment'].astype(float)

        # Convert publishedAt to just date for display and charting
        filtered_data['publishedAt'] = filtered_data['publishedAt'].dt.date

        # Rebase prices to 100 from base date
        filtered_data = rebase_prices(filtered_data)

    # Display results
    st.subheader("📰 Filtered Sentiment Data with Closing Price")
    if not filtered_data.empty:
        st.dataframe(
            filtered_data[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'closing_price', 'rebased_price', 'source', 'url']].sort_values(by='publishedAt', ascending=False),
            use_container_width=True
        )

        # --- Combined Altair Chart ---
        st.subheader("📈 Chart: Rebased Price (100 on 2025-05-19) and Sentiment")

        base = alt.Chart(filtered_data).encode(x='publishedAt:T', color='ticker:N')

        sentiment_line = base.mark_line(strokeDash=[5, 5]).encode(
            y=alt.Y('combined_sentiment:Q', axis=alt.Axis(title='Sentiment Score')),
            tooltip=['publishedAt', 'ticker', 'combined_sentiment']
        )

        rebased_price_line = base.mark_line().encode(
            y=alt.Y('rebased_price:Q', axis=alt.Axis(title='Price (Rebased to 100)')),
            tooltip=['publishedAt', 'ticker', 'rebased_price']
        )

        chart = alt.layer(rebased_price_line, sentiment_line).resolve_scale(y='independent').properties(
            width='container',
            height=400,
            title="Rebased Price vs Sentiment Score"
        )

        st.altair_chart(chart, use_container_width=True)

        # --- Download Button ---
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name=f"filtered_sentiment_with_prices_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No results match the selected filters.")


if __name__ == "__main__":
    main()
